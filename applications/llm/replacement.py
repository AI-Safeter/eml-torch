"""Explicit local contribution replacement using upstream-only predictors."""

import json

import torch
import torch.nn as nn
from common import OUT
from fit_equations import design

from emltorch._ast import _EML, _Add, _Const, _Div, _Exp, _Mul, _parse_inner, _Sub, _Var
from emltorch.operator import safe_eml


def emit(node):
    if isinstance(node, _Const):
        return repr(node.value)
    if isinstance(node, _Var):
        assert node.name in ["z0", "z1", "z2"]
        return node.name
    if isinstance(node, _EML):
        return f"E({emit(node.left)}, {emit(node.right)}, z0)"
    if isinstance(node, _Exp):
        return f"torch.exp({emit(node.arg)})"
    return (
        f"({emit(node.left)} "
        + {_Add: "+", _Sub: "-", _Mul: "*", _Div: "/"}[type(node)]
        + f" {emit(node.right)})"
    )


def E(a, b, ref):
    if not isinstance(a, torch.Tensor):
        a = torch.full_like(ref, a)
    if not isinstance(b, torch.Tensor):
        b = torch.full_like(ref, b)
    return safe_eml(a, b)


class Replacement:
    def __init__(self):
        self.component = torch.load(OUT / "component.pt", weights_only=True)
        self.state = torch.load(OUT / "replacements.pt", weights_only=True)
        self.layer = self.component["layer"]
        self.d = self.component["direction"].cuda()
        self.random_d = self.component["random_direction"].cuda()
        self.normsq = self.d.double().square().sum().float()
        self.random_d -= self.d * (self.random_d.double() @ self.d.double()).float() / self.normsq
        self.random_d /= self.random_d.norm()
        self.mean = self.component["input_mean"].cuda()
        self.encoder = self.component["encoder"].cuda()
        for k in ["zmean", "zstd", "ymean", "ystd"]:
            self.state[k] = self.state[k].cuda()
        self.baselines = {k: v.cuda() for k, v in self.state["baselines"].items()}
        self.network = (
            nn.Sequential(
                nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1)
            )
            .cuda()
            .eval()
        )
        self.network.load_state_dict(self.state["network"])
        info = json.loads((OUT / "fits.json").read_text())
        self.formulas = {}
        for mode, fit in info["selected"].items():
            scope = {"torch": torch, "E": E}
            source = (
                "def f(z):\n    z0,z1,z2 = z.unbind(1)\n    return "
                + emit(_parse_inner(fit["expression"]))
                + "\n"
            )
            exec(compile(source, "<EML-expression>", "exec"), scope)
            self.formulas["eml_" + mode] = scope["f"]

    def coordinates(self, h):
        return (((h - self.mean) @ self.encoder) - self.state["zmean"]) / self.state["zstd"]

    def coefficient(self, h, kind):
        z = self.coordinates(h)
        if kind in self.formulas:
            y = self.formulas[kind](z)
        elif kind == "network":
            y = self.network(z).squeeze(1)
        else:
            y = design(z, kind) @ self.baselines[kind]
        return y * self.state["ystd"] + self.state["ymean"]

    def hooks(self, model, kind, mixed_input=None):
        cache = {}
        state = {"active": False}

        def pre(module, args):
            h = args[0]
            state["active"] = h.shape[1] > 1
            if not state["active"]:
                return
            cache["input"] = (h[:, -1, :] if mixed_input is None else mixed_input).detach()
            return (h,)

        def post(module, args, out):
            if not state["active"]:
                return out
            h = cache["input"]
            base_c = out[:, -1, :] @ self.d
            # Direct forward avoids recursively invoking this module's hooks.
            c = base_c if mixed_input is None else module.forward(mixed_input) @ self.d
            cache["true_coefficient"] = c.detach()
            if kind in ["original", "restore"]:
                pred = c
            elif kind in ["mean", "matched_random"]:
                pred = torch.full_like(c, self.component["component_mean"])
            elif kind == "zero":
                pred = torch.zeros_like(c)
            elif kind == "shuffle":
                pred = c.roll(1)
            else:
                pred = self.coefficient(h, kind)
            cache["predicted_coefficient"] = pred.detach()
            if kind == "original" and mixed_input is None:
                return out
            result = out.clone()
            direction = (
                self.random_d / self.normsq.sqrt()
                if kind == "matched_random"
                else self.d / self.normsq
            )
            delta = pred - base_c
            if kind == "matched_random" and mixed_input is not None:
                delta = c - base_c
            result[:, -1, :] += delta[:, None] * direction
            return result

        layer = model.model.layers[self.layer].mlp
        return [layer.register_forward_pre_hook(pre), layer.register_forward_hook(post)], cache
