"""Budgeted parallel SiLU/EML corrections to a full-width affine shortcut."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from emltorch.operator import safe_eml

KINDS = ("silu", "eml", "silu_silu", "silu_eml")


class Branch(nn.Module):
    def __init__(self, width, hidden, kind):
        super().__init__()
        self.kind = kind
        self.arguments = nn.Linear(width, hidden * (2 if kind == "eml" else 1))
        self.readout = nn.Linear(hidden, width)
        nn.init.normal_(self.arguments.weight, std=0.2 / math.sqrt(width))
        nn.init.zeros_(self.arguments.bias)
        if kind == "eml":
            with torch.no_grad():
                self.arguments.bias[hidden:].fill_(0.3)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, h):
        value = self.arguments(h)
        if self.kind == "eml":
            left, right = value.chunk(2, -1)
            value = safe_eml(left, 1 + right.square(), clamp_val=12.0) - 1
        else:
            value = F.silu(value)
        return self.readout(value)


class Hybrid(nn.Module):
    """One nonlinear depth in every arm; two-branch arms share encoder and decoder.

    f(x) = A x + b + D [S(LN(E x)) + g C(LN(E x))].
    The nonlinear correction lies in range(D), of dimension at most width.
    The affine shortcut removes a mandatory low-rank *total* output subspace,
    but does not remove this nonlinear subspace or the encoder's nullspace.
    """

    def __init__(self, kind, statistics=None, dimension=1536, width=512, budget=6000000):
        super().__init__()
        assert kind in KINDS
        self.kind, self.dimension, self.width, self.budget = kind, dimension, width, budget
        statistics = statistics or {
            "xmean": torch.zeros(dimension),
            "xstd": torch.ones(dimension),
            "ymean": torch.zeros(dimension),
            "yscale": torch.tensor(1.0),
        }
        for k, v in statistics.items():
            self.register_buffer(k, v.detach().clone())
        self.shortcut = nn.Linear(dimension, dimension)
        self.encoder = nn.Linear(dimension, width)
        self.norm = nn.LayerNorm(width)
        self.decoder = nn.Linear(width, dimension, bias=False)
        two = "_" in kind
        if two:
            self.gate = nn.Parameter(torch.tensor(0.01))
        else:
            self.register_parameter("gate", None)
        available = budget - self.coefficients()

        def count(hidden, activation):
            return hidden * ((3 * width + 2) if activation == "eml" else (2 * width + 1)) + width

        main_kind = kind.split("_")[0]
        correction_kind = kind.split("_")[-1] if two else None
        correction_budget = budget // 10 if two else 0
        if two:
            unit = count(1, correction_kind) - width
            correction_hidden = (correction_budget - width) // unit
            correction_budget = count(correction_hidden, correction_kind)
        main_hidden = (available - correction_budget - width) // (count(1, main_kind) - width)
        assert main_hidden > 0
        self.main = Branch(width, main_hidden, main_kind)
        self.correction = Branch(width, correction_hidden, correction_kind) if two else None
        assert budget * 0.999 < self.coefficients() <= budget

    def coefficients(self):
        return sum(p.numel() for p in self.parameters()) + sum(b.numel() for b in self.buffers())

    def specification(self):
        return {k: getattr(self, k) for k in ("kind", "dimension", "width", "budget")}

    def normalized(self, x):
        h = self.norm(self.encoder(x))
        update = self.main(h)
        if self.correction is not None:
            update = update + self.gate * self.correction(h)
        return self.shortcut(x) + self.decoder(update)

    def forward(self, x):
        return self.normalized((x - self.xmean) / self.xstd) * self.yscale + self.ymean


def initialize(model, init):
    with torch.no_grad():
        model.shortcut.weight.copy_(init["ridge"][:-1].T)
        model.shortcut.bias.copy_(init["ridge"][-1])
        model.encoder.weight.copy_(init["x_basis"][:, : model.width].T)
        model.encoder.bias.zero_()
        model.decoder.weight.copy_(init["residual_basis"][:, : model.width])


class DeployedHybrid(nn.Module):
    """Fold affine statistics, branch biases, and constant gate for every arm."""

    def __init__(self, model, ablate=False):
        super().__init__()
        m = self.network = copy.deepcopy(model)
        with torch.no_grad():
            for layer in (m.encoder, m.shortcut):
                weight = layer.weight.double() / m.xstd.double()
                layer.bias.copy_(layer.bias.double() - weight @ m.xmean.double())
                layer.weight.copy_(weight)
            m.shortcut.weight.mul_(m.yscale)
            m.shortcut.bias.mul_(m.yscale).add_(m.ymean)
            m.decoder.weight.mul_(m.yscale)
            if m.correction is not None:
                if ablate:
                    m.correction = None
                else:
                    m.correction.readout.weight.mul_(m.gate)
                    m.correction.readout.bias.mul_(m.gate)
            for branch in (m.main, m.correction):
                if branch is not None:
                    m.shortcut.bias.add_(m.decoder.weight @ branch.readout.bias)
                    branch.readout.register_parameter("bias", None)
            m.register_parameter("gate", None)
            for key in ("xmean", "xstd", "ymean", "yscale"):
                delattr(m, key)

    def forward(self, x):
        m = self.network
        h = m.norm(m.encoder(x))
        update = m.main(h)
        if m.correction is not None:
            update = update + m.correction(h)
        return m.shortcut(x) + m.decoder(update)


def load_hybrid(path, device="cuda", dtype=torch.bfloat16, ablate=False):
    row = torch.load(path, map_location="cpu", weights_only=True)
    model = Hybrid(**row["specification"])
    model.load_state_dict(row["state"])
    return DeployedHybrid(model, ablate).to(device=device, dtype=dtype).eval().requires_grad_(False)
