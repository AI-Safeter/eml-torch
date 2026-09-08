"""Complete MLP students with compositional depth and counted normalization."""

import math

import torch
from torch import nn

from emltorch.operator import safe_eml


class Stage(nn.Module):
    def __init__(self, width, hidden, kind, depth):
        super().__init__()
        self.kind, self.scale = kind, depth**-0.5
        self.norm = nn.LayerNorm(width)
        self.arguments = nn.Linear(width, hidden * (2 if kind == "eml" else 1))
        self.readout = nn.Linear(hidden, width)
        nn.init.normal_(self.arguments.weight, std=0.2 / math.sqrt(width))
        nn.init.zeros_(self.arguments.bias)
        if kind == "eml":
            with torch.no_grad():
                self.arguments.bias[hidden:].fill_(0.3)
        nn.init.normal_(self.readout.weight, std=0.001 / math.sqrt(hidden))
        nn.init.zeros_(self.readout.bias)
        self.diagnostics = False
        self.clamps, self.arguments_seen, self.max_argument = 0, 0, 0.0

    def forward(self, x):
        a = self.arguments(self.norm(x))
        if self.kind == "eml":
            left, right = a.chunk(2, dim=-1)
            if self.diagnostics:
                self.clamps += int((left.abs() > 12).sum())
                self.arguments_seen += left.numel()
                self.max_argument = max(self.max_argument, float(left.abs().max()))
            y = safe_eml(left, 1 + right.square(), clamp_val=12.0) - 1
        else:
            y = torch.nn.functional.silu(a)
        return x + self.scale * self.readout(y)


class Student(nn.Module):
    def __init__(self, dimension, budget, width, depth, kind, statistics=None):
        super().__init__()
        assert kind in {"eml", "silu", "linear"}
        self.dimension, self.budget = dimension, budget
        self.width, self.depth, self.kind = width, depth, kind
        statistics = statistics or {
            "xmean": torch.zeros(dimension),
            "xstd": torch.ones(dimension),
            "ymean": torch.zeros(dimension),
            "yscale": torch.tensor(1.0),
        }
        for name in ["xmean", "xstd", "ymean", "yscale"]:
            self.register_buffer(name, statistics[name].detach().clone())
        constants = sum(x.numel() for x in self.buffers())
        if kind == "linear":
            width = min(dimension, (budget - constants - dimension) // (2 * dimension + 1))
            self.width, self.depth, self.hidden = width, 0, 0
            self.stages = nn.ModuleList()
        else:
            fixed = constants + 2 * dimension * width + width + dimension + depth * 3 * width
            per_hidden = (3 * width + 2) if kind == "eml" else (2 * width + 1)
            self.hidden = (budget - fixed) // (depth * per_hidden)
            assert self.hidden > 0
            self.stages = nn.ModuleList(
                Stage(width, self.hidden, kind, depth) for _ in range(depth)
            )
        self.encoder = nn.Linear(dimension, width)
        self.decoder = nn.Linear(width, dimension)
        assert self.coefficients() <= budget

    def normalized(self, x):
        h = self.encoder(x)
        for stage in self.stages:
            h = stage(h)
        return self.decoder(h)

    def forward(self, x):
        return self.normalized((x - self.xmean) / self.xstd) * self.yscale + self.ymean

    def coefficients(self):
        return sum(x.numel() for x in self.parameters()) + sum(x.numel() for x in self.buffers())

    def specification(self):
        return {
            "dimension": self.dimension,
            "budget": self.budget,
            "width": self.width,
            "depth": self.depth,
            "kind": self.kind,
        }


def load_student(path, device="cuda", dtype=torch.bfloat16):
    record = torch.load(path, map_location="cpu", weights_only=True)
    model = Student(**record["specification"])
    model.load_state_dict(record["state"])
    return model.to(device=device, dtype=dtype).eval()


def install(model, student, layer=26):
    """Physically replace the module; never retain a teacher reference in the model."""
    native_layer = model.model.language_model.layers[layer]
    old = native_layer.mlp
    old_parameters = {id(p) for p in old.parameters()}
    native_layer.mlp = student
    assert not old_parameters.intersection(id(p) for p in model.parameters())
    return old


def contribution(y, weight, eps=1e-6):
    """Native Gemma RMSNorm arithmetic for the training-only contribution loss."""
    dtype = y.dtype
    value = y.float()
    value = value * torch.pow(value.square().mean(-1, keepdim=True) + eps, -0.5)
    return (value * weight.float()).to(dtype)
