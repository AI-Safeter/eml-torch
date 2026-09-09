"""Three fully counted MLP replacements with matched activation budgets."""

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

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


class StructuredProjection(nn.Module):
    """Independent diagonal-plus-low-rank maps, with packed argument GEMMs."""

    def __init__(self, dimension, rank, groups=1, zero=False):
        super().__init__()
        self.dimension, self.rank, self.groups = dimension, rank, groups
        self.down = nn.Parameter(torch.randn(groups, rank, dimension) / math.sqrt(dimension))
        self.up = nn.Parameter(
            torch.randn(groups, dimension, rank) * (0 if zero else 0.02) / math.sqrt(rank)
        )
        self.diagonal = nn.Parameter(torch.full((groups, dimension), 0.0 if zero else 0.2))
        self.bias = nn.Parameter(torch.zeros(groups, dimension))

    def forward(self, x):
        low = F.linear(x, self.down.flatten(0, 1))
        if self.groups == 1:
            value = F.linear(low, self.up[0]) + x * self.diagonal[0]
            return value if self.bias is None else value + self.bias[0]
        shape = x.shape[:-1]
        low = low.reshape(-1, self.groups, self.rank).transpose(0, 1)
        value = torch.bmm(low, self.up.transpose(1, 2)).transpose(0, 1)
        value = value.reshape(*shape, self.groups, self.dimension)
        value = value + x.unsqueeze(-2) * self.diagonal
        return value if self.bias is None else value + self.bias

    def matrices(self):
        return self.up @ self.down + torch.diag_embed(self.diagonal)


class StructuredStage(nn.Module):
    def __init__(self, dimension, rank, activation, depth):
        super().__init__()
        self.kind, self.scale = activation, depth**-0.5
        self.norm = nn.LayerNorm(dimension)
        self.arguments = StructuredProjection(dimension, rank, 2 if activation == "eml" else 1)
        if activation == "eml":
            with torch.no_grad():
                self.arguments.bias[1].fill_(0.3)
        self.readout = StructuredProjection(dimension, rank, zero=True)
        self.diagnostics = False
        self.clamps = self.arguments_seen = 0
        self.max_argument = 0.0

    def forward(self, x):
        arguments = self.arguments(self.norm(x))
        if self.kind == "eml":
            left, right = arguments.unbind(-2)
            if self.diagnostics:
                self.clamps += int((left.abs() > 12).sum())
                self.arguments_seen += left.numel()
                self.max_argument = max(self.max_argument, float(left.abs().max()))
            value = safe_eml(left, 1 + right.square(), clamp_val=12.0) - 1
        else:
            value = F.silu(arguments)
        return x + self.scale * self.readout(value)


class Replacement(nn.Module):
    def __init__(
        self,
        architecture,
        activation,
        depth=1,
        budget=6000000,
        dimension=1536,
        width=512,
        statistics=None,
    ):
        super().__init__()
        assert architecture in {"bottleneck", "shortcut", "structured"}
        assert activation in {"eml", "silu"} and depth > 0
        self.architecture, self.activation = architecture, activation
        self.depth, self.budget, self.dimension, self.width = depth, budget, dimension, width
        statistics = statistics or {
            "xmean": torch.zeros(dimension),
            "xstd": torch.ones(dimension),
            "ymean": torch.zeros(dimension),
            "yscale": torch.tensor(1.0),
        }
        for key, value in statistics.items():
            self.register_buffer(key, value.detach().clone())
        constants = sum(v.numel() for v in self.buffers())
        if architecture == "structured":
            # Dense full-width output map + full-width structured residual stages.
            maps = 3 if activation == "eml" else 2
            fixed = (
                constants
                + dimension**2
                + dimension
                + depth * (2 * dimension + maps * 2 * dimension)
            )
            self.rank = (budget - fixed) // (depth * maps * 2 * dimension)
            assert self.rank > 0
            self.stages = nn.ModuleList(
                StructuredStage(dimension, self.rank, activation, depth) for _ in range(depth)
            )
            self.decoder = nn.Linear(dimension, dimension)
        else:
            self.encoder = nn.Linear(dimension, width)
            self.decoder = nn.Linear(width, dimension, bias=architecture == "bottleneck")
            if architecture == "shortcut":
                self.shortcut = nn.Linear(dimension, dimension)
            base = sum(p.numel() for p in self.parameters()) + constants
            per_hidden = (3 * width + 2) if activation == "eml" else (2 * width + 1)
            self.hidden = (budget - base - depth * 3 * width) // (depth * per_hidden)
            assert self.hidden > 0
            self.stages = nn.ModuleList(
                Stage(width, self.hidden, activation, depth) for _ in range(depth)
            )
            # Matched exact affine starting functions; all nonlinear updates start at zero.
            for stage in self.stages:
                nn.init.zeros_(stage.readout.weight)
                nn.init.zeros_(stage.readout.bias)
        assert self.coefficients() <= budget

    def normalized(self, x):
        if self.architecture == "structured":
            h = x
            for stage in self.stages:
                h = stage(h)
            return self.decoder(h)
        h = self.encoder(x)
        if self.architecture == "bottleneck":
            for stage in self.stages:
                h = stage(h)
            return self.decoder(h)
        # Accumulate updates directly, avoiding cancellation in h_final - h_initial.
        update = torch.zeros_like(h)
        for stage in self.stages:
            arguments = stage.arguments(stage.norm(h))
            if self.activation == "eml":
                left, right = arguments.chunk(2, -1)
                if stage.diagnostics:
                    stage.clamps += int((left.abs() > 12).sum())
                    stage.arguments_seen += left.numel()
                    stage.max_argument = max(stage.max_argument, float(left.abs().max()))
                value = safe_eml(left, 1 + right.square(), clamp_val=12.0) - 1
            else:
                value = F.silu(arguments)
            delta = stage.scale * stage.readout(value)
            update = update + delta
            h = h + delta
        return self.shortcut(x) + self.decoder(update)

    def forward(self, x):
        return self.normalized((x - self.xmean) / self.xstd) * self.yscale + self.ymean

    def coefficients(self):
        return sum(p.numel() for p in self.parameters()) + sum(b.numel() for b in self.buffers())

    def specification(self):
        return {
            k: getattr(self, k)
            for k in ["architecture", "activation", "depth", "budget", "dimension", "width"]
        }


class Deployed(nn.Module):
    def __init__(self, trained):
        super().__init__()
        # Use the exact architecture forward while removing training-only statistics.
        self.network = copy.deepcopy(trained)
        self.architecture = trained.architecture
        with torch.no_grad():
            if trained.architecture == "structured":
                # LayerNorm prevents folding channelwise input normalization through it.
                self.register_buffer("xmean", trained.xmean.clone())
                self.register_buffer("xstd", trained.xstd.clone())
            else:
                for key in ["encoder"] + (
                    ["shortcut"] if trained.architecture == "shortcut" else []
                ):
                    original, layer = getattr(trained, key), getattr(self.network, key)
                    weight = original.weight.double() / trained.xstd.double()
                    layer.weight.copy_(weight)
                    layer.bias.copy_(original.bias.double() - weight @ trained.xmean.double())
            self.network.decoder.weight.mul_(trained.yscale)
            if trained.architecture == "shortcut":
                self.network.shortcut.weight.mul_(trained.yscale)
                self.network.shortcut.bias.mul_(trained.yscale).add_(trained.ymean)
            else:
                self.network.decoder.bias.mul_(trained.yscale).add_(trained.ymean)
            # The last residual bias has no later nonlinear consumer. Fold it
            # through the decoder for every architecture and activation.
            last = self.network.stages[-1]
            target = (
                self.network.shortcut.bias
                if trained.architecture == "shortcut"
                else self.network.decoder.bias
            )
            target.copy_(
                target.double()
                + last.scale
                * (self.network.decoder.weight.double() @ last.readout.bias.double().flatten())
            )
            last.readout.register_parameter("bias", None)
        for key in ["xmean", "xstd", "ymean", "yscale"]:
            delattr(self.network, key)

    def forward(self, x):
        if self.architecture == "structured":
            x = (x - self.xmean) / self.xstd
        return self.network.normalized(x)


def load_replacement(path, deployed=False, device="cuda", dtype=torch.bfloat16):
    record = torch.load(path, map_location="cpu", weights_only=True)
    model = Replacement(**record["specification"])
    model.load_state_dict(record["state"])
    if deployed:
        model = Deployed(model)
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)
