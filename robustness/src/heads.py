"""Explicit learned-argument EML heads and parameter-comparable neural controls."""

import torch
from torch import nn

from emltorch.operator import safe_eml


class Head(nn.Module):
    def __init__(self, kind, rank, width):
        super().__init__()
        self.kind, self.rank, self.width = kind, rank, width
        self.skip = nn.Linear(rank, 1)
        if kind in ["eml_affine", "eml_square"]:
            self.left = nn.Linear(rank, width)
            self.right = nn.Linear(rank, width)
            self.out = nn.Linear(width, 1, bias=False)
            nn.init.normal_(self.left.weight, std=0.2 / rank**0.5)
            nn.init.normal_(self.right.weight, std=0.3 / rank**0.5)
            nn.init.zeros_(self.left.bias)
            nn.init.constant_(self.right.bias, 2.0 if kind == "eml_affine" else 0.3)
        elif kind == "eml_log":
            self.right = nn.Linear(rank, width)
            self.out = nn.Linear(width, 1, bias=False)
            nn.init.normal_(self.right.weight, std=0.5 / rank**0.5)
            nn.init.normal_(self.right.bias, std=0.3)
        elif kind == "silu":
            self.hidden = nn.Linear(rank, 2 * width)
            self.out = nn.Linear(2 * width, 1, bias=False)
        elif kind == "silu_two":
            self.hidden = nn.Linear(rank, width)
            self.hidden2 = nn.Linear(width, width)
            self.out = nn.Linear(width, 1, bias=False)
        else:
            raise ValueError(kind)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, x):
        if self.kind == "eml_affine":
            hidden = safe_eml(self.left(x), self.right(x))
        elif self.kind == "eml_square":
            hidden = safe_eml(self.left(x), 1 + self.right(x).square())
        elif self.kind == "eml_log":
            right = self.right(x)
            hidden = safe_eml(torch.zeros_like(right), 1 + right.square())
        else:
            hidden = torch.nn.functional.silu(self.hidden(x))
            if self.kind == "silu_two":
                hidden = torch.nn.functional.silu(self.hidden2(hidden))
        return (self.skip(x) + self.out(hidden)).squeeze(1)

    @torch.no_grad()
    def initialize_linear(self, x, y):
        a = torch.cat([torch.ones(len(x), 1, device=x.device), x], 1).double()
        ridge = torch.eye(a.shape[1], device=x.device, dtype=torch.float64) * 0.001
        ridge[0, 0] = 0
        w = torch.linalg.solve(a.T @ a + ridge, a.T @ y.double()).float()
        self.skip.bias.copy_(w[:1])
        self.skip.weight.copy_(w[1:][None])


def objective(pred, y, cases, weight):
    absolute = (pred - y).square().mean()
    p, t = pred.reshape(5, cases), y.reshape(5, cases)
    response = ((p[1:] - p[:1]) - (t[1:] - t[:1])).square().mean()
    return absolute + weight * response, absolute, response


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
