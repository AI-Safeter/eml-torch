"""Vector students that execute without consulting the teacher MLP."""

import torch
from torch import nn

from emltorch.operator import safe_eml


class Student(nn.Module):
    def __init__(self, dimension, width, kind, statistics):
        super().__init__()
        self.kind, self.width, self.dimension = kind, width, dimension
        for name, value in statistics.items():
            self.register_buffer(name, value.clone())
        rank = width * 3 // 2 if kind == "linear" else width
        self.left = nn.Linear(dimension, rank)
        if kind != "linear":
            self.right = nn.Linear(dimension, rank)
            nn.init.normal_(self.right.weight, std=0.3 / dimension**0.5)
            nn.init.constant_(self.right.bias, 0.3)
        self.out = nn.Linear(rank, dimension)
        nn.init.normal_(self.left.weight, std=0.1 / dimension**0.5)
        nn.init.zeros_(self.left.bias)
        nn.init.normal_(self.out.weight, std=0.02 / rank**0.5)
        nn.init.zeros_(self.out.bias)

    def normalized(self, x):
        left = self.left(x)
        if self.kind == "eml":
            hidden = safe_eml(left, 1 + self.right(x).square())
        elif self.kind == "swiglu":
            hidden = torch.nn.functional.silu(left) * self.right(x)
        elif self.kind == "linear":
            hidden = left
        else:
            raise ValueError(self.kind)
        return self.out(hidden)

    def forward(self, x):
        return self.normalized((x - self.xmean) / self.xstd) * self.yscale + self.ymean

    def stored_coefficients(self):
        return sum(x.numel() for x in self.state_dict().values())
