"""Fold training normalization into affine weights for actual deployment."""

import copy

import torch
from torch import nn

from emltorch.operator import safe_eml


class FoldedStudent(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.kind = student.kind
        for name in ["left", "right", "out"]:
            if hasattr(student, name):
                setattr(self, name, copy.deepcopy(getattr(student, name)))
        with torch.no_grad():
            for name in ["left", "right"]:
                if not hasattr(self, name):
                    continue
                original = getattr(student, name)
                folded = getattr(self, name)
                # Accumulate affine folding in float64, then deploy in the original dtype.
                weight = original.weight.double() / student.xstd.double()
                folded.weight.copy_(weight)
                folded.bias.copy_(original.bias.double() - weight @ student.xmean.double())
            self.out.weight.copy_(student.out.weight.double() * student.yscale.double())
            self.out.bias.copy_(
                student.out.bias.double() * student.yscale.double() + student.ymean.double()
            )

    def forward(self, x):
        left = self.left(x)
        if self.kind == "eml":
            hidden = safe_eml(left, 1 + self.right(x).square())
        elif self.kind == "swiglu":
            hidden = torch.nn.functional.silu(left) * self.right(x)
        else:
            hidden = left
        return self.out(hidden)

    def stored_coefficients(self):
        return sum(v.numel() for v in self.state_dict().values())
