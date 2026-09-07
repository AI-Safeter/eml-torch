"""Trainable affine-argument EML heads for neural models."""

import torch
from torch import Tensor, nn

from ._validation import positive_int
from .operator import safe_eml


class EMLHead(nn.Module):
    """A weighted EML expansion with an optional linear residual.

    Each hidden unit computes ``eml(left(x), 1 + right(x)**2)``. The
    logarithm argument is positive for finite real inputs. ``safe_eml``
    supplies the same numerical guards used by the symbolic tree API.

    Inputs have shape ``(*, in_features)`` and outputs have shape
    ``(*, out_features)``. Arguments and readout weights train with ordinary
    PyTorch optimizers; this module does not run symbolic tree discovery.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = 1,
        *,
        linear_skip: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        positive_int("in_features", in_features)
        positive_int("hidden_features", hidden_features)
        positive_int("out_features", out_features)
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.out_features = int(out_features)
        factory = {"device": device, "dtype": dtype}
        self.skip = nn.Linear(in_features, out_features, **factory) if linear_skip else None
        self.left = nn.Linear(in_features, hidden_features, **factory)
        self.right = nn.Linear(in_features, hidden_features, **factory)
        self.out = nn.Linear(hidden_features, out_features, bias=not linear_skip, **factory)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize small affine arguments and a small nonlinear readout."""
        if self.skip is not None:
            self.skip.reset_parameters()
        nn.init.normal_(self.left.weight, std=0.2 / self.in_features**0.5)
        nn.init.zeros_(self.left.bias)
        nn.init.normal_(self.right.weight, std=0.3 / self.in_features**0.5)
        nn.init.constant_(self.right.bias, 0.3)
        nn.init.normal_(self.out.weight, std=0.02)
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor) -> Tensor:
        hidden = safe_eml(self.left(x), 1 + self.right(x).square())
        result = self.out(hidden)
        return result if self.skip is None else result + self.skip(x)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, hidden_features={self.hidden_features}, "
            f"out_features={self.out_features}, linear_skip={self.skip is not None}"
        )
