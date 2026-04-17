"""End-to-end tests for emltorch.fit() — the public API."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import emltorch as eml   # noqa: E402


DEVICE = "cuda:7" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("fn_name,fn,depth,low,high", [
    ("exp(x)", torch.exp,                      1, -2.0, 2.0),
    ("e - x",  lambda x: math.e - x,           2,  0.1, 2.0),
    ("ln(x)",  torch.log,                      3,  0.5, 5.0),
])
def test_known_recovery(fn_name, fn, depth, low, high):
    """emltorch.fit should recover standard elementary functions."""
    x = torch.linspace(low, high, 512)
    y = fn(x)
    result = eml.fit(x, y, depth=depth, device=DEVICE)
    assert result.r2 > 0.99, (
        f"{fn_name} at depth {depth} failed: R²={result.r2:.4f}, "
        f"expr={result.expression}"
    )


def test_linear_signal_via_random():
    """Pure linear signals should resolve via random search at shallow depth."""
    x = torch.linspace(-2.0, 2.0, 256)
    y = 2 * x + 1
    result = eml.fit(x, y, depth=2, strategy="random", device=DEVICE)
    assert result.r2 > 0.99


def test_neg_x_via_evolution_affine():
    """−x should resolve at depth 4 (paper says) or depth 2 + affine (our win)."""
    x = torch.linspace(-2.0, 2.0, 256)
    y = -x
    result = eml.fit(x, y, depth=4, strategy="evolution", device=DEVICE,
                     population=1024, generations=10)
    assert result.r2 > 0.99


def test_result_has_metadata():
    x = torch.linspace(0.5, 5.0, 256)
    y = torch.log(x)
    result = eml.fit(x, y, depth=3, device=DEVICE)
    assert hasattr(result, "expression")
    assert hasattr(result, "r2")
    assert hasattr(result, "time_s")
    assert hasattr(result, "a")
    assert hasattr(result, "b")
    assert result.time_s > 0
