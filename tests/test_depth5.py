"""
Depth 5 stress tests — evolution + crossover + polish.

Targets:
    x * y      (paper says depth 5)
    x^2        (depth 3 nominally; depth 5 with constants flexibility)
    cos(x)     (depth 8 pure EML; with affine + constants should go lower)
"""

import sys
import math
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import emltorch as eml

DEVICE = "cuda:7" if torch.cuda.is_available() else "cpu"


def _make_2d(fn, low, high, n=32):
    xs = torch.linspace(low, high, n)
    g = torch.stack(torch.meshgrid([xs, xs], indexing='ij'),
                    dim=0).reshape(2, -1)
    y = fn(g[0], g[1])
    return g, y


def test_multiplication_d5():
    """x*y should reach R² > 0.95 with evolution+crossover+polish."""
    x, y = _make_2d(lambda a, b: a * b, 0.5, 2.0)
    result = eml.fit(x, y, depth=5, strategy="evolution",
                     population=2048, generations=25,
                     polish=True, polish_iters=3000,
                     device=DEVICE, r2_target=0.999)
    print(f"  x*y R²={result.r2:.4f}  strategy={result.strategy}")
    print(f"  {result.expression}")
    assert result.r2 > 0.95


def test_x_squared_d3():
    """x² at depth 3 should be near-perfect with polish."""
    x_ = torch.linspace(0.5, 3.0, 256)
    y = x_ ** 2
    result = eml.fit(x_, y, depth=3, strategy="evolution",
                     population=1024, generations=15,
                     polish=True,
                     device=DEVICE, r2_target=0.999)
    print(f"  x² R²={result.r2:.4f}  strategy={result.strategy}")
    print(f"  {result.expression}")
    assert result.r2 > 0.99


def test_cos_d5():
    """cos(x) at depth 5 — paper says depth 8. With polish should pass 0.95."""
    x_ = torch.linspace(-math.pi, math.pi, 512)
    y = torch.cos(x_)
    result = eml.fit(x_, y, depth=5, strategy="evolution",
                     population=2048, generations=30,
                     polish=True, polish_iters=3000,
                     device=DEVICE, r2_target=0.999)
    print(f"  cos(x) R²={result.r2:.4f}  strategy={result.strategy}")
    print(f"  {result.expression}")
    assert result.r2 > 0.95


if __name__ == "__main__":
    test_multiplication_d5()
    test_x_squared_d3()
    test_cos_d5()
    print("\nAll depth-5 tests passed!")
