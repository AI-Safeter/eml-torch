"""Tests for the pairwise-combo leaf extension (a+b, a-b, b-a).

The combo extension enables 2-variable targets that require a linear
pre-combination (sum or difference) of input variables, most notably
softmax[0] = sigmoid(a - b), which is blocked at every depth without it.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import emltorch as eml  # noqa: E402
from emltorch.tree import BatchedEMLTree, build_base, enumerate_combos, num_combos  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_combo_enumeration():
    assert enumerate_combos(1) == []
    assert num_combos(1) == 0
    v2 = enumerate_combos(2)
    # 1 sum (unordered) + 2 diffs (ordered, i!=j)
    assert v2 == [("add", 0, 1), ("sub", 0, 1), ("sub", 1, 0)]
    assert num_combos(2) == 3
    # V=3: 3 sums + 6 diffs = 9
    assert num_combos(3) == 9


def test_build_base_values():
    x = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])  # (B=1, V=2, N=3)
    base = build_base(x, 2, torch.float32)
    # Expected rows: [1], [x1], [x2], [x1+x2], [x1-x2], [x2-x1]
    assert base.shape == (1, 6, 3)
    assert torch.allclose(base[0, 0], torch.ones(3))
    assert torch.allclose(base[0, 1], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(base[0, 2], torch.tensor([10.0, 20.0, 30.0]))
    assert torch.allclose(base[0, 3], torch.tensor([11.0, 22.0, 33.0]))
    assert torch.allclose(base[0, 4], torch.tensor([-9.0, -18.0, -27.0]))
    assert torch.allclose(base[0, 5], torch.tensor([9.0, 18.0, 27.0]))


def test_v1_untouched():
    """V=1 path must be byte-identical to pre-combo behavior:
    leaf_choices=2, internal_choices=3, base=[1, x]."""
    tree = BatchedEMLTree(num_trees=4, depth=2, num_vars=1, device=DEVICE)
    assert tree.leaf_logits.shape[-1] == 2
    assert tree.internal_logits[0].shape[-1] == 3
    assert tree.f_child_idx == 2
    assert tree.n_combo == 0


def test_v2_choice_counts():
    tree = BatchedEMLTree(num_trees=4, depth=2, num_vars=2, device=DEVICE)
    # 1 + V + n_combo = 1 + 2 + 3 = 6 leaf choices; 7 internal
    assert tree.leaf_logits.shape[-1] == 6
    assert tree.internal_logits[0].shape[-1] == 7
    assert tree.f_child_idx == 6
    assert tree.n_combo == 3


def test_exp_of_diff_depth1():
    """y = exp(x1 - x2) should recover near-perfectly at depth=1.

    Requires picking the `x1 - x2` combo at the leaf's left input and
    the constant `1` at the right, then eml(x1-x2, 1) = exp(x1-x2).
    This wouldn't be reachable at depth 1 without the combo.
    """
    torch.manual_seed(0)
    n = 16
    xs = torch.linspace(-1.0, 1.0, n)
    g = torch.stack(torch.meshgrid([xs, xs], indexing="ij"), dim=0).reshape(2, -1)
    y = torch.exp(g[0] - g[1])
    result = eml.fit(g, y, depth=1, strategy="evolution",
                     population=512, generations=10,
                     polish=True, device=DEVICE, r2_target=0.9999)
    assert result.r2 > 0.99, (
        f"exp(x1-x2) at d=1 failed: R²={result.r2:.4f}, expr={result.expression}"
    )
    # Symbolic output should reference the combo
    assert "x1 - x2" in result.expression or "x2 - x1" in result.expression or \
           "x1 + x2" in result.expression, \
        f"combo not surfaced in expression: {result.expression}"
