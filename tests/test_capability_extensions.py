"""Unit tests for the 2026-04-24 EML capability extensions.

Covers:
  - `use_mul`   pair combos `x_i * x_j`               (tree.py)
  - `use_mul3`  triple combos `x_i * x_j * x_k`       (tree.py)
  - `BatchedEMLMulTree` forward + snap roundtrip      (hybrid_mul.py)
  - `evolve_hybrid_mul` end-to-end on exp(a*b)        (hybrid_mul.py)
  - `polish_hybrid_mul` target-normalization fix      (hybrid_mul.py)
  - Nested-paren gradient parser for `eml(((a*b)*c), 1)`  (gradient.py)
  - Z3 bridge accepts `*` combos                      (smt.py)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import emltorch as eml  # noqa: E402
from emltorch.tree import (  # noqa: E402
    BatchedEMLTree,
    build_base,
    enumerate_combos,
    enumerate_triples,
    num_combos,
)
from emltorch.hybrid_mul import (  # noqa: E402
    BatchedEMLMulTree,
    HybridMulConfig,
    evolve_hybrid_mul,
    polish_hybrid_mul,
    safe_eml as safe_eml_hyb,
    safe_mul,
)
from emltorch.gradient import diff_formula, gradient_at, torch_gradient_fn  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ─── 1. use_mul combo enumeration + sizing ──────────────────────────────────


def test_enumerate_combos_default():
    """use_mul=False keeps the original add+sub pair layout."""
    assert enumerate_combos(1) == []
    assert enumerate_combos(2) == [
        ("add", 0, 1),
        ("sub", 0, 1),
        ("sub", 1, 0),
    ]
    assert num_combos(2, use_mul=False) == 3


def test_enumerate_combos_use_mul():
    """use_mul=True appends (mul, i, j) for i<j AFTER the add/sub block."""
    combos = enumerate_combos(3, use_mul=True)
    # Layout: 3 adds, 6 subs, 3 muls = 12 total
    assert len(combos) == 12
    # Last three should be mul pairs in i<j order
    assert combos[-3:] == [("mul", 0, 1), ("mul", 0, 2), ("mul", 1, 2)]
    assert num_combos(3, use_mul=True) == 12


def test_enumerate_triples():
    """use_mul3=True gives C(V, 3) triples in strict i<j<k order."""
    assert enumerate_triples(2, use_mul3=True) == []  # V<3 -> empty
    assert enumerate_triples(3, use_mul3=True) == [(0, 1, 2)]
    # V=4 -> C(4,3) = 4 triples
    assert enumerate_triples(4, use_mul3=True) == [
        (0, 1, 2),
        (0, 1, 3),
        (0, 2, 3),
        (1, 2, 3),
    ]
    assert enumerate_triples(3, use_mul3=False) == []


def test_num_combos_matches_enumeration():
    """num_combos() must match len(enumerate_combos) + len(enumerate_triples)."""
    for V in range(1, 5):
        for um in (False, True):
            for um3 in (False, True):
                expected = len(enumerate_combos(V, use_mul=um)) + len(
                    enumerate_triples(V, use_mul3=um3)
                )
                assert num_combos(V, use_mul=um, use_mul3=um3) == expected, (
                    f"V={V} use_mul={um} use_mul3={um3}: "
                    f"expected {expected}, got {num_combos(V, use_mul=um, use_mul3=um3)}"
                )


def test_num_combos_v4_all():
    """V=4 with all flags: C(4,2)=6 add + V*(V-1)=12 sub + C(4,2)=6 mul
    + C(4,3)=4 triples = 28."""
    assert num_combos(4, use_mul=True, use_mul3=True) == 28


# ─── 2. build_base column layout ────────────────────────────────────────────


def test_build_base_mul_columns():
    """build_base with use_mul=True emits the mul columns AFTER add/sub."""
    x = torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]])  # (B=1, V=2, N=3)
    base = build_base(x, 2, torch.float32, use_mul=True)
    # Expected: [1, x0, x1, x0+x1, x0-x1, x1-x0, x0*x1]
    assert base.shape == (1, 7, 3)
    assert torch.allclose(base[0, -1], torch.tensor([10.0, 40.0, 90.0]))


def test_build_base_triple_columns():
    """build_base with use_mul=True, use_mul3=True: triples after pair combos."""
    x = torch.tensor(
        [[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [100.0, 200.0, 300.0]]]
    )  # (B=1, V=3, N=3)
    base = build_base(x, 3, torch.float32, use_mul=True, use_mul3=True)
    # V=3 use_mul=True use_mul3=True -> num_combos = 13. Columns = 1 + V + 13 = 17.
    assert base.shape == (1, 17, 3)
    # Last column is x0*x1*x2.
    expected_triple = torch.tensor(
        [1.0 * 10.0 * 100.0, 2.0 * 20.0 * 200.0, 3.0 * 30.0 * 300.0]
    )
    assert torch.allclose(base[0, -1], expected_triple)


# ─── 3. BatchedEMLTree with use_mul / use_mul3 ──────────────────────────────


def test_batched_tree_use_mul_shapes():
    """BatchedEMLTree with use_mul=True / use_mul3=True has correct choice counts."""
    # V=3, no flags: leaf_choices = 1 + 3 + 9 = 13
    t = BatchedEMLTree(num_trees=2, depth=2, num_vars=3, device=DEVICE)
    assert t.leaf_logits.shape[-1] == 13
    # V=3, use_mul=True: leaf_choices = 1 + 3 + 12 = 16
    t = BatchedEMLTree(num_trees=2, depth=2, num_vars=3, device=DEVICE, use_mul=True)
    assert t.leaf_logits.shape[-1] == 16
    # V=3, use_mul=True, use_mul3=True: leaf_choices = 1 + 3 + 13 = 17
    t = BatchedEMLTree(
        num_trees=2, depth=2, num_vars=3, device=DEVICE, use_mul=True, use_mul3=True
    )
    assert t.leaf_logits.shape[-1] == 17


def test_batched_tree_forward_no_nan_on_mul3():
    """Forward pass with triple combos produces finite outputs."""
    torch.manual_seed(0)
    t = BatchedEMLTree(
        num_trees=8, depth=2, num_vars=3, device=DEVICE, use_mul=True, use_mul3=True
    )
    t.snap()
    x = torch.randn(8, 3, 32, device=DEVICE)
    with torch.no_grad():
        y = t(x)
    assert y.shape == (8, 32)
    assert torch.isfinite(y).all()


# ─── 4. BatchedEMLMulTree + evolve_hybrid_mul ───────────────────────────────


def test_hybrid_mul_tree_forward_shape():
    """BatchedEMLMulTree forward produces (B, N)."""
    torch.manual_seed(0)
    t = BatchedEMLMulTree(num_trees=4, depth=3, num_vars=2, device=DEVICE)
    t.snap()
    x = torch.randn(4, 2, 16, device=DEVICE)
    with torch.no_grad():
        y = t(x)
    assert y.shape == (4, 16)
    assert torch.isfinite(y).all()


def test_hybrid_mul_tree_snapped_roundtrip():
    """snapped() returns valid indices; topology is deterministic after snap."""
    torch.manual_seed(0)
    t = BatchedEMLMulTree(num_trees=4, depth=3, num_vars=2, device=DEVICE)
    t.snap()
    leaf_in, leaf_op, int_in, int_op = t.snapped()
    # leaf_in: (B, L, 2); values in [0, leaf_choices)
    assert leaf_in.shape == (4, 4, 2)
    leaf_choices = t.leaf_logits.shape[-1]
    assert (leaf_in >= 0).all() and (leaf_in < leaf_choices).all()
    # leaf_op: (B, L); values in {0, 1}
    assert leaf_op.shape == (4, 4)
    assert ((leaf_op == 0) | (leaf_op == 1)).all()


def test_evolve_hybrid_mul_exp_ab():
    """exp(a*b) should be recovered by the hybrid tree at depth 2 with
    leaf-level mul combos (via use_mul=True). Sanity test: HELDOUT R² > 0.99
    in ~15 generations."""
    torch.manual_seed(0)
    N = 256
    a = torch.randn(N) * 0.4
    b = torch.randn(N) * 0.4
    x = torch.stack([a, b], 0)
    y = torch.exp(a * b).float()

    cfg = HybridMulConfig(
        depth=2,
        num_vars=2,
        population=1024,
        generations=15,
        use_mul=True,
        device=DEVICE,
    )
    res = evolve_hybrid_mul(x, y, cfg)
    # exp(a*b) recovers structurally at R² ≈ 1.0 (see 2026-04-24 writeup)
    assert res.r2 > 0.99, f"exp(a*b) hybrid R² too low: {res.r2:.4f}"


def test_polish_hybrid_mul_large_magnitude():
    """polish_hybrid_mul must NOT regress on large-magnitude targets.

    Chinchilla-style target (~magnitude 50-100): without target normalization,
    Adam + grad-clip 1.0 was observed to collapse R² from 0.99 (evolve) to
    -6.77 (polish) on depth-5 topologies. The fix: normalize y internally,
    denormalize a, b at return.
    """
    torch.manual_seed(0)
    N = 256
    x_np = torch.randn(2, N).float()
    # Large-magnitude target with simple structure
    y_np = (50.0 + 20.0 * x_np[0] * x_np[1].pow(2) + 0.5 * torch.randn(N)).float()

    cfg = HybridMulConfig(
        depth=4,
        num_vars=2,
        population=512,
        generations=10,
        use_mul=True,
        device=DEVICE,
    )
    res = evolve_hybrid_mul(x_np, y_np, cfg)
    evo_r2 = res.r2

    pol = polish_hybrid_mul(
        res.tree,
        res.idx,
        x_np,
        y_np,
        n_iters=300,
        warm_a=res.a,
        warm_b=res.b,
        normalize_target=True,
    )
    # Polish must never be worse than evolve's R² (NaN-revert guarantee).
    assert (
        pol.r2 >= evo_r2 - 1e-3
    ), f"polish ({pol.r2:.4f}) regressed below evolve ({evo_r2:.4f})"
    # And polish should still be a finite number (not NaN/-inf).
    assert math.isfinite(pol.r2)


# ─── 5. Gradient parser — nested parens from use_mul3 triples ───────────────


def test_gradient_parses_nested_paren_triple():
    """Gradient parser handles `eml(((a*b)*c), 1)` emitted for use_mul3 triples."""
    # d/dc eml(((a*b)*c), 1) = exp((a*b)*c) * d/dc((a*b)*c) = exp(a*b*c) * (a*b)
    grad = diff_formula("eml(((a * b) * c), 1)", wrt="c")
    # Smoke test: produces a non-empty string with "exp" in it.
    assert "exp" in grad
    assert "a" in grad and "b" in grad


def test_gradient_at_nested_triple_matches_analytic():
    """Numerically check gradient of eml(((a*b)*c), 1) at a test point."""
    # ∂/∂c exp(a*b*c) = a*b * exp(a*b*c)
    vals = {"a": 0.5, "b": 0.3, "c": 0.7}
    analytic = vals["a"] * vals["b"] * math.exp(vals["a"] * vals["b"] * vals["c"])
    got = gradient_at("eml(((a * b) * c), 1)", wrt="c", values=vals)
    assert abs(got - analytic) < 1e-6, f"expected {analytic:.6f}, got {got:.6f}"


def test_gradient_at_nested_triple_partial_a():
    """∂/∂a exp(a*b*c) = b*c * exp(a*b*c)."""
    vals = {"a": 0.4, "b": -0.25, "c": 0.6}
    analytic = vals["b"] * vals["c"] * math.exp(vals["a"] * vals["b"] * vals["c"])
    got = gradient_at("eml(((a * b) * c), 1)", wrt="a", values=vals)
    assert abs(got - analytic) < 1e-6


def test_torch_gradient_fn_nested_triple():
    """Vectorized torch gradient of exp(a*b*c) w.r.t. c."""
    fn = torch_gradient_fn("eml(((a * b) * c), 1)", wrt="c")
    a = torch.tensor([0.1, 0.3, -0.2])
    b = torch.tensor([0.5, -0.4, 0.6])
    c = torch.tensor([0.0, 0.2, 0.1])
    got = fn({"a": a, "b": b, "c": c})
    expected = a * b * torch.exp(a * b * c)
    assert torch.allclose(got, expected, atol=1e-5), f"got={got} exp={expected}"


def test_gradient_pair_mul_combo():
    """Flat pair combo `(a*b)` still works for d/da → b (product rule)."""
    vals = {"a": 0.3, "b": 0.7}
    got = gradient_at("eml((a * b), 1)", wrt="a", values=vals)
    # ∂/∂a exp(a*b) = b * exp(a*b)
    analytic = vals["b"] * math.exp(vals["a"] * vals["b"])
    assert abs(got - analytic) < 1e-6


# ─── 6. Z3 / SMT bridge accepts `*` combo and nested parens ─────────────────


def _z3_available():
    try:
        import z3  # noqa: F401

        return True
    except ImportError:
        return False


def _z3_transcendentals_available():
    try:
        import z3
    except ImportError:
        return False
    return hasattr(z3, "Exp") and hasattr(z3, "Ln")


@pytest.mark.skipif(not _z3_available(), reason="z3 not installed")
def test_z3_bridge_combo_mul_is_parsed():
    """AST parse path must accept `*` combos and nested triple parens without
    raising, even on Z3 builds that lack Exp/Ln. The EML -> Z3 path is
    covered by the full-bridge tests below when transcendentals are present.
    """
    from emltorch.gradient import (
        _parse_inner,
        _Combo,
        _Mul,
    )

    # Pair combo parses to _Combo with op="*"
    node_pair = _parse_inner("(a * b)")
    assert isinstance(node_pair, _Combo)
    assert node_pair.op == "*"
    assert node_pair.left == "a" and node_pair.right == "b"

    # Nested triple parses to _Mul (non-flat AST) — the use_mul3 emitted form
    node_triple = _parse_inner("((a * b) * c)")
    assert isinstance(node_triple, _Mul)


@pytest.mark.skipif(
    not _z3_transcendentals_available(),
    reason="z3 lacks Exp/Ln transcendentals (build-dependent)",
)
def test_z3_bridge_eml_with_mul_combo():
    """Full EML→Z3 roundtrip with a pair `*` combo (requires z3 Exp/Ln)."""
    import z3
    from emltorch.smt import eml_formula_to_z3

    a_z, b_z = z3.Reals("a b")
    expr = eml_formula_to_z3("eml((a * b), 1.0)", {"a": a_z, "b": b_z})
    assert isinstance(expr, z3.ArithRef)


@pytest.mark.skipif(
    not _z3_transcendentals_available(),
    reason="z3 lacks Exp/Ln transcendentals (build-dependent)",
)
def test_z3_bridge_nested_triple_parens():
    """Full EML→Z3 roundtrip with a nested triple `((a*b)*c)` form."""
    import z3
    from emltorch.smt import eml_formula_to_z3

    a_z, b_z, c_z = z3.Reals("a b c")
    expr = eml_formula_to_z3("eml(((a * b) * c), 1.0)", {"a": a_z, "b": b_z, "c": c_z})
    assert isinstance(expr, z3.ArithRef)


# ─── 7. Safe operators ──────────────────────────────────────────────────────


def test_safe_mul_clamp_prevents_overflow():
    """safe_mul clamps each operand and the product so nested mul in a
    hybrid tree can't blow up toward inf."""
    left = torch.tensor([1e10, -1e8])
    right = torch.tensor([1e10, 1e8])
    out = safe_mul(left, right)
    # Each operand clamped to 1e6, product to 1e6 -> output magnitudes ≤ 1e6.
    assert (out.abs() <= 1e6 + 1e-3).all()


def test_safe_eml_clamp_prevents_overflow():
    """safe_eml clamps exp() arg at 60 and log arg at >=1e-6."""
    left = torch.tensor([100.0, 200.0])
    right = torch.tensor([-1.0, 1e-30])
    out = safe_eml_hyb(left, right)
    # exp(60) ≈ 1.14e26; log(1e-6) ≈ -13.8. Values must be finite.
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    import pytest as _pytest

    _pytest.main([__file__, "-v"])
