"""
Tests for the L-BFGS constant-optimization mode in polish().

Tests are ordered so the gating backward-compat test (test 1) must pass
before any lbfgs feature is validated.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import emltorch as eml
from emltorch.evolution import evolve, EvolutionConfig
from emltorch.polish import polish


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _make_target(device="cpu"):
    """y = 2.5*exp(x) - 1.3 on x in [0.2, 3.0], 256 points."""
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.linspace(0.2, 3.0, 256, device=device)
    y = 2.5 * torch.exp(x) - 1.3
    return x, y


def _evolve_tree(x, y, depth=3, seed=42):
    """Run a small evolution pass and return (tree, idx, a, b)."""
    cfg = EvolutionConfig(
        depth=depth,
        num_vars=1,
        population=256,
        generations=10,
        device="cpu",
        verbose=False,
        r2_target=0.999,
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    res = evolve(x, y, cfg)
    return res.best_tree, res.best_idx, res.best_a, res.best_b


# ---------------------------------------------------------------------------
# Test 1 — gating backward-compat test
# ---------------------------------------------------------------------------


def test_adam_mode_is_unchanged_default():
    """
    polish(..., optimizer="adam") must return bit-identical results to
    polish(...) called WITHOUT the optimizer kwarg.

    This is the gating test: it proves the default path is completely
    untouched by the new kwarg.
    """
    x, y = _make_target()
    tree, idx, warm_a, warm_b = _evolve_tree(x, y)

    # Call without kwarg (default behaviour)
    torch.manual_seed(7)
    res_default = polish(
        tree,
        idx,
        x,
        y,
        var_names=["x"],
        device="cpu",
        warm_a=warm_a,
        warm_b=warm_b,
    )

    # Call with explicit optimizer="adam"
    torch.manual_seed(7)
    res_adam = polish(
        tree,
        idx,
        x,
        y,
        var_names=["x"],
        device="cpu",
        warm_a=warm_a,
        warm_b=warm_b,
        optimizer="adam",
    )

    assert (
        abs(res_default.r2 - res_adam.r2) < 1e-9
    ), f"r2 mismatch: default={res_default.r2:.10f}  adam={res_adam.r2:.10f}"
    for i, (c1, c2) in enumerate(zip(res_default.constants, res_adam.constants)):
        assert abs(c1 - c2) < 1e-9, f"constants[{i}] mismatch: default={c1}  adam={c2}"


# ---------------------------------------------------------------------------
# Test 2 — lbfgs runs and returns finite values
# ---------------------------------------------------------------------------


def test_lbfgs_runs_and_is_finite():
    """
    optimizer="lbfgs" must return finite r2 and predictions,
    and r2 must be >= warm-start r2 (never-worse invariant).
    """
    x, y = _make_target()
    tree, idx, warm_a, warm_b = _evolve_tree(x, y)

    # Compute warm-start r2 directly
    from emltorch.polish import _FixedTopologyTree

    leaf_idx_all, internal_idx_all = tree.snapped_choices()
    leaf_choices = leaf_idx_all[idx].to("cpu")
    internal_choices = [c[idx].to("cpu") for c in internal_idx_all]
    fixed = _FixedTopologyTree(
        leaf_choices=leaf_choices,
        internal_choices=internal_choices,
        num_vars=1,
    )
    with torch.no_grad():
        ws_pred = fixed(x.unsqueeze(0))
        ws_fit = warm_a + warm_b * ws_pred
        ws_mse = (ws_fit - y).pow(2).mean().item()
    N = y.shape[0]
    ss_tot = ((y - y.mean()).pow(2).sum()).item()
    warm_r2 = 1 - ws_mse * N / max(ss_tot, 1e-12)

    res = polish(
        tree,
        idx,
        x,
        y,
        var_names=["x"],
        device="cpu",
        warm_a=warm_a,
        warm_b=warm_b,
        optimizer="lbfgs",
    )

    assert math.isfinite(res.r2), f"r2 is not finite: {res.r2}"
    assert math.isfinite(res.mse), f"mse is not finite: {res.mse}"
    assert (
        res.r2 >= warm_r2 - 1e-9
    ), f"LBFGS returned worse than warm-start: lbfgs_r2={res.r2:.6f}  warm_r2={warm_r2:.6f}"


# ---------------------------------------------------------------------------
# Test 3 — lbfgs at least matches adam on constant-sensitive target
# ---------------------------------------------------------------------------


def test_lbfgs_at_least_matches_adam_on_constant_sensitive_target():
    """
    On y = 2.5*exp(x) - 1.3, L-BFGS should at least match Adam's r2.
    Report the actual delta.
    """
    x, y = _make_target()
    tree, idx, warm_a, warm_b = _evolve_tree(x, y)

    torch.manual_seed(7)
    res_adam = polish(
        tree,
        idx,
        x,
        y,
        var_names=["x"],
        device="cpu",
        warm_a=warm_a,
        warm_b=warm_b,
        optimizer="adam",
    )

    res_lbfgs = polish(
        tree,
        idx,
        x,
        y,
        var_names=["x"],
        device="cpu",
        warm_a=warm_a,
        warm_b=warm_b,
        optimizer="lbfgs",
    )

    delta = res_lbfgs.r2 - res_adam.r2
    assert res_lbfgs.r2 >= res_adam.r2 - 1e-6, (
        f"LBFGS r2={res_lbfgs.r2:.8f} is worse than Adam r2={res_adam.r2:.8f} "
        f"by delta={delta:.2e} (exceeds -1e-6 tolerance)"
    )
    # Informational: report the actual delta
    print(
        f"\n[test3] LBFGS r2={res_lbfgs.r2:.8f}  Adam r2={res_adam.r2:.8f}  delta={delta:+.2e}"
    )


# ---------------------------------------------------------------------------
# Test 4 — adam+lbfgs at least matches adam
# ---------------------------------------------------------------------------


def test_adam_plus_lbfgs_at_least_matches_adam():
    """
    optimizer="adam+lbfgs" r2 must be >= optimizer="adam" r2 - 1e-6.
    """
    x, y = _make_target()
    tree, idx, warm_a, warm_b = _evolve_tree(x, y)

    torch.manual_seed(7)
    res_adam = polish(
        tree,
        idx,
        x,
        y,
        var_names=["x"],
        device="cpu",
        warm_a=warm_a,
        warm_b=warm_b,
        optimizer="adam",
    )

    torch.manual_seed(7)
    res_combo = polish(
        tree,
        idx,
        x,
        y,
        var_names=["x"],
        device="cpu",
        warm_a=warm_a,
        warm_b=warm_b,
        optimizer="adam+lbfgs",
    )

    delta = res_combo.r2 - res_adam.r2
    assert res_combo.r2 >= res_adam.r2 - 1e-6, (
        f"adam+lbfgs r2={res_combo.r2:.8f} is worse than adam r2={res_adam.r2:.8f} "
        f"by delta={delta:.2e} (exceeds -1e-6 tolerance)"
    )
    print(
        f"\n[test4] adam+lbfgs r2={res_combo.r2:.8f}  adam r2={res_adam.r2:.8f}  delta={delta:+.2e}"
    )


# ---------------------------------------------------------------------------
# Test 5 — non-finite safety: never-worse invariant under risky topology
# ---------------------------------------------------------------------------


def test_nonfinite_safety():
    """
    When the target has extreme curvature (could cause LBFGS divergence),
    polish with optimizer="lbfgs" must still return finite r2 and must
    never be worse than the warm-start r2.
    """
    # Use a target with very large dynamic range that stresses the optimizer
    torch.manual_seed(99)
    x = torch.linspace(0.1, 5.0, 128)
    # exp(exp(x)) grows very fast — stresses any constant optimization
    y = torch.exp(torch.exp(x.clamp(max=2.0))) * 0.001 - 500.0

    tree, idx, warm_a, warm_b = _evolve_tree(x, y, depth=3, seed=99)

    # Compute warm-start r2
    from emltorch.polish import _FixedTopologyTree

    leaf_idx_all, internal_idx_all = tree.snapped_choices()
    leaf_choices = leaf_idx_all[idx].to("cpu")
    internal_choices = [c[idx].to("cpu") for c in internal_idx_all]
    fixed_ws = _FixedTopologyTree(
        leaf_choices=leaf_choices,
        internal_choices=internal_choices,
        num_vars=1,
    )
    with torch.no_grad():
        ws_pred = fixed_ws(x.unsqueeze(0))
        ws_fit = warm_a + warm_b * ws_pred
        ws_mse = (ws_fit - y).pow(2).mean().item()
    N = y.shape[0]
    ss_tot = ((y - y.mean()).pow(2).sum()).item()
    warm_r2 = 1 - ws_mse * N / max(ss_tot, 1e-12)

    res = polish(
        tree,
        idx,
        x,
        y,
        var_names=["x"],
        device="cpu",
        warm_a=warm_a,
        warm_b=warm_b,
        optimizer="lbfgs",
    )

    assert math.isfinite(res.r2), f"r2 is not finite: {res.r2}"
    assert math.isfinite(res.mse), f"mse is not finite: {res.mse}"
    assert res.r2 >= warm_r2 - 1e-9, (
        f"LBFGS returned worse than warm-start: "
        f"lbfgs_r2={res.r2:.6f}  warm_r2={warm_r2:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 6 — empirical proof-of-value: LBFGS at small iter budget
# ---------------------------------------------------------------------------


def test_lbfgs_at_small_iter_budget_matches_or_beats_adam():
    """Empirical evidence that the LBFGS mode does something distinct from Adam.

    At the default n_iters=2000 both methods saturate to the same fixed point on
    smooth analytic targets, so they tie (the honest result reported in test 3).
    The discriminator is the SMALL-iter regime: LBFGS's strong-Wolfe line
    search takes quasi-Newton-class steps, so at constrained budgets it reaches
    a lower MSE than Adam at lr=1e-2 even when both safeguards keep things
    finite. This test runs polish at n_iters=30 with a depth-4 evolved tree
    (more leaf constants → bigger optimization space) and asserts LBFGS reaches
    MSE no worse than Adam, while printing the actual ratio as evidence.

    Discipline: this asserts the safety property (LBFGS never strictly worse at
    small budget); the test does NOT claim LBFGS always beats Adam, because on
    fully affine-absorbable targets they correctly tie at the saturated optimum.
    """
    x, y = _make_target("cpu")
    tree, idx, a, b = _evolve_tree(x, y, depth=4, seed=42)
    n_small = 30  # well below the n_iters=2000 saturation regime

    pol_adam = polish(
        tree, idx, x, y, var_names=["x"],
        n_iters=n_small, lr=1e-2, device="cpu",
        warm_a=a, warm_b=b, optimizer="adam",
    )
    pol_lbfgs = polish(
        tree, idx, x, y, var_names=["x"],
        n_iters=n_small, lr=1e-2, device="cpu",
        warm_a=a, warm_b=b, optimizer="lbfgs",
    )

    mse_a, mse_l = pol_adam.mse, pol_lbfgs.mse
    r2_a, r2_l = pol_adam.r2, pol_lbfgs.r2
    # Evidence emitted for the test log (visible with pytest -s).
    print(
        f"\n  n_iters={n_small}  adam: r2={r2_a:.6f} mse={mse_a:.3e}  "
        f"lbfgs: r2={r2_l:.6f} mse={mse_l:.3e}  Δr2={r2_l-r2_a:+.2e}"
    )

    # Safety properties — both must hold for the LBFGS mode to be acceptable.
    assert math.isfinite(r2_l) and math.isfinite(mse_l)
    # LBFGS may equal Adam at saturation; it must not regress beyond float noise.
    assert mse_l <= mse_a * 1.001 + 1e-9, (
        f"At small-budget polish, LBFGS reached worse MSE than Adam: "
        f"adam_mse={mse_a:.6e}, lbfgs_mse={mse_l:.6e}"
    )
