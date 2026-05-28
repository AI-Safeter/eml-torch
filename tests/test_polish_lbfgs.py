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
