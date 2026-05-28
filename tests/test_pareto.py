"""Tests for emltorch.fit_pareto() — the Pareto-front accuracy/complexity API.

Mirrors the discipline of test_fit_multi_seed.py: TDD-first, run red,
then implement.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import emltorch as eml  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Test 1 — basic shape and subset invariant
# ---------------------------------------------------------------------------


def test_front_nonempty_and_subset():
    """The Pareto front is non-empty and every front point is in all_evaluated."""
    x = torch.linspace(0.5, 5.0, 256)
    y = torch.log(x)

    result = eml.fit_pareto(x, y, depths=(1, 2, 3, 4), seeds_per_depth=1, device=DEVICE)

    assert isinstance(result, eml.ParetoResult)
    assert len(result.front) > 0, "front should be non-empty"
    assert len(result.all_evaluated) > 0, "all_evaluated should be non-empty"

    # all_evaluated length == number of depths tried
    assert len(result.all_evaluated) == 4

    # Every front tuple must be in all_evaluated
    all_set = set((c, round(r2, 10)) for c, r2, _ in result.all_evaluated)
    for c, r2, fit in result.front:
        assert (
            c,
            round(r2, 10),
        ) in all_set, f"Front point (c={c}, r2={r2}) not found in all_evaluated"

    # Each entry has the right shape: (int, float, FitResult)
    for c, r2, fit in result.front:
        assert isinstance(c, int), f"complexity should be int, got {type(c)}"
        assert isinstance(r2, float), f"r2 should be float, got {type(r2)}"
        assert isinstance(fit, eml.FitResult)

    # Summary does not crash
    s = result.summary()
    assert "->" in s, "summary should contain 'complexity->r2' pairs"


# ---------------------------------------------------------------------------
# Test 2 — strict Pareto property
# ---------------------------------------------------------------------------


def test_pareto_property():
    """Along the sorted front, complexity strictly increases AND r2 strictly increases.
    No front point dominates another.
    """
    x = torch.linspace(0.5, 5.0, 256)
    y = torch.log(x)

    result = eml.fit_pareto(x, y, depths=(1, 2, 3, 4), seeds_per_depth=1, device=DEVICE)

    front = result.front
    # Front must be sorted by complexity
    complexities = [c for c, _, _ in front]
    assert complexities == sorted(
        complexities
    ), f"Front not sorted by complexity: {complexities}"

    if len(front) > 1:
        for i in range(len(front) - 1):
            c_i, r2_i, _ = front[i]
            c_j, r2_j, _ = front[i + 1]
            assert c_j > c_i, (
                f"Complexity not strictly increasing at indices {i},{i+1}: "
                f"{c_i} -> {c_j}"
            )
            assert r2_j > r2_i, (
                f"R² not strictly increasing at indices {i},{i+1}: "
                f"{r2_i:.6f} -> {r2_j:.6f}"
            )

    # No front point dominates another:
    # P_i dominates P_j iff c_i <= c_j and r2_i >= r2_j and (c_i<c_j or r2_i>r2_j)
    for i, (ci, ri, _) in enumerate(front):
        for j, (cj, rj, _) in enumerate(front):
            if i == j:
                continue
            # Check P_i does NOT dominate P_j
            dominates = (ci <= cj) and (ri >= rj) and (ci < cj or ri > rj)
            assert not dominates, (
                f"Front point {i} (c={ci}, r2={ri:.6f}) dominates "
                f"front point {j} (c={cj}, r2={rj:.6f}) — front is not Pareto-optimal"
            )


# ---------------------------------------------------------------------------
# Test 3 — dominated points are excluded from front, kept in all_evaluated
# ---------------------------------------------------------------------------


def test_dominated_depth_excluded():
    """A dominated point must NOT appear on the front but MUST appear in all_evaluated.

    Construction. On ``y = log(x)``, depth 1 already finds the exact identity
    ``eml(1, x) = e − log(x)`` (R² ≈ 1.0 − float-noise). Deeper depths re-find
    the same identity wrapped in extra layers, yielding (complexity > 1, R² ≈ 1).
    Under the float-tolerance domination rule (``rtol_r2=1e-9``), every deeper
    point with R² within tolerance of depth-1's R² is dominated by depth-1
    (strictly smaller complexity, equal-within-tol R²) and must be excluded
    from the front.
    """
    x = torch.linspace(0.5, 5.0, 256)
    y = torch.log(x)

    result = eml.fit_pareto(x, y, depths=(1, 2, 3, 4), seeds_per_depth=1, device=DEVICE)

    # Confirm at least one dominated point exists in this construction
    # (depth-1 hits R²≈1 first; deeper depths re-find it at higher complexity).
    all_pts = [(c, r2) for c, r2, _ in result.all_evaluated]
    rtol = 1e-9
    dominated_pts = [
        (ci, ri)
        for ci, ri in all_pts
        for cj, rj in all_pts
        if (ci, ri) != (cj, rj)
        and cj <= ci
        and rj >= ri - rtol
        and (cj < ci or rj > ri + rtol)
    ]
    assert dominated_pts, (
        "Construction precondition: expected at least one dominated point on the "
        "log(x) target under depths=(1,2,3,4). all_evaluated="
        f"{[(c, round(r, 6)) for c, r, _ in result.all_evaluated]}"
    )

    # Every dominated point must be excluded from the front but kept in all_evaluated.
    front_set = set((c, round(r2, 10)) for c, r2, _ in result.front)
    all_eval_set = set((c, round(r2, 10)) for c, r2, _ in result.all_evaluated)
    for ci, ri in dominated_pts:
        assert (
            ci,
            round(ri, 10),
        ) not in front_set, (
            f"Dominated point (c={ci}, r2={ri:.10f}) must NOT be on front."
        )
        assert (
            ci,
            round(ri, 10),
        ) in all_eval_set, (
            f"Dominated point (c={ci}, r2={ri:.10f}) must remain in all_evaluated."
        )


def test_float_tolerance_collapses_near_tied_r2():
    """A near-tied R² (within rtol_r2) should not place both points on the front.

    Concrete case: on ``y = log(x)``, depth-1 R² and depth-≥2 R² typically differ
    by ~1e-15 (float noise). With the default ``rtol_r2 = 1e-9`` the higher-
    complexity point is dominated. With ``rtol_r2 = 0`` it would survive — that
    is the degenerate behaviour we are explicitly guarding against here.
    """
    x = torch.linspace(0.5, 5.0, 256)
    y = torch.log(x)

    tol = eml.fit_pareto(x, y, depths=(1, 2, 3, 4), seeds_per_depth=1, device=DEVICE)
    no_tol = eml.fit_pareto(
        x, y, depths=(1, 2, 3, 4), seeds_per_depth=1, device=DEVICE, rtol_r2=0.0
    )

    # With tolerance, the front should be a SUBSET of the no-tolerance front
    # (tolerance can only DROP points, never add).
    tol_pts = set((c, round(r2, 10)) for c, r2, _ in tol.front)
    no_tol_pts = set((c, round(r2, 10)) for c, r2, _ in no_tol.front)
    assert tol_pts.issubset(no_tol_pts), (
        f"Tolerant front must be a subset of zero-tolerance front. "
        f"tol={tol_pts}, no_tol={no_tol_pts}"
    )
    # And on this construction the tolerance MUST actually drop at least one
    # point (otherwise the tolerance fix has no observable effect on the
    # advisor-flagged case).
    assert len(tol_pts) < len(no_tol_pts) or len(no_tol_pts) == 1, (
        f"Expected rtol_r2=1e-9 to collapse a near-tied front on log(x); "
        f"got tol_front_size={len(tol_pts)} == no_tol_front_size={len(no_tol_pts)}"
    )


# ---------------------------------------------------------------------------
# Test 4 — select(), best(), predict()
# ---------------------------------------------------------------------------


def test_select_and_best():
    """select(max_complexity) returns a FitResult with complexity <= max_complexity
    and the highest r2 within that budget; best() returns max-r2 front fit;
    predict() returns a tensor of the right length."""
    x = torch.linspace(0.5, 5.0, 256)
    y = torch.log(x)

    result = eml.fit_pareto(x, y, depths=(1, 2, 3, 4), seeds_per_depth=1, device=DEVICE)

    # best() returns highest-r2 front fit
    best = result.best()
    assert isinstance(best, eml.FitResult)
    best_r2 = best.r2
    for _, r2, _ in result.front:
        assert (
            r2 <= best_r2 + 1e-9
        ), f"best().r2={best_r2:.6f} is not max on front (found {r2:.6f})"

    # predict() delegates to best() and returns correct length
    x_new = torch.linspace(1.0, 4.0, 64)
    yp = result.predict(x_new)
    assert isinstance(yp, torch.Tensor)
    assert yp.shape == (64,), f"Expected shape (64,), got {yp.shape}"

    # select(max_complexity) — choose a budget that should work (minimum complexity)
    min_complexity = result.front[0][0]
    selected = result.select(max_complexity=min_complexity)
    assert (
        selected is not None
    ), "select(min_complexity) should return the first front point"
    assert isinstance(selected, eml.FitResult)
    assert eml._expression_complexity(selected.expression) <= min_complexity

    # select(max_complexity) with budget that excludes all front points
    # Use -1 or 0 if min is 0; else use min_complexity - 1
    too_tight = min_complexity - 1
    none_result = result.select(max_complexity=too_tight)
    assert (
        none_result is None
    ), f"select({too_tight}) should return None when min front complexity={min_complexity}"

    # select(large budget) returns max-r2 within budget — should match best()
    large_budget = 1000
    selected_large = result.select(max_complexity=large_budget)
    assert selected_large is not None
    assert selected_large.r2 >= best_r2 - 1e-9


# ---------------------------------------------------------------------------
# Test 5 — __repr__ delegates to summary()
# ---------------------------------------------------------------------------


def test_repr_and_summary():
    """__repr__ and summary() return consistent non-empty strings."""
    x = torch.linspace(0.5, 5.0, 128)
    y = torch.log(x)

    result = eml.fit_pareto(x, y, depths=(1, 2), seeds_per_depth=1, device=DEVICE)

    s = result.summary()
    assert isinstance(s, str)
    assert len(s) > 0
    assert repr(result) == s
