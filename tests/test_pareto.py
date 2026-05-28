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
    """A dominated point (higher-or-equal complexity, lower-or-equal r2 vs another)
    must NOT appear on the front, but MUST appear in all_evaluated."""
    x = torch.linspace(0.5, 5.0, 256)
    y = torch.log(x)

    result = eml.fit_pareto(x, y, depths=(1, 2, 3, 4), seeds_per_depth=1, device=DEVICE)

    # Find dominated points by definition
    all_pts = [(c, r2) for c, r2, _ in result.all_evaluated]
    front_set = set((c, round(r2, 10)) for c, r2, _ in result.front)

    dominated_found = False
    for ci, ri in all_pts:
        for cj, rj in all_pts:
            if (ci, ri) == (cj, rj):
                continue
            # cj dominates ci
            if cj <= ci and rj >= ri and (cj < ci or rj > ri):
                dominated_found = True
                # ci is dominated — should NOT be on front
                assert (ci, round(ri, 10)) not in front_set, (
                    f"Dominated point (c={ci}, r2={ri:.6f}) should not be on front, "
                    f"but it is. It is dominated by (c={cj}, r2={rj:.6f})."
                )
                # But MUST be in all_evaluated
                all_eval_pts = set(
                    (c, round(r2, 10)) for c, r2, _ in result.all_evaluated
                )
                assert (
                    ci,
                    round(ri, 10),
                ) in all_eval_pts, (
                    f"Dominated point (c={ci}, r2={ri:.6f}) should be in all_evaluated"
                )

    if not dominated_found:
        pytest.skip(
            "No dominated points found in this run — all depths produced distinct "
            "Pareto-optimal points. This is valid; the test can't assert on absent data."
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
