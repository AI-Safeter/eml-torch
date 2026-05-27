"""Tests for emltorch.fit_residual_boost() — gradient-boosting-style residual fit.

Verifies that:
  - n_stages stages are produced
  - cumulative train R² is non-decreasing across stages
  - predict() returns the additive sum of stage predictions
  - the combined fit beats single-stage on targets that mix elementary
    families (e.g. exp + log)
  - invalid n_stages raises ValueError
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import emltorch as eml  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _r2(y_true, y_pred):
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    ss_res = float(np.sum((y_true_np - y_pred_np) ** 2))
    ss_tot = float(np.sum((y_true_np - y_true_np.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def test_boost_basic_shape():
    """fit_residual_boost returns BoostedResult with n_stages FitResults."""
    x = torch.linspace(0.5, 5.0, 256)
    y = torch.log(x)
    result = eml.fit_residual_boost(x, y, n_stages=2, depth=3, device=DEVICE)

    assert isinstance(result, eml.BoostedResult)
    assert result.n_stages == 2
    assert len(result.stage_fits) == 2
    assert all(isinstance(s, eml.FitResult) for s in result.stage_fits)


def test_boost_cumulative_train_r2_nondecreasing():
    """Each additional stage adds to cumulative train fit — R² monotone up."""
    torch.manual_seed(0)
    x = torch.linspace(0.5, 5.0, 256)
    # exp(x) + 0.3 * log(x): two-family target where one EML can't capture
    # both at low depth without boosting
    y = torch.exp(x * 0.5) + 0.3 * torch.log(x)

    result = eml.fit_residual_boost(x, y, n_stages=3, depth=3, device=DEVICE)
    cum = result.cumulative_r2_train
    assert len(cum) == 3
    # Each stage should at least maintain (residual-fit can't make MSE
    # worse on train; small numerical slack allowed).
    assert cum[1] >= cum[0] - 1e-6
    assert cum[2] >= cum[1] - 1e-6


def test_boost_predict_is_additive_sum():
    """result.predict(x) == sum of stage_fits[k].predict(x)."""
    x = torch.linspace(-1.0, 1.0, 128)
    y = torch.exp(x) + 0.2 * torch.sin(x)  # depth-3 mixed

    result = eml.fit_residual_boost(x, y, n_stages=2, depth=3, device=DEVICE)
    x_new = torch.linspace(-0.5, 0.5, 32)

    combined = result.predict(x_new)
    by_stage = [s.predict(x_new) for s in result.stage_fits]
    expected = by_stage[0]
    for s in by_stage[1:]:
        expected = expected + s

    assert torch.allclose(combined, expected, atol=1e-5)


def test_boost_beats_single_stage_on_mixed_target():
    """On a target where one EML tree at fixed depth can't fully capture
    the structure, residual boosting should add measurable R² over the
    single-stage fit. Bar set deliberately low so test is robust to seed."""
    torch.manual_seed(0)
    x = torch.linspace(0.1, 3.0, 300)
    # Target = exp(x) + 0.5*log(x). Single depth-2 EML usually fits one
    # well but residuals show pattern of the other.
    y = torch.exp(x) + 0.5 * torch.log(x)

    single = eml.fit(x, y, depth=2, device=DEVICE)
    boosted = eml.fit_residual_boost(x, y, n_stages=3, depth=2, device=DEVICE)
    # Boosted final train R² should be ≥ single (with small slack).
    assert boosted.final_r2_train >= single.r2 - 1e-3


def test_boost_n_stages_one_matches_single_fit():
    """n_stages=1 should equal calling fit() once (same seed)."""
    torch.manual_seed(0)
    x = torch.linspace(0.5, 5.0, 128)
    y = torch.log(x)

    # fit_residual_boost uses seed_start (default 0) for stage 0
    boosted = eml.fit_residual_boost(x, y, n_stages=1, depth=3, device=DEVICE)
    # Mirror its seeding manually
    torch.manual_seed(0)
    np.random.seed(0)
    single = eml.fit(x, y, depth=3, device=DEVICE)

    # Should land at the same train R² (same seed, same fit() call)
    assert abs(boosted.stage_fits[0].r2 - single.r2) < 1e-6


def test_boost_summary_string():
    """summary() returns a single-line audit string with key fields."""
    x = torch.linspace(0.5, 5.0, 128)
    y = torch.log(x)
    result = eml.fit_residual_boost(x, y, n_stages=2, depth=3, device=DEVICE)

    s = result.summary()
    assert "n_stages=2" in s
    assert "stage1_train_r2" in s
    assert "final_train_r2" in s
    # __repr__ delegates
    assert repr(result) == s


def test_boost_expression_is_additive():
    """expression returns a parenthesized sum of stage expressions: each
    stage's string is wrapped in `(...)`, joined by `) + (`."""
    x = torch.linspace(0.5, 5.0, 128)
    y = torch.log(x)
    result = eml.fit_residual_boost(x, y, n_stages=2, depth=3, device=DEVICE)

    expr = result.expression
    assert expr.startswith("(")
    assert expr.endswith(")")
    # Exactly one `) + (` join between the 2 stage expressions.
    assert expr.count(") + (") == result.n_stages - 1
    # All stage expressions appear verbatim in the combined output.
    for stage in result.stage_fits:
        assert stage.expression in expr


def test_boost_invalid_n_stages_raises():
    """n_stages < 1 raises ValueError."""
    x = torch.linspace(0.5, 5.0, 64)
    y = torch.log(x)
    with pytest.raises(ValueError, match="n_stages"):
        eml.fit_residual_boost(x, y, n_stages=0, depth=3, device=DEVICE)


def test_boost_invalid_seeds_per_stage_raises():
    """seeds_per_stage < 1 raises ValueError."""
    x = torch.linspace(0.5, 5.0, 64)
    y = torch.log(x)
    with pytest.raises(ValueError, match="seeds_per_stage"):
        eml.fit_residual_boost(
            x, y, n_stages=2, seeds_per_stage=0, depth=3, device=DEVICE
        )


def test_boost_seeds_per_stage_runs_and_shapes():
    """seeds_per_stage > 1 still produces n_stages FitResults with a valid
    additive predictor (best-of-seeds selection per stage)."""
    torch.manual_seed(0)
    x = torch.linspace(0.1, 3.0, 200)
    y = torch.exp(x) + 0.4 * torch.log(x)

    result = eml.fit_residual_boost(
        x, y, n_stages=2, seeds_per_stage=3, depth=3, device=DEVICE
    )
    assert result.n_stages == 2
    assert len(result.stage_fits) == 2
    # best-of-seeds selection can only match-or-beat single-seed train fit
    single = eml.fit_residual_boost(
        x, y, n_stages=2, seeds_per_stage=1, depth=3, device=DEVICE
    )
    assert result.final_r2_train >= single.final_r2_train - 1e-3


def test_boost_diminishing_returns_on_pure_signal():
    """If the target is exactly recoverable by a single EML (e.g. log(x)),
    stage 2 has nothing to fit — its R² contribution to the train residual
    should be near zero, and combined R² remains near 1.0."""
    x = torch.linspace(0.5, 5.0, 256)
    y = torch.log(x)

    result = eml.fit_residual_boost(x, y, n_stages=3, depth=3, device=DEVICE)
    # Stage 1 should already nail this (R² > 0.95). Stage 2 has tiny gain.
    assert result.cumulative_r2_train[0] > 0.95
    # Final should still be near 1.0 — boost doesn't degrade.
    assert result.final_r2_train >= result.cumulative_r2_train[0] - 1e-3
