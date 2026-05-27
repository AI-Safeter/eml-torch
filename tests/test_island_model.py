"""Tests for the island-model (multi-population) evolution path.

Verifies:
  - n_islands=1 is byte-identical to the default panmictic path (same seed
    → same expression and R²), so the feature is zero-risk to existing runs.
  - n_islands>1 runs end-to-end and recovers a simple elementary target.
  - population is rounded up to a multiple of n_islands by the top-level fit.
  - evolve() raises a clear error on a non-divisible population.
  - invalid n_islands is rejected.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import emltorch as eml  # noqa: E402
from emltorch.evolution import EvolutionConfig, evolve  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _seeded_fit(**kw):
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.linspace(0.5, 5.0, 256)
    y = torch.log(x)
    return eml.fit(x, y, depth=3, device=DEVICE, **kw)


def test_n_islands_one_matches_default():
    """n_islands=1 takes the original panmictic code path; with the same
    seed it must reproduce the no-island result exactly."""
    base = _seeded_fit()
    one = _seeded_fit(n_islands=1)
    assert one.expression == base.expression
    assert abs(one.r2 - base.r2) < 1e-9


def test_islands_run_and_recover_log():
    """A 4-island fit recovers log(x) to high R² (sanity: the island path
    is functional, not just non-crashing)."""
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.linspace(0.5, 5.0, 256)
    y = torch.log(x)
    res = eml.fit(x, y, depth=3, device=DEVICE, n_islands=4, population=512)
    assert res.r2 > 0.9
    assert torch.isfinite(res.predict(x)).all()


def test_population_rounded_up_for_islands():
    """A population not divisible by n_islands is rounded up by fit() so the
    island reshape is valid (no error)."""
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.linspace(0.5, 5.0, 128)
    y = torch.log(x)
    # 100 is not divisible by 3 → fit() must round to 102 internally.
    res = eml.fit(x, y, depth=3, device=DEVICE, n_islands=3, population=100)
    assert res.r2 > 0.8


def test_evolve_nondivisible_population_raises():
    """Constructing EvolutionConfig directly with a non-divisible population
    and n_islands>1 raises a clear ValueError in evolve()."""
    x = torch.linspace(0.5, 5.0, 64)
    y = torch.log(x)
    cfg = EvolutionConfig(
        depth=3,
        num_vars=1,
        population=10,
        generations=2,
        n_islands=3,
        device=DEVICE,
    )
    with pytest.raises(ValueError, match="divisible by n_islands"):
        evolve(x, y, cfg)


def test_invalid_n_islands_raises():
    """n_islands < 1 is rejected by fit()."""
    x = torch.linspace(0.5, 5.0, 64)
    y = torch.log(x)
    with pytest.raises(ValueError, match="n_islands"):
        eml.fit(x, y, depth=3, device=DEVICE, n_islands=0)


def test_islands_predict_shape():
    """predict() on the island-fit result returns the right length."""
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.linspace(0.5, 5.0, 200)
    y = torch.log(x)
    res = eml.fit(x, y, depth=3, device=DEVICE, n_islands=4, population=256)
    out = res.predict(torch.linspace(0.5, 5.0, 37))
    assert out.shape[0] == 37
