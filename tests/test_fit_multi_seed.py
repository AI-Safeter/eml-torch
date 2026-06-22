"""Tests for emltorch.fit_multi_seed(), the topology-stability discipline API.

Operationalizes the honest-stability check: run `fit()` with N independent
RNG seeds and ask "does the same closed-form keep emerging?" Byte-equality
topology counting (advisor-recommended precision, same string, not just
structurally similar).
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import emltorch as eml  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_multi_seed_basic_shape():
    """fit_multi_seed returns MultiSeedResult with N FitResults."""
    x = torch.linspace(0.5, 5.0, 256)
    y = torch.log(x)
    result = eml.fit_multi_seed(x, y, n_seeds=3, depth=3, device=DEVICE)

    assert result.n_seeds == 3
    assert len(result.all_results) == 3
    assert all(isinstance(r, eml.FitResult) for r in result.all_results)
    # MultiSeedResult exposes the best-R² FitResult
    assert isinstance(result.best_fit, eml.FitResult)


def test_multi_seed_aggregate_metrics():
    """best_r2 / median_r2 / mean_r2 / std_r2 are coherent."""
    x = torch.linspace(0.5, 5.0, 256)
    y = torch.log(x)
    result = eml.fit_multi_seed(x, y, n_seeds=3, depth=3, device=DEVICE)

    r2s = [r.r2 for r in result.all_results]
    assert abs(result.best_r2 - max(r2s)) < 1e-9
    assert result.median_r2 <= result.best_r2
    assert result.mean_r2 <= result.best_r2
    assert result.std_r2 >= 0.0


def test_multi_seed_topology_counting():
    """topology_counts is a byte-equality count of expression strings."""
    x = torch.linspace(0.5, 5.0, 256)
    y = torch.log(x)
    result = eml.fit_multi_seed(x, y, n_seeds=4, depth=3, device=DEVICE)

    # The Counter should sum to n_seeds
    assert sum(result.topology_counts.values()) == 4
    # top_topology must be a key with the maximum count
    assert result.top_topology in result.topology_counts
    assert result.top_topology_count == max(result.topology_counts.values())
    assert result.n_unique_topologies == len(result.topology_counts)
    # Stability fraction is in [0, 1]
    assert 0.0 < result.topology_stability <= 1.0
    expected = result.top_topology_count / 4
    assert abs(result.topology_stability - expected) < 1e-12


def test_multi_seed_predict_uses_best_fit():
    """predict() delegates to best_fit's predict()."""
    x = torch.linspace(0.5, 5.0, 256)
    y = torch.log(x)
    result = eml.fit_multi_seed(x, y, n_seeds=3, depth=3, device=DEVICE)

    x_new = torch.linspace(1.0, 4.0, 64)
    yp_multi = result.predict(x_new)
    yp_best = result.best_fit.predict(x_new)
    assert torch.allclose(yp_multi, yp_best)


def test_multi_seed_recovers_known_function():
    """At least one seed of N should hit a sensible R² for a recoverable
    elementary function. Same standard as test_fit.test_known_recovery."""
    x = torch.linspace(-2.0, 2.0, 256)
    y = torch.exp(x)
    result = eml.fit_multi_seed(x, y, n_seeds=3, depth=1, device=DEVICE)
    assert result.best_r2 > 0.99, (
        f"exp(x) at depth 1 failed: best R²={result.best_r2:.4f}, "
        f"expressions={list(result.topology_counts.keys())}"
    )


def test_multi_seed_seed_start():
    """seed_start shifts the RNG sequence, different start gives different results."""
    x = torch.linspace(0.5, 5.0, 256)
    y = torch.log(x)
    r0 = eml.fit_multi_seed(x, y, n_seeds=2, depth=3, seed_start=0, device=DEVICE)
    r100 = eml.fit_multi_seed(x, y, n_seeds=2, depth=3, seed_start=100, device=DEVICE)
    # Different seed ranges may or may not produce the same expressions, but
    # the n_seeds + structure are consistent. This test mainly exercises the
    # `seed_start` arg path.
    assert r0.n_seeds == r100.n_seeds == 2
    assert sum(r0.topology_counts.values()) == 2
    assert sum(r100.topology_counts.values()) == 2


def test_multi_seed_invalid_n_seeds():
    """n_seeds < 1 should raise ValueError."""
    x = torch.linspace(0.5, 5.0, 64)
    y = torch.log(x)
    with pytest.raises(ValueError, match="n_seeds"):
        eml.fit_multi_seed(x, y, n_seeds=0, depth=3, device=DEVICE)


def test_multi_seed_summary_string():
    """summary() returns a non-empty, parseable single-line string."""
    x = torch.linspace(0.5, 5.0, 256)
    y = torch.log(x)
    result = eml.fit_multi_seed(x, y, n_seeds=3, depth=3, device=DEVICE)

    s = result.summary()
    assert "n_seeds=3" in s
    assert "best_r2" in s
    assert "topology_stability" in s
    # __repr__ delegates to summary
    assert repr(result) == s
