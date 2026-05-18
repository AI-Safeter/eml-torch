"""Tests for the scikit-learn compatible EMLRegressor wrapper."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emltorch.sklearn import EMLRegressor  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_ln_recovery_sklearn_api():
    """EMLRegressor should recover ln(x) via fit/predict, sklearn-style."""
    rng = np.random.default_rng(0)
    X = np.linspace(0.5, 5.0, 256).reshape(-1, 1)
    y = np.log(X).reshape(-1) + 1e-6 * rng.standard_normal(256)

    model = EMLRegressor(depth=3, device=DEVICE)
    model.fit(X, y)

    assert hasattr(model, "expression_")
    assert "eml" in model.expression_, (
        f"expected 'eml' in expression, got {model.expression_!r}"
    )
    assert model.r2_ > 0.9, (
        f"Expected R2 > 0.9, got {model.r2_:.4f} "
        f"(expression={model.expression_!r})"
    )

    # predict() should also yield reasonable values on the same X.
    pred = model.predict(X)
    assert pred.shape == y.shape
    # sklearn-style score
    score = model.score(X, y)
    assert score > 0.9, f"Expected score > 0.9, got {score:.4f}"
