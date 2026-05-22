"""Tests for numerical stability guards and input normalization."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emltorch.operator import safe_eml as std_safe_eml, safe_eml_param as std_safe_eml_param  # noqa: E402
from emltorch.hybrid_mul import safe_eml as hybrid_safe_eml, safe_mul as hybrid_safe_mul, HybridMulConfig, evolve_hybrid_mul  # noqa: E402
from emltorch.sklearn import EMLRegressor  # noqa: E402
from emltorch.evolution import EvolutionConfig, evolve  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_core_operator_nan_inf_guards():
    """Verify that operator.py functions clean inputs/outputs and clamp correctly under extreme values."""
    # Test values
    nan_tensor = torch.tensor([float("nan"), 1.0, float("inf"), -float("inf")])
    extreme_tensor = torch.tensor([1e38, -1e38, 1e20, -1e20])

    # 1. Standard safe_eml
    out_nan = std_safe_eml(nan_tensor, nan_tensor)
    assert torch.isfinite(out_nan).all(), f"expected finite outputs, got {out_nan}"

    out_ext = std_safe_eml(extreme_tensor, extreme_tensor)
    assert torch.isfinite(out_ext).all(), f"expected finite outputs, got {out_ext}"

    # 2. Parameterized safe_eml_param
    alpha = torch.tensor(1.5)
    beta = torch.tensor(0.8)
    out_param_nan = std_safe_eml_param(nan_tensor, nan_tensor, alpha, beta)
    assert torch.isfinite(out_param_nan).all()

    out_param_ext = std_safe_eml_param(extreme_tensor, extreme_tensor, alpha, beta)
    assert torch.isfinite(out_param_ext).all()


def test_hybrid_operator_nan_inf_guards():
    """Verify that hybrid_mul.py operators clean inputs/outputs and clamp correctly under extreme values."""
    nan_tensor = torch.tensor([float("nan"), 1.0, float("inf"), -float("inf")])
    extreme_tensor = torch.tensor([1e38, -1e38, 1e20, -1e20])

    # 1. Hybrid safe_eml
    out_eml_nan = hybrid_safe_eml(nan_tensor, nan_tensor)
    assert torch.isfinite(out_eml_nan).all()

    out_eml_ext = hybrid_safe_eml(extreme_tensor, extreme_tensor)
    assert torch.isfinite(out_eml_ext).all()

    # 2. Hybrid safe_mul
    out_mul_nan = hybrid_safe_mul(nan_tensor, nan_tensor)
    assert torch.isfinite(out_mul_nan).all()

    out_mul_ext = hybrid_safe_mul(extreme_tensor, extreme_tensor)
    assert torch.isfinite(out_mul_ext).all()


def test_input_normalization_evolution():
    """Verify that EvolutionConfig + evolve fits correctly under extreme feature scale when normalize_inputs is True."""
    rng = np.random.default_rng(42)
    # Inputs scaled to 1e9 - 1e12 range
    X_np = np.linspace(1e9, 1e12, 100).reshape(1, -1)
    # y = exp((x - mean)/std) - ln((x - mean)/std) or similar shape
    X_norm = (X_np - X_np.mean()) / (X_np.std() + 1e-8)
    y_np = np.exp(np.clip(X_norm[0], -2.0, 2.0))

    x_tensor = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
    y_tensor = torch.tensor(y_np, dtype=torch.float32, device=DEVICE)

    cfg = EvolutionConfig(
        depth=2,
        num_vars=1,
        population=128,
        generations=5,
        normalize_inputs=True,
        device=DEVICE
    )

    # Should run and return results without NaN / overflow crashes
    res = evolve(x_tensor, y_tensor, cfg)
    assert res.best_r2 > 0.5, f"Expected reasonable R2 under normalized inputs, got {res.best_r2:.4f}"
    
    # Check tree forward evaluation works and applies cached buffers
    preds = res.best_tree(x_tensor.unsqueeze(0).expand(128, 1, -1))
    assert torch.isfinite(preds).all()


def test_input_normalization_hybrid_evolution():
    """Verify that HybridMulConfig + evolve_hybrid_mul fits correctly under extreme scale when normalize_inputs is True."""
    X_np = np.linspace(1e9, 1e12, 100).reshape(1, -1)
    X_norm = (X_np - X_np.mean()) / (X_np.std() + 1e-8)
    y_np = np.exp(np.clip(X_norm[0], -2.0, 2.0))

    x_tensor = torch.tensor(X_np, dtype=torch.float32, device=DEVICE)
    y_tensor = torch.tensor(y_np, dtype=torch.float32, device=DEVICE)

    cfg = HybridMulConfig(
        depth=2,
        num_vars=1,
        population=128,
        generations=5,
        normalize_inputs=True,
        device=DEVICE
    )

    res = evolve_hybrid_mul(x_tensor, y_tensor, cfg)
    assert res.r2 > 0.5, f"Expected reasonable R2 under normalized hybrid inputs, got {res.r2:.4f}"
    
    # Check tree forward works and has cached buffers
    preds = res.tree(x_tensor.unsqueeze(0).expand(128, 1, -1))
    assert torch.isfinite(preds).all()


def test_input_normalization_sklearn_api():
    """Verify that EMLRegressor supports and correctly applies normalize_inputs during fit and predict."""
    X = np.linspace(1e9, 1e12, 100).reshape(-1, 1)
    X_norm = (X - X.mean()) / (X.std() + 1e-8)
    y = np.exp(np.clip(X_norm, -2.0, 2.0)).reshape(-1)

    model = EMLRegressor(depth=2, normalize_inputs=True, population=128, generations=5, device=DEVICE)
    model.fit(X, y)

    assert hasattr(model, "tree_")
    assert model.tree_.normalize_inputs.item() is True
    
    # Check that predict() works and returns correct scale
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert np.isfinite(preds).all()
    
    # sklearn-style score
    score = model.score(X, y)
    assert score > 0.5, f"Expected score > 0.5 under input normalization, got {score:.4f}"
