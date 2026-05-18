"""Regression: emltorch.fit accepts numpy / list / torch, both shape
conventions ((N, V) sklearn and (V, N)), and 1D x."""

import numpy as np
import pytest
import torch

import emltorch


def _make_xy(N=64, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(N, 1)).astype(np.float32)
    y = np.exp(x[:, 0]).astype(np.float32)
    return x, y


@pytest.mark.parametrize(
    "wrap_x,wrap_y",
    [
        (np.asarray, np.asarray),
        (list, list),
        (torch.as_tensor, torch.as_tensor),
    ],
)
def test_fit_accepts_array_types(wrap_x, wrap_y):
    x, y = _make_xy()
    x_in = wrap_x(x.tolist()) if wrap_x is list else wrap_x(x)
    y_in = wrap_y(y.tolist()) if wrap_y is list else wrap_y(y)
    res = emltorch.fit(x_in, y_in, depth=2, population=64, generations=2)
    assert res.r2 == res.r2  # not NaN


def test_fit_1d_x():
    x, y = _make_xy()
    res = emltorch.fit(x[:, 0], y, depth=2, population=64, generations=2)
    assert res.r2 == res.r2


def test_fit_sklearn_shape_NV():
    """x shape (N, V) — the common sklearn convention."""
    x, y = _make_xy(N=80)
    res = emltorch.fit(
        torch.as_tensor(x), torch.as_tensor(y), depth=2, population=64, generations=2
    )
    assert res.r2 == res.r2


def test_fit_legacy_shape_VN():
    """x shape (V, N) — original convention, still supported."""
    x, y = _make_xy(N=80)
    x_vn = torch.as_tensor(x).t().contiguous()  # (1, 80)
    res = emltorch.fit(x_vn, torch.as_tensor(y), depth=2, population=64, generations=2)
    assert res.r2 == res.r2


def test_fit_column_vector_y():
    """y as (N, 1) is squeezed to (N,)."""
    x, y = _make_xy()
    y_col = y.reshape(-1, 1)
    res = emltorch.fit(x, y_col, depth=2, population=64, generations=2)
    assert res.r2 == res.r2


def test_fit_rejects_incompatible_shapes():
    """Clear error when x and y can't be aligned."""
    x = torch.randn(50, 3)
    y = torch.randn(100)
    with pytest.raises(ValueError, match="incompatible"):
        emltorch.fit(x, y, depth=2, population=64, generations=2)


def test_fit_rejects_empty_input():
    """Non-empty x/y is a hard requirement (downstream evolve crashes opaquely)."""
    with pytest.raises(ValueError, match="non-empty"):
        emltorch.fit(torch.empty(0), torch.empty(0), depth=2)


def test_fit_rejects_nan_inf():
    """NaN/Inf in inputs must error early, not produce NaN R² silently."""
    x = torch.linspace(-1.0, 1.0, 64)
    y = torch.exp(x)
    y_nan = y.clone()
    y_nan[5] = float("nan")
    with pytest.raises(ValueError, match="NaN/Inf"):
        emltorch.fit(x, y_nan, depth=2, population=64, generations=2)

    x_inf = x.clone()
    x_inf[3] = float("inf")
    with pytest.raises(ValueError, match="NaN/Inf"):
        emltorch.fit(x_inf, y, depth=2, population=64, generations=2)


def test_fit_warns_on_square_input():
    """Ambiguous square input should warn, not silently transpose."""
    N = 32
    x = torch.randn(N, N)
    y = torch.randn(N)
    with pytest.warns(UserWarning, match="square"):
        emltorch.fit(x, y, depth=2, population=64, generations=2)


def test_fit_predict_round_trip():
    """FitResult.predict(x) recovers training values; in-sample R² matches."""
    torch.manual_seed(0)
    x = torch.linspace(-2.0, 2.0, 128)
    y = torch.exp(x)
    r = emltorch.fit(x, y, depth=3, population=512, generations=20)
    y_pred = r.predict(x)
    ss_res = ((y - y_pred) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item()
    r2_eval = 1 - ss_res / max(ss_tot, 1e-12)
    # predict should agree with reported R² to within numerical noise
    assert abs(r2_eval - r.r2) < 0.01, (r2_eval, r.r2)


def test_fit_predict_ood():
    """Discovered exp(x) extrapolates exactly off-distribution."""
    torch.manual_seed(0)
    x_tr = torch.linspace(-3.0, 0.0, 256)
    y_tr = torch.exp(x_tr)
    r = emltorch.fit(x_tr, y_tr, depth=3, population=1024, generations=20)
    if r.r2 < 0.999:
        pytest.skip(f"train R²={r.r2:.4f} < 0.999; structural recovery missed")
    x_te = torch.linspace(-10.0, -5.0, 256)
    y_te = torch.exp(x_te)
    y_pred = r.predict(x_te)
    ss_res = ((y_te - y_pred) ** 2).sum().item()
    ss_tot = ((y_te - y_te.mean()) ** 2).sum().item()
    r2_ood = 1 - ss_res / max(ss_tot, 1e-12)
    assert (
        r2_ood > 0.99
    ), f"OOD R²={r2_ood} (expected > 0.99 on structural exp recovery)"
