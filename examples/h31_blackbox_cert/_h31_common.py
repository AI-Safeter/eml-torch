"""Shared helpers for the H31 example scripts.

Centralizes feature layout, train/test split, R² scoring, measurement
loading, and the EML predict bridge so the same logic doesn't drift
across the 8 H31 scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"

# Feature layout matches emltorch's default x1..x5 variable naming.
FEAT_ORDER = ["T", "L", "n_rep", "entropy_top50", "log_tok_id"]


def load_measurements(tag: str) -> list[dict]:
    """Load JSONL measurements file by tag (e.g., 'qwen36')."""
    out = []
    with (OUT_DIR / f"measurements_{tag}.jsonl").open() as f:
        for line in f:
            out.append(json.loads(line))
    return out


def build_features(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) where X has columns FEAT_ORDER and y is p_target."""
    T = np.array([r["T"] for r in rows], dtype=np.float64)
    L = np.array([r["L"] for r in rows], dtype=np.float64)
    n_rep = np.array([r["n_repeats"] for r in rows], dtype=np.float64)
    entropy = np.array([r["entropy_top50"] for r in rows], dtype=np.float64)
    log_tok = np.log(
        np.array([r["target_token_id"] for r in rows], dtype=np.float64) + 1.0
    )
    X = np.stack([T, L, n_rep, entropy, log_tok], axis=1)
    y = np.array([r["p_target"] for r in rows], dtype=np.float64)
    return X, y


def random_split(X, y, frac: float = 0.25, seed: int = 42):
    """Permutation split. Default frac=0.25, seed=42 matches the H31 fit."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.permutation(n)
    n_te = max(2, int(n * frac))
    return X[idx[n_te:]], y[idx[n_te:]], X[idx[:n_te]], y[idx[:n_te]]


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


def predict(r, X: np.ndarray) -> np.ndarray:
    """Evaluate an emltorch FitResult on raw numpy X. Returns numpy 1-D."""
    import torch

    yp = r.predict(torch.tensor(X, dtype=torch.float32))
    if hasattr(yp, "cpu"):
        yp = yp.cpu().numpy()
    return np.asarray(yp)


def assert_no_hooks(model) -> None:
    """Verify no forward / pre-forward / backward hooks on any submodule."""
    for name, mod in model.named_modules():
        if mod._forward_hooks:
            raise AssertionError(f"Hook found on {name} (forward)")
        if mod._forward_pre_hooks:
            raise AssertionError(f"Hook found on {name} (pre-forward)")
        if mod._backward_hooks:
            raise AssertionError(f"Hook found on {name} (backward)")
