"""Generate `figure_eml_wins_v2.png` — a target where EML's structural
recovery beats polynomial OLS.

Target: ``y = exp(a * b)``, a ∈ [-1, 1.2], b ∈ [-1, 1.2], n=300.

With ``use_mul=True``, the EML search space has the multiplicative
pre-feature ``a*b`` at every leaf, so depth-1 evolution discovers the
canonical closed form ``eml(a*b, 1) = exp(a*b) - 0 = exp(a*b)`` exactly
(R² → 1.0). Polynomial OLS K=5 cannot express ``exp(·)`` and plateaus
around R² ≈ 0.997 with 21 cross-terms whose extreme-tail residuals are
visibly biased.

This figure is the README's "where EML wins" anchor — complementing the
Feynman-subset figure where EML loses on R² to both poly and PySR.

Usage:
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \\
        python3 -u make_eml_wins_figure.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from emltorch.evolution import evolve, EvolutionConfig  # noqa: E402


def _poly_K_features(X: np.ndarray, K: int) -> np.ndarray:
    """Total-degree-≤K monomials from V features in X (N, V). Includes bias."""
    from itertools import combinations_with_replacement

    N, V = X.shape
    cols = [np.ones((N, 1))]
    for k in range(1, K + 1):
        for combo in combinations_with_replacement(range(V), k):
            col = np.ones(N)
            for j in combo:
                col = col * X[:, j]
            cols.append(col.reshape(-1, 1))
    return np.concatenate(cols, axis=1)


def main():
    rng = np.random.default_rng(2026)
    a = rng.uniform(-1.0, 1.2, 300)
    b = rng.uniform(-1.0, 1.2, 300)
    y = np.exp(a * b)
    X = np.stack([a, b], axis=1)  # (300, 2)

    # 75/25 split, same protocol as run.py
    idx = np.random.default_rng(42).permutation(len(y))
    n_te = max(2, len(y) // 4)
    te, tr = idx[:n_te], idx[n_te:]
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]
    # No standardization on this demo — the exp(a·b) target is intrinsically
    # multiplicative on raw inputs; standardizing would distort the relation
    # to y = exp((sd_a·a_std + mu_a)·(sd_b·b_std + mu_b)), preventing the
    # clean depth-1 structural recovery this figure exists to demonstrate.
    Xtr_s = Xtr
    Xte_s = Xte

    # ── EML d=1, use_mul=True ──────────────────────────────────────────────
    # Avoid torch.from_numpy (broken in this venv due to numpy/torch ABI gap);
    # build tensors directly from python lists.
    device = "cpu"
    torch.manual_seed(0)
    x_t = torch.tensor(Xtr_s.T.tolist(), dtype=torch.float32, device=device)  # (V, N)
    y_t = torch.tensor(ytr.tolist(), dtype=torch.float32, device=device)
    x_te_t = torch.tensor(Xte_s.T.tolist(), dtype=torch.float32, device=device)

    cfg = EvolutionConfig(
        depth=1,
        population=1024,
        generations=30,
        elite_fraction=0.1,
        range_penalty=0.0,
        r2_target=0.99999,
        use_mul=True,
        device=device,
    )
    evo = evolve(x_t, y_t, cfg)
    expr = evo.best_expression
    a_aff, b_aff = evo.best_a, evo.best_b

    # Manual predict — broadcast HELDOUT/train x to all pop trees, pick best_idx
    def _predict(xv: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            B = evo.best_tree.num_trees
            V = xv.shape[0]
            N = xv.shape[-1]
            x_b = xv.unsqueeze(0).expand(B, V, N).contiguous()
            preds = evo.best_tree.forward(x_b)  # (B, N)
            tree_pred = preds[evo.best_idx]
        return np.array((a_aff + b_aff * tree_pred).cpu().tolist())

    yp_tr = _predict(x_t)
    yp_te = _predict(x_te_t)

    eml_r2_tr = 1.0 - ((ytr - yp_tr) ** 2).sum() / ((ytr - ytr.mean()) ** 2).sum()
    eml_r2_te = 1.0 - ((yte - yp_te) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    eml_complexity = expr.count("eml(") + expr.count("exp(") + expr.count("ln(")
    print(f"EML d=1 use_mul=True:")
    print(f"  expression: {expr}")
    print(f"  affine wrapper: a={a_aff:.4f}, b={b_aff:.4f}")
    print(f"  train R² = {eml_r2_tr:.6f}, test R² = {eml_r2_te:.6f}")
    print(f"  complexity (eml/exp/ln node count): {eml_complexity}")

    # ── poly K=5 baseline ──────────────────────────────────────────────────
    Phi_tr = _poly_K_features(Xtr_s, 5)
    Phi_te = _poly_K_features(Xte_s, 5)
    coef, *_ = np.linalg.lstsq(Phi_tr, ytr, rcond=None)
    yp_tr_poly = Phi_tr @ coef
    yp_te_poly = Phi_te @ coef
    poly_r2_tr = 1.0 - ((ytr - yp_tr_poly) ** 2).sum() / ((ytr - ytr.mean()) ** 2).sum()
    poly_r2_te = 1.0 - ((yte - yp_te_poly) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    poly_n_terms = Phi_tr.shape[1]
    print(
        f"poly K=5: train R²={poly_r2_tr:.6f}, test R²={poly_r2_te:.6f}, "
        f"n_terms={poly_n_terms}"
    )

    # ── figure ────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Left: predicted-vs-actual on HELDOUT
    lo = min(yte.min(), yp_te.min(), yp_te_poly.min())
    hi = max(yte.max(), yp_te.max(), yp_te_poly.max())
    ax1.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1, label="y = ŷ")
    ax1.scatter(
        yte,
        yp_te_poly,
        s=35,
        alpha=0.7,
        color="#a0aec0",
        edgecolors="black",
        linewidths=0.4,
        label=f"poly K=5 (R²={poly_r2_te:.4f}, {poly_n_terms} terms)",
    )
    ax1.scatter(
        yte,
        yp_te,
        s=35,
        alpha=0.85,
        color="#2b6cb0",
        edgecolors="black",
        linewidths=0.4,
        label=f"EML d=1 (R²={eml_r2_te:.4f}, 1 eml node)",
    )
    ax1.set_xlabel("actual y = exp(a·b)", fontsize=9)
    ax1.set_ylabel("predicted ŷ on HELDOUT", fontsize=9)
    ax1.set_title("Predicted vs actual on HELDOUT (n=75)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle=":")

    # Right: residuals as a function of |a·b| (the regime where poly breaks)
    ab_te = Xte[:, 0] * Xte[:, 1]
    resid_poly = yte - yp_te_poly
    resid_eml = yte - yp_te
    order = np.argsort(np.abs(ab_te))
    ax2.plot(
        np.abs(ab_te)[order],
        resid_poly[order],
        "o-",
        markersize=4,
        linewidth=0.8,
        color="#a0aec0",
        alpha=0.8,
        label=f"poly K=5 (max |resid| = {np.abs(resid_poly).max():.3f})",
    )
    ax2.plot(
        np.abs(ab_te)[order],
        resid_eml[order],
        "o-",
        markersize=4,
        linewidth=0.8,
        color="#2b6cb0",
        alpha=0.9,
        label=f"EML d=1 (max |resid| = {np.abs(resid_eml).max():.2e})",
    )
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_xlabel("|a·b|  (extremes are where exp(·) curvature dominates)", fontsize=9)
    ax2.set_ylabel("residual (y − ŷ) on HELDOUT", fontsize=9)
    ax2.set_title(
        "Residuals vs |a·b| — poly K=5 fails to capture exp(·) curvature",
        fontsize=10,
    )
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle=":")

    fig.suptitle(
        "Where EML wins: structural recovery of  y = exp(a · b)  "
        "(use_mul=True, depth-1 search)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = os.path.join(SCRIPT_DIR, "figure_eml_wins_v2.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
