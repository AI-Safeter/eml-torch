"""Generate two minimal figures next to the SRBench-Feynman example:

  figure_benchmark.png   2-panel summary of the 8-equation Feynman benchmark
                         (per-equation HELDOUT R² heat-strip + per-method
                         mean R² vs median fit time scatter)
  figure_pareto_demo.png accuracy/complexity Pareto front discovered by
                         emltorch.fit_pareto on Feynman I.6.20a exp(-theta^2/2)

Both figures read from this directory:
  - results.json (the live benchmark output shipped alongside)
  - pysr_multithreading_retime.json (the multithreaded re-time)

Usage:
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        python3 -u make_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

# Headless backend so this works on a server with no display.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Allow running from a fresh git clone without `pip install -e .`.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import emltorch as eml  # noqa: E402


METHOD_COLORS = {
    "eml_d3": "#2b6cb0",  # deep blue
    "eml_d4": "#4a90c2",  # lighter blue
    "poly_K2": "#a0aec0",  # gray
    "poly_K5": "#718096",  # darker gray
    "pysr": "#dd6b20",  # orange
}
METHOD_LABEL = {
    "eml_d3": "EML d=3",
    "eml_d4": "EML d=4",
    "poly_K2": "poly K=2",
    "poly_K5": "poly K=5",
    "pysr": "PySR (serial)",
}


def make_benchmark_figure():
    with open(os.path.join(SCRIPT_DIR, "results.json")) as f:
        data = json.load(f)

    methods = ["eml_d3", "eml_d4", "poly_K2", "poly_K5", "pysr"]
    eq_names = [k for k in data.keys() if k != "aggregate"]

    # Per-(equation, method) HELDOUT R² matrix.
    # results.json shape: data[eq] = {"desc": ..., "methods": {method: {"heldout_r2": ...}}}
    r2_mat = np.full((len(methods), len(eq_names)), np.nan)
    for j, eq in enumerate(eq_names):
        eq_methods = data[eq].get("methods", {}) if isinstance(data[eq], dict) else {}
        for i, m in enumerate(methods):
            cell = eq_methods.get(m, {})
            if isinstance(cell, dict) and isinstance(
                cell.get("heldout_r2"), (int, float)
            ):
                r2_mat[i, j] = cell["heldout_r2"]

    agg = data["aggregate"]
    means = [agg[m]["mean_heldout_r2"] for m in methods]
    medians_t = [agg[m]["median_fit_time_s"] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # ── LEFT: per-equation R² heat-strip ──────────────────────────────────
    im = ax1.imshow(
        r2_mat,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
    )
    ax1.set_xticks(range(len(eq_names)))
    ax1.set_xticklabels(eq_names, rotation=45, ha="right", fontsize=8)
    ax1.set_yticks(range(len(methods)))
    ax1.set_yticklabels([METHOD_LABEL[m] for m in methods], fontsize=9)
    ax1.set_title("HELDOUT R² per equation (8 Feynman targets, n=300)", fontsize=10)
    # Annotate each cell with the R² value
    for i in range(len(methods)):
        for j in range(len(eq_names)):
            v = r2_mat[i, j]
            if np.isfinite(v):
                ax1.text(
                    j,
                    i,
                    f"{v:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if v > 0.85 else "white",
                )
    cbar = fig.colorbar(im, ax=ax1, shrink=0.85, pad=0.02)
    cbar.set_label("R²", fontsize=9)

    # ── RIGHT: aggregate R² vs time scatter ───────────────────────────────
    for m, mu, t in zip(methods, means, medians_t):
        ax2.scatter(
            max(t, 1e-4),
            mu,
            s=140,
            color=METHOD_COLORS[m],
            edgecolors="black",
            linewidths=0.6,
            label=METHOD_LABEL[m],
            zorder=3,
        )
        ax2.annotate(
            METHOD_LABEL[m],
            (max(t, 1e-4), mu),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
        )
    ax2.set_xscale("log")
    ax2.set_xlabel("Median fit time (s, log scale)", fontsize=9)
    ax2.set_ylabel("Mean HELDOUT R²  (across 8 equations)", fontsize=9)
    ax2.set_title(
        "Accuracy vs speed (lower-right = bad, upper-left = ideal)", fontsize=10
    )
    ax2.grid(True, which="both", alpha=0.3, linestyle=":")
    ax2.set_ylim(0.85, 1.005)
    ax2.set_xlim(1e-4, 1e2)

    fig.suptitle(
        "emltorch vs polynomial OLS vs PySR — Feynman subset",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(SCRIPT_DIR, "figure_benchmark.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def make_pareto_demo_figure():
    """Run fit_pareto on Feynman I.6.20a (Gaussian) and plot the front."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(1.0, 3.0, 300)
    x = theta.reshape(-1, 1)
    y = np.exp(-(theta**2) / 2.0)

    # Train standardization (same protocol as run.py)
    rng2 = np.random.default_rng(42)
    idx = rng2.permutation(len(x))
    te = max(2, len(x) // 4)
    Xtr, Xte = x[idx[te:]], x[idx[:te]]
    ytr, yte = y[idx[te:]], y[idx[:te]]
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-12
    Xtr_s = (Xtr - mu) / sd

    pareto = eml.fit_pareto(
        Xtr_s,
        ytr,
        depths=(1, 2, 3, 4, 5),
        seeds_per_depth=3,
        device="cpu",
    )
    print(f"Pareto front on Feynman I.6.20a: {pareto.summary()}")

    # All evaluated + front
    all_c = [c for c, _, _ in pareto.all_evaluated]
    all_r2 = [r for _, r, _ in pareto.all_evaluated]
    front_c = [c for c, _, _ in pareto.front]
    front_r2 = [r for _, r, _ in pareto.front]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # Dominated points
    dom_c, dom_r2 = [], []
    front_set = set((c, round(r, 10)) for c, r, _ in pareto.front)
    for c, r in zip(all_c, all_r2):
        if (c, round(r, 10)) not in front_set:
            dom_c.append(c)
            dom_r2.append(r)
    if dom_c:
        ax.scatter(
            dom_c,
            dom_r2,
            s=80,
            color="lightgray",
            edgecolors="gray",
            linewidths=0.7,
            label="dominated (kept in `all_evaluated`)",
            zorder=2,
        )
    # Front
    order = np.argsort(front_c)
    fc = [front_c[i] for i in order]
    fr = [front_r2[i] for i in order]
    ax.plot(fc, fr, "-", color="#2b6cb0", linewidth=1.5, zorder=3)
    ax.scatter(
        fc,
        fr,
        s=140,
        color="#2b6cb0",
        edgecolors="black",
        linewidths=0.8,
        label="Pareto front",
        zorder=4,
    )
    for c, r in zip(fc, fr):
        ax.annotate(
            f"(c={c}, R²={r:.4f})",
            (c, r),
            textcoords="offset points",
            xytext=(8, -3),
            fontsize=8,
        )
    ax.set_xlabel("Expression complexity (# of eml operators)", fontsize=9)
    ax.set_ylabel("HELDOUT R² on standardized train", fontsize=9)
    ax.set_title(
        "`fit_pareto(...)` on Feynman I.6.20a:  exp(−θ²/2)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, "figure_pareto_demo.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    make_benchmark_figure()
    make_pareto_demo_figure()
