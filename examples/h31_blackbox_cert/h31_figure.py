#!/usr/bin/env python3
"""H31 headline figure v2 — focused on the discovered formula.

Left: scatter of (H, P_target) on Qwen3.6-27B factual, EML formula line overlay.
Right: cert pair summary text box.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"


def main() -> None:
    with (OUT_DIR / "measurements_qwen36.jsonl").open() as f:
        rows = [
            json.loads(line) for line in f if json.loads(line)["circuit"] == "factual"
        ]
    H = np.array([r["entropy_top50"] for r in rows])
    y = np.array([r["p_target"] for r in rows])

    # The EML formula was fit with normalize_inputs=True, so x2, x4 inside
    # the symbolic expression are (L, H) standardized by the training
    # column means and stds. On the factual training slice L=0 is constant,
    # so the formula's algebraic value collapses to a + b·H_norm, which on
    # raw H is a linear function P = a' + b'·H. We recover (a', b') by the
    # closed-form OLS fit that the formula reduces to on this data
    # (verified Δ R² = 0.0000 vs EML in h_only_refit_results.json).
    A = np.stack([np.ones(len(H)), H], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a_lin, b_lin = float(coef[0]), float(coef[1])

    H_grid = np.linspace(H.min() - 0.1, H.max() + 0.1, 200)
    eml_pred = a_lin + b_lin * H_grid

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 2]}
    )

    ax1.scatter(
        H,
        y,
        s=40,
        alpha=0.65,
        color="#3a7fbd",
        edgecolor="black",
        linewidth=0.5,
        label=r"Qwen3.6-27B factual (n=50)",
    )
    ax1.plot(
        H_grid,
        eml_pred,
        color="#d94d4d",
        linewidth=2.2,
        label=rf"EML formula evaluated: $P \approx {a_lin:.2f}{b_lin:+.2f} \cdot H$",
    )
    ax1.set_xlabel(r"$H$ = entropy of top-50 logprobs", fontsize=11)
    ax1.set_ylabel(r"$P(\mathrm{target} \mid \mathrm{prompt})$", fontsize=11)
    ax1.set_title("Black-box derivation on Qwen3.6-27B factual recall", fontsize=12)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(-0.05, 0.85)

    ax2.axis("off")
    txt = [
        r"$\mathbf{Formula \; (depth\!-\!4 \; EML)}$",
        "",
        r"$P_{\mathrm{target}} \approx 0.5954 - 0.1353 \cdot $",
        r"$\quad \mathrm{eml}(L,\, \mathrm{eml}(L\!-\!H,\, 1))$",
        "",
        r"with $\mathrm{eml}(x, y) = e^x - \ln y$",
        r"HELDOUT $R^2 = 0.89$",
        "",
        r"$\mathbf{Cert \; pair \;\; (dual\!-\!verified)}$",
        "",
        r"$P > 0.10$ over working box:",
        r"   z3: $\mathbf{unsat}$ (12 ms)",
        r"   cvc5: $\mathbf{unsat}$ (4 ms)",
        "",
        r"$P > 0.10$ over failure box:",
        r"   z3: $\mathbf{sat}$ + counterexample",
        r"   cvc5: $\mathbf{sat}$ + counterexample",
        "",
        r"$\mathbf{Black\!-\!box}$: no hooks, no",
        r"$\mathrm{output\_attentions}$, top-K only",
    ]
    ax2.text(
        0.0,
        0.97,
        "\n".join(txt),
        va="top",
        ha="left",
        fontsize=10.5,
        family="monospace",
    )

    plt.tight_layout()
    out_png = OUT_DIR / "headline_figure.png"
    out_pdf = OUT_DIR / "headline_figure.pdf"
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
