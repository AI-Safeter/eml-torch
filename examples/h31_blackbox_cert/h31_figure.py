#!/usr/bin/env python3
"""H31 headline figure: cross-vendor behavioral fingerprint + cert."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "outputs"


def load_measurements(tag: str) -> list[dict]:
    out = []
    with (OUT_DIR / f"measurements_{tag}.jsonl").open() as f:
        for line in f:
            out.append(json.loads(line))
    return out


CIRCUITS = ["induction", "factual", "copy_oneshot", "ioi", "syntactic"]
LABELS = {
    "induction": "induction",
    "factual": "factual",
    "copy_oneshot": "copy (one-shot)",
    "ioi": "IOI",
    "syntactic": "syntactic",
}


def main() -> None:
    qwen = load_measurements("qwen36")
    gemma = load_measurements("gemma4")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [3, 2]}
    )

    # ── Left: per-circuit P_target distribution as violin/box ────────────
    positions = np.arange(len(CIRCUITS))
    width = 0.36
    qcolor = "#3a7fbd"
    gcolor = "#d94d4d"

    for i, c in enumerate(CIRCUITS):
        qp = [m["p_target"] for m in qwen if m["circuit"] == c]
        gp = [m["p_target"] for m in gemma if m["circuit"] == c]
        bp1 = ax1.boxplot(
            qp,
            positions=[i - width / 2],
            widths=width,
            patch_artist=True,
            showmeans=True,
            manage_ticks=False,
            medianprops={"color": "white", "linewidth": 2},
            boxprops={"facecolor": qcolor, "edgecolor": "black"},
        )
        bp2 = ax1.boxplot(
            gp,
            positions=[i + width / 2],
            widths=width,
            patch_artist=True,
            showmeans=True,
            manage_ticks=False,
            medianprops={"color": "white", "linewidth": 2},
            boxprops={"facecolor": gcolor, "edgecolor": "black"},
        )

    ax1.set_xticks(positions)
    ax1.set_xticklabels([LABELS[c] for c in CIRCUITS], rotation=0)
    ax1.set_ylabel("P(target | prompt)  —  black-box top-50 logprobs", fontsize=11)
    ax1.set_title(
        "H31 — same probes, different vendor behavioral fingerprint", fontsize=12
    )
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.grid(axis="y", alpha=0.3)

    # Legend
    qpatch = plt.Rectangle(
        (0, 0), 1, 1, fc=qcolor, ec="black", label="Qwen/Qwen3.6-27B  (hybrid)"
    )
    gpatch = plt.Rectangle(
        (0, 0), 1, 1, fc=gcolor, ec="black", label="google/gemma-4-31b-it"
    )
    ax1.legend(handles=[qpatch, gpatch], loc="upper right", fontsize=10)

    # ── Right: cert summary box ──────────────────────────────────────────
    ax2.axis("off")
    txt = [
        r"$\mathbf{Qwen3.6\!-\!27B \;\; factual}$  (R² = 0.89, depth-4 EML)",
        "",
        r"$\mathrm{P_{target}} \approx 0.5954 - 0.1353 \cdot$",
        r"$\quad \mathrm{eml}(L,\, \mathrm{eml}(L\!-\!H,\, 1))$",
        "",
        r"$L$ = induction lag (here = 0)",
        r"$H$ = entropy of top-50 (working: 1.94–2.40)",
        "",
        r"$\mathbf{Cert\;pair}$  ($\tau = 0.10$):",
        r"  $\bullet$  $P > \tau$ over IQR working box",
        r"      z3:   $\mathbf{unsat}$  (12 ms)",
        r"      cvc5: $\mathbf{unsat}$  (4 ms)",
        r"  $\bullet$  $P > \tau$ over failure box",
        r"      z3:   $\mathbf{sat}$  (3 ms)  +  model",
        r"      cvc5: $\mathbf{sat}$  (4 ms)  +  model",
        "",
        r"$\mathbf{Black\!-\!box\;discipline}$:",
        r"  no hooks, no $\mathrm{output\_attentions}$,",
        r"  no $\mathrm{output\_hidden\_states}$, top-K only",
    ]
    ax2.text(
        0.0,
        0.98,
        "\n".join(txt),
        va="top",
        ha="left",
        fontsize=10.5,
        family="monospace",
    )

    plt.suptitle(
        "H31 — black-box LLM behavioral interpreter on 2 frontier LLMs\n"
        "256 probes • 5 circuit classes • 0 hooks • dual-verified z3 + cvc5",
        fontsize=12.5,
        y=1.00,
    )
    plt.tight_layout()
    out_path = OUT_DIR / "headline_figure.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
