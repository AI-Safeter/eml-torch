"""Render tables and a standalone research figure from audited numerical artifacts."""

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import HERE, digest, root


def main():
    out = root()
    s = json.loads((out / "summary.json").read_text())
    destination = HERE / "results"
    destination.mkdir(exist_ok=True)

    def number(v, n=4):
        return f"{v:.{n}f}"

    table = [
        "| Design | Activation | Raw MSE | Contribution MSE | Multiply % | ARC % | CE nats | Fit seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    account = [
        "| Design | Activation | Trainable parameters | Training buffers | Deployed parameters | Deployed buffers |",
        "|---|---|---:|---:|---:|---:|",
    ]
    timing = [
        "| Design | Activation | Original E2E ms | Replacement E2E ms | Reduction % [95% block CI] |",
        "|---|---|---:|---:|---:|",
    ]
    error = [
        "| Design | Activation | Train raw MSE | Outside decoder MSE | Within decoder MSE | Output covariance rank |",
        "|---|---|---:|---:|---:|---:|",
    ]
    training = []
    timing_rows = []
    for arch in ["bottleneck", "shortcut", "structured"]:
        for act in ["eml", "silu"]:
            key = f"{arch}-{act}-d1-s1103-n12000"
            r = s["fits"][key]
            q = s["quality"][key]["point"]
            d = s["diagnostics"][key]["train"]
            raw = sum(v["raw_mse"] for v in r["selection"].values()) / 2
            contrib = sum(v["contribution_mse"] for v in r["selection"].values()) / 2
            table.append(
                f"| {arch} | {act} | {raw:.6f} | {contrib:.6f} | {q['multiply'] * 100:.2f} | {q['arc'] * 100:.2f} | {q['ce']:.4f} | {r['training_seconds']:.1f} |"
            )
            counts = r["accounting"]
            deploy = r["deployed_accounting"]
            account.append(
                f"| {arch} | {act} | {counts['parameters']:,} | {counts['buffers']:,} | {deploy['parameters']:,} | {deploy['buffers']:,} |"
            )
            outside = "—" if d["outside_decoder_mse"] is None else f"{d['outside_decoder_mse']:.6f}"
            inside = "—" if d["within_decoder_mse"] is None else f"{d['within_decoder_mse']:.6f}"
            error.append(
                f"| {arch} | {act} | {d['raw_mse']:.6f} | {outside} | {inside} | {d['output_covariance_rank_relative_1e_minus_6']} |"
            )
            training.append(
                {
                    "method": key,
                    "raw_mse": raw,
                    "contribution_mse": contrib,
                    **q,
                    "training_seconds": r["training_seconds"],
                    "selected_step": r["selected_step"],
                }
            )
            primary = s["timings"][key + "-b8-p512"]["timing"]["end_to_end_seconds"]
            lo, hi = primary["relative_reduction_95_block_bootstrap"]
            timing.append(
                f"| {arch} | {act} | {primary['original_mean'] * 1000:.2f} | {primary['replacement_mean'] * 1000:.2f} | {primary['relative_reduction'] * 100:.2f} [{lo * 100:.2f}, {hi * 100:.2f}] |"
            )
    for key, r in s["timings"].items():
        for phase, v in r["timing"].items():
            timing_rows.append(
                {
                    "method": r["method"],
                    "batch": r["batch"],
                    "prefill_tokens": r["prefill_tokens"],
                    "decode_steps": 32,
                    "phase": phase,
                    **{k: value for k, value in v.items() if not isinstance(value, list)},
                    "ci_low": v["relative_reduction_95_block_bootstrap"][0],
                    "ci_high": v["relative_reduction_95_block_bootstrap"][1],
                }
            )
    for filename, rows in [("development.csv", training), ("timing.csv", timing_rows)]:
        with (destination / filename).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    # Standalone artifact: no image generation or alteration of experimental data.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), layout="constrained")
    colors = {"eml": "#b44432", "silu": "#286d9f"}
    for i, arch in enumerate(["bottleneck", "shortcut", "structured"]):
        for act, offset in [("eml", -0.12), ("silu", 0.12)]:
            r = next(v for v in training if v["method"] == f"{arch}-{act}-d1-s1103-n12000")
            axes[0].scatter(
                i + offset, r["raw_mse"], color=colors[act], label=act.upper() if i == 0 else None
            )
            axes[1].scatter(i + offset, r["arc"] * 100, color=colors[act])
            t = s["timings"][r["method"] + "-b8-p512"]["timing"]["end_to_end_seconds"]
            mean = t["relative_reduction"] * 100
            lo, hi = [v * 100 for v in t["relative_reduction_95_block_bootstrap"]]
            axes[2].plot([i + offset] * 2, [lo, hi], color=colors[act])
            axes[2].scatter(i + offset, mean, color=colors[act])
    axes[0].set_ylabel("Development raw MLP MSE")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("Development ARC accuracy (%)")
    axes[1].axhline(
        s["quality"]["original"]["point"]["arc"] * 100, color="0.4", ls="--", label="Original"
    )
    axes[1].legend(frameon=False)
    axes[2].set_ylabel("End-to-end latency reduction (%)")
    axes[2].axhline(0, color="0.4", ls="--")
    for ax in axes:
        ax.set_xticks(range(3), ["Bottleneck", "Shortcut", "Structured"], rotation=15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("One-seed development screen; timing: B8, 512-token prefill, 32 decode steps")
    fig.savefig(destination / "architecture-screen.png", dpi=180)
    fig.savefig(destination / "architecture-screen.pdf")
    plt.close(fig)
    sections = {
        "development_table": "\n".join(table),
        "accounting_table": "\n".join(account),
        "timing_table": "\n".join(timing),
        "error_table": "\n".join(error),
    }
    (destination / "tables.json").write_text(json.dumps(sections, indent=2) + "\n")
    for filename in [
        "summary.json",
        "release-audit.json",
        "decision.json",
        "budget.json",
        "validation.json",
        "integration.json",
        "screen-d1-n12000.json",
    ]:
        shutil.copy2(out / filename, destination / filename)
    for sub in ["evaluation/development", "benchmark", "exports"]:
        target = destination / Path(sub).name
        target.mkdir(exist_ok=True)
        for p in sorted((out / sub).glob("*.json")):
            shutil.copy2(p, target / p.name)
    (destination / "render-provenance.json").write_text(
        json.dumps(
            {
                "source_sha256": digest(HERE / "report.py"),
                "summary_sha256": digest(out / "summary.json"),
                "audit_sha256": digest(out / "release-audit.json"),
            },
            indent=2,
        )
        + "\n"
    )
    print("REPORT TABLES AND FIGURE RENDERED", destination)


if __name__ == "__main__":
    main()
