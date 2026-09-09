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

    table = [
        "| Design | Activation | Raw MSE | Contribution MSE | Add % | Multiply % | Divide % | ARC % | CE nats | Fit seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                f"| {arch} | {act} | {raw:.6f} | {contrib:.6f} | {q['add'] * 100:.2f} | {q['multiply'] * 100:.2f} | {q['divide'] * 100:.2f} | {q['arc'] * 100:.2f} | {q['ce']:.4f} | {r['training_seconds']:.1f} |"
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
    phase_table = [
        "| Design / activation | Prefill ms, original → replacement | Decode ms, original → replacement | Output tokens/s, original → replacement | Peak allocated GiB, original → replacement |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in training:
        method = row["method"]
        r = s["timings"][method + "-b8-p512"]
        pre, dec = r["timing"]["prefill_seconds"], r["timing"]["decode_seconds"]
        tp = r["throughput_mean"]
        mem = r["isolated_memory"]
        phase_table.append(
            f"| {method.split('-d1')[0]} | {pre['original_mean'] * 1000:.2f} → {pre['replacement_mean'] * 1000:.2f} | "
            f"{dec['original_mean'] * 1000:.2f} → {dec['replacement_mean'] * 1000:.2f} | "
            f"{tp['original']['output_tokens_per_second']:.2f} → {tp[method]['output_tokens_per_second']:.2f} | "
            f"{mem['original']['peak_allocated_bytes'] / 2**30:.4f} → {mem[method]['peak_allocated_bytes'] / 2**30:.4f} |"
        )
    sections = {
        "development_table": "\n".join(table),
        "accounting_table": "\n".join(account),
        "timing_table": "\n".join(timing),
        "error_table": "\n".join(error),
        "phase_table": "\n".join(phase_table),
    }
    (destination / "tables.json").write_text(json.dumps(sections, indent=2) + "\n")
    for filename in [
        "summary.json",
        "release-audit.json",
        "decision.json",
        "budget.json",
        "validation.json",
        "integration.json",
        "precision-audit.json",
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
    ledger = json.loads((out / "budget.json").read_text())
    hours = (
        sum(v.get("elapsed_seconds", v["allowance_seconds"]) for v in ledger["jobs"].values())
        / 3600
    )
    original = s["quality"]["original"]["point"]
    report = f"""# Removing the decoder bottleneck: architecture screen

Removing the fixed output subspace improves reconstruction, but it does not
restore the model's quality at this budget. The affine shortcut lowers development
raw MLP error by 4.7% for EML and 5.2% for SiLU. This is an architectural
improvement in the observed screen, not an EML-specific advantage. Neither new
architecture passes the preregistered development gate.

The experiment therefore stops before longer training, depth exploration,
three-seed confirmation, or multiple-block replacement. Fresh confirmation data
remain unopened. This study does **not** establish that EML is inherently
inefficient, that optimization has converged, or that one cause explains most
of the previous quality failure.

![Architecture screen](architecture-screen.png)

## Scope, fairness, and the stop decision

The exact local model is `google/gemma-4-E2B-it`, revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`. Configuration SHA-256 is
`1b28f3d2c3100f6c594754b81107428bd7b822a7f48272ca681dae9d2ec38330`.
Configuration, safetensors metadata, and native CUDA registration agree:
5,104,297,504 unique model parameters and 56,623,104 parameters in layer 26's MLP.
All six installed replacements forbid the original MLP forward and exclude its
parameters. Teacher activations are used only for offline supervision.

The [protocol](../PROTOCOL.md) was frozen before fitting. All six candidates use
seed **1103**, nonlinear depth **one**, approximately **6M coefficients**,
12,000 FP32 AdamW updates, identical per-seed minibatch streams, and the same
raw-plus-native-contribution objective. Each update samples 256 arithmetic and
256 language positions. The best development activation checkpoint is selected
at a common 200-update interval. Equal affine starting functions within each
architecture were verified on CUDA. Initializations differ across architectures
as declared in the protocol.

Each new architecture needed at least 10% improvement in both reconstruction
terms, at least 3 points of multiplication gain versus its matching bottleneck,
no more than 2 points of loss on the other development accuracy endpoints, and
at most 0.02 nats of language CE increase versus the original. None passes.
Raw/contribution gains are 4.70%/3.96% for shortcut EML, 5.24%/4.12% for shortcut
SiLU, 1.96%/1.76% for structured EML, and 6.87%/6.15% for structured SiLU.

The error threshold also prevents the conditional longer-training diagnostic
from activating. All selected checkpoints are at update 11,800. Curves continue
improving near the limit, so inadequate optimization remains possible.
**Seeds 2207 and 3301 were not run**, and no seed-variance estimate is supplied.
Depth two is implemented and its numerical deployment transformations were
tested, but it was not trained because the architecture gate did not pass.

## All completed seed results

Errors below use previously inspected selection activations with equal domain
weights. Raw MSE is divided by the training target's average centered variance;
contribution MSE is divided by the native contribution's average second moment.
These are different diagnostics, not interchangeable quality scores.

Development uses 64 operand groups per operation, each in code and reversed
formats; 32 larger-operand groups per operation are diagnostic. It also uses
96 ARC questions and 64 documents of BOS plus 256 tokens. Generation is capped
at 24 tokens and malformed responses remain wrong. Previously inspected data
are deliberately treated as development, including the former final split.

Original development scores: addition **{original["add"] * 100:.2f}%**,
multiplication **{original["multiply"] * 100:.2f}%**, division
**{original["divide"] * 100:.2f}%**, ARC **{original["arc"] * 100:.2f}%**,
and document CE **{original["ce"]:.4f} nats**.

{sections["development_table"]}

All full-width variants lose 4.17–5.21 ARC points relative to their matching
bottleneck control. Language CE improves relative to the original for every
student, while ARC and multiplication deteriorate. This is direct evidence
that lower activation error or document CE alone is insufficient for this
replacement's quality decision.

Paired 95% bootstrap intervals, grouped by operand pair across prompt formats,
are in `summary.json` and the raw predictions are in `development/`. For shortcut
EML, the multiplication loss versus the original is 7.81 points [2.34, 13.28]
and ARC loss is 10.42 points [5.21, 16.67]. Shortcut SiLU loses 6.25 multiplication
points [0.78, 12.50] and 9.38 ARC points [4.17, 15.62]. These are **descriptive
intervals on inspected development data**, conditional on the fitted seed;
they do not replace confirmation or account for candidate selection.

The direct EML-versus-SiLU multiplication intervals include zero in every
architecture. No consistent downstream EML advantage is established.
Code-format multiplication also drops: original 46/64, shortcut EML 38/64,
shortcut SiLU 40/64. All three parse 63/64 responses, so this diagnostic loss
cannot be explained by an increased count of malformed outputs alone.
All format, parsing, and shifted-operand results remain in the JSON evidence.
No arithmetic mechanism is inferred from these measurements.

## What the architectural change removes

The [derivation](../THEORY.md) distinguishes output-subspace restrictions from
input information loss. The bottleneck output lies in a learned affine subspace
of dimension at most 512; its input encoder also discards directions. The
shortcut removes the complete prediction's fixed output-subspace restriction,
but nonlinear dependence remains confined to a compact encoder and decoder.
Outside its correction decoder, prediction must remain affine. Structured
residual stages keep 1,536-dimensional states and diagonal-plus-low-rank maps;
they remove the mandatory 512-state and rank-512 output constraints. They still
have finite depth, structured mixing, normalization, and a limited budget.

The correctly recomputed empirical normalized raw-MSE floors are **0.141368**
for the bottleneck and **0.104122** for the shortcut. The latter uses the
*unregularized optimal affine residual*, not a ridge residual. The FP64 normal
equation condition number is 8,311.4; relative normal-equation residual is
1.50e-15 and direct residual-covariance disagreement is 2.06e-14. The structured
design has no positive floor implied solely by a fixed output subspace; its
general bound here is zero, which is not evidence that its finite network can
achieve zero error. These are training-distribution bounds, not bounds on
contribution error, unseen data, answer quality, or EML in general.

{sections["error_table"]}

For EML, shortcut outside-decoder error falls from 0.148124 to 0.123445, while
inside-decoder error rises from 0.077124 to 0.088821. Its best-affine floor outside
the **learned** decoder is 0.113672, leaving a 0.009773 gap attributable to the
learned affine component relative to that empirical optimum. The complete
shortcut floor also allows the decoder to move and is therefore lower.
Inside/outside quantities use each model's own decoder, not identical axes.

The shortcut's full affine map uses 2,360,832 coefficients. At the fixed budget,
the compact EML nonlinear width drops from 2,873 to 1,339 units, and SiLU width
from 4,311 to 2,009. Structured EML uses rank 393 per map and structured SiLU rank
590: EML pays for two independent argument maps, while both retain the same
1,536-dimensional state. Matching budget therefore does not mean matching
inner width or factor rank.

This supports a concrete capacity-allocation explanation: part of the recovered
output-space capacity is offset by a harder or less capable nonlinear fit at
the same coefficient budget. It does not uniquely separate encoder loss,
nonlinear capacity, initialization, optimization, and the competing contribution
objective. Structured SiLU fits better than structured EML in this seed, whereas
EML fits slightly better within the bottleneck and shortcut families. The
activation ranking depends on architecture.

Covariance rank is measured at a relative eigenvalue cutoff of 1e-6. Shortcut
predictions reach rank 1,536 and structured predictions approximately 1,485,
yet general-knowledge quality worsens. Full covariance rank does not guarantee
injectivity or preserved information; even powers of one input coordinate can
have full covariance rank while discarding other coordinates. Random
Jacobian-direction sensitivities and structured projection spectra are included
as diagnostics, not semantic information certificates.

There is no evidence of widespread exponential clamping in the sampled training
telemetry: bottleneck and shortcut EML have zero clamps; structured EML has
7 clamps in 47,185,920 sampled arguments and clips gradient norm in 40/12,000
updates. This limited telemetry does not rule out optimization difficulties.
Training versus selection error gaps persist for every design; the experiment
does not assign them uniquely to distribution shift or capacity.

A separate CUDA precision audit evaluates all 32,768 selection positions in
unfolded FP32, folded FP32, folded BF16, and unfolded BF16. Folding alone changes
normalized predictions by at most 4.02e-12 MSE. BF16 deployment changes raw target
MSE by at most 0.000226 in absolute value, below 0.1% of each corresponding
FP32 raw MSE. It therefore does not explain the bulk reconstruction gap.
Small numerical changes can still alter individual downstream answers; this
check is not an end-to-end precision-invariance guarantee. See
`precision-audit.json` for every architecture and activation.

## Complete accounting and measured inference cost

{sections["accounting_table"]}

The native pre/post-feedforward normalizations retain 3,072 parameters in the
complete model. Structured deployment retains 3,072 input-normalization values;
the other designs fold them into projections. Every shortcut, bias, projection,
normalization, decoder, and retained buffer is included. The release audit lists
every deployed tensor. The whole model retains the checkpoint's embeddings and
multimodal components even though the evaluated workload is text-only.

Each replacement removes approximately **0.992% of total model parameters** and
compresses the individual block by about **9.44×**. Thus this single-block result
cannot satisfy the 20% total-model target. No multiple-block expansion was run.

Timing uses the actual installed path with the whole model on GPU: native eager
SDPA for both paths, the same algebraic folding policy, ten warm-ups, thirty
alternating paired repeats, and synchronized CUDA/wall clocks. The main workload
is **batch 8, 512 prefill tokens, 32 fixed cached decoding steps**. The uncertainty
resamples six contiguous blocks of five pairs with 10,000 bootstrap replicates.
Positive reduction means faster replacement. All four batch/prefill workloads
are recorded separately in `timing.csv` and `benchmark/`.

{sections["timing_table"]}

{sections["phase_table"]}

Prefill, decoding, output/total/prefill throughput, CUDA event time, peak allocated
and reserved memory, and unique cache storage are separate fields in the raw
records and summary. Isolated peak memory comes from each path's first warm-up
with unused allocator cache cleared. Timed switches retain the warmed allocator;
reserved memory in those samples can include previous comparison allocations.
The original 113,246,208-byte MLP is staged on CPU for paired comparisons, outside
the installed GPU model and outside timing. Comparison transfers, loading,
tokenization, service queuing, and warm-up are excluded. Fixed-token decoding
measures a declared workload, not a production request-latency guarantee.

Other experiments continued on the H100s. GPU UUIDs and before/after utilization
and free-memory telemetry accompany every timing record. These intervals
quantify within-run variation under shared-device conditions, not independent
serving deployments. Training times also depend on GPU sharing; equal update
counts do not imply equal elapsed time. Quality evaluation uses the previous
bitwise-validated CPU PLE lookup in every method; latency benchmarks keep that
table on GPU and include its storage.

## Budget, reproducibility, and defensible claim

The declared cap was eight H100 device-hours. This completed run charged
**{hours:.4f} device-hours**, counting elapsed time of our GPU subprocesses,
including startup, training, checks, exports, shared-device delays, and timing.
The ledger preserves every job. No other experiment was stopped. Unused budget
does not override the prospective screen's stopping rules.

Run [the reproduction commands](../README.md). `reproduce.py --release` resumes
completed jobs, enforces memory admission and the budget, applies the observed
screen-stop decision, exports all students, and audits results. CUDA checks cover
dense/structured forward and derivative agreement, FP32 folding and BF16
roundoff, original-block removal in prefill and cached decoding, exact exported
reload, equal minibatch streams, raw-error decomposition, and agreement of the
vectorized paired statistics with the previous reference implementation. A
standalone exported shortcut also generated successfully without activation data.
Large checkpoints and activation tensors remain outside Git; hashes and
per-tensor accounting are included with runnable regeneration commands.

Fresh confirmation was frozen before fitting: 1,024 ordinary groups per
operation, 128 shifted groups per operation, 1,024 previously unused ARC
questions, and 256 documents from previously unused hostnames. Historical
operand pairs/donors, document hashes/hostnames, and question IDs/text hashes
were excluded. Freshness is relative to this study, not model pretraining.
The audit verifies that no confirmation outputs or selection-for-confirmation
file exist. Frozen confirmation criteria remain available for a future
qualifying architecture; they were not relaxed after this screen.

**Strongest defensible claim:** the old affine-decoder constraint is a real,
measurable contributor to reconstruction error. Removing it yields modest
improvements shared by EML and SiLU, while consuming capacity elsewhere and
failing to restore downstream quality in this fixed-budget screen. This rules
out treating high output covariance rank as a sufficient remedy. It does not
establish a general EML limitation or identify optimization versus finite
capacity as the sole remaining cause. No deployment success or recovered
arithmetic algorithm is claimed.
"""
    (destination / "REPORT.md").write_text(report)
    print("REPORT TABLES AND FIGURE RENDERED", destination)


if __name__ == "__main__":
    main()
