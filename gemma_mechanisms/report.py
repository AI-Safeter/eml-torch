"""Render the completed, provenance-bound replacement and causal study."""

import argparse
import collections
import json
import statistics

import matplotlib
from scipy.stats import beta

from .quality_summary import accuracy, bootstrap
from .runtime import HERE, SPEC, digest, root, save, setup

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def mean(xs):
    return statistics.mean(xs)


def carry_chain(a, b):
    carry = run = longest = 0
    while a or b or carry:
        carry = int(a % 10 + b % 10 + carry >= 10)
        run = run + 1 if carry else 0
        longest = max(longest, run)
        a, b = a // 10, b // 10
    return longest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--mechanisms-output", required=True)
    args = parser.parse_args()
    setup(73)
    out, mechanism = root(args.output), root(args.mechanisms_output)
    assert json.loads((out / "confirmation-status.json").read_text())["status"] == "Complete"
    destination = HERE / "results"
    destination.mkdir(exist_ok=True)
    provenance = {}

    def read(path):
        key = (
            ("replacement/" + str(path.relative_to(out)))
            if path.is_relative_to(out)
            else ("mechanisms/" + str(path.relative_to(mechanism)))
        )
        provenance[key] = digest(path)
        return json.loads(path.read_text())

    assert read(out / "release-audit.json")["status"] == "complete"

    selection = read(out / "training/selection.json")
    gate, final = [read(out / "evaluation" / f"{s}-summary.json") for s in ["gate", "final"]]
    fits = [read(p) for p in sorted((out / "training").glob("*-b*-d*-s*.json"))]
    audits = {p.stem: read(p) for p in sorted((out / "fit-audits").glob("*.json"))}
    deployed = read(out / "deployment-validation.json")
    assert len(fits) == len(audits) == len(deployed["records"]) == 42
    groups = collections.defaultdict(list)
    for fit in fits:
        spec = fit["specification"]
        groups[(spec["kind"], spec["budget"], spec["depth"])].append(fit)
    training = []
    for (kind, budget, depth), rows in sorted(groups.items()):
        values = [audits[r["name"]] for r in rows]
        training.append(
            {
                "kind": kind,
                "budget": budget,
                "depth": depth,
                "seeds": [r["seed"] for r in rows],
                "selection_objective_mean": mean(r["selection_objective"] for r in rows),
                "selection_objectives": [r["selection_objective"] for r in rows],
                "training_raw_mse_mean": mean(r["training_raw_mse"] for r in values),
                "rank_floor": values[0]["training_raw_mse_lower_bound"],
                "rank": values[0]["output_rank"],
                "training_seconds_total": sum(r["training_seconds"] for r in rows),
                "training_tokens_total": sum(r["training_tokens"] for r in rows),
                "gradient_clip_fraction": sum(r["gradient_clipped_steps"] for r in rows)
                / sum(r["gradient_steps"] for r in rows),
                "exponent_clamps": sum(
                    s["clamps"] for r in rows for s in r["exponent_diagnostics"]
                ),
                "exponent_arguments_sampled": sum(
                    s["sampled_arguments"] for r in rows for s in r["exponent_diagnostics"]
                ),
                "peak_allocated_bytes": max(r["peak_allocated_bytes"] for r in rows),
            }
        )
    methods = ["original", *final["methods"]]
    diagnostics = {}
    for split in ["final", "shift"]:
        raw = {
            name: read(out / "evaluation" / split / f"{name}.json")["results"] for name in methods
        }
        diagnostics[split] = {}
        for name in methods[1:]:
            record = {}
            for field in ["arithmetic", "new_formats"]:
                for op in SPEC["arithmetic"]["operations"]:
                    t = [r for r in raw["original"][field] if r["op"] == op]
                    s = [r for r in raw[name][field] if r["op"] == op]
                    record[f"{field}/{op}"] = (
                        final["methods"][name]["accuracy"][op]
                        if split == "final" and field == "arithmetic"
                        else accuracy(t, s, 0.05, True)
                    )
                for label, predicate in [
                    ("long_carry", lambda r: r["op"] == "add" and carry_chain(r["a"], r["b"]) >= 3),
                    ("no_carry", lambda r: r["op"] == "add" and carry_chain(r["a"], r["b"]) == 0),
                    (
                        "five_plus_digits",
                        lambda r: r["op"] == "add" and max(len(str(r["a"])), len(str(r["b"]))) >= 5,
                    ),
                ]:
                    t = [r for r in raw["original"][field] if predicate(r)]
                    s = [r for r in raw[name][field] if predicate(r)]
                    if t:
                        record[f"{field}/{label}"] = accuracy(t, s, 0.05, True)
            diagnostics[split][name] = record
    timing = []
    for path in sorted((out / "benchmark").glob("*-b*-p*.json")):
        result = read(path)
        name = result["method"]
        samples = result["samples"]
        assert len(samples) == 50
        speedups = [
            1 - s[name]["end_to_end_seconds"] / s["original"]["end_to_end_seconds"] for s in samples
        ]
        entry = {
            "method": name,
            "batch": result["batch"],
            "prefill_tokens": result["prefill_tokens"],
            "device_uuid": result["device_uuid"],
            "paired_speedup": bootstrap(speedups),
            "five_pair_block_speedup": bootstrap(
                [mean(speedups[i : i + 5]) for i in range(0, 50, 5)]
            ),
            "student_module_accounting": result["student_module_accounting"],
        }
        for label, method in [("original", "original"), ("student", name)]:
            entry[label] = {
                key: statistics.median(s[method][key] for s in samples)
                for key in [
                    "prefill_seconds",
                    "decode_seconds",
                    "end_to_end_seconds",
                    "prefill_gpu_ms",
                    "decode_gpu_ms",
                    "prefill_tokens_per_second",
                    "decode_tokens_per_second",
                    "output_tokens_per_second",
                    "peak_allocated_bytes",
                    "peak_reserved_bytes",
                ]
            }
        timing.append(entry)
    routing, equations = {}, {}
    for split in ["gate", "shift"]:
        for style in ["known", "new"]:
            cohort = f"{split}-256-{style}"
            data = read(mechanism / "state-sufficiency/v3" / f"{cohort}.json")
            by_kind = collections.defaultdict(list)
            for row in data["records"]:
                by_kind[row["kind"]].append(row)
            summary = {}
            for kind, rows in by_kind.items():
                ids = sorted({r["id"] for r in rows})
                per_group = [[r for r in rows if r["id"] == identity] for identity in ids]
                failures = sum(
                    any(not r["all_logits_bitwise_equal_to_donor"] for r in g) for g in per_group
                )
                successes = sum(any(r["target_digit_correct"] for r in g) for g in per_group)
                n = len(ids)
                summary[kind] = {
                    "cases": len(rows),
                    "target_digit_accuracy": mean(r["target_digit_correct"] for r in rows),
                    "target_digit_accuracy_bootstrap": bootstrap(
                        [mean(r["target_digit_correct"] for r in g) for g in per_group]
                    ),
                    "any_format_success_upper95": 1.0
                    if successes == n
                    else float(beta.ppf(0.95, successes + 1, n - successes)),
                    "exact_logit_cases": sum(r["all_logits_bitwise_equal_to_donor"] for r in rows),
                    "exact_logit_group_failure_upper95": 1.0
                    if failures == n
                    else float(beta.ppf(0.95, failures + 1, n - failures)),
                    "consistent_prefix_cases": sum(r["prefix_consistent_with_donor"] for r in rows),
                    "consistent_prefix_accuracy": mean(
                        r["target_digit_correct"] for r in rows if r["prefix_consistent_with_donor"]
                    )
                    if any(r["prefix_consistent_with_donor"] for r in rows)
                    else None,
                }
            routing[cohort] = {
                "variants": summary,
                "all_original_groups": data["all_original_groups"],
                "eligible_original_groups": data["eligible_original_groups"],
                "evaluated_groups": data["evaluated_groups"],
            }
            equations[cohort] = read(mechanism / "equation-responses" / f"{cohort}.json")
    read(out / "data/freeze.json")
    read(out / "training/freeze.json")
    read(out / "evaluation/gate/freeze.json")
    read(out / "evaluation/final/freeze.json")
    module_timing = []
    for r in read(out / "microbenchmark.json")["records"]:
        module_timing.append(
            {
                "method": r["method"],
                "tokens": r["tokens"],
                **{
                    f"{kind}_{metric}": statistics.median(s[kind][metric] for s in r["samples"])
                    for kind in ["original", "student"]
                    for metric in ["gpu_ms_per_call", "wall_ms_per_call"]
                },
            }
        )
    chosen = selection["families"]["eml"]
    chosen_training = next(
        r
        for r in training
        if r["kind"] == "eml" and r["budget"] == chosen["budget"] and r["depth"] == chosen["depth"]
    )
    failure_analysis = {
        "selected_eml_training": chosen_training,
        "selected_eml_domain_errors": {
            name: {"train": audits[name]["train"], "selection": audits[name]["selection"]}
            for name in chosen["checkpoints"]
        },
        "raw_training_error_fraction_at_rank_floor": chosen_training["rank_floor"]
        / chosen_training["training_raw_mse_mean"],
        "selected_eml_failed_gate_endpoints": {
            name: [k for k, v in gate["methods"][name]["accuracy"].items() if not v["passes"]]
            + (["language"] if not gate["methods"][name]["language_ce_increase"]["passes"] else [])
            + (
                ["hostname_language"]
                if not gate["methods"][name]["language_ce_increase"]["hostname_sensitivity_passes"]
                else []
            )
            for name in chosen["checkpoints"]
        },
        "selected_eml_parameter_reduction": {
            name: final["methods"][name]["storage"]["model_parameter_reduction"]
            for name in chosen["checkpoints"]
        },
        "benchmark_comparison_cpu_reference_mlp_bytes": 56623104 * 2,
        "comparison_reference_note": "Held outside the registered model only for alternating benchmarks/evaluation; absent from exported inference. CPU staging and transfers are outside measured execution.",
    }
    summary = {
        "selection": selection,
        "gate": gate,
        "final": final,
        "training": training,
        "deployment_validation": deployed,
        "diagnostics": diagnostics,
        "timing": timing,
        "routing": routing,
        "equations": equations,
        "module_timing": module_timing,
        "failure_analysis": failure_analysis,
    }
    save(destination / "summary.json", summary)
    lines = [
        "# Gemma complete-MLP replacement and causal arithmetic tests",
        "",
        "This study physically replaces Gemma's entire layer-26 MLP. The removed block cannot execute in replacement evaluation, benchmarks, or the exported inference path. Whether the replacement preserves quality is a separate, held-out question.",
        "",
        "Exact checkpoint: `google/gemma-4-E2B-it`, revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`. Native model: **5,104,297,504 unique parameters**. Target block: **56,623,104**. E2B refers to effective size, not the total stored parameter count.",
        "",
        f"The frozen one-block expansion gate **{'passed' if gate['eml_expansion_allowed'] else 'failed'}**. The ≥20% total-parameter, ≥10% latency, and ≤1pp accuracy-loss deployment targets are **not established**. No multi-block expansion followed a failed gate.",
        "",
        "## Held-out quality",
        "",
        "Every row below is a training seed, not a selected favorable run. Accuracy uses the full original-format final cohort. The linked JSON includes paired 95% loss intervals, conservative multiplicity-adjusted regression bounds, new formats, longer operands, and carry diagnostics.",
        "",
        "| Replacement | Addition % | Multiplication % | Division % | ARC % | Language Δ nats | Gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    original_accuracy = final["original_accuracy"]
    first = next(iter(final["methods"].values()))
    lines.append(
        f"| Original | {100 * original_accuracy['add']:.2f} | {100 * original_accuracy['multiply']:.2f} | {100 * original_accuracy['divide']:.2f} | {100 * first['accuracy']['arc']['teacher_accuracy']:.2f} | 0 | Reference |"
    )
    for name, r in final["methods"].items():
        a = r["accuracy"]
        cells = " | ".join(
            f"{100 * a[k]['student_accuracy']:.2f}" for k in ["add", "multiply", "divide", "arc"]
        )
        lines.append(
            f"| {name} | {cells} | {r['language_ce_increase']['mean']:+.4f} | {'Pass' if gate['methods'][name]['quality_gate_passes'] else 'Fail'} |"
        )
    lines += [
        "",
        "Division uses the preregistered 24-token integer-only output contract. The original model often emits prose; a zero baseline is uninformative about preservation of division competence. No parser or generation-budget change was made after observing that problem.",
        "",
        "The initial grid omitted BOS from raw documents. A selection-only native check found 9.764632 versus 4.795804 nats/token without/with BOS. Those 42 fits were retained as diagnostics. The reported grid recollects language activations with BOS and retrains all 42 candidates; no replacement gate/final outputs were opened before the correction. Arithmetic chat tokenization already included BOS. See [BOS amendment](../BOS_AMENDMENT.md).",
        "",
        "## Depth, capacity, and training cost",
        "",
        "Budgets count every encoder, nonlinear argument/readout projection, decoder, bias, norm, and statistic. EML/SiLU comparisons match sequential nonlinear depth and coefficient ceilings. Deployment folds affine statistics; the cheapest linear factorization is collapsed to one affine map. Native surrounding norms and all other model components remain counted.",
        "",
        "| Family | Budget | Depth | Output rank | Selection objective, mean of 3 seeds | Train raw MSE | Rank floor | Fit minutes, 3 seeds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in training:
        lines.append(
            f"| {r['kind']} | {r['budget'] / 1e6:g}M | {r['depth']} | {r['rank']} | {r['selection_objective_mean']:.5f} | {r['training_raw_mse_mean']:.5f} | {r['rank_floor']:.5f} | {r['training_seconds_total'] / 60:.1f} |"
        )
    lines += [
        "",
        f"Corrected-grid fit time totals **{sum(r['training_seconds_total'] for r in training) / 3600:.2f} device-hours of elapsed training time**. This includes validation and shared-device delays, and is not a hardware active-compute counter. It excludes teacher collection, initialization, the superseded BOS diagnostic grid, and final evaluation. Their individual timings remain in the run records.",
        "",
        "The covariance-tail floor applies to raw MLP outputs confined to an affine decoder subspace of the stated rank. It bounds any such decoder on this training distribution, regardless of nonlinear depth; it does not bound EML architectures generally or the normalized residual contribution. The JSON also reports clipping, sampled exponent clamps, precision conversion, and actual CUDA memory peaks.",
        "",
        f"For the selected EML configuration, the raw training MSE is {chosen_training['training_raw_mse_mean']:.5f} and its rank floor is {chosen_training['rank_floor']:.5f}. The floor is {100 * failure_analysis['raw_training_error_fraction_at_rank_floor']:.1f}% of the observed raw error. Deeper nonlinear stages cannot remove that affine-output-rank constraint. Residual error above the floor, domain-specific errors, clipping and seed spread remain separate capacity/optimization diagnostics; they do not identify a universal EML limitation.",
        "",
        "Selected EML gate failures: "
        + "; ".join(
            f"`{name}`: {', '.join(endpoints) or 'none'}"
            for name, endpoints in failure_analysis["selected_eml_failed_gate_endpoints"].items()
        )
        + ". The detailed bounds distinguish measured net degradation from failure to certify a one-percentage-point limit.",
        "",
        "## Actual replacement inference cost",
        "",
        "Fully GPU-resident BF16, native eager SDPA, identical optimization of baselines. Each workload uses 10 warm-ups and 50 alternating paired runs with 32 forced decode steps. Model loading, comparison transfers, tokenization, and warm-up are excluded; prefill, decoding, and phase synchronization are included. Other users share these H100s. Intervals describe variability within these runs, not exclusive-serving or between-run uncertainty.",
        "",
        "| Method | Batch/prefill | Prefill ms | Decode ms | End-to-end ms | Output tokens/s | Peak allocated GiB | E2E speedup %, block-bootstrap 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in timing:
        v = r["student"]
        b = r["five_pair_block_speedup"]
        lo, hi = b["two_sided_95"]
        lines.append(
            f"| {r['method']} | {r['batch']}/{r['prefill_tokens']} | {1000 * v['prefill_seconds']:.1f} | {1000 * v['decode_seconds']:.1f} | {1000 * v['end_to_end_seconds']:.1f} | {v['output_tokens_per_second']:.1f} | {v['peak_allocated_bytes'] / 2**30:.2f} | {100 * b['mean']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] |"
        )
    lines += [
        "",
        "The JSON reports each paired original baseline, separate GPU event timings, prefill/decode throughput, peak allocated/reserved memory, device UUID, and counted student state. End-to-end speedup is a measured paired ratio; parameter reduction alone is not evidence of speedup.",
        "",
        "Alternating comparisons retain an extra **113,246,208-byte CPU reference MLP** outside the registered replacement model. That apparatus is explicitly separate from deployed parameter counts and is absent from the export. Its transfers are outside timing; GPU execution and peak working memory are measured on the installed replacement path.",
        "",
        "Isolated module timings below diagnose token-count dependence. They omit the rest of Gemma and cannot establish an end-to-end speedup. All five tested token counts and both CUDA-event/wall measurements are in the JSON.",
        "",
        "| Method | Tokens | Original module GPU ms | Replacement module GPU ms | Original wall ms | Replacement wall ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in module_timing:
        if r["tokens"] in [1, 4096]:
            lines.append(
                f"| {r['method']} | {r['tokens']} | {r['original_gpu_ms_per_call']:.4f} | {r['student_gpu_ms_per_call']:.4f} | {r['original_wall_ms_per_call']:.4f} | {r['student_wall_ms_per_call']:.4f} |"
            )
    lines += [
        "",
        "## Causal result",
        "",
        "Carry and digit probes are hypotheses. High decoding accuracy did not make the operand-orthogonal carry direction a mediator. The confirmation tests transplant native residual/KV state, use same-carry and matched random controls, and block the downstream carry readout. All rates include native errors and fixed prefixes inconsistent with the donor's true answer; that prefix-consistent subset is reported separately.",
        "",
        "| Cohort | Eligible / original groups | Both residual+KV: exact logits / cases | Residual only: donor digit % | Earlier sliding V: donor digit % | Random KV % | Carry direction % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in routing.items():
        v = r["variants"]
        lines.append(
            f"| {name} | {r['eligible_original_groups']}/{r['all_original_groups']} | {v['both']['exact_logit_cases']}/{v['both']['cases']} | {100 * v['hidden_only']['target_digit_accuracy']:.2f} | {100 * v['sliding_other_values']['target_digit_accuracy']:.2f} | {100 * v['matched_random_sliding_kv']['target_digit_accuracy']:.2f} | {100 * v['carry_direction']['target_digit_accuracy']:.2f} |"
        )
    lines += [
        "",
        "An exact match of the source residual can coexist with different downstream states and logits when reused KV differs. A deterministic equation of that residual alone therefore lacks a required input in these interventions. Each equation-response file quantifies the best possible average squared error on these conflicting-input pairs and tests the actual trained EML/SiLU, linear, symbolic, and identity equations. Cache-only interventions provide a direct test: unchanged equation inputs imply zero predicted change despite a nonzero native response.",
        "",
        "This is evidence about conditional causal routing and input sufficiency. It is not a recovered addition algorithm, an explanation of multiplication/division, or proof that EML cannot succeed with a different state representation. Failed carry mediation prevents using that interpretation to justify a mechanism-based replacement.",
        "",
        "The omitted-KV limitation applies to the cross-layer equation. The native MLP itself is a deterministic function of its complete input, so missing KV does not explain the compact MLP's reconstruction error. See [the bound derivations and their scope](../THEORY_LIMITS.md).",
        "",
        "## Reproduction and evidence",
        "",
        "See [commands](../README.md), the [prospective protocol](../PROTOCOL.md), [evaluation details](../EVALUATION_PLAN.md), and [causal confirmation plan](../CAUSAL_CONFIRMATION.md). `summary.json` contains all seed results and diagnostics. `provenance.json` binds source and scientific inputs by SHA-256. Large activation tensors, raw evaluation traces, and checkpoints are retained in the external run roots; the compact export needs the pinned base checkpoint and replacement weights, with no teacher activations at inference.",
    ]
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = {"eml": "#007c91", "silu": "#d97922"}
    for kind in ["eml", "silu"]:
        for budget, marker in [(3000000, "o"), (6000000, "s")]:
            rs = [r for r in training if r["kind"] == kind and r["budget"] == budget]
            axes[0].plot(
                [r["depth"] for r in rs],
                [r["selection_objective_mean"] for r in rs],
                marker=marker,
                color=colors[kind],
                linestyle="-" if budget == 6000000 else "--",
                label=f"{kind.upper()} {budget / 1e6:g}M",
            )
    axes[0].set(
        xlabel="Nonlinear depth",
        ylabel="Selection objective (mean of 3 seeds)",
        title="Depth at matched coefficient budgets",
        xticks=[1, 2, 4],
    )
    axes[0].legend(fontsize=8)
    labels = [
        "hidden_only",
        "sliding_keys_only",
        "sliding_values_only",
        "matched_random_sliding_kv",
        "both",
    ]
    v = routing["gate-256-known"]["variants"]
    heights = [100 * v[k]["target_digit_accuracy"] for k in labels]
    axes[1].bar(
        range(len(labels)), heights, color=["#777777", "#aaaaaa", "#007c91", "#d97922", "#245b35"]
    )
    axes[1].set(
        xticks=range(len(labels)),
        xticklabels=["Residual", "Keys", "Values", "Random KV", "Residual+KV"],
        ylabel="Donor target digit (%)",
        title="Original-model causal interventions",
        ylim=(0, 100),
    )
    axes[1].tick_params(axis="x", rotation=20)
    fig.savefig(destination / "primary.png", dpi=180)
    fig.savefig(destination / "primary.pdf")
    plt.close(fig)
    save(
        destination / "provenance.json",
        {
            "scientific_inputs": provenance,
            "source_sha256": {p.name: digest(p) for p in sorted(HERE.glob("*.py"))},
            "files": {
                name: digest(destination / name)
                for name in ["REPORT.md", "summary.json", "primary.png", "primary.pdf"]
            },
        },
    )
    print("REPORT COMPLETE", destination, flush=True)


if __name__ == "__main__":
    main()
