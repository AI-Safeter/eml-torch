"""Render the completed, provenance-bound replacement and causal study."""

import argparse
import collections
import json
import statistics

import matplotlib
import torch
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


def carry_readouts(by_kind):
    """Check edit efficacy separately from the hypothesis of causal mediation."""
    rows = {
        kind: sorted(values, key=lambda r: (r["id"], r["style"]))
        for kind, values in by_kind.items()
    }
    identities = [(r["id"], r["style"]) for r in rows["recipient"]]
    assert all([(r["id"], r["style"]) for r in rs] == identities for rs in rows.values())
    decoded = {
        kind: torch.tensor(
            [[r["quantities"][layer][4] for layer in ["26", "32"]] for r in rs], device="cuda"
        )
        for kind, rs in rows.items()
    }
    result = {"native_accuracy": {}, "edits": {}}
    for kind, operand in [("recipient", "a"), ("donor", "donor_a")]:
        truth = torch.tensor(
            [r[operand] % 10 + r["b"] % 10 >= 10 for r in rows[kind]], device="cuda"
        )
        result["native_accuracy"][kind] = (
            ((decoded[kind] >= 0.5) == truth[:, None]).float().mean(0).tolist()
        )
    for kind, reference, column in [
        ("carry_direction", "donor", 0),
        ("values_then_carry_block", "recipient", 1),
        ("values_then_random_block", "recipient", 1),
    ]:
        value, target = decoded[kind][:, column], decoded[reference][:, column]
        result["edits"][kind] = {
            "layer": [26, 32][column],
            "reference": reference,
            "readout_mean_absolute_error": float((value - target).abs().mean()),
            "readout_max_absolute_error": float((value - target).abs().max()),
            "threshold_class_agreement": float(((value >= 0.5) == (target >= 0.5)).float().mean()),
        }
    for kind in ["values_then_carry_block", "values_then_random_block"]:
        paired = collections.defaultdict(list)
        for edited, base in zip(rows[kind], rows["sliding_values_only"]):
            paired[base["id"]].append(
                int(edited["target_digit_correct"]) - int(base["target_digit_correct"])
            )
        result["edits"][kind]["target_digit_accuracy_change"] = bootstrap(
            [mean(v) for v in paired.values()]
        )
    result["scope"] = (
        "Post hoc edit-efficacy diagnostics on fixed interventions. Readout equality does not imply equality of a causally meaningful variable; native readout accuracy is reported by distribution."
    )
    return result


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
    diagnostics, matched = {}, {}
    for split in ["final", "shift"]:
        raw = {
            name: read(out / "evaluation" / split / f"{name}.json")["results"] for name in methods
        }
        diagnostics[split] = {}
        matched[split] = {}
        selected_eml = selection["families"]["eml"]
        for seed in SPEC["replacement"]["seeds"]:
            eml_name = f"eml-b{selected_eml['budget']}-d{selected_eml['depth']}-s{seed}"
            silu_name = f"silu-b{selected_eml['budget']}-d{selected_eml['depth']}-s{seed}"
            comparison = {}
            for field in ["arithmetic", "new_formats"]:
                for op in SPEC["arithmetic"]["operations"]:
                    s = [r for r in raw[silu_name][field] if r["op"] == op]
                    e = [r for r in raw[eml_name][field] if r["op"] == op]
                    value = accuracy(s, e, 0.05, True)
                    lo, hi = value["net_loss_bootstrap"]["two_sided_95"]
                    comparison[f"{field}/{op}"] = {
                        "eml_accuracy_gain": -value["net_loss"],
                        "gain_95ci": [-hi, -lo],
                        "groups": value["groups"],
                    }
            if split == "final":
                value = accuracy(raw[silu_name]["arc"], raw[eml_name]["arc"], 0.05)
                lo, hi = value["net_loss_bootstrap"]["two_sided_95"]
                comparison["arc"] = {
                    "eml_accuracy_gain": -value["net_loss"],
                    "gain_95ci": [-hi, -lo],
                    "groups": value["groups"],
                }
                comparison["language_ce_eml_minus_silu"] = bootstrap(
                    [
                        e["ce_nats"] - s["ce_nats"]
                        for e, s in zip(raw[eml_name]["language"], raw[silu_name]["language"])
                    ]
                )
            matched[split][str(seed)] = {
                "eml": eml_name,
                "silu": silu_name,
                "metrics": comparison,
                "scope": "Descriptive paired 95% intervals, unadjusted. No model selection or acceptance decision uses these comparisons.",
            }
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
            entry[label]["isolated_memory"] = result["isolated_memory"][method]
            entry[label]["peak_allocated_bytes"] = max(
                result["isolated_memory"][method]["peak_allocated_bytes"],
                max(s[method]["peak_allocated_bytes"] for s in samples),
            )
            entry[label]["peak_reserved_bytes"] = result["isolated_memory"][method][
                "peak_reserved_bytes"
            ]
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
                    "donor_top_token_agreement": mean(r["donor_top_token_match"] for r in rows),
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
                "carry_readouts": carry_readouts(by_kind),
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
    equation_observational = read(mechanism / "residual/equations/observational-results.json")
    extension = [read(p) for p in sorted((out / "optimization-extension").glob("*-b*-s*.json"))]
    assert len(extension) == 6
    compiler = [read(p) for p in sorted((out / "compiler-diagnostic").glob("*-b*-s*.json"))]
    if compiler:
        read(out / "compiler-diagnostic/freeze.json")
    superseded = [read(p) for p in sorted((mechanism / "training").glob("*-b*-s*.json"))]
    assert not superseded or len(superseded) == 42
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
    eml_training = [r for r in training if r["kind"] == "eml"]
    stability = {
        "gradient_clip_fraction": mean(r["gradient_clip_fraction"] for r in eml_training),
        "sampled_exponent_clamps": sum(r["exponent_clamps"] for r in eml_training),
        "sampled_exponent_arguments": sum(r["exponent_arguments_sampled"] for r in eml_training),
    }
    failure_analysis["eml_training_stability"] = stability
    summary = {
        "selection": selection,
        "gate": gate,
        "final": final,
        "training": training,
        "deployment_validation": deployed,
        "diagnostics": diagnostics,
        "matched_eml_silu": matched,
        "timing": timing,
        "routing": routing,
        "equations": equations,
        "equation_observational": equation_observational,
        "module_timing": module_timing,
        "failure_analysis": failure_analysis,
        "optimization_extension": extension,
        "compiler_diagnostic": compiler,
        "superseded_bos_grid_fit_seconds": sum(r["training_seconds"] for r in superseded),
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
        "Division uses the preregistered 24-token integer-only output contract. The original model often emits prose and has very low accuracy under this contract, providing weak evidence about preservation of division competence. No parser or generation-budget change was made after observing that problem.",
        "",
        "The initial grid omitted BOS from raw documents. A selection-only native check found 9.764632 versus 4.795804 nats/token without/with BOS. Those 42 fits were retained as diagnostics. The reported grid recollects language activations with BOS and retrains all 42 candidates; no replacement gate/final outputs were opened before the correction. Arithmetic chat tokenization already included BOS. See [BOS amendment](../BOS_AMENDMENT.md).",
        "",
        "Matched-depth EML versus SiLU on the final original-format tasks follows. Positive differences favor EML. Intervals resample operand groups (or ARC questions); they are descriptive unadjusted 95% intervals, not a post hoc selection rule.",
        "",
        "| Seed | Addition difference pp [95% CI] | Multiplication difference pp [95% CI] | Division difference pp [95% CI] | ARC difference pp [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for seed, row in matched["final"].items():
        cells = []
        for key in ["arithmetic/add", "arithmetic/multiply", "arithmetic/divide", "arc"]:
            v = row["metrics"][key]
            lo, hi = v["gain_95ci"]
            cells.append(f"{100 * v['eml_accuracy_gain']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]")
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Generalization below keeps every generated answer, including malformed outputs and native errors. Each student entry is the mean and range over the three selected EML seeds. These summaries do not replace the paired per-seed intervals in the JSON.",
        "",
        "| Operand split / prompts | Task | Original % | EML %, mean [seed range] | Groups per seed |",
        "|---|---|---:|---|---:|",
    ]
    for split in ["final", "shift"]:
        for field in ["arithmetic", "new_formats"]:
            for op in SPEC["arithmetic"]["operations"]:
                rs = [diagnostics[split][name][f"{field}/{op}"] for name in chosen["checkpoints"]]
                values = [100 * r["student_accuracy"] for r in rs]
                lines.append(
                    f"| {split} / {'known' if field == 'arithmetic' else 'new'} | {op} | {100 * rs[0]['teacher_accuracy']:.2f} | {mean(values):.2f} [{min(values):.2f}, {max(values):.2f}] | {rs[0]['groups']} |"
                )
    lines += [
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
        f"Across the EML primary grid, gradient clipping occurred on {100 * stability['gradient_clip_fraction']:.4f}% of updates, and telemetry found {stability['sampled_exponent_clamps']} clamped exponent arguments among {stability['sampled_exponent_arguments']:,} sampled arguments. These observations do not support widespread exponential saturation as the main measured failure. They do not establish convergence or rule out difficult optimization geometry.",
        "",
        "| Selected EML domain | Train raw MSE | Selection raw MSE | Train contribution MSE | Selection contribution MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for domain in ["arithmetic", "language"]:
        cells = [
            mean(audits[name][split][domain][metric] for name in chosen["checkpoints"])
            for metric in ["raw_mse", "contribution_mse"]
            for split in ["train", "selection"]
        ]
        lines.append(f"| {domain} | " + " | ".join(f"{v:.5f}" for v in cells) + " |")
    lines += [
        "",
        "Domain errors use the training scales of the fitted objective; differences can reflect both target variation and approximation quality. Teacher-forced activation fit also differs from free generation after the replacement changes a token. The final/new-format/shift tables test behavior under distribution changes; this study does not isolate every source of rollout error.",
        "",
        "All 36 nonlinear primary fits reached the 12,000-update cap and selected their best checkpoint within the last 1,000 updates. Optimization was therefore not demonstrated to converge. A separate diagnostic restarted the selected EML architecture and matched-depth SiLU for 24,000 updates, with the same initialization and minibatch stream. These weights were never substituted into the frozen held-out roster.",
        "",
        "| Diagnostic | Primary 12k objective | Best 24k objective | Best update | Extra run minutes |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in extension:
        lines.append(
            f"| {r['name']} | {r['prefix_replay']['primary_best_objective']:.5f} | {r['selection_objective']:.5f} | {r['selected_step']} | {r['training_seconds'] / 60:.1f} |"
        )
    lines += [
        "",
        (
            f"The withdrawn BOS grid cost {sum(r['training_seconds'] for r in superseded) / 3600:.2f} additional device-hours of elapsed fit time. "
            if superseded
            else "The optional historical BOS diagnostic grid is absent from this reproduction. "
        )
        + f"The six doubled-budget diagnostic runs cost {sum(r['training_seconds'] for r in extension) / 3600:.2f} device-hours. They replayed the first 12k objective before continuing; their saved final objectives were independently replayed on CUDA. Improved activation fit on selection data does not establish improved held-out answer quality.",
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
        "The export contains replacement-module weights. Parameter reduction describes the installed runtime model after removing the original MLP; the supplied loader still reads the full pinned base checkpoint. The export is not a standalone compressed full-model download.",
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
    if compiler:
        lines += [
            "",
            "An additional module diagnostic applies identical `torch.compile` Inductor/default/fullgraph settings to every native/student pair. The table includes every configuration. Setup times and precision differences are recorded; compilation can change BF16 rounding. These results do not replace the eager full-model benchmark or establish compiled-model answer quality. CUDA-event intervals include any gaps between submitted kernels, not just kernel execution time.",
            "",
            "| Method | Tokens | Original eager / compiled GPU ms | Student eager / compiled GPU ms | Student compiled-vs-eager NRMSE |",
            "|---|---:|---|---|---:|",
        ]
        for result in compiler:
            for r in result["records"]:
                if r["status"] == "failed":
                    lines.append(
                        f"| {result['method']} | {r['tokens']} | Failed: {r['error_type']} | See JSON | — |"
                    )
                    continue
                cells = [
                    " / ".join(
                        f"{statistics.median(s[f'{kind}_{mode}']['gpu_ms_per_call'] for s in r['samples']):.4f}"
                        for mode in ["eager", "compiled"]
                    )
                    for kind in ["original", "student"]
                ]
                lines.append(
                    f"| {result['method']} | {r['tokens']} | {cells[0]} | {cells[1]} | {r['precision']['student']['output_nrmse']:.5f} |"
                )
    lines += [
        "",
        "## Causal result",
        "",
        "Carry and digit probes are hypotheses. High decoding accuracy did not establish the operand-orthogonal carry direction as a sufficient or dominant mediator. The confirmation tests transplant native residual/KV state, use same-carry and matched random controls, and block the downstream carry readout. All rates include native errors and fixed prefixes inconsistent with the donor's true answer; that prefix-consistent subset is reported separately.",
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
        "The four cohorts contain 2,048 prompt cases but 512 distinct operand groups: known/new formats reuse each split's 256 groups. Exact-logit failure bounds therefore group formats by operand pair. With zero failures in a 256-group cohort, the one-sided 95% binomial upper bound is about 1.16%, rather than treating every logit or prompt as independent evidence.",
        "",
        "The edit-efficacy check separates the probe readout from behavior. The upstream edit nearly matches the donor's carry readout; the downstream carry block restores the recipient's thresholded readout in every case. The next table reports native decoding accuracy and the paired behavioral effect of blocking, including uncertainty clustered by operand group. Near-chance upstream decoding on shifted operands further limits its interpretation as a general carry variable.",
        "",
        "| Cohort | Native recipient carry accuracy at 26 / 32 % | Upstream edit: donor readout class match % | Downstream block: recipient readout class match % | Blocking effect on donor-digit accuracy pp [95% CI] |",
        "|---|---|---:|---:|---|",
    ]
    for name, r in routing.items():
        c = r["carry_readouts"]
        edit = c["edits"]["values_then_carry_block"]
        effect = edit["target_digit_accuracy_change"]
        lo, hi = effect["two_sided_95"]
        native = " / ".join(f"{100 * v:.2f}" for v in c["native_accuracy"]["recipient"])
        lines.append(
            f"| {name} | {native} | {100 * c['edits']['carry_direction']['threshold_class_agreement']:.2f} | {100 * edit['threshold_class_agreement']:.2f} | {100 * effect['mean']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] |"
        )
    lines += [
        "",
        "An exact match of the source residual can coexist with different downstream states and logits when reused KV differs. A deterministic equation of that residual alone therefore lacks a required input in these interventions. Each equation-response file quantifies the best possible average squared error on these conflicting-input pairs and tests the actual trained EML/SiLU, linear, symbolic, and identity equations. Cache-only interventions provide a direct test: unchanged equation inputs imply zero predicted change despite a nonzero native response.",
        "",
        "Equation response NRMSE below uses the full gate/known cohort. The three-seed mean is reported for every tested depth; no intervention result selects a depth. The complete JSON includes all four cohorts, seeds, simpler controls, absolute response energy, MSE intervals and input-collision bounds. Large normalized errors for a small native response should be read alongside its absolute energy.",
        "",
        "| Equation | Coefficients | Selection observational MSE | Natural donor response NRMSE | Residual-only response NRMSE | V-only response NRMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    eq_groups = [(kind, [kind]) for kind in ["linear", "symbolic", "identity"]]
    eq_groups += [
        (
            f"{kind} depth {depth}",
            [f"{kind}-d{depth}-s{seed}" for seed in SPEC["replacement"]["seeds"]],
        )
        for kind in ["eml", "silu"]
        for depth in [1, 2, 4]
    ]
    obs = equation_observational["results"]
    for label, names in eq_groups:
        coeff = obs[names[0]].get("coefficients", 72 if label == "linear" else 0)
        entries = equations["gate-256-known"]["results"]
        response = [
            mean(entries[name][kind]["response_nrmse"] for name in names)
            for kind in ["donor", "hidden_only", "sliding_values_only"]
        ]
        cells = " | ".join(f"{v:.4f}" for v in response)
        lines.append(
            f"| {label} | {coeff} | {mean(obs[name]['mean'] for name in names):.4f} | {cells} |"
        )
    lines += [
        "",
        "The equation input readout adds **12,296 coefficients**; measuring downstream quantities adds another **12,296**, reported separately from the equation. Inputs include estimates of carry and result digits already present in the upstream state. This is a propagation fit, not derivation of arithmetic from operand labels.",
        "",
        "This is evidence about conditional causal routing and input sufficiency. It is not a recovered addition algorithm, an explanation of multiplication/division, or proof that EML cannot succeed with a different state representation. The small blocking effects can include a partial causal contribution; they do not validate the proposed carry algorithm or justify a mechanism-based replacement.",
        "",
        "The omitted-KV limitation applies to the cross-layer equation. The native MLP itself is a deterministic function of its complete input, so missing KV does not explain the compact MLP's reconstruction error. See [the bound derivations and their scope](../THEORY_LIMITS.md).",
        "",
        "Shared-KV dependence is expected from Gemma's architecture. [Related work](../RELATED_WORK.md) connects this study to Goodfire's interventions and parameter decomposition, causal abstraction, and arithmetic heuristic research without claiming a new algorithm or an advantage over those methods.",
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
    fig.savefig(destination / "diagnostics.png", dpi=180)
    fig.savefig(destination / "diagnostics.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    display = [
        ("EML", list(chosen["checkpoints"]), "#007c91"),
        (
            f"SiLU d{chosen['depth']}",
            [
                f"silu-b{chosen['budget']}-d{chosen['depth']}-s{s}"
                for s in SPEC["replacement"]["seeds"]
            ],
            "#d97922",
        ),
    ]
    silu_selected = selection["families"]["silu"]
    if silu_selected["depth"] != chosen["depth"]:
        display.append(
            (f"SiLU d{silu_selected['depth']}", list(silu_selected["checkpoints"]), "#925821")
        )
    display.append(
        (
            "Linear rank 974",
            [f"linear-b3000000-d0-s{s}" for s in SPEC["replacement"]["seeds"]],
            "#777777",
        )
    )
    endpoints = ["add", "multiply", "divide", "arc"]
    bar_width = 0.8 / len(display)
    for j, (label, names, color) in enumerate(display):
        values = [
            [100 * final["methods"][name]["accuracy"][task]["net_loss"] for name in names]
            for task in endpoints
        ]
        centers = [mean(v) for v in values]
        errors = [
            [c - min(v) for c, v in zip(centers, values)],
            [max(v) - c for c, v in zip(centers, values)],
        ]
        positions = [i - 0.4 + bar_width / 2 + j * bar_width for i in range(4)]
        axes[0].bar(positions, centers, bar_width, yerr=errors, color=color, label=label, capsize=2)
    axes[0].axhline(1, color="black", linestyle="--", linewidth=1, label="1pp target")
    axes[0].set(
        xticks=range(4),
        xticklabels=["Add", "Multiply", "Divide†", "ARC"],
        ylabel="Final net accuracy loss (percentage points)",
        title="Complete MLP replacement",
    )
    axes[0].legend(fontsize=8)
    primary = [r for r in timing if r["batch"] == 8 and r["prefill_tokens"] == 512]
    for j, r in enumerate(primary):
        name = r["method"]
        label = name.split("-b")[0].upper() + " " + name.split("-d")[1].split("-")[0]
        if name.startswith("linear"):
            label = "Linear " + ("974" if "b3000000" in name else "1536")
        b = r["five_pair_block_speedup"]
        center, (lo, hi) = 100 * b["mean"], [100 * v for v in b["two_sided_95"]]
        axes[1].errorbar(
            center,
            j,
            xerr=[[max(0, center - lo)], [max(0, hi - center)]],
            fmt="o",
            color="#007c91" if name.startswith("eml") else "#777777",
            capsize=3,
        )
        axes[1].text(center, j + 0.12, label, fontsize=8, ha="center")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].axvline(10, color="#245b35", linestyle="--", label="10% target")
    axes[1].set(
        yticks=[],
        xlabel="Paired end-to-end speedup (%)",
        title="Batch 8 / prefill 512 / decode 32",
        ylim=(-0.5, len(primary) - 0.5),
    )
    axes[1].legend(fontsize=8)
    cohorts = list(routing)
    for j, (kind, label, color) in enumerate(
        [
            ("hidden_only", "Residual", "#888888"),
            ("sliding_other_values", "Earlier sliding V", "#007c91"),
            ("matched_random_sliding_kv", "Random KV", "#d97922"),
            ("both", "Residual + KV", "#245b35"),
        ]
    ):
        values = [100 * routing[c]["variants"][kind]["donor_top_token_agreement"] for c in cohorts]
        axes[2].bar([i - 0.3 + j * 0.2 for i in range(4)], values, 0.2, color=color, label=label)
    axes[2].set(
        xticks=range(4),
        xticklabels=["Gate\nknown", "Gate\nnew", "Shift\nknown", "Shift\nnew"],
        ylabel="Donor top-token agreement (%)",
        title="Original-model intervention response",
        ylim=(0, 105),
    )
    axes[2].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Quality bars: mean and range over three seeds. Timing: within-run block bootstrap. †Division has a weak original baseline.",
        fontsize=10,
    )
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
                for name in [
                    "REPORT.md",
                    "summary.json",
                    "primary.png",
                    "primary.pdf",
                    "diagnostics.png",
                    "diagnostics.pdf",
                ]
            },
        },
    )
    print("REPORT COMPLETE", destination, flush=True)


if __name__ == "__main__":
    main()
