"""Render the completed, audited study without choosing favorable cells."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from runtime import HERE, RUNS

PRIMARY = "heads-active-r32-g0.1/eml"
CONTROL = "heads-active-r32-g0.1/neural"
ROSTER = json.loads((HERE / "study.json").read_text())
LABELS = ROSTER["primary_models"]


def number(value, digits=3):
    return "undefined" if value is None else f"{value:.{digits}f}"


def interval(value):
    return "undefined" if value is None else f"[{number(value[0])}, {number(value[1])}]"


def figures(cells, destination):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), layout="constrained", sharey=True)
    for index, row in enumerate(cells):
        for key, color, offset, label in [
            ("eml", "#166e67", -0.12, "EML"),
            ("neural", "#bb601b", 0.12, "SiLU"),
        ]:
            metric = row[key]
            ci = metric["nrmse_simultaneous_ci95"]
            if metric["nrmse"] is not None and ci is not None:
                axes[0].plot(ci, [index + offset] * 2, color=color)
                axes[0].scatter(
                    metric["nrmse"],
                    index + offset,
                    color=color,
                    s=22,
                    label=label if index == 0 else None,
                )
        axes[1].scatter(row["ordinary"]["accuracy_loss_upper_pp"], index, color="#166e67")
        axes[2].scatter(
            row["ordinary"]["original_accuracy"] * 100,
            index,
            color="#555555",
            label="Original" if index == 0 else None,
        )
        axes[2].scatter(
            row["ordinary"]["accuracy"] * 100,
            index,
            color="#166e67",
            marker="x",
            label="Replacement" if index == 0 else None,
        )
    axes[0].axvline(0.2, color="#a33", linestyle="--", linewidth=1)
    axes[1].axvline(1, color="#a33", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Response NRMSE (lower is better)")
    axes[1].set_xlabel("Upper bound on accuracy loss (pp)")
    axes[2].set_xlabel("Complete-answer accuracy (%)")
    axes[0].set_yticks(
        range(len(cells)), [f"{LABELS[r['model']]} / {r['operation']}" for r in cells]
    )
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    axes[2].legend(frameon=False)
    fig.suptitle("Fresh scalar-component confirmation: every model and operation")
    for suffix in ["png", "pdf"]:
        fig.savefig(destination / f"primary.{suffix}", dpi=180)
    plt.close(fig)


def explorer(results, destination):
    payload = json.dumps(results, allow_nan=False).replace("<", "\\u003c")
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Component replacement evidence</title><style>
body{font:16px/1.5 system-ui,sans-serif;background:#f5f4ef;color:#222;margin:0}main{max-width:1180px;margin:auto;padding:32px}h1{font-size:30px}select{font:inherit;padding:8px;margin:8px 16px 8px 0}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:10px;border-bottom:1px solid #ddd}th{background:#e7ece9}td{font-variant-numeric:tabular-nums}p{max-width:850px}.scroll{overflow-x:auto}img{width:100%;height:auto}.note{color:#555}a{color:#166e67}
</style><main><h1>Component replacement evidence</h1><p>All nine model/operation cells, including failed criteria and stress tests. Scalar replacement retains the original MLP's other contributions. Whole-block results are reported separately in REPORT.md.</p>
<img src="primary.png" alt="Primary response fidelity, accuracy-loss bounds, and complete-answer accuracy for all nine cells">
<label>Model <select id="model"></select></label><label>Operation <select id="operation"><option>add</option><option>multiply</option><option>divide</option></select></label>
<p id="decision"></p><p class="note">Response intervals are adjusted across nine cells per metric. Stress and seed results do not redefine the primary criteria. Accuracy bounds assume independent operand groups and preserve dependence between prompt formats.</p>
<div class="scroll"><table><thead><tr><th>Suite</th><th>Method</th><th>Response NRMSE</th><th>Response correlation</th><th>Answer accuracy</th><th>Upper accuracy loss (pp)</th></tr></thead><tbody id="rows"></tbody></table></div><p><a href="REPORT.md">Full report</a> · <a href="summary.json">Summary data</a></p></main>
<script>const DATA=__DATA__;const PRIMARY='heads-active-r32-g0.1/eml';const LABELS=__LABELS__;const model=document.querySelector('#model');const operation=document.querySelector('#operation');for(const [key,label] of Object.entries(LABELS)){const o=document.createElement('option');o.value=key;o.textContent=label;model.append(o)}
const fmt=x=>x===null||x===undefined?'—':Number(x).toFixed(3);
function render(){const d=DATA[model.value+'/'+operation.value];document.querySelector('#decision').textContent='Primary criteria: '+(d.primary_checks.all_pass?'all passed':'one or more failed')+'. This is a fidelity test; it does not establish a semantic arithmetic algorithm.';const body=document.querySelector('#rows');body.replaceChildren();const suites=new Set([...Object.keys(d.ordinary).map(x=>x.replace(/^ordinary-/,'')),...Object.keys(d.interventions).map(x=>x.replace(/^interventions-/,''))]);for(const suite of suites){const normal=d.ordinary['ordinary-'+suite]||{};const causal=d.interventions['interventions-'+suite]||d.interventions[suite]||{models:{}};const names=new Set([...Object.keys(normal),...Object.keys(causal.models)]);for(const name of names){if(name==='original'||name==='restore')continue;const n=normal[name]||{},c=causal.models[name]||{};const row=document.createElement('tr');for(const value of [suite,name,fmt(c.nrmse),fmt(c.correlation),n.accuracy===undefined?'—':fmt(100*n.accuracy)+'%',fmt(n.accuracy_loss_upper_pp)]){const td=document.createElement('td');td.textContent=value;row.append(td)}body.append(row)}}}
model.addEventListener('change',render);operation.addEventListener('change',render);render();</script></html>"""
    (destination / "explorer.html").write_text(
        html.replace("__DATA__", payload).replace(
            "__LABELS__", json.dumps(LABELS).replace("<", "\\u003c")
        )
    )


def main():
    audit = json.loads((RUNS / "audit.json").read_text())
    assert audit["status"] == "complete" and audit["roster"] == ROSTER
    assert set(audit["scalar"]) == {
        f"{model}/{op}" for model in LABELS for op in ["add", "multiply", "divide"]
    }
    destination = HERE / "results"
    destination.mkdir(exist_ok=True)
    results, cells = {}, []
    for model in LABELS:
        for op in ["add", "multiply", "divide"]:
            component = torch.load(RUNS / model / op / "component-active.pt", weights_only=True)
            selection = json.loads((RUNS / model / op / "selection.json").read_text())
            stored = {
                key: component[key].numel()
                for key in ["encoder", "input_mean", "direction", "zmean", "zstd", "ymean", "ystd"]
            }
            stored["head"] = selection["heads"][PRIMARY]["parameters"]
            data = json.loads((RUNS / model / op / "robust-results.json").read_text())
            per_style = json.loads((RUNS / model / op / "per-style-results.json").read_text())
            for category in ["ordinary", "interventions"]:
                data[category].update(per_style[category])
            results[f"{model}/{op}"] = data
            causal = data["interventions"]["interventions-test-known-formats"]["models"]
            cells.append(
                {
                    "model": model,
                    "operation": op,
                    "checks": data["primary_checks"],
                    "stored_scalar_coefficients": stored,
                    "eml": causal[PRIMARY],
                    "neural": causal[CONTROL],
                    "paired_comparison": data["interventions"]["interventions-test-known-formats"][
                        "comparisons"
                    ][PRIMARY],
                    "ordinary": data["ordinary"]["ordinary-test-known-formats"][PRIMARY],
                    "unconditioned": data["ordinary"]["ordinary-unconditioned"][PRIMARY],
                    "all_tokens": data["ordinary"]["ordinary-all-tokens-unconditioned"][PRIMARY],
                    "new_formats": data["ordinary"]["ordinary-test-new-formats"][PRIMARY],
                    "new_formats_response": data["interventions"]["interventions-test-new-formats"][
                        "models"
                    ][PRIMARY],
                    "gradient_capture": json.loads(
                        (RUNS / model / op / "gradient-audit.json").read_text()
                    )["feature_gradient_energy_fraction"],
                }
            )
    whole = json.loads((RUNS / "whole-block/utility-results.json").read_text())
    timing = json.loads((RUNS / "whole-block/latency.json").read_text())
    block_timing = json.loads((RUNS / "whole-block/block-latency.json").read_text())
    storage = json.loads((RUNS / "whole-block/model-storage.json").read_text())
    output_fidelity = json.loads((RUNS / "whole-block/output-fidelity.json").read_text())
    summary = {
        "primary": cells,
        "roster": ROSTER,
        "superseded": audit["superseded"],
        "whole_block": whole,
        "latency": timing,
        "standalone_block_latency": block_timing,
        "model_storage": storage,
        "whole_output_fidelity": output_fidelity,
        "scalar_passed": sum(c["checks"]["all_pass"] for c in cells),
        "scalar_total": 9,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Fresh component-replacement study",
        "",
        f"The validation-selected EML scalar replacement passes all declared primary bounds in **{summary['scalar_passed']} of 9** model/operation cells. All cells are reported below. Whole-block utility criteria **{'pass' if whole['eml']['all_utility_checks_pass'] else 'do not pass'}** for the selected EML vector student.",
        "",
        "These experiments measure preservation of model behavior. A compact formula in learned coordinates does not by itself identify an arithmetic algorithm. Scalar replacement keeps the original dense MLP active and provides no model compression claim.",
        "",
        "## Primary scalar confirmation",
        "",
        "NRMSE intervals below are adjusted across nine cells. Accuracy-loss upper bounds account for three correlated prompt formats within each operand group and remain positive when no regressions are observed.",
        "",
        "| Model / operation | EML NRMSE [interval] | EML correlation [interval] | SiLU NRMSE | Original → EML accuracy | Upper accuracy loss (pp) | All primary bounds |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for c in cells:
        n = c["ordinary"]
        lines.append(
            f"| {LABELS[c['model']]} / {c['operation']} | {number(c['eml']['nrmse'])} {interval(c['eml']['nrmse_simultaneous_ci95'])} | {number(c['eml']['correlation'])} {interval(c['eml']['correlation_simultaneous_ci95'])} | {number(c['neural']['nrmse'])} | {100 * n['original_accuracy']:.2f}% → {100 * n['accuracy']:.2f}% | {number(n['accuracy_loss_upper_pp'])} | {'pass' if c['checks']['all_pass'] else 'fail'} |"
        )
    lines += [
        "",
        "The paired comparison below uses the same operand groups and active features with derivative-loss weight .1. Negative differences favor EML. These 95% intervals are per comparison, without adjustment across cells; an isolated favorable result does not establish operator superiority.",
        "",
        "| Model / operation | EML minus SiLU response MSE | Paired 95% interval |",
        "|---|---:|---:|",
    ]
    for c in cells:
        comparison = c["paired_comparison"]
        lo, hi = comparison["paired_difference_ci95"]
        lines.append(
            f"| {LABELS[c['model']]} / {c['operation']} | {comparison['eml_minus_control_response_mse']:.6g} | [{lo:.6g}, {hi:.6g}] |"
        )
    lines += [
        "",
        "The scalar predictor includes a dense input projection and normalization/direction constants. Head size alone is not its deployment footprint:",
        "",
        "| Model / operation | Head coefficients | Projection coefficients | Total predictor/patch coefficients |",
        "|---|---:|---:|---:|",
    ]
    for c in cells:
        stored = c["stored_scalar_coefficients"]
        lines.append(
            f"| {LABELS[c['model']]} / {c['operation']} | {stored['head']:,} | {stored['encoder']:,} | {sum(stored.values()):,} |"
        )
    lines += [
        "",
        "## Retained input sensitivity",
        "",
        "Validation gradient energy inside each frozen rank-32 input subspace is a local diagnostic. It does not guarantee accuracy on distant operands or arbitrary edits. The active subspace was fitted using training gradients only.",
        "",
        "| Model / operation | PLS retained gradient energy | Active retained gradient energy |",
        "|---|---:|---:|",
    ]
    for c in cells:
        capture = c["gradient_capture"]
        lines.append(
            f"| {LABELS[c['model']]} / {c['operation']} | {100 * capture['pls32']['validation']:.2f}% | {100 * capture['active32']['validation']:.2f}% |"
        )
    lines += [
        "",
        "## Unconditioned prompts and later generated tokens",
        "",
        "The primary contrast cohort is conditioned on answer-token differences. The separate ordinary cohort has no such condition. These original-model accuracies limit any claim about retained arithmetic ability.",
        "",
        "| Model / operation | Original accuracy | Prefill replacement | Replacement through decoding | Upper all-token loss (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in cells:
        n, a = c["unconditioned"], c["all_tokens"]
        lines.append(
            f"| {LABELS[c['model']]} / {c['operation']} | {100 * n['original_accuracy']:.2f}% | {100 * n['accuracy']:.2f}% | {100 * a['accuracy']:.2f}% | {number(a['accuracy_loss_upper_pp'])} |"
        )
    lines += [
        "",
        "The [interactive explorer](explorer.html) includes all feature/loss controls, every fitted seed, held-out formats, shifted operands, carry patterns, both-operand edits, and ambient/nullspace diagnostics. Primary success does not imply unrestricted equivalence under arbitrary input edits.",
        "",
        "Accuracy bounds in the smaller stress cohorts have limited resolution. With three prompt formats and the frozen nine-cell adjustment, even zero regressions give an upper loss bound of 1.22 percentage points for 512 operand groups and 2.43 points for 256 groups. Failure to certify one-point retention in these cohorts is not evidence that the actual loss exceeds one point. Stress results do not redefine the primary decision.",
        "",
        "## Held-out prompt formats",
        "",
        "These four formats were excluded from fitting. Complete-answer accuracy uses the same strict numeric parser as the primary cohort, so extra prose and nonnumeric outputs count as errors. Low replacement error can coexist with low teacher accuracy: preserving a component's response does not repair the teacher's arithmetic or formatting failures.",
        "",
        "| Model / operation | Original answer accuracy | EML answer accuracy | EML response NRMSE |",
        "|---|---:|---:|---:|",
    ]
    for c in cells:
        normal, causal = c["new_formats"], c["new_formats_response"]
        lines.append(
            f"| {LABELS[c['model']]} / {c['operation']} | {100 * normal['original_accuracy']:.2f}% | {100 * normal['accuracy']:.2f}% | {number(causal['nrmse'])} |"
        )
    lines += [
        "",
        "## Whole-block utility",
        "",
        "| Student | Stored coefficients | Block reduction | Language CE increase | Upper CE increase | Utility criteria |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for kind in ["eml", "swiglu", "linear"]:
        r = whole[kind]
        lines.append(
            f"| {kind} | {r['stored_coefficients']:,} | {r['block_storage_reduction']:.2f}× | {number(r['language_increase'])} | {number(r['language_increase_upper95'])} | {'pass' if r['all_utility_checks_pass'] else 'fail'} |"
        )
    lines += [
        "",
        "Complete MLP-output fidelity is measured on real, unpadded held-out token positions. Its NRMSE denominator is RMS original MLP output, including the mean; this differs from the downstream margin-response normalization in the scalar study.",
        "",
        "| Token domain | EML output NRMSE | SwiGLU output NRMSE | Linear output NRMSE |",
        "|---|---:|---:|---:|",
    ]
    for domain, values in output_fidelity.items():
        lines.append(
            "| "
            + domain
            + " | "
            + " | ".join(number(values[k]["output_nrmse"]) for k in ["eml", "swiglu", "linear"])
            + " |"
        )
    lines += [
        "",
        "Whole-block complete-answer accuracy on the unconditioned cohort:",
        "",
        "| Operation | Original | EML | SwiGLU | Linear |",
        "|---|---:|---:|---:|---:|",
    ]
    for op in ["add", "multiply", "divide"]:
        name = op + "/unconditioned"
        accuracies = [whole["original"]["arithmetic"][name]["original_accuracy"]] + [
            whole[k]["arithmetic"][name]["replacement_accuracy"]
            for k in ["eml", "swiglu", "linear"]
        ]
        lines.append("| " + op + " | " + " | ".join(f"{100 * a:.2f}%" for a in accuracies) + " |")
    lines += [
        "",
        "The selected original MLP does not execute in this replacement path. The rest of the model remains frozen. Retention requires a language cross-entropy increase upper bound ≤ .02 nats/token and an accuracy-loss upper bound ≤ 1 percentage point for each operation on unconditioned prompts, in addition to ≥ 4× block storage reduction and finite outputs.",
        "",
        f"The EML replacement reduces the entire model's parameter count by **{storage['students']['eml']['full_model_parameter_reduction_percent']:.2f}%**. The block reduction must not be read as a whole-model compression ratio. These are architecture counts; paired benchmarks keep the original and replacement alternatives resident and do not measure peak-memory savings.",
        "",
        "## End-to-end latency",
        "",
        "| Student | Batch | Phase | Original ms | Replacement ms | Speed ratio [paired 95% interval] |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in timing["records"]:
        lines.append(
            f"| {r['student']} | {r['batch']} | {r['phase']} | {number(r['baseline_ms_mean'])} | {number(r['replacement_ms_mean'])} | {number(r['speedup_mean_ratio'])} {interval(r['paired_ratio_ci95'])} |"
        )
    lines += [
        "",
        "Ratios above one favor replacement. These are eager float32 measurements on shared H100 hardware, using alternating paired order and identical fixed token budgets. A speed gain does not compensate for a failed fidelity criterion.",
        "",
        "Standalone block timing is a separate measurement:",
        "",
        "| Student | Batch | Sequence length | Original µs/call | Replacement µs/call | Block speed ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in block_timing["records"]:
        pairs = r["paired_microseconds_per_call"]
        base = sum(p["original"] for p in pairs) / len(pairs)
        replacement = sum(p["replacement"] for p in pairs) / len(pairs)
        lines.append(
            f"| {r['student']} | {r['batch']} | {r['sequence_length']} | {base:.2f} | {replacement:.2f} | {r['ratio_of_means']:.3f} |"
        )
    lines += [
        "",
        "A block can be faster on a large token batch while being slower for a single decode token. Neither result determines whole-model latency; the end-to-end measurements above include all remaining layers.",
        "",
        "## Evidence and limits",
        "",
        "All 270 primary scalar candidates, 90 superseded scalar candidates, and 27 vector candidates are accounted for by the study audit, including any failed fits. Validation objectives are recomputed from the saved checkpoints on GPU. Original snapshots remain available. The user-requested Gemma amendment was frozen before Gemma fitting/evaluation, after partial Qwen and superseded SmolLM2 results had been observed. Selection hashes precede the corresponding evaluations; the normalization-folding repair is documented. The previous exploratory release is preserved separately.",
        "",
        "See [protocol](../PROTOCOL.md), [evaluation details](../EVALUATION_DETAILS.md), [reproduction](../README.md), and [prior work](../RELATED_WORK.md). Symbolic component replacement, arithmetic neuron interventions, and geometric explanations have substantial prior work. This study alone establishes neither priority nor an EML-specific advantage over neural controls. Individual operand values may repeat in-range; independent problem pairs and larger-range tests support different generalization claims.",
    ]
    lines += [
        "",
        "## Superseded model arm",
        "",
        "The user requested Gemma E2B after SmolLM2 addition ordinary results had been observed. Gemma uses the same frozen data and criteria, with a separate pinned PyTorch/Transformers backend. Cross-family differences therefore do not isolate backend effects. See [the amendment](../GEMMA_AMENDMENT.md).",
        "",
        "SmolLM2's remaining inference was stopped at the user's request. Its 90 completed fits and completed traces are retained in the evidence bundle. It is incomplete and excluded from the amended nine-cell decision; stopping its evaluation is not a fit failure.",
        "",
    ]
    for model, arm in audit["superseded"].items():
        for op, metrics in arm["ordinary_primary_metrics"].items():
            eml = metrics[PRIMARY]
            neural = metrics[CONTROL]
            lines.append(
                f"{arm['label']} / {op}: ordinary accuracy was {100 * eml['original_accuracy']:.3f}% for the original, {100 * eml['accuracy']:.3f}% for EML, and {100 * neural['accuracy']:.3f}% for SiLU. Causal and stress evaluation is incomplete; this is not a primary pass claim."
            )
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n")
    figures(cells, destination)
    explorer(results, destination)
    print("AUDITED REPORT RENDERED", destination, flush=True)


if __name__ == "__main__":
    main()
