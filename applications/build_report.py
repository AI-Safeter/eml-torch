"""Build the application report, standalone interactive demo, and scientific figure."""

import html
import importlib.util
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from emltorch._ast import _parse_inner

from .shared import ROOT, expression_source


def main():
    results = {
        k: json.loads((ROOT / "results" / k / "metrics.json").read_text())
        for k in ["distillation", "surrogate"]
    }
    validation = json.loads((ROOT / "results/validation.json").read_text())
    llm = json.loads((ROOT / "llm/replay/summary.json").read_text())
    pysr = {k: json.loads((ROOT / "results" / k / "pysr.json").read_text()) for k in results}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    labels = ["EML", "Neural", "Cubic", "Spline", "PySR"]
    methods = ["eml", "network", "polynomial3", "spline"]
    for ax, name, title in zip(
        axes[:2],
        results,
        ["Airfoil: distillation loses accuracy", "Pendulum: accurate within range"],
    ):
        values = [results[name]["suites"]["test"][k]["rmse"] for k in methods] + [
            pysr[name]["suites"]["test"]["rmse"]
        ]
        ax.bar(labels, values, color=["#cf693c", "#3b7187", "#849fab", "#a9bbc2", "#538274"])
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_ylabel("Held-out RMSE (dB)" if name == "distillation" else "Held-out energy RMSE")
        ax.spines[["top", "right"]].set_visible(False)
    methods_llm = ["eml_intervention", "eml_deeper", "eml_shallow_search", "network"]
    axes[2].bar(
        ["EML 3", "EML 5", "EML 4", "Neural"],
        [llm["pairs-prose"]["interventions"][k]["response_correlation"] for k in methods_llm],
        color=["#cf693c"] * 3 + ["#3b7187"],
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_title("LLM: causal fidelity remains limited", fontsize=11, weight="bold")
    axes[2].set_ylabel("Intervention-response correlation")
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.savefig(ROOT / "results/comparison.png", dpi=180)
    fig.savefig(ROOT / "results/comparison.pdf")
    plt.close(fig)
    lines = [
        "# Three emltorch applications",
        "",
        "All three workflows run. The pendulum example meets its in-range surrogate target; the airfoil distillation and LLM causal-replacement targets are not met. These are application demonstrations, not evidence that EML outperforms established alternatives.",
        "",
        "[Open the interactive demo](index.html) · [Scientific figure](results/comparison.png)",
        "",
        "## 1. Small-model distillation",
        "",
        "A 5 → 32 → 32 → 1 SiLU network learns the UCI Airfoil Self-Noise measurements. EML and the regression controls fit the teacher's outputs. Entire operating configurations stay together across splits; maximum free-stream velocity is held out for shift testing. Training has 614 rows, validation 215, test 209, and shift 465.",
        "",
        "NASA measurements: Brooks, Pope, and Marcolini (1989), [UCI Airfoil Self-Noise](https://doi.org/10.24432/C5VW2C), CC BY 4.0. Inputs are frequency, attack angle, chord, free-stream velocity, and displacement thickness. Target units are dB.",
        "",
        "| Model | Test RMSE | Shift RMSE |",
        "|---|---:|---:|",
    ]
    for k in methods:
        d = results["distillation"]["suites"]
        lines.append(f"| {k} | {d['test'][k]['rmse']:.4f} | {d['shift'][k]['rmse']:.4f} |")
    lines.append(
        f"| PySR | {pysr['distillation']['suites']['test']['rmse']:.4f} | {pysr['distillation']['suites']['shift']['rmse']:.4f} |"
    )
    lines += [
        "",
        "**Distillation acceptance: failed.** The EML student exceeds the allowed 5% increase in teacher test MSE. Its much larger shifted error also rules out extrapolation. The runnable export demonstrates deployment, but this student should not replace the teacher for this task.",
        "",
        "## 2. Scientific surrogate",
        "",
        "The simulator integrates `theta'' + damping*theta' + sin(theta) = 0` to time 6 with zero initial angular velocity. The target is final energy. Training uses 2,048 parameter draws, validation 512, test 512, and amplitude-shift testing 512. Initial angle is 0.1–2.0 radians and damping 0.02–0.3 in the evaluated range; shifted angles are 2.0–2.7.",
        "",
        "An initial direct-energy fit had test RMSE 0.02646 and very poor extrapolation. That attempt is retained in `results/surrogate-unstructured`. The follow-up fits `log(final_energy / initial_energy)` and reconstructs energy with the known initial energy `(1 - cos(initial_angle))`. All learned controls use the same transformation. This is an explicitly supplied physical prior, not a discovered conservation law. The follow-up uses newly generated parameter draws, with the method fixed before their evaluation.",
        "",
        "| Model | Test energy RMSE |",
        "|---|---:|",
    ]
    for k in methods + ["small_angle_envelope"]:
        lines.append(f"| {k} | {results['surrogate']['suites']['test'][k]['rmse']:.6f} |")
    lines.append(f"| PySR | {pysr['surrogate']['suites']['test']['rmse']:.6f} |")
    sr = results["surrogate"]
    sv = validation["applications"]["surrogate"]
    lines += [
        "",
        f"**In-range surrogate acceptance: passed.** EML test R² is {sr['suites']['test']['eml']['r2']:.6f}, normalized RMSE is {sr['test_nrmse']:.2%}, and maximum test error is {sr['suites']['test']['eml']['max_error']:.5f}. The acceptance rule was normalized RMSE ≤5% plus faster scalar deployment than the numerical reference. The neural, cubic, and PySR controls have lower test error; this is not a comparative accuracy win.",
        "",
        "**Extrapolation: failed.** 29.3% of shifted EML predictions are nonfinite; the remaining shifted predictions are not certified accurate. Exported models reject coordinates outside their training bounds. A bounding box cannot detect every distribution shift or guarantee accuracy inside it.",
        "",
        "## Deployment and validation",
        "",
        "Each regression application exports `equation.py`, which uses only Python's standard library, with named input order, finite-input checks, normalization, and range rejection. It is also included in the browser demo. Raw equation JSON, search candidates, fitted neural checkpoints, regression controls, predictions, and data provenance are retained.",
        "",
        "| Single-input CPU evaluation | Pendulum median |",
        "|---|---:|",
        f"| Exported equation | {sr['deployment']['cpu_single']['median_us']:.2f} µs |",
        f"| Neural model, eager PyTorch | {sr['deployment']['network_cpu_single']['median_us']:.2f} µs |",
        f"| Neural model, NumPy implementation | {sv['numpy_network_cpu_single']['median_us']:.2f} µs |",
        f"| SciPy adaptive RK45 | {sv['scipy_cpu_single']['median_us']:.2f} µs |",
        f"| Reference 600-step PyTorch CPU RK4 | {sr['deployment']['solver_cpu_single']['median_us']:.2f} µs |",
        "",
        "These are warm, single-input, local CPU microbenchmarks. The two solver implementations are not optimized native solvers; timings do not establish GPU or batched speedups. Export source bytes and teacher raw parameter bytes are reported separately, not equated as identical storage formats.",
        "",
        "Training, simulation generation, equation scoring, paired bootstrap analysis, and LLM checks ran on H100 GPUs with float32 learned models and TF32 disabled. Simulation generation and convergence checks used float64. The 600-step GPU RK4 reference passed a 1,200-step convergence check; an independently implemented SciPy solver also agreed with the GPU reference. Scalar Python and NumPy deployment checks were compared with GPU outputs. Test RMSE intervals use 5,000 GPU bootstrap samples: operating groups for airfoil and independent parameter draws for the pendulum.",
        "",
        "EML searched depths 2/3/4, two seeds per residual stage, populations of 1,024, 80 generations, and up to four stages, with 200 L-BFGS polishing iterations. Training fits candidates; validation chooses candidates and stops stages. Neural candidates use three seeds and validation early stopping. Polynomial degrees 1–3 and additive cubic regression splines select ridge strength on validation. PySR used one serial CPU seed, 40 iterations, eight populations of 32, and a 120-second search timeout; equations were selected and scored on CUDA using the same splits and targets. This is not a compute-matched benchmark or a broad multi-dataset comparison.",
        "",
        "## 3. LLM component analysis",
        "",
        "The runnable pipeline performs arithmetic activation patching, selects an MLP contribution, fits upstream-only equations, replaces that contribution, and measures clean answers and intervention responses. The model is Qwen3-1.7B at pinned revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. It operates on block 22, one output direction, and three supervised input coordinates.",
        "",
        "The included replay reran all 308 prompt-format cases, including 1,848 intervention points per method, with exact-restoration, finite-logit, and alpha-zero assertions. It reuses the frozen prior test cohort and is a replication, not a new untouched test set. The additional four-node shallow-search control was selected solely by validation error. Prior held-out carry results are retained; no new eligible carry set was available.",
        "",
        "| Predictor | Fresh-cohort replay response correlation |",
        "|---|---:|",
    ]
    for label, k in zip(
        [
            "Prior three-node EML",
            "Five-node deeper EML",
            "Four-node shallow search winner",
            "Small neural control",
        ],
        methods_llm,
    ):
        lines.append(
            f"| {label} | {llm['pairs-prose']['interventions'][k]['response_correlation']:.3f} |"
        )
    lines += [
        "",
        "**Causal-replacement acceptance: failed.** EML does not meet the preceding study's correlation ≥0.9 and normalized response error ≤0.2 criteria. The workflow is useful for testing and rejecting explanatory hypotheses. These latent coordinates are not human-readable arithmetic concepts; no recovered addition algorithm, model compression, Goodfire integration, or inference speedup is claimed. Multiplication and division were not tested.",
        "",
        "[LLM commands and artifact guide](llm/README.md) · [GPU validation record](results/validation.json)",
        "",
        "## Reproduce",
        "",
        "Run from the repository root on the `applications` branch:",
        "",
        "```bash",
        "python -m pip install -e . -r applications/requirements.txt",
        "CUDA_VISIBLE_DEVICES=0 python -m applications.regression distillation",
        "CUDA_VISIBLE_DEVICES=0 python -m applications.regression surrogate --energy-prior",
        "CUDA_VISIBLE_DEVICES=0 python applications/llm/run.py",
        "CUDA_VISIBLE_DEVICES=0 python applications/llm/validate_pipeline.py",
        "CUDA_VISIBLE_DEVICES=0 python -m applications.validate",
        "# Optional PySR comparison; requires a Julia-capable environment:",
        "python -m pip install -r applications/requirements-pysr.txt",
        "CUDA_VISIBLE_DEVICES=0 python -m applications.pysr_baseline distillation",
        "CUDA_VISIBLE_DEVICES=0 python -m applications.pysr_baseline surrogate",
        "python -m applications.build_report",
        "```",
        "",
        "The report builder consumes the included comparison results when PySR is not rerun. Model weights download from Hugging Face at the pinned revision, or set `EMLTORCH_MODEL_PATH` to a local snapshot. Commands overwrite results; use a separate clone to preserve this run. Random search, timeout budgets, and floating-point differences can change results. The recorded environment used NVIDIA's custom Torch 2.5 build. PySR 2.2.1 declares SymPy ≥1.14 whereas that custom Torch declares 1.13.1; both ran here with 1.13.1, but that combination is outside PySR's declared dependency range. Use a separate environment with a compatible modern Torch/SymPy combination for PySR reproduction.",
        "",
        "The `core-cleanup` branch retains the core-only package. Applications, reports, and validation scripts live only on this branch. See `manifest.json` for artifact hashes.",
    ]
    (ROOT / "README.md").write_text("\n".join(lines) + "\n")
    payload = {"metrics": results, "models": {}, "llm": {}}
    functions = []
    for name in results:
        directory = ROOT / "results" / name
        model = json.loads((directory / "equation.json").read_text())
        spec = importlib.util.spec_from_file_location("equation", directory / "equation.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = torch.load(directory / "predictions.pt", weights_only=True)["test"]
        model.update(
            {
                "low": module.LOW,
                "high": module.HIGH,
                "sample": data["x"][0].tolist(),
                "sample_target": float(data["target"][0]),
                "sample_network": float(data["network"][0]),
            }
        )
        payload["models"][name] = model
        expr = str(model["bias"]) + "".join(
            " + (" + expression_source(_parse_inner(e), "scalar") + ")"
            for e in model["expressions"]
        )
        functions.append(f"function {name}Equation(x){{return {expr};}}")
    for path in (ROOT / "llm/replay").glob("interventions-*.json"):
        data = json.loads(path.read_text())
        payload["llm"][path.stem.removeprefix("interventions-")] = {
            k: data[k] for k in ["original", "eml_intervention", "network"]
        }
    template = (ROOT / "dashboard.html").read_text()
    template = template.replace(
        "__PAYLOAD__", json.dumps(payload, allow_nan=False).replace("</", "<\\/")
    )
    template = template.replace("__FUNCTIONS__", "\n".join(functions))
    template = template.replace(
        "__FORMULAS__", html.escape("\n\n".join(payload["models"]["surrogate"]["expressions"]))
    )
    (ROOT / "index.html").write_text(template)
    print("Report, scientific figure, and interactive applications written")


if __name__ == "__main__":
    main()
