"""Build tables and exportable figures directly from frozen measurement files."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OPS = ["add", "multiply", "divide"]


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def estimate(row):
    lo, hi = row["response_nrmse_ci95"]
    return f"{row['response_nrmse']:.4f} [{lo:.4f}, {hi:.4f}]"


def main():
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    blocks = [
        "# Equations that survive arithmetic interventions\n",
        "We replaced one localized scalar contribution inside two frozen Qwen models with small EML networks. All six operation–model combinations meet the declared primary fidelity criteria, including their grouped confidence-interval checks. This is a useful component-distillation result. It does not establish that an EML unit represents addition or that the complete language model can be replaced by an equation.\n",
        "The useful change was to choose upstream coordinates from the target's input gradients. A coordinate system optimized to predict observed values can discard directions needed to reproduce interventions. Gradient-based coordinates retain those directions. Both EML and ordinary neural heads benefit.\n",
        "Open [the interactive explorer](explorer.html) to change operands, prompt formats, and interpolation strength. Its coefficient equation runs locally in JavaScript; the downstream model responses are frozen GPU measurements. Read [the protocol](PROTOCOL.md) and [reproduction instructions](REPRODUCE.md) for exact scope.\n",
        "## Primary held-out results\n",
        "The table consistently reports the active rank-32 family: gradient-loss weight 0.1 for 1.7B and 0 for the prespecified smaller-model replication. It does not pick the best test result across feature variants. Each family selects its head and seed by validation alone. NRMSE is error RMS divided by the original model's margin-response RMS; lower is better. Brackets are 95% intervals from 5,000 paired operand-group bootstrap resamples. Accuracy is strict complete-answer accuracy.\n",
    ]
    primary, controls, stress, compact, captures, seeds = [], [], [], [], [], []
    results = {}
    for label, folder, method in [
        ("1.7B", ROOT, "heads-active-r32-g0.1"),
        ("0.6B", ROOT / "replication-0.6b", "heads-active-r32-g0"),
    ]:
        for op in OPS:
            out = folder / op
            d, selection = read(out / "results.json"), read(out / "selection.json")
            results[label, op] = d
            name = method + "/eml"
            p = d["test-known-formats"]
            n, c = p["ordinary"][name], p["interventions"][name]
            primary.append(
                [
                    label,
                    op,
                    f"{100 * p['ordinary']['original']['exact_answer_accuracy']:.2f}%",
                    f"{100 * n['exact_answer_accuracy']:.2f}%",
                    estimate(c),
                    f"{c['response_correlation']:.5f}",
                ]
            )
            for key in ["mean", "linear", "sparse16", "sparse_selected", *selection["heads"]]:
                if key in p["interventions"]:
                    controls.append(
                        [
                            label,
                            op,
                            key,
                            estimate(p["interventions"][key]),
                            f"{d['coordinate-interventions']['interventions'][key]['response_nrmse']:.4f}",
                        ]
                    )
            for suite in ["test-new-format", "shift", "carry"]:
                if suite in d:
                    stress.append(
                        [
                            label,
                            op,
                            suite,
                            estimate(d[suite]["interventions"][name]),
                            f"{100 * d[suite]['ordinary']['original']['exact_answer_accuracy']:.2f}%",
                            f"{100 * d[suite]['ordinary'][name]['exact_answer_accuracy']:.2f}%",
                        ]
                    )
            audit = read(out / "gradient-audit.json")["feature_gradient_energy_fraction"]
            captures.append(
                [
                    label,
                    op,
                    *[f"{100 * audit[k]['validation']:.2f}%" for k in ["pls32", "active32"]],
                ]
            )
            if "minimal-fresh" in d:
                frozen = read(out / "minimal-selection.json")
                spec = frozen["selection"]["heads"]["minimal/eml"]
                c = d["minimal-fresh"]["interventions"]["minimal/eml"]
                compact.append(
                    [
                        label,
                        op,
                        spec["rank"],
                        spec["width"],
                        spec["parameters"],
                        spec["encoder_coefficients"],
                        estimate(c),
                        "yes" if frozen["small_head_validation_threshold_met"] else "no",
                    ]
                )
            if "seed-replication" in d:
                for seed in [59, 71, 83, 97]:
                    p = d["seed-replication"]
                    seeds.append(
                        [
                            seed,
                            estimate(p["interventions"][f"seed-{seed}/eml"]),
                            estimate(p["interventions"][f"seed-{seed}/neural"]),
                            f"{p['ordinary'][f'seed-{seed}/eml']['accuracy_change_pp']:+.3f}",
                        ]
                    )
    blocks += [
        table(
            [
                "Model",
                "Operation",
                "Original accuracy",
                "EML accuracy",
                "Response NRMSE [95% CI]",
                "Correlation",
            ],
            primary,
        ),
        "\nThese tests use 256 independent operand pairs for addition and multiplication and 192 for division, each in three formats. The four evaluated interior interpolation strengths were absent from fitting. Accuracy-change bootstrap intervals can collapse to [0, 0] when every observed group difference is zero. A declared interval-check pass is descriptive and does not certify the absence of rare regressions or formal one-point noninferiority. A low error means that the replacement follows the original model's behavior; it does not mean the original answer was correct. The 0.6B model is particularly weak on these prompts.\n",
        "![Primary fidelity and gradient capture](figures/primary.png)\n",
        "## Why the feature representation matters\n",
        table(
            [
                "Model",
                "Operation",
                "Validation gradient energy in PLS-32",
                "In active-32",
            ],
            captures,
        ),
        "\nFor any differentiable predictor of linear coordinates, its input gradient lies in their span. Omitted target-gradient energy therefore bounds the best possible local derivative fit for that representation. It does not bound natural arithmetic accuracy. On 1.7B addition, a perturbation nearly invisible to the PLS features (maximum standardized coordinate change 3.58e-7) changes the true scalar by 0.542 RMS. This is a representation limitation that adding head depth cannot repair.\n",
        "## Neural, linear, and original-neuron controls\n",
        table(
            [
                "Model",
                "Operation",
                "Replacement",
                "Primary NRMSE [95% CI]",
                "Coordinate-edit NRMSE",
            ],
            controls,
        ),
        "\nReplacing the scalar with its training mean lowers 1.7B accuracy by 2.60 points on addition, 7.68 on multiplication, and 9.37 on division. This control shows that the selected contribution affects behavior. EML is competitive with the same-feature neural controls, and individual comparisons favor different families. Sparse original SwiGLU neurons can be excellent approximators, especially at the larger 64-neuron budget. Sixteen such neurons have approximately the same input-weight storage as a rank-32 encoder. The linear control is fit on training data and selected by validation; on 0.6B it was added after the initial evaluation and is a disclosed supplementary control.\n",
        "The coordinate diagnostic edits active coordinates 0, 1, 7, and 31 by ±0.25 training standard deviations. These are deliberately local and potentially off the natural activation manifold. They establish fidelity on the tested edits, not unrestricted causal equivalence.\n",
        "## Smaller equations on an additional operand cohort\n",
        table(
            [
                "Model",
                "Operation",
                "Input rank",
                "EML units",
                "Head coefficients",
                "Encoder coefficients",
                "Fresh NRMSE [95% CI]",
                "Validation size criterion met",
            ],
            compact,
        ),
        "\nEach row uses 96 additional operand pairs, disjoint from all original splits, with three formats and four unseen interior interpolation strengths. The grid and pairs were frozen before compact fitting. Selection takes the smallest encoder-plus-head meeting validation value and response MSE ≤0.005, otherwise the best validation head with an explicit failure label. The 0.6B division head misses that validation target. The 1.7B addition follow-up uses the same new pairs as 0.6B; it is a model replication.\n",
        "On the extra cohort, the 1.7B eight-unit head changes accuracy from 97.22% to 97.57%. The 0.6B four-unit head changes 46.88% to 47.22%, but its accuracy-change interval extends to −1.04 percentage points, narrowly missing the conservative −1.00-point criterion. The four-unit result contains 49 head coefficients, plus 4,096 encoder coefficients. It is not a 49-number replacement for the whole model. Encoder and head counts above exclude normalization statistics, the output direction, and the model that supplies upstream activations. Frozen `stored_coefficients` fields likewise mean encoder plus head; the run's normalization metadata remain in each component checkpoint.\n",
        "## New initialization replication\n",
        table(
            [
                "Seed",
                "EML NRMSE [95% CI]",
                "SiLU NRMSE [95% CI]",
                "EML accuracy change (pp)",
            ],
            seeds,
        )
        if seeds
        else "Replication analysis pending.",
        "\nAll four additional seeds are reported, with no selection among seeds. They use the fixed 1.7B addition encoder, rank 32, width 32, and gradient weight 0.1. Validation selects a checkpoint within each seed. Evaluation uses the same 96-pair compact cohort. This checks initialization sensitivity conditional on a fitted encoder; it is not four independent repetitions of localization and data sampling.\n",
        "## Where the replacements fail\n",
        table(
            [
                "Model",
                "Operation",
                "Held-out condition",
                "EML NRMSE [95% CI]",
                "Original accuracy",
                "EML accuracy",
            ],
            stress,
        ),
        "\nThe smaller model's addition replacement fails under the operand-range shift: response NRMSE rises to about 0.94. The larger model's multiplication replacement also misses the 0.20 response-error threshold, at 0.226; the smaller multiplication model is below it by point estimate but not by its interval upper bound. In-range and prompt-format success do not justify unrestricted extrapolation. The smaller model's shifted multiplication accuracy is itself only a few percent. The contrast-conditioned sample, one scalar direction, one prefill position, small number of models, and extensive validation search all limit generalization.\n",
        "![Generalization across prompt and operand shifts](figures/generalization.png)\n",
        "## Diagnosing the smaller-model range failure\n",
        "An exploratory diagnostic, fixed after observing the failure and without refitting, compares the first 64 addition pairs per split in all three formats at strengths 0, 0.5, and 1. In 0.6B, the active basis captures 99.58% of target-gradient energy in range but 84.10% under the operand shift. About 62.3% of shifted mixed inputs leave at least one training coordinate range, versus 1.0% in range. No EML numerical guard activates. The 1.7B addition comparison retains 97.33% of shifted gradient energy. A separate 1.7B multiplication diagnostic retains 91.24%, with only 5.4% of shifted inputs outside the coordinate box; a simple range check is therefore not a fidelity certificate. These observations are consistent with a combination of changed local sensitivities and extrapolation; they do not isolate each contribution. Adding depth to a fixed low-rank head cannot recover omitted local input directions. Raw diagnostic metrics distinguish scalar-coefficient error from the primary downstream-margin metric.\n",
        "## A deployable real-data surrogate\n",
    ]
    airfoil = read(ROOT / "airfoil-replay/results.json")
    portable = read(ROOT / "airfoil-replay/portable-validation.json")
    rows = []
    for split, m in airfoil["metrics"]["eml"].items():
        rows.append(
            [
                split,
                f"{m['teacher_rmse']:.3f}",
                f"{m['rmse']:.3f}",
                f"{m['mse_ratio_to_teacher']:.3f}",
                str([round(x, 3) for x in m["mse_ratio_ci95"]]),
            ]
        )
    blocks += [
        table(
            [
                "Split",
                "Teacher RMSE (dB)",
                "EML RMSE (dB)",
                "MSE ratio",
                "Ratio 95% CI",
            ],
            rows,
        ),
        "\nA 422-coefficient EML head distills a frozen 1,281-parameter airfoil-noise teacher. The test RMSE is 3.140 dB versus 3.193 dB. The test MSE-ratio interval includes 1 and exceeds the declared 1.05 upper target, so there is no statistical superiority or established 5% noninferiority. The validation fidelity target was also missed. This is a replay of an earlier evaluated dataset split, explicitly not a fresh confirmatory test.\n",
        f"Both predictors were compiled to C with the same flags. Median amortized scalar latency was {portable['timings']['eml']['scalar_c_median_us']:.3f} µs for EML and {portable['timings']['teacher']['scalar_c_median_us']:.3f} µs for the teacher, measured in seven groups on a shared CPU host. The standalone C equation agrees with the CUDA reference on all 1,503 rows; the standard-library Python API was checked on all 1,038 in-range rows and rejects inputs outside the training bounds. Bounds are not an accuracy certificate.\n",
        "Use [airfoil_equation.py](airfoil-replay/airfoil_equation.py) or compile [airfoil_equation.c](airfoil-replay/airfoil_equation.c). C entry points are unchecked primitives. The [UCI Airfoil Self-Noise dataset](https://doi.org/10.24432/C5VW2C) is by Brooks, Pope, and Marcolini (1989), CC BY 4.0.\n",
        "## Python versus native CUDA\n",
    ]
    timing = read(ROOT / "compiled-comparison.json")["models"]["eml_square"]["batches"]
    timing_rows = [
        [
            batch,
            *[f"{row[key]['median_us']:.2f}" for key in ["reference", "packed_compiled", "native"]],
        ]
        for batch, row in timing.items()
    ]
    blocks += [
        table(
            [
                "Batch",
                "Reference Python/Torch (µs)",
                "Packed torch.compile (µs)",
                "Native CUDA (µs)",
            ],
            timing_rows,
        ),
        "\nThese timings include the raw-input projection for the same frozen standalone scalar head, using CUDA graphs containing repeated calls. They are not complete-LLM speedups. Compiled Python wins at batch 1 and 1,024; the custom kernel wins at intermediate batches. Keep Python as the core API and use compilation where measured. The experimental native kernel stays in the research bundle. Slower split-kernel measurements and superseded host-replay timings remain labeled in the evidence.\n",
        "## Relation to interpretability research\n",
        "Goodfire's [parameter-decomposition work](https://www.goodfire.com/research/interpreting-lm-parameters) emphasizes faithful mechanistic components. This study contributes a smaller operational test: whether a fitted scalar component continues to predict downstream changes under interventions. We did not implement Goodfire's decomposition, extract named semantic features, or show that our active coordinates are human-interpretable concepts. Goodfire's [manifold-steering discussion](https://www.goodfire.com/research/manifold-steering) also motivates distinguishing natural activation changes from off-manifold coordinate edits.\n",
        "Arithmetic-specific prior work includes [Arithmetic Without Algorithms](https://arxiv.org/abs/2410.21272) and [form-invariance across symbols, text, and code](https://arxiv.org/abs/2607.16693). Our cross-format tests do not establish a new discovery of shared arithmetic neurons. Gradient [active subspaces](https://arxiv.org/abs/1408.0545), [equation learning](https://arxiv.org/abs/1912.04825), and [learned EML networks](https://arxiv.org/abs/2604.13871) are prior methods. The original searches evaluated 962 network candidates across primary fits, compact follow-ups, new seeds, and the airfoil replay; computational reproduction runs are additional. This is extensive exploratory tuning, documented in `search-budget.json`. The contribution here is the controlled implementation, measured repair of feature-dependent intervention failures, compact replacements, portable real-data export, and reproducible negative results.\n",
        "## Core contribution and validation\n",
        "The core commit adds `EMLHead`, an ordinary trainable PyTorch module, in 80 added lines across three files. It supports arbitrary leading batch dimensions, a positive log argument, an optional linear skip, serialization, autograd, and `torch.compile`. CUDA checks covered input and parameter gradients against the frozen research head, first- and second-order numerical gradient checks, four floating-point dtypes, compilation, empty/batched inputs, optimizer integration, and an isolated wheel installation. Research artifacts are published separately so the core stays focused.\n",
        "Computational reproduction rebuilt the six model–operation evaluations in fresh directories: all 77 raw output files match byte for byte. A fresh 0.6B localization/feature/training run reproduces 102 tensor archives and all 36 candidate records (excluding runtime). All three arithmetic problem files regenerate exactly. The airfoil replay reproduces all 48 candidates and all 1,503 teacher/student predictions; two interval endpoints differ by at most 2.23e-16. These are computational reproductions of the same data and software, not independent statistical confirmations. The evaluator verifies zero-delta restoration of logits and generated text, metric analytic controls, exact intended coordinate edits, frozen checkpoint hashes, and disjoint operand groups. The offline explorer independently evaluates the exported equation in JavaScript against GPU measurements. Full-model inference and numerical validation ran on H100s; CPU timing and browser interaction checks necessarily use their respective runtimes.\n",
    ]
    (ROOT / "REPORT.md").write_text("\n\n".join(blocks).rstrip() + "\n")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    colors = ["#187e83", "#b26727"]
    x = np.arange(3)
    for index, (label, method) in enumerate(
        [("1.7B", "heads-active-r32-g0.1"), ("0.6B", "heads-active-r32-g0")]
    ):
        metrics = [
            results[label, op]["test-known-formats"]["interventions"][method + "/eml"] for op in OPS
        ]
        y = np.array([m["response_nrmse"] for m in metrics])
        bounds = np.array([m["response_nrmse_ci95"] for m in metrics]).T
        axes[0].errorbar(
            x + (index - 0.5) * 0.12,
            y,
            yerr=np.stack([y - bounds[0], bounds[1] - y]),
            fmt="o",
            capsize=4,
            label=label,
            color=colors[index],
        )
    axes[0].axhline(0.2, color="#999999", linestyle="--", label="Declared threshold")
    axes[0].set(
        xticks=x,
        xticklabels=OPS,
        ylabel="Margin-response NRMSE (lower is better)",
        title="Held-out intervention fidelity",
        ylim=(0, 0.215),
    )
    axes[0].legend(frameon=False)
    for index, (key, label) in enumerate([("pls32", "PLS-32"), ("active32", "Active-32")]):
        values = [
            read(ROOT / op / "gradient-audit.json")["feature_gradient_energy_fraction"][key][
                "validation"
            ]
            for op in OPS
        ]
        axes[1].bar(
            x + (index - 0.5) * 0.32,
            values,
            width=0.3,
            label=label,
            color=colors[index],
        )
    axes[1].set(
        xticks=x,
        xticklabels=OPS,
        ylim=(0, 1.08),
        ylabel="Fraction of validation gradient energy",
        title="Coordinates retain different local sensitivities (1.7B)",
    )
    axes[1].legend(frameon=False, loc="upper left")
    for extension in ["png", "pdf"]:
        fig.savefig(figures / f"primary.{extension}")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, (label, method) in zip(
        axes, [("1.7B", "heads-active-r32-g0.1"), ("0.6B", "heads-active-r32-g0")]
    ):
        for index, (suite, description, color) in enumerate(
            [
                ("test-known-formats", "Held-out operands, known formats", "#187e83"),
                ("test-new-format", "New instruction format", "#7763a6"),
                ("shift", "Larger operands", "#b26727"),
            ]
        ):
            metrics = [results[label, op][suite]["interventions"][method + "/eml"] for op in OPS]
            y = np.array([m["response_nrmse"] for m in metrics])
            limits = np.array([m["response_nrmse_ci95"] for m in metrics]).T
            ax.errorbar(
                x + (index - 1) * 0.14,
                y,
                yerr=np.stack([y - limits[0], limits[1] - y]),
                fmt="o",
                capsize=3,
                label=description,
                color=color,
            )
        ax.axhline(0.2, color="#999999", linestyle="--")
        ax.set(
            yscale="log",
            ylim=(0.006, 1.5),
            xticks=x,
            xticklabels=OPS,
            ylabel="Margin-response NRMSE (log scale)",
            title=f"Qwen3-{label}",
        )
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    for extension in ["png", "pdf"]:
        fig.savefig(figures / f"generalization.{extension}")
    plt.close(fig)
    print("REPORT AND FIGURES BUILT")


if __name__ == "__main__":
    main()
