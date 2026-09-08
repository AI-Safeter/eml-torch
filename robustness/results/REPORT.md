# Fresh component-replacement study

The validation-selected EML scalar replacement passes all declared primary bounds in **9 of 9** model/operation cells. All cells are reported below. Whole-block utility criteria **do not pass** for the selected EML vector student.

These experiments measure preservation of model behavior. A compact formula in learned coordinates does not by itself identify an arithmetic algorithm. Scalar replacement keeps the original dense MLP active and provides no model compression claim.

## Primary scalar confirmation

NRMSE intervals below are adjusted across nine cells. Accuracy-loss upper bounds account for three correlated prompt formats within each operand group and remain positive when no regressions are observed.

| Model / operation | EML NRMSE [interval] | EML correlation [interval] | SiLU NRMSE | Original → EML accuracy | Upper accuracy loss (pp) | All primary bounds |
|---|---:|---:|---:|---:|---:|---|
| Qwen3-1.7B / add | 0.023 [0.019, 0.027] | 1.000 [1.000, 1.000] | 0.025 | 96.81% → 96.74% | 0.748 | pass |
| Qwen3-1.7B / multiply | 0.015 [0.013, 0.017] | 1.000 [1.000, 1.000] | 0.019 | 66.47% → 66.47% | 0.686 | pass |
| Qwen3-1.7B / divide | 0.037 [0.032, 0.044] | 0.999 [0.999, 0.999] | 0.047 | 52.18% → 52.25% | 0.686 | pass |
| Qwen3-4B / add | 0.013 [0.009, 0.018] | 1.000 [1.000, 1.000] | 0.015 | 99.45% → 99.45% | 0.613 | pass |
| Qwen3-4B / multiply | 0.034 [0.028, 0.042] | 0.999 [0.999, 1.000] | 0.038 | 88.93% → 88.90% | 0.686 | pass |
| Qwen3-4B / divide | 0.030 [0.026, 0.036] | 1.000 [0.999, 1.000] | 0.043 | 88.38% → 88.41% | 0.613 | pass |
| Gemma 4 E2B IT / add | 0.019 [0.014, 0.024] | 1.000 [1.000, 1.000] | 0.019 | 99.58% → 99.58% | 0.613 | pass |
| Gemma 4 E2B IT / multiply | 0.072 [0.054, 0.090] | 0.997 [0.996, 0.998] | 0.076 | 88.90% → 88.90% | 0.613 | pass |
| Gemma 4 E2B IT / divide | 0.015 [0.013, 0.016] | 1.000 [1.000, 1.000] | 0.013 | 71.91% → 71.91% | 0.613 | pass |

The paired comparison below uses the same operand groups and active features with derivative-loss weight .1. Negative differences favor EML. These 95% intervals are per comparison, without adjustment across cells; an isolated favorable result does not establish operator superiority.

Both heads contain 2,177 parameters, but EML uses one nonlinear stage with two affine arguments and the comparator uses two sequential SiLU stages. This comparison therefore includes architectural and depth differences. Equal parameter counts do not guarantee equal runtime. See the [mathematical and implementation limits](../RELATED_WORK.md#eml-versus-silu-what-the-comparison-establishes).

| Model / operation | EML minus SiLU response MSE | Paired 95% interval |
|---|---:|---:|
| Qwen3-1.7B / add | -0.0036706 | [-0.0086139, 0.000235679] |
| Qwen3-1.7B / multiply | -0.00408913 | [-0.00605304, -0.00225781] |
| Qwen3-1.7B / divide | -0.00831392 | [-0.012745, -0.00450508] |
| Qwen3-4B / add | -0.00090149 | [-0.00195969, 6.74438e-05] |
| Qwen3-4B / multiply | -0.000948599 | [-0.00247204, 0.000405752] |
| Qwen3-4B / divide | -0.0170256 | [-0.0263832, -0.00935886] |
| Gemma 4 E2B IT / add | 0.000296824 | [-0.00383578, 0.00452931] |
| Gemma 4 E2B IT / multiply | -0.00327279 | [-0.0142783, 0.00538948] |
| Gemma 4 E2B IT / divide | 0.000410307 | [0.000241347, 0.000588659] |

A constant coefficient can sometimes retain answer accuracy while losing the original intervention response. The following baseline table reports both metrics. Each entry is strict answer accuracy / response NRMSE. The selected native-neuron control may use more coefficients; the 16-neuron control approximates the primary predictor's total storage.

| Model / operation | Constant mean | Selected linear | 16 native neurons | Selected native neurons |
|---|---:|---:|---:|---:|
| Qwen3-1.7B / add | 92.38% / 1.000 | 96.32% / 0.385 | 96.74% / 0.116 | 96.78% / 0.029 |
| Qwen3-1.7B / multiply | 54.56% / 1.000 | 66.54% / 0.227 | 66.28% / 0.105 | 66.54% / 0.017 |
| Qwen3-1.7B / divide | 50.81% / 1.000 | 51.99% / 0.544 | 52.18% / 0.108 | 52.12% / 0.025 |
| Qwen3-4B / add | 97.98% / 1.000 | 99.35% / 0.249 | 99.45% / 0.070 | 99.45% / 0.019 |
| Qwen3-4B / multiply | 88.96% / 1.000 | 89.06% / 0.357 | 89.10% / 0.302 | 88.96% / 0.041 |
| Qwen3-4B / divide | 87.79% / 1.000 | 88.28% / 0.472 | 88.31% / 0.100 | 88.35% / 0.029 |
| Gemma 4 E2B IT / add | 97.43% / 1.000 | 99.61% / 0.285 | 99.61% / 0.134 | 99.54% / 0.027 |
| Gemma 4 E2B IT / multiply | 89.00% / 1.000 | 88.90% / 0.361 | 88.93% / 0.231 | 88.90% / 0.072 |
| Gemma 4 E2B IT / divide | 71.91% / 1.000 | 71.91% / 0.300 | 71.91% / 0.036 | 71.91% / 0.023 |

Every primary-family training seed is included below. These are observed minimum–maximum response NRMSE values across seeds 101, 211, 307, 401, and 503, not confidence intervals. All seeds use the same test operands. The validation-selected seed remains the primary result regardless of its position in this range.

| Model / operation | EML five-seed range | SiLU five-seed range |
|---|---:|---:|
| Qwen3-1.7B / add | 0.0215–0.0246 | 0.0236–0.0257 |
| Qwen3-1.7B / multiply | 0.0151–0.0161 | 0.0190–0.0229 |
| Qwen3-1.7B / divide | 0.0371–0.0423 | 0.0454–0.0490 |
| Qwen3-4B / add | 0.0124–0.0168 | 0.0152–0.0179 |
| Qwen3-4B / multiply | 0.0323–0.0375 | 0.0351–0.0377 |
| Qwen3-4B / divide | 0.0304–0.0409 | 0.0427–0.0548 |
| Gemma 4 E2B IT / add | 0.0165–0.0191 | 0.0189–0.0219 |
| Gemma 4 E2B IT / multiply | 0.0584–0.0778 | 0.0655–0.0790 |
| Gemma 4 E2B IT / divide | 0.0143–0.0156 | 0.0127–0.0136 |

The scalar predictor includes a dense input projection and normalization/direction constants. Head size alone is not its deployment footprint:

| Model / operation | Head coefficients | Projection coefficients | Total predictor/patch coefficients |
|---|---:|---:|---:|
| Qwen3-1.7B / add | 2,177 | 65,536 | 71,875 |
| Qwen3-1.7B / multiply | 2,177 | 65,536 | 71,875 |
| Qwen3-1.7B / divide | 2,177 | 65,536 | 71,875 |
| Qwen3-4B / add | 2,177 | 81,920 | 89,283 |
| Qwen3-4B / multiply | 2,177 | 81,920 | 89,283 |
| Qwen3-4B / divide | 2,177 | 81,920 | 89,283 |
| Gemma 4 E2B IT / add | 2,177 | 49,152 | 54,467 |
| Gemma 4 E2B IT / multiply | 2,177 | 49,152 | 54,467 |
| Gemma 4 E2B IT / divide | 2,177 | 49,152 | 54,467 |

## Retained input sensitivity

Validation gradient energy inside each frozen rank-32 input subspace is a local diagnostic. It does not guarantee accuracy on distant operands or arbitrary edits. The active subspace was fitted using training gradients only.

| Model / operation | PLS retained gradient energy | Active retained gradient energy |
|---|---:|---:|
| Qwen3-1.7B / add | 25.83% | 99.63% |
| Qwen3-1.7B / multiply | 24.70% | 99.64% |
| Qwen3-1.7B / divide | 24.26% | 98.29% |
| Qwen3-4B / add | 41.91% | 99.74% |
| Qwen3-4B / multiply | 14.83% | 97.70% |
| Qwen3-4B / divide | 24.69% | 99.42% |
| Gemma 4 E2B IT / add | 38.10% | 99.50% |
| Gemma 4 E2B IT / multiply | 26.35% | 97.26% |
| Gemma 4 E2B IT / divide | 28.81% | 98.38% |

## Unconditioned prompts and later generated tokens

The primary contrast cohort is conditioned on answer-token differences. The separate ordinary cohort has no such condition. These original-model accuracies limit any claim about retained arithmetic ability.

| Model / operation | Original accuracy | Prefill replacement | Replacement through decoding | Upper all-token loss (pp) |
|---|---:|---:|---:|---:|
| Qwen3-1.7B / add | 96.58% | 96.45% | 96.32% | 1.109 |
| Qwen3-1.7B / multiply | 44.82% | 44.82% | 44.82% | 1.454 |
| Qwen3-1.7B / divide | 55.14% | 55.18% | 55.47% | 1.774 |
| Qwen3-4B / add | 99.48% | 99.48% | 99.48% | 0.613 |
| Qwen3-4B / multiply | 78.26% | 78.26% | 78.09% | 2.228 |
| Qwen3-4B / divide | 88.25% | 88.25% | 88.05% | 1.517 |
| Gemma 4 E2B IT / add | 99.51% | 99.51% | 99.51% | 0.613 |
| Gemma 4 E2B IT / multiply | 80.34% | 80.37% | 80.31% | 1.428 |
| Gemma 4 E2B IT / divide | 73.01% | 73.01% | 73.01% | 0.913 |

The [interactive explorer](explorer.html) includes all feature/loss controls, every fitted seed, held-out formats, shifted operands, carry patterns, both-operand edits, and ambient/nullspace diagnostics. Primary success does not imply unrestricted equivalence under arbitrary input edits.

## Held-out prompt formats

These four formats were excluded from fitting. Complete-answer accuracy uses the same strict numeric parser as the primary cohort, so extra prose and nonnumeric outputs count as errors. Low replacement error can coexist with low teacher accuracy: preserving a component's response does not repair the teacher's arithmetic or formatting failures.

| Model / operation | Original answer accuracy | EML answer accuracy | EML response NRMSE |
|---|---:|---:|---:|
| Qwen3-1.7B / add | 46.53% | 46.48% | 0.032 |
| Qwen3-1.7B / multiply | 23.27% | 23.29% | 0.020 |
| Qwen3-1.7B / divide | 21.02% | 21.00% | 0.060 |
| Qwen3-4B / add | 96.19% | 96.22% | 0.026 |
| Qwen3-4B / multiply | 73.24% | 73.27% | 0.040 |
| Qwen3-4B / divide | 43.92% | 43.92% | 0.050 |
| Gemma 4 E2B IT / add | 96.29% | 96.24% | 0.031 |
| Gemma 4 E2B IT / multiply | 71.14% | 71.14% | 0.083 |
| Gemma 4 E2B IT / divide | 61.21% | 61.21% | 0.017 |

## Larger operands and off-subspace edits

Each response entry is EML / SiLU NRMSE for the validation-selected primary-family heads. Each suite normalizes by its own original margin-response RMS. Ambient and nullspace diagnostics edit the MLP input outside the natural clean–corrupted interpolation path. A predictor that only sees the retained features cannot in general reproduce sensitivity to omitted directions. Relative error near one can still correspond to a small absolute margin change; the original response RMS values provide that scale. These diagnostics do not redefine the primary acceptance criteria; their complete intervals remain in the explorer.
The [fixed-feature bound](../RELATED_WORK.md#a-fixed-feature-projection-imposes-a-separate-limit) explains why depth alone cannot recover an omitted direction: an exactly unchanged feature vector produces zero student response, hence NRMSE one when the original response is nonzero. This limits the representation shared by the primary EML and SiLU heads, rather than ranking their activations.

The smaller stress cohorts cannot certify a one-percentage-point accuracy loss under the frozen conservative bound. Even with zero observed regressions, its minimum upper loss is 1.2213 points for the 512-group shifted/carry cohorts and 2.4277 points for the 256-group both-operand cohorts. The zero-regression case needs at least 627 groups with three formats and the nine-cell adjustment. Matching observed accuracy alone is therefore insufficient; the primary cohorts have 1,024 groups. Failure to certify this bound does not establish that the actual loss exceeds one point. These sample sizes and criteria remain fixed.

| Model / operation | Larger operands | Both operands edited | Ambient directions | Nullspace directions | Original response RMS: ambient / nullspace | Shifted answer accuracy: original → EML | Shifted upper loss (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B / add | 0.087 / 0.103 | 0.034 / 0.043 | 0.117 / 0.110 | 1.000 / 1.000 | 0.0532 / 0.0035 | 94.01% → 93.68% | 1.8599 |
| Qwen3-1.7B / multiply | 0.118 / 0.138 | 0.014 / 0.017 | 0.392 / 0.478 | 1.000 / 1.000 | 0.1071 / 0.0219 | 11.59% → 11.59% | 1.2213 |
| Qwen3-1.7B / divide | 0.306 / 0.375 | 0.041 / 0.040 | 0.393 / 0.476 | 1.000 / 1.000 | 0.0306 / 0.0068 | 26.63% → 26.56% | 2.2996 |
| Qwen3-4B / add | 0.036 / 0.045 | 0.012 / 0.018 | 0.106 / 0.194 | 1.000 / 1.001 | 0.0133 / 0.0024 | 99.74% → 99.61% | 1.4914 |
| Qwen3-4B / multiply | 0.180 / 0.178 | 0.049 / 0.053 | 0.220 / 0.210 | 1.000 / 1.000 | 0.0321 / 0.0038 | 33.92% → 33.98% | 1.2213 |
| Qwen3-4B / divide | 0.073 / 0.123 | 0.039 / 0.042 | 0.245 / 0.292 | 1.000 / 0.999 | 0.0146 / 0.0018 | 61.59% → 61.59% | 1.3666 |
| Gemma 4 E2B IT / add | 0.045 / 0.042 | 0.018 / 0.029 | 0.140 / 0.153 | 1.000 / 1.000 | 0.0229 / 0.0014 | 99.22% → 99.22% | 1.2213 |
| Gemma 4 E2B IT / multiply | 0.340 / 0.351 | 0.086 / 0.105 | 0.315 / 0.317 | 1.000 / 1.000 | 0.0206 / 0.0032 | 38.93% → 39.00% | 1.2213 |
| Gemma 4 E2B IT / divide | 0.238 / 0.207 | 0.015 / 0.012 | 0.230 / 0.182 | 1.000 / 1.000 | 0.0413 / 0.0051 | 40.17% → 40.17% | 1.2213 |

## Whole-block utility

| Student | Stored coefficients | Block reduction | Language CE increase | Upper CE increase | Utility criteria |
|---|---:|---:|---:|---:|---|
| eml | 3,148,800 | 11.99× | 0.117 | 0.126 | fail |
| swiglu | 3,148,800 | 11.99× | 0.101 | 0.109 | fail |
| linear | 3,148,544 | 11.99× | 0.118 | 0.126 | fail |

Complete MLP-output fidelity is measured on real, unpadded held-out token positions. Its NRMSE denominator is RMS original MLP output, including the mean; this differs from the downstream margin-response normalization in the scalar study.

| Token domain | EML output NRMSE | SwiGLU output NRMSE | Linear output NRMSE |
|---|---:|---:|---:|
| language | 0.752 | 0.705 | 0.758 |
| arithmetic | 0.319 | 0.262 | 0.389 |

Whole-block complete-answer accuracy on the unconditioned cohort:

| Operation | Original | EML | SwiGLU | Linear |
|---|---:|---:|---:|---:|
| add | 96.58% | 92.29% | 95.51% | 3.35% |
| multiply | 44.82% | 29.95% | 38.05% | 15.89% |
| divide | 55.14% | 36.69% | 47.07% | 16.63% |

The selected original MLP does not execute in this replacement path. The rest of the model remains frozen. Retention requires a language cross-entropy increase upper bound ≤ .02 nats/token and an accuracy-loss upper bound ≤ 1 percentage point for each operation on unconditioned prompts, in addition to ≥ 4× block storage reduction and finite outputs.

The EML replacement reduces the entire model's parameter count by **2.01%**. The block reduction must not be read as a whole-model compression ratio. These are architecture counts; paired benchmarks keep the original and replacement alternatives resident and do not measure peak-memory savings.

## End-to-end latency

| Student | Batch | Phase | Original ms | Replacement ms | Speed ratio [paired 95% interval] |
|---|---:|---|---:|---:|---:|
| eml | 1 | prefill | 19.035 | 19.152 | 0.994 [0.989, 1.000] |
| eml | 1 | decode | 509.347 | 505.723 | 1.007 [0.980, 1.036] |
| eml | 8 | prefill | 71.624 | 70.907 | 1.010 [0.988, 1.026] |
| eml | 8 | decode | 543.074 | 561.362 | 0.967 [0.944, 0.992] |
| swiglu | 1 | prefill | 19.740 | 18.868 | 1.046 [0.971, 1.146] |
| swiglu | 1 | decode | 530.173 | 539.521 | 0.983 [0.957, 1.008] |
| swiglu | 8 | prefill | 72.426 | 70.242 | 1.031 [0.991, 1.083] |
| swiglu | 8 | decode | 561.433 | 568.405 | 0.988 [0.972, 1.004] |
| linear | 1 | prefill | 20.350 | 20.327 | 1.001 [0.881, 1.138] |
| linear | 1 | decode | 559.428 | 553.261 | 1.011 [0.980, 1.043] |
| linear | 8 | prefill | 71.556 | 70.241 | 1.019 [1.009, 1.027] |
| linear | 8 | decode | 516.791 | 530.614 | 0.974 [0.949, 1.000] |

Ratios above one favor replacement. These are eager float32 measurements on shared H100 hardware, using alternating paired order and identical fixed token budgets. A speed gain does not compensate for a failed fidelity criterion.

Standalone block timing is a separate measurement:

| Student | Batch | Sequence length | Original µs/call | Replacement µs/call | Block speed ratio |
|---|---:|---:|---:|---:|---:|
| eml | 1 | 1 | 70.88 | 124.62 | 0.569 |
| eml | 1 | 128 | 279.07 | 147.94 | 1.886 |
| eml | 8 | 1 | 133.52 | 150.23 | 0.889 |
| eml | 8 | 128 | 1604.67 | 220.76 | 7.269 |
| swiglu | 1 | 1 | 70.66 | 59.89 | 1.180 |
| swiglu | 1 | 128 | 284.22 | 86.69 | 3.279 |
| swiglu | 8 | 1 | 133.51 | 76.83 | 1.738 |
| swiglu | 8 | 128 | 1599.58 | 185.21 | 8.636 |
| linear | 1 | 1 | 70.75 | 30.27 | 2.337 |
| linear | 1 | 128 | 287.39 | 45.92 | 6.259 |
| linear | 8 | 1 | 133.74 | 38.34 | 3.488 |
| linear | 8 | 128 | 1602.23 | 166.04 | 9.650 |

A block can be faster on a large token batch while being slower for a single decode token. Neither result determines whole-model latency; the end-to-end measurements above include all remaining layers.

## Evidence and limits

All 270 primary scalar candidates, 90 superseded scalar candidates, and 27 vector candidates are accounted for by the study audit, including any failed fits. Validation objectives are recomputed from the saved checkpoints on GPU. Original snapshots remain available. The user-requested Gemma amendment was frozen before Gemma fitting/evaluation, after partial Qwen and superseded SmolLM2 results had been observed. Selection hashes precede the corresponding evaluations; the normalization-folding repair is documented. The previous exploratory release is preserved separately.

Every scalar summary, per-format result, confidence interval, paired comparison, and primary decision is recomputed from the raw traces using the frozen CUDA statistics. Saved values must agree exactly. The report checks that the replay records still match the source and evidence hashes. This checks faithful calculation of the declared metrics; it is not an independent validation of their statistical assumptions.

See [protocol](../PROTOCOL.md), [evaluation details](../EVALUATION_DETAILS.md), [reproduction](../README.md), and [prior work](../RELATED_WORK.md). Symbolic component replacement, arithmetic neuron interventions, and geometric explanations have substantial prior work. This study alone establishes neither priority nor an EML-specific advantage over neural controls. Individual operand values may repeat in-range; independent problem pairs and larger-range tests support different generalization claims.

## Superseded model arm

The user requested Gemma E2B after SmolLM2 addition ordinary results had been observed. Gemma uses the same frozen data and criteria, with a separate pinned PyTorch/Transformers backend. Cross-family differences therefore do not isolate backend effects. See [the amendment](../GEMMA_AMENDMENT.md).

SmolLM2's remaining inference was stopped at the user's request. Its 90 completed fits and completed traces are retained in the evidence bundle. It is incomplete and excluded from the amended nine-cell decision; stopping its evaluation is not a fit failure.

SmolLM2-1.7B-Instruct / add: ordinary accuracy was 90.202% for the original, 90.169% for EML, and 90.169% for SiLU. Causal and stress evaluation is incomplete; this is not a primary pass claim.
