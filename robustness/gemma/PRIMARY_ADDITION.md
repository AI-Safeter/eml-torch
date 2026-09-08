# Gemma addition: primary scalar fidelity passes

All planned Gemma addition selected-head, seed, decoding, stress, and geometry
evaluations are complete. Their saved summaries have been recomputed exactly
on GPU. The **nine-cell study and its final audit remain unfinished**.
[The machine-readable audit](primary-addition.json) includes all controls,
source hashes, trace checks and the frozen validation selection.

The cohort contains 1,024 unseen operand pairs in three familiar prompt formats
(3,072 prompts). Pairs are conditioned on different first answer tokens for clean
and corrupted operands. Prompts use the fixed assistant prefill `The answer is `.
Unconditioned-operand decoding, seed, and stress results are available below.

The selected EML and SiLU heads each have 2,177 parameters and consume 32 active
features. These counts exclude the feature projection and normalization. Each
head was selected on validation data, before the test results were available.
The patch replaces one scalar contribution at the final prompt position; the
original dense MLP still executes.

| Replacement | Strict answer accuracy | Response NRMSE | Response correlation |
|---|---:|---:|---:|
| Original | 99.5768% | 0 | 1 |
| EML, active features with derivative loss | 99.5768% | 0.01907 | 0.999806 |
| SiLU, same feature/parameter budget | 99.5443% | 0.01885 | 0.999811 |
| Validation-selected linear | 99.6094% | 0.28477 | 0.955888 |
| 16 native neurons | 99.6094% | 0.13389 | 0.990350 |
| Validation-selected native neurons | 99.5443% | 0.02702 | 0.999612 |

The EML accuracy-loss upper bound is **0.6125 percentage points**, below the
one-point criterion. Its simultaneous NRMSE interval is **[0.01446, 0.02401]**,
and its simultaneous correlation interval is **[0.999696, 0.999889]**. Both pass
their fixed bounds. These intervals account for the planned nine primary cells,
grouping the three formats by operand pair.

The paired EML-minus-SiLU response-MSE difference is **+0.000297**, with 95%
interval **[−0.003836, +0.004529]**. This does not establish EML superiority.
Ordinary answer accuracy also hides substantial differences in intervention
fidelity, as the linear control illustrates.

Response error concerns changes in the downstream answer-token logit margin
under held-out interpolation strengths. It is not a reconstruction metric for
the entire model, identification of an arithmetic algorithm, or a speed claim.

## Replacement throughout decoding

The all-token test reapplies the same scalar replacement at each generation
step. The separate unconditioned cohort contains 1,024 fresh operand pairs and
three familiar formats (3,072 prompts), without conditioning on a change in the
answer's first token. It retains the fixed assistant prefill.

| Cohort and patch scope | Original accuracy | EML accuracy | SiLU accuracy | EML upper accuracy loss (pp) |
|---|---:|---:|---:|---:|
| Primary operands, all generated tokens | 99.5768% | 99.5768% | 99.5443% | 0.6125 |
| Unconditioned operands, prefill only | 99.5117% | 99.5117% | 99.5117% | 0.6125 |
| Unconditioned operands, all generated tokens | 99.5117% | 99.5117% | 99.5117% | 0.6125 |

These diagnostics meet the one-percentage-point accuracy-loss bound using the
frozen statistics. They support retention under repeated scalar replacement
on these addition prompts; they do not establish an advantage over SiLU,
performance without the assistant prefill, or equivalent behavior on other
operations. The source hashes, trace checks and GPU-recomputed metrics are
included in the machine-readable audit. All 12 methods remain in the raw traces.

Accuracy retention does not mean identical generated text. With the EML patch
applied throughout decoding, 65 primary-cohort continuations and 91
unconditioned-cohort continuations changed text, while no prompt changed its
correct/incorrect label under the frozen numeric parser.

## All five seeds and stress conditions

The table shows the predeclared primary architecture and feature/loss setting.
Validation selected EML seed 307 and SiLU seed 101. Those choices remain fixed.

| Seed | EML response NRMSE | SiLU response NRMSE | EML accuracy | SiLU accuracy |
|---|---:|---:|---:|---:|
| 101 | 0.01806 | 0.01885 | 99.5768% | 99.5443% |
| 211 | 0.01682 | 0.01968 | 99.5768% | 99.5768% |
| 307 | 0.01907 | 0.02113 | 99.5768% | 99.5768% |
| 401 | 0.01652 | 0.01997 | 99.5768% | 99.5443% |
| 503 | 0.01776 | 0.02187 | 99.5768% | 99.5768% |

These fits share the same test operands; five training seeds are not five
independent data replications. The seed diagnostics do not replace the
inconclusive validation-selected EML–SiLU comparison.

| Stress cohort | EML response NRMSE | SiLU response NRMSE | Original accuracy | EML accuracy | EML upper loss (pp) |
|---|---:|---:|---:|---:|---:|
| Four unseen formats | 0.03072 | 0.02912 | 96.2891% | 96.2402% | 0.7437 |
| Larger operands | 0.04513 | 0.04200 | 99.2188% | 99.2188% | 1.2213 |
| Held-out carry patterns | 0.01526 | 0.02066 | 99.8698% | 99.8698% | 1.2213 |
| Both operands edited | 0.01829 | 0.02868 | 99.3490% | 99.3490% | 2.4277 |

Matching observed accuracy does not certify the one-percentage-point loss bound
in the smaller stress cohorts: larger operands and carry use 512 operand groups,
and both-operand edits use 256. Their conservative upper bounds exceed one point.
These diagnostics preserve the original criteria and do not establish a uniform
EML advantage across conditions.

The [geometry audit](geometry-addition.json) reports ambient-direction NRMSE
0.1400 for EML and 0.1527 for SiLU. Both have NRMSE near one in the active-feature
nullspace, where the original margin-response RMS is only 0.001368. The
[fixed-feature argument](../RELATED_WORK.md#a-fixed-feature-projection-imposes-a-separate-limit)
explains why deeper heads alone cannot recover those omitted directions.
