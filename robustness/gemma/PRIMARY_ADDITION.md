# Gemma addition: primary scalar fidelity passes

This is a completed primary cohort within an **unfinished nine-cell study**.
The full study report still requires every operation, seed and stress condition.
[The machine-readable audit](primary-addition.json) includes all controls,
source hashes, trace checks and the frozen validation selection.

The cohort contains 1,024 unseen operand pairs in three familiar prompt formats
(3,072 prompts). Pairs are conditioned on different first answer tokens for clean
and corrupted operands. Prompts use the fixed assistant prefill `The answer is `.
Unconditioned-operand decoding results are now available below. Full seed and
stress analysis remains necessary before the study report can be completed.

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
