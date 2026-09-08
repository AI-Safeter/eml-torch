# Gemma addition: primary scalar fidelity passes

This is a completed primary cohort within an **unfinished nine-cell study**.
The full study report still requires every operation, seed and stress condition.
[The machine-readable audit](primary-addition.json) includes all controls,
source hashes, trace checks and the frozen validation selection.

The cohort contains 1,024 unseen operand pairs in three familiar prompt formats
(3,072 prompts). Pairs are conditioned on different first answer tokens for clean
and corrupted operands. Prompts use the fixed assistant prefill `The answer is `.
The separate unconditioned-operand and new-format tests remain necessary.

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
