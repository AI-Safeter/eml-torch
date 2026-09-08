# J-lens feasibility: no reliable semantic swap

**Decision: stop before equation fitting.** On Gemma 4 E2B IT, swapping two
selected-token Jacobian directions did not change the top answer to the intended
target in any of the 16 discovery prompts at any tested layer or nonzero strength.
This remains true when number words such as “Eight” count as eight.

The protocol and corpus were frozen before collecting intervention outcomes.
An early two-document computation preflight used “octopus”; this was replaced
with “tick” before the 64-document calibration or any answer measurements, to
avoid conflating arms with legs. Those preflight gradients were discarded.

## Fixed gate at swap strength 1

| Block output (zero-based) | Median split-half direction cosine | Positive target-margin change | Intended target becomes top answer | Mean margin change: semantic / random |
|---|---:|---:|---:|---:|
| 10 | 0.9324 | 13/16 | 0/16 | +1.4503 / −0.1126 |
| 17 | 0.9854 | 10/16 | 0/16 | +0.0918 / −0.0483 |
| 24 | 0.9837 | 10/16 | 0/16 | +0.3155 / +0.0992 |

The native model returned the correct digit on 13/16 prompts. On the other
three it returned the correct number word. The literal digit accuracy passed
the fixed 80% baseline threshold; treating words as numbers does not rescue the
intervention result. All layers failed the required 50% target-answer rate.
Strengths 0.5 and 2 also produced zero intended answer changes.

Each random control uses a direction orthogonal to the two concept vectors and
matches the semantic edit's per-position norm before BF16 rounding. The table
is descriptive: these are eight ordered swaps, including reverse pairs, with
two related templates, not 48 independent factual questions. Discovery results
are not held-out generalization evidence.

Full direction stability is also stronger than contrast stability. Across
discovery pairs, split-half cosine for the difference between normalized concept
directions ranges from 0.909–0.988 at block 10, 0.843–0.903 at block 17, and
0.903–0.948 at block 24. Similar individual directions alone do not establish a
stable semantic control interface.

## Numerical checks and architecture

Native text-only forwarding and a zero edit reproduce wrapper logits bitwise.
Direct projected Jacobians agree with an independent weighted sum of coordinate
Jacobian rows to relative error at most 0.0136 in BF16 and 0.0000021 in an FP32
suffix using the same checkpoint weights. Finite-difference step sweeps validate
the FP32 derivatives to relative error below 0.00005 at all three layers.

The step sweep matters. On the audited calibration paragraph, at block 10 the
FP32 local directional derivative is approximately +5.09, yet a symmetric edit
of 0.1% of the mean residual norm gives a finite slope of approximately −1.73.
Smaller steps converge to the local derivative. This is a measured nonlinear
response in one direction on one paragraph, not an estimated universal radius
or an EML result. Both full BF16 and FP32 sweeps are retained.

The GPU gradient audit also confirms an architectural restriction: earlier token
positions influence a later output through block 10, but their gradient is
exactly zero through the residual outputs of blocks 17 and 24 in the same
forward pass. Gemma's later layers reuse keys and values already constructed by
blocks 13 and 14. Editing those later residual outputs can change their own
positions, while previously computed shared KV states stay fixed. This does not
claim that early context stops influencing the model, or that future generated
tokens are unaffected.

The reusable contribution here is the selected-token estimator and its GPU
audit. No EML formula, model replacement, inference speedup, or consciousness
claim follows from these results. The held-back animals and formats were not
used to search for a passing result.
