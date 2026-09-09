# emltorch

`emltorch` fits symbolic expressions and trains neural heads in PyTorch using
`eml(x, y) = exp(x) - log(y)`.

The LLM experiments have not established a consistent EML advantage over SiLU or
met the deployment targets. A later interpretability screen also failed its
prerequisite for fitting causal equations. The library, experiment code, and
measurements remain available; compression research is paused.

Install with Python 3.10 or later and a PyTorch build suited to your CUDA version:

```bash
pip install -e .
# Optional Z3 and cvc5 support for formula export:
pip install -e '.[smt]'
```

To fit an expression on a GPU:

```python
import torch
import emltorch as eml

torch.manual_seed(7)
x = torch.linspace(-1, 1, 128, device="cuda")
result = eml.fit(x, torch.exp(x), depth=1, device="cuda")
print(result.expression)
prediction = result.predict(x)
```

`fit` accepts tensors, NumPy arrays, or lists. It selects CUDA when available if
you omit `device`; set `device="cpu"` to use a CPU. Predictions use the original
input coordinates, including when the fit normalizes its inputs. The package also
supports constant polishing, repeated seeds, residual fitting, and Pareto selection.
See the signatures and docstrings in [api.py](emltorch/api.py).

For a neural module that trains with ordinary PyTorch optimizers:

```python
head = eml.EMLHead(32, 16, out_features=1, device="cuda")
inputs = torch.randn(64, 32, device="cuda")
loss = head(inputs).square().mean()
loss.backward()
```

Each hidden unit computes `eml(left(x), 1 + right(x)**2)`. A learned readout and
an optional linear shortcut produce the output. This module learns weights;
symbolic expression search runs through `fit`. Numerical guards clip exponential
arguments and protect the logarithm. The SMT exports describe real-valued formulas;
they do not certify floating-point or approximation error.

The latest experiment tested weekday features in the local Gemma model, inspired by
[Goodfire's arithmetic study](https://arxiv.org/abs/2605.01148). We fitted two- and
six-dimensional feature spaces from source-weekday token states, then patched
natural donor coordinates while holding the offset fixed. The model, feature
construction, and patch site differ from Goodfire's Llama experiment.

Calibration used 140 prompts. Development used 84 prompts covering 28 held-out
weekday/offset groups in three formats, including one new format. Each source had
two donor weekdays. All 2,520 scored interventions are included in the
[results](gemma_mechanisms/results/weekdays.json), with paired group bootstrap
intervals and conditional diagnostics. Unconditional counterfactual accuracy was:

| Block | Feature dimensions | Feature edit | Matched random edit | Full source-state patch |
|---|---:|---:|---:|---:|
| 4 | 2 | 11.9% | 11.3% | 13.7% |
| 4 | 6 | 13.1% | 7.1% | 13.7% |
| 10 | 2 | 13.1% | 10.1% | 13.1% |
| 10 | 6 | 11.9% | 11.9% | 13.1% |
| 17 | 2 | 11.3% | 11.3% | 11.3% |
| 17 | 6 | 11.3% | 11.3% | 11.3% |

The original model answered 53/84 prompts correctly under the declared first-token
criterion: 63.1%, with an exploratory 95% interval of 47.6–77.4%. This measures the
full-vocabulary next-token argmax, not accuracy after generating a complete answer.
The frozen gate required at least 80% native accuracy and 70% counterfactual
accuracy. No candidate passed. EML fitting and fresh confirmation stayed closed.

At blocks 4 and 10, the six-dimensional features decoded the source weekday with
100% development accuracy. Their weak interventions, including the full-state
controls, show that the chosen source-token interface was inadequate for this task.
The six Fourier contrasts span all centered means of seven weekday labels; this
removes a constraint on class-mean variation but says nothing about variation within
a label or preservation of the model's computation. Earlier blocks may already
have copied weekday information elsewhere in the prompt. That explanation remains
a hypothesis because those other paths were not localized.

The late-layer failure has a specific architectural explanation. Gemma's final 20
blocks reuse keys and values written at blocks 13 and 14. Once those are computed,
changing only an earlier source token's block output cannot change other tokens'
queries or the stored keys and values. Twelve source-state edits after blocks 14
and 17 left every answer logit bitwise unchanged across three prompts and both SDPA
and eager attention. Six edits after block 13 changed logits. This result concerns
source-token-only edits; it does not establish an arithmetic algorithm or an EML
limitation.

The initial padded SDPA run was invalidated when the same prompt produced a weekday
alone and whitespace in a mixed-length batch. The screen was rerun at batch 1 with
the same scientific criteria, and the runner now rejects padding. Eager attention
preserved the two audited padded answers. The SDPA anomaly's kernel-level cause
remains unresolved. The result file retains the invalid run's metadata, both batch
audits, and two failed diagnostic attempts. Total charged cost was 0.134 H100-hours
against a one-hour cap, including conservative charges for those failed attempts.

Run the screen and its posthoc implementation audit inside the repository:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-gemma/bin/python -m gemma_mechanisms.weekdays \
  --run-dir .artifacts/gemma-weekdays-reproduce
CUDA_VISIBLE_DEVICES=1 .venv-gemma/bin/python -m gemma_mechanisms.weekday_audit \
  --run-dir .artifacts/gemma-weekdays-reproduce
```

These commands require the pinned local model and environment described below.
The [protocol](gemma_mechanisms/weekday-protocol.json) records the gate, budget,
batching correction, and reserved confirmation data. Feature bases contain 3,072
or 9,216 FP32 coefficients. Causal tests also require donor model activations;
they are not deployment replacements. No EML comparison was warranted by this run.

The MLP replacement experiments replace layer 26's complete MLP in `google/gemma-4-E2B-it`,
revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`. The local configuration identifies
its native activation as GELU. The checkpoint contains 5,104,297,504 unique parameters;
the original MLP contains 56,623,104. During replacement inference, that MLP is removed
and teacher activations are not inputs.

The hybrid screen tested a SiLU network with a small EML correction against SiLU,
EML, and SiLU with a SiLU correction. Each candidate used a full-width affine
shortcut, a shared 512-wide nonlinear path, and one nonlinear stage. Two-branch
candidates allocated about 600k coefficients to the correction, with a learned gate
initialized at 0.01. Total counts, including buffers, were 5,999,066–5,999,773.

All four fits used seed 1103, identical training batches, and 12,000 updates.
Development results were:

| Replacement | MLP error ↓ | Contribution error ↓ | Arithmetic | ARC |
|---|---:|---:|---:|---:|
| Original | — | — | 64.6% | 65.6% |
| SiLU | 0.245214 | 0.257763 | 61.5% | 59.4% |
| EML | 0.243637 | 0.255766 | 61.5% | 59.4% |
| SiLU + SiLU | 0.240731 | 0.253263 | 63.5% | 53.1% |
| SiLU + EML | 0.247306 | 0.259337 | 62.5% | 56.3% |

MLP error is mean squared error divided by training target variance. Contribution
error measures the native normalized MLP contribution, scaled by its training
second moment. Both average arithmetic and language domains equally. Quality
scores include every prompt: 48 operand groups in two formats and 32 ARC questions.
These are development results from one seed, with substantial sampling uncertainty.
Reported bootstrap intervals are exploratory and unadjusted. Zero observed paired
disagreements can give a [0, 0] interval; that does not certify equivalence.

Removing the trained EML branch reduced arithmetic accuracy from 62.5% to 51.0%
(a paired drop of 11.5 percentage points; bootstrap 95% interval 3.1–20.8). This drop
came from addition. Removing the branch improved shifted arithmetic from
33.3% to 41.7%, so its contribution did not transfer consistently.
The complete hybrid had 2.7% higher reconstruction error than SiLU + SiLU. Its shifted arithmetic accuracy was
33.3%, versus 37.5% for SiLU + SiLU and 41.7% for SiLU. The hybrid had the lowest
language cross-entropy on 16 development documents, 4.8719 nats, compared with
4.8773 for SiLU + SiLU. That small difference does not establish a language benefit.

The hybrid failed the frozen advancement gate. No additional seeds were trained,
and fresh confirmation stayed closed. Three fits selected their last checkpoint;
training may still be incomplete. About 56% of the hybrid's development error lay
outside its learned decoder subspace. The full-width shortcut still leaves the
nonlinear correction limited to 512 output directions. These results cannot separate
remaining architectural limits, optimization, and the choice to spend part of the
budget on EML. They do not establish a general limitation of the operator.

At batch 1 with 128 prompt tokens and 16 decoding steps, the hybrid averaged
68.3 ms for prefill, 786.4 ms for decoding, and 854.6 ms end to end. SiLU averaged
844.9 ms end to end. The paired hybrid/SiLU latency ratio was 1.013, with a 95%
interval of [0.987, 1.039]. These measurements used a shared H100, native eager
execution, and CPU lookup of Gemma's auxiliary embedding table. They do not
establish a serving speed advantage. The two-SiLU branches could also be merged
algebraically; that optimization was not applied in this screen. Full timing,
throughput, memory, and training-cost records are in the result file.

[The hybrid results](gemma_architecture/results/hybrid.json) include every completed
fit, per-example predictions, branch ablations, paired uncertainty, and timing.
The [frozen protocol](gemma_architecture/hybrid-protocol.json) records the two H100-hour
cap and stopping rules. The screen and inference check used 0.266 H100 device
hours, including shared-device delays and process startup. Earlier activation
collection is outside that total. The training caches, initialization files, and
fitted checkpoints have since been deleted. Repeating this study requires
regenerating those inputs using their recorded source revisions. The published
results retain their input hashes and per-example measurements.

The earlier [architecture screen](gemma_architecture/results/summary.json) found
that removing a mandatory output bottleneck helped both EML and SiLU modestly.
None of its six candidates passed its quality gate. Earlier
[complete-block results](gemma_mechanisms/results/summary.json) and
[quantization-correction evidence](kv_correction/evidence) also failed to establish
a consistent EML benefit. The deployment targets remain 20% fewer total parameters,
10% lower end-to-end latency, and at most one percentage point of accuracy loss.
Activation fitting here has not established a recovered arithmetic algorithm.

The local Gemma environment uses PyTorch 2.9.0+cu128 and Transformers 5.16.1, with
dependencies from the existing shared Python environments. The pinned model remains
in the local Hugging Face cache; it is not included in a fresh clone. The weekday
screen above can regenerate its inputs from that model. Hybrid inference requires
a new checkpoint because the failed replacement weights were deleted.

Keep environments and run outputs inside this repository. `.venv-gemma/` and
`.artifacts/` are ignored by Git. Cleanup removed 3.79 GiB of local run artifacts
and 1,088 tracked tensor files (52.5 MiB), including old activation dumps and fitted
heads. Generated research tensors are now ignored. Published reports, protocols,
and per-example measurements remain; their artifact paths and hashes are historical
references. Removed research tensors are listed in [OMITTED.json](research/OMITTED.json).

Historical runners that hash Markdown protocols need their recorded source
revision. The [archived documentation](https://github.com/AI-Safeter/eml-torch/tree/29886f0cab0d45857075f9456ef1c075de980e2c)
includes the full reports and reproduction commands. This README is the only
Markdown document in the current source tree.

The EML library is [MIT licensed](LICENSE). Andrzej Odrzywołek introduced the EML operator
and its [universality construction](https://arxiv.org/abs/2603.21852).
The Jacobian-lens adaptation credits Anthropic in [jspace/NOTICE](jspace/NOTICE)
and retains its [Apache 2.0 license](jspace/LICENSE).
Included Qwen-derived artifacts retain the
[Apache 2.0 license](research/licenses/Qwen-Apache-2.0.txt).

The included Airfoil Self-Noise data are by Thomas F. Brooks, D. Stuart Pope, and
Michael A. Marcolini, distributed by UCI under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
[DOI 10.24432/C5VW2C](https://doi.org/10.24432/C5VW2C). SQuAD/Wikipedia excerpts
retain the attribution and CC BY-SA 4.0 terms recorded in `jspace/NOTICE` and
their evidence files. Model weights and other datasets retain their own terms.
