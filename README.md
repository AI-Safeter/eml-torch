# emltorch

`emltorch` fits symbolic expressions and trains neural heads in PyTorch using
`eml(x, y) = exp(x) - log(y)`.

Research is paused after the architecture and hybrid screens. The LLM experiments
have not established a consistent EML advantage over SiLU or met the deployment
targets. The library, replacement code, and measurements remain available.

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

The Gemma experiments replace layer 26's complete MLP in `google/gemma-4-E2B-it`,
revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`. The local configuration identifies
its native activation as GELU. The checkpoint contains 5,104,297,504 unique parameters;
the original MLP contains 56,623,104. During replacement inference, that MLP is removed
and teacher activations are not inputs.

The latest screen tested a SiLU network with a small EML correction against SiLU,
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
collection is outside that total. With the retained activation files and pinned
environment, a new run can be started inside the repository:

```bash
EML_HYBRID_RUNS="$PWD/.artifacts/gemma-hybrid-reproduce" \
  .venv-gemma/bin/python -m gemma_architecture.hybrid run --gpu 0 --eval-gpu 1
```

Choose GPUs with available memory. The runner preserves completed jobs and stops
at its budget or quality gate. The development data were inspected in previous
studies; confirmation files are opened only after the gate passes and all selected
checkpoints are frozen. A fresh clone also needs the local prerequisite activations
and initialization recorded in the result's input hashes.

The earlier [architecture screen](gemma_architecture/results/summary.json) found
that removing a mandatory output bottleneck helped both EML and SiLU modestly.
None of its six candidates passed its quality gate. Earlier
[complete-block results](gemma_mechanisms/results/summary.json) and
[quantization-correction evidence](kv_correction/evidence) also failed to establish
a consistent EML benefit. The deployment targets remain 20% fewer total parameters,
10% lower end-to-end latency, and at most one percentage point of accuracy loss.
Activation fitting here has not established a recovered arithmetic algorithm.

For local inference with the hybrid checkpoint, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-gemma/bin/python -m gemma_architecture.infer \
  --hybrid --checkpoint .artifacts/gemma-hybrid/training/silu_eml-s1103.pt \
  --prompt 'Say hello in one short sentence.' --tokens 24
```

This command requires the pinned model in the local Hugging Face cache and the
trained checkpoint. Neither is included in a fresh clone. The local Gemma environment
uses PyTorch 2.9.0+cu128 and Transformers 5.16.1, with dependencies from the existing
shared Python environments. These replacement weights failed the quality gate.

Keep environments and run outputs inside this repository. `.venv-gemma/` and
`.artifacts/` are ignored by Git. The latter holds the architecture and hybrid runs,
retained exports, and prerequisite activation data. Older fitted weights were
deleted. Published records retain their original paths and hashes.

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
