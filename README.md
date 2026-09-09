# emltorch

`emltorch` fits symbolic expressions and trains neural heads in PyTorch using
`eml(x, y) = exp(x) - log(y)`.

Research is paused. The LLM experiments did not establish a consistent advantage
over SiLU or meet the deployment targets. The library and experimental replacement
code remain available, along with the measurements behind that decision.

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

The last experiment replaced layer 26's complete MLP in `google/gemma-4-E2B-it`,
revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`. The checkpoint contains
5,104,297,504 unique parameters; the original MLP contains 56,623,104. Each
replacement uses about six million coefficients. During replacement inference,
the original MLP is removed and teacher activations are not inputs.

We compared a 512-state bottleneck, a full-width affine shortcut with a nonlinear
correction, and a network with structured full-width states. Both EML and SiLU
used one nonlinear stage, seed 1103, and 12,000 training updates. Development
reconstruction errors were:

| Architecture | EML error | SiLU error |
|---|---:|---:|
| Bottleneck | 0.255290 | 0.257667 |
| Affine shortcut | 0.243286 | 0.244154 |
| Structured states | 0.250279 | 0.239963 |

These errors are mean squared MLP errors divided by training target variance.
The shortcut reduced error by 4.7% for EML and 5.2% for SiLU. Both benefited from
the architectural change. Answer quality still fell: ARC accuracy was 73.96% for
the original model, 63.54% for shortcut EML, and 64.58% for shortcut SiLU.

Shortcut EML reduced end-to-end latency by 0.18%, with a 95% interval of
[-1.18%, 1.87%], at batch 8, 512 input tokens, and 32 cached decoding steps.
That measurement does not establish a speed improvement. The deployment targets
of 20% fewer total parameters, 10% lower latency, and at most one percentage point
of accuracy loss remain unmet.

None of the six candidates passed the development gate. The study stopped before
additional seeds, deeper training, or fresh confirmation. It used 1.416 H100 device
hours. Training curves were still improving, so the results do not separate
remaining capacity limits from incomplete optimization. They also do not establish
a general mathematical limit on EML.

[The result data](gemma_architecture/results/summary.json) contain all completed
fits, quality measurements, and uncertainty estimates. The
[protocol](gemma_architecture/protocol.json) records the budget and acceptance
criteria. Earlier [complete-block results](gemma_mechanisms/results/summary.json)
and the [quantization-correction evidence](kv_correction/evidence) remain available.
Those experiments also failed to establish a consistent EML benefit. Activation
fitting in this repository has not established a recovered arithmetic algorithm.

For local inference with one of the retained experimental exports, run from the
repository root and choose a GPU with enough free memory:

```bash
CUDA_VISIBLE_DEVICES=1 .venv-gemma/bin/python -m gemma_architecture.infer \
  --checkpoint .artifacts/gemma-architecture-release-5c87a61/exports/shortcut-eml-d1-s1103-n12000.pt \
  --prompt 'Say hello in one short sentence.' --tokens 24
```

This command requires the pinned model in the local Hugging Face cache and the
retained export. Neither is included in a fresh clone. The local Gemma environment
uses PyTorch 2.9.0+cu128 and Transformers 5.16.1, with dependencies from the existing
shared Python environments. These replacement weights failed the quality gate.

Keep environments and run outputs inside this repository. `.venv-gemma/` and
`.artifacts/` are ignored by Git. The latter holds the final run, its six exports,
and prerequisite activation data. Older fitted weights were deleted. Published
records retain their original paths and hashes.

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
