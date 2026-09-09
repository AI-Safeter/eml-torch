# Full-width Gemma MLP replacement study

This stage compares a 512-state bottleneck, a full affine shortcut with a compact
nonlinear correction, and a full-width structured residual network. Each has EML
and SiLU variants at approximately six million coefficients and matched nonlinear
depth. Replacement inference removes the complete layer-26 MLP.

The checkpoint is **google/gemma-4-E2B-it**, revision
`3e22461f65e89153144f8adb70e3b8c2cc9845a7`. The loaded checkpoint has
5,104,297,504 unique parameters, including retained embeddings and multimodal
components. The replaced MLP has 56,623,104 parameters. Model naming is not used
as a substitute for actual parameter accounting.

Read [the result](results/REPORT.md), [the frozen protocol](PROTOCOL.md), and
[the mathematical constraints](THEORY.md). The eight-device-hour cap was declared
before training. Previously inspected evaluations are development data. Fresh
operand groups, documents, and questions were frozen before fitting.

All six depth-one fits completed 12,000 updates. None passed the prospective
screen. Consequently this release contains development evidence from seed 1103;
seeds 2207 and 3301, longer training, depth two, and fresh confirmation were not
run. The stopping decision is a result, not a successful quality certification.
The available training and evaluation commands also support the preregistered
continuation and confirmation paths; those paths have not been exercised by
this stopped study.

## Reproduce this stage

Run commands from the repository root in the pinned Gemma environment:

```bash
export EML_ARCHITECTURE_RUNS=/home/ubuntu/samuel/emltorch-gemma-architecture-runs
EML_PY=/home/ubuntu/samuel/emltorch-gemma-env/bin/python
"$EML_PY" -m gemma_architecture.reproduce \
  --fit-gpu 0 --model-gpu 3 --audit-gpu 2 --release
```

The driver preserves completed budget-journal entries and checks free GPU memory
before launching another job. Do not run two copies against the same run root.
It never terminates another experiment. Select GPU indices based on current
availability; the observed study initially spread independent fits across GPUs
0, 1, and 2 and ran model evaluation/timing on GPU 3.

For a fresh reproduction, set `EML_ARCHITECTURE_RUNS` to a new empty directory.
The prior corrected study at
`/home/ubuntu/samuel/emltorch-gemma-replacement-runs-bos` and its historical
companion run are required: activation datasets, native normalization weights,
old evaluation records, and dataset shards supply training and exclusions.
Their hashes are bound in `data-freeze.json` and the training freeze. See
[the preceding study's commands](../gemma_mechanisms/COMMANDS.md) to regenerate
those inputs. Fresh preparation rewrites the local stage's identity/data
manifest with its new absolute paths; numerical sources and the protocol stay
unchanged. Exact results also depend on the pinned data/model artifacts.

The environment uses Python 3.10, PyTorch 2.9.0+cu128, Transformers 5.16.1,
NumPy 2.2.6, SciPy 1.15.3, PyArrow 23.0.0, Safetensors 0.8.0, and
Tokenizers 0.23.2. Model and dataset caches must already be accessible. Existing
shared environments are not upgraded by these commands.

Each GPU subprocess is metered, including startup, training, validation, and
shared-device delays. For example:

```bash
"$EML_PY" -m gemma_architecture.budget \
  --phase screen --label fit-shortcut-eml --gpu 0 --seconds 900 \
  train --architecture shortcut --activation eml --depth 1 --seed 1103
```

All activation variants use the same minibatch stream per seed. The checkpoint
stores the best development activation objective; a separate continuation file
stores the final weights, optimizer state, and exact minibatch RNG state.
`--steps 24000` resumes the corresponding 12,000-step run. Depth, effort, and
confirmation must follow `PROTOCOL.md`; do not use fresh confirmation results
for selection. The reproduction driver handles this release's observed stop
path and refuses to silently advance a different outcome.

## Use an exported replacement

Exports are stored under `$EML_ARCHITECTURE_RUNS/exports`; large tensor artifacts
remain outside Git. Their hashes, byte sizes, tensor accounting, and bitwise
reload checks are in the result manifests. The inference command loads only the
base checkpoint and the exported replacement; it does not read activation data.

```bash
"$EML_PY" -m gemma_architecture.budget \
  --phase benchmark_and_audit --label demo-shortcut --gpu 3 --seconds 120 \
  infer --checkpoint "$EML_ARCHITECTURE_RUNS/exports/shortcut-eml-d1-s1103-n12000.pt" \
  --prompt 'Say hello in one short sentence.' --tokens 24
```

The base checkpoint is loaded before swapping out the MLP. The removed module
is absent during generation and its forward is forbidden. These experimental
weights have **not** passed a deployment quality gate.

## Files and accounting

`model.py` implements the three designs and folded deployment modules.
`train.py` implements initialization, certified empirical bounds, fitting, and
continuation. `evaluate.py` runs native arithmetic, ARC, and document scoring.
`diagnose.py` separates raw errors and measures covariance/local sensitivities.
`benchmark.py` measures fully GPU-resident prefill and cached decoding.
`export.py`, `infer.py`, and `audit.py` verify the standalone replacement path.
`statistics.py` and `summarize.py` preserve paired uncertainty and every outcome.

Raw numerical results, source/input hashes, the budget ledger, and per-tensor
accounting accompany the report. The former MLP's surrounding native
normalizations remain counted in the complete model. Structured deployment also
retains 3,072 input-normalization values; the other designs fold these values
into their input projections. Every design folds its output normalization and
last readout bias. None of the factored matrix products is collapsed when that
would increase the matrix multiply count at these dimensions.

No arithmetic-mechanism discovery is part of this stage. Covariance rank and
activation fitting do not establish preserved information or a recovered
algorithm.
