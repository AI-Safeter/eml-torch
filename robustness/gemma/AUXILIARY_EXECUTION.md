# Isolated division evaluation

`auxiliary_evaluate.py` lets a spare GPU precompute the selected-head Gemma
division cohorts while the main queue finishes multiplication. It invokes the
existing, validated Gemma execution drivers. It changes output routing and
scheduling; model computation, precision, examples, checkpoints, and criteria
stay fixed.

Inputs and selected checkpoint directories are linked into a separate staging
directory outside the repository and authoritative run directory. After each
scientific JSON file closes, its hash enters the private staging manifest.
Publication copies that complete file and installs the copy with an atomic
link operation. Existing results must have identical bytes and are never
overwritten by the auxiliary worker. Job and completion markers remain private.

Before publishing each cohort, the worker checks the main queue's live child
process. If that queue has reached division, the auxiliary worker yields. The
main evaluator skips cohorts already present and runs the remaining stages,
including all seeds and analysis, normally. A transition during computation can
duplicate work on one cohort; it does not authorize a change to the evidence.
An interrupted, unrecorded staging file is rejected on restart.

GPU validation runs the native-generation and cached-generation drivers into
separate directories, then compares all four validation files byte for byte:
2,880 method/condition records, including ordinary generation, all-token
patching, natural interventions, and coordinate interventions. These are not
2,880 independent examples. The validator also checks atomic publication,
preservation and rejection of existing outputs, and yielding to the main queue.

Example from the repository root, with the pinned Gemma Python environment:

```bash
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" robustness/gemma/auxiliary_evaluate.py \
  --operation divide --output /absolute/private-native --engine native --validate-only
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" robustness/gemma/auxiliary_evaluate.py \
  --operation divide --output /absolute/private-cached --engine cached --validate-only
"$GEMMA_PYTHON" robustness/gemma/validate_auxiliary.py \
  --native /absolute/private-native --cached /absolute/private-cached \
  --output /absolute/auxiliary-validation.json
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" robustness/gemma/auxiliary_evaluate.py \
  --operation divide --output /absolute/private-cached --engine cached --publish
```

The publication command requires the committed validation record and source
freeze. Final auditing checks this execution freeze as well as the model
traces, checkpoint provenance, and recomputed metrics. The staging directory
is an execution aid; the normal run directory contains the published evidence.
This concurrency change is not an EML acceleration or serving benchmark.
