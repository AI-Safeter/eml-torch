# Reproduction commands

Run from the repository root in the pinned Gemma environment. Check GPU memory
before assigning devices; do not terminate other users' experiments. Offline
fitting uses approximately 1 GiB beyond the existing workload. Full-model
quality needs about 6–8 GiB with CPU PLE; resident/native placement validation
temporarily needs about 16 GiB. Adjust device placement for another machine.

## Use a trained replacement

```bash
GEMMA_PYTHON=/home/ubuntu/samuel/emltorch-gemma-env/bin/python
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.deploy \
  --checkpoint /path/to/selected-training-checkpoint.pt \
  --text 'What is 573 + 846? Return only the integer answer.'

CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.export \
  --checkpoint /path/to/selected-training-checkpoint.pt \
  --output /path/to/replacement.pt \
  --validation-input /path/to/collection/language-selection.pt
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.export \
  --run /path/to/replacement.pt --text 'What is 573 + 846?'
```

The validation input is used only to check export equivalence. Inference uses
the compact replacement and pinned base weights, without teacher activations.

## Fresh corrected study

Use a new run directory. The current preparer already includes BOS; the
historical `prepare_corrected` migration is unnecessary for a fresh run.

```bash
export EML_GEMMA_MECHANISMS_RUNS=/path/to/fresh-study-bos
"$GEMMA_PYTHON" -m gemma_mechanisms.prepare
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.validate --full-model
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.collect
CUDA_VISIBLE_DEVICES=2 "$GEMMA_PYTHON" -m gemma_mechanisms.train --prepare-only
```

Run the following disjoint shards in separate terminals after initialization:

```bash
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.train_grid --output "$EML_GEMMA_MECHANISMS_RUNS" --shard 0
CUDA_VISIBLE_DEVICES=2 "$GEMMA_PYTHON" -m gemma_mechanisms.train_grid --output "$EML_GEMMA_MECHANISMS_RUNS" --shard 1
CUDA_VISIBLE_DEVICES=3 "$GEMMA_PYTHON" -m gemma_mechanisms.train_grid --output "$EML_GEMMA_MECHANISMS_RUNS" --shard 2
```

After all shards finish, `train` preserves their fits and freezes selection:

```bash
CUDA_VISIBLE_DEVICES=2 "$GEMMA_PYTHON" -m gemma_mechanisms.train
CUDA_VISIBLE_DEVICES=2 "$GEMMA_PYTHON" -m gemma_mechanisms.audit_fits
CUDA_VISIBLE_DEVICES=2 "$GEMMA_PYTHON" -m gemma_mechanisms.validate_deployment
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.validate_evaluation --deployed
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.evaluate --split gate
CUDA_VISIBLE_DEVICES=2 "$GEMMA_PYTHON" -m gemma_mechanisms.quality_summary --split gate
```

Final/shift evaluation requires a completed gate. A failed one-block gate stops
expansion but does not discard the candidate's final evaluation. A passing gate
requires a fresh multi-block protocol before final evaluation.

```bash
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.evaluate --split final
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.evaluate --split shift
CUDA_VISIBLE_DEVICES=2 "$GEMMA_PYTHON" -m gemma_mechanisms.quality_summary --split final
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.benchmark
CUDA_VISIBLE_DEVICES=2 "$GEMMA_PYTHON" -m gemma_mechanisms.microbenchmark
```

`evaluate --method NAME` and `benchmark --method NAME` support disjoint workers.
Never launch two writers for the same result. Do not mutate frozen source or
inputs while training/evaluation is running. All failed fits remain recorded.

## Causal branch

Create a companion root without the `-bos` suffix, with read-only-use symlinks
to the corrected data and collection. The naming relation lets confirmation
verify that replacement selection is frozen before shared gate operands open.

```bash
mkdir /path/to/fresh-study
ln -s /path/to/fresh-study-bos/data /path/to/fresh-study/data
ln -s /path/to/fresh-study-bos/collection /path/to/fresh-study/collection
export EML_GEMMA_MECHANISMS_RUNS=/path/to/fresh-study
CUDA_VISIBLE_DEVICES=2 "$GEMMA_PYTHON" -m gemma_mechanisms.probes
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.causal --split selection
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.collect_residual
CUDA_VISIBLE_DEVICES=2 "$GEMMA_PYTHON" -m gemma_mechanisms.probes --output "$EML_GEMMA_MECHANISMS_RUNS/residual"
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.causal --site residual --output "$EML_GEMMA_MECHANISMS_RUNS/residual" --split selection
CUDA_VISIBLE_DEVICES=2 "$GEMMA_PYTHON" -m gemma_mechanisms.equations
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.state_sufficiency --split gate --groups 256
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.state_sufficiency --split gate --groups 256 --new-formats
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.state_sufficiency --split shift --groups 256
CUDA_VISIBLE_DEVICES=0 "$GEMMA_PYTHON" -m gemma_mechanisms.state_sufficiency --split shift --groups 256 --new-formats
CUDA_VISIBLE_DEVICES=2 "$GEMMA_PYTHON" -m gemma_mechanisms.equation_responses --cohort gate-256-known
```

Run equation-response analysis for all four combinations of `gate`/`shift` and
`known`/`new`. These are fixed-prefix original-model interventions, distinct
from unconditional answer generation.

## Local automation and report

After preparing the causal branch, the local scheduler can wait for the
replacement grid and perform confirmation. It uses GPU 0 twice for concurrent
quality workers, GPUs 1/3 for other quality workers, GPU 2 for audits, and
separate GPU 0/3 workers for resident benchmarks. Review this placement before
reusing the command elsewhere. It stops for a new protocol if the gate passes.

```bash
"$GEMMA_PYTHON" -m gemma_mechanisms.run_confirmation \
  --output /path/to/fresh-study-bos --mechanisms-output /path/to/fresh-study
CUDA_VISIBLE_DEVICES=2 EML_GEMMA_MECHANISMS_RUNS=/path/to/fresh-study-bos \
  "$GEMMA_PYTHON" -m gemma_mechanisms.microbenchmark
CUDA_VISIBLE_DEVICES=2 EML_GEMMA_MECHANISMS_RUNS=/path/to/fresh-study-bos \
  "$GEMMA_PYTHON" -m gemma_mechanisms.longer_training
TORCHINDUCTOR_COMPILE_THREADS=2 CUDA_VISIBLE_DEVICES=2 \
  EML_GEMMA_MECHANISMS_RUNS=/path/to/fresh-study-bos \
  "$GEMMA_PYTHON" -m gemma_mechanisms.compile_probe
CUDA_VISIBLE_DEVICES=2 "$GEMMA_PYTHON" -m gemma_mechanisms.audit_release \
  --output /path/to/fresh-study-bos --mechanisms-output /path/to/fresh-study
CUDA_VISIBLE_DEVICES=2 "$GEMMA_PYTHON" -m gemma_mechanisms.report \
  --output /path/to/fresh-study-bos --mechanisms-output /path/to/fresh-study
```

The report command requires completed confirmation. Exact execution inputs,
checkpoint identities and source hashes are retained in per-phase freezes.
The original BOS diagnostic run is reproduced from historical source revisions,
not by rerunning today's corrected preparer into its existing directory.

The compiler diagnostic applies identical Inductor settings to the native MLP
and every first-seed replacement. It records setup time, numerical differences,
and both eager/compiled module timings. These optional measurements do not
replace the frozen eager end-to-end benchmark or establish compiled-model
quality. Unsupported configurations remain recorded as failures.
