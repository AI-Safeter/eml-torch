# Component robustness study

Status: experiments in progress. Training/validation results are not held-out findings. The earlier exploratory release remains under `../research/`; its manifest is unchanged. The small `emltorch` package is unchanged by this study.

Read [the scalar protocol](PROTOCOL.md), [the whole-block protocol](WHOLE_BLOCK_PROTOCOL.md), [evaluation details](EVALUATION_DETAILS.md), and [related work](RELATED_WORK.md). Four source snapshots record the implementation and the documented normalization-folding repair. Run `python verify_sources.py` to verify them together.

## Environment and data

The recorded environment uses Python 3.10, PyTorch 2.5.0a0 (NVIDIA CUDA 12.6 build), Transformers 4.51.3, NumPy 1.24.4, SciPy 1.14.0, safetensors 0.4.5, huggingface-hub 0.36.2, and PyArrow. CUDA validation uses H100 GPUs. Install the core package from this checkout. Research additionally needs Transformers, SciPy, PyArrow, safetensors and huggingface-hub. Model snapshots are pinned in `models.json`; the language corpus revision is pinned in `WHOLE_BLOCK_PROTOCOL.md` and `language-freeze.json`.

From this directory:

```bash
python make_data.py --output /tmp/eml-fresh-data
python prepare_models.py qwen17b
python prepare_models.py qwen4b
python prepare_models.py smollm
python prepare_language.py
CUDA_VISIBLE_DEVICES=0 python validate_adapter.py qwen17b
CUDA_VISIBLE_DEVICES=0 python robust_statistics.py
CUDA_VISIBLE_DEVICES=0 python validate_extensions.py
```

`make_data.py` checks all split overlaps and bundled historical exclusions. Regeneration must match the committed arithmetic data exactly. `prepare_language.py` fetches the official WikiText-2 source and reconstructs the frozen token blocks; its text retains CC BY-SA 3.0 / GFDL licensing. Test tokenization is permitted; no model test outputs are used during preparation or fitting.

## Scalar study

Use one GPU per model where memory permits. Actual free memory on shared hardware matters; the 4B model is loaded in float32. Activations, checkpoints, and raw results live in the sibling directory `emltorch-robustness-runs/`, keeping them separate from the core library.

```bash
CUDA_VISIBLE_DEVICES=0 python run_collection.py qwen17b --output ../../emltorch-robustness-runs/qwen17b
CUDA_VISIBLE_DEVICES=0 python validate_generation.py qwen17b
CUDA_VISIBLE_DEVICES=0 python run_fits.py qwen17b
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py qwen17b
```

For this checkout layout, pass the absolute sibling run directory to `--output` (or `../../emltorch-robustness-runs/qwen17b` when starting inside `robustness/`). `run_fits.py` and `run_evaluation.py` use the location defined in `runtime.py`. Repeat for `qwen4b` and `smollm` on other GPUs. The evaluator checks that all ten candidates per operation/method have finished, creates missing sparse/linear controls, validates restoration and all-token hooks, then evaluates every selected head and every seed. It does not choose a model on test performance.

The primary result is `MODEL/OPERATION/robust-results.json`. Original dense MLPs still execute in scalar replacement. This path cannot demonstrate full-block compression or a model speedup.

An independent feature-pipeline replay can be run after collection:

```bash
CUDA_VISIBLE_DEVICES=0 python replay_collection.py smollm --output /tmp/eml-smollm-replay
```

This recollects training/validation activations and recomputes the projections and derivatives. It compares all 24 tensor archives on GPU and records file-byte equality separately. Use a different output directory from the original run.

## Whole-block study

Wait for Qwen3-1.7B addition localization before collection. Run these stages sequentially on an available GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python whole_collect.py
CUDA_VISIBLE_DEVICES=0 python whole_fit.py
CUDA_VISIBLE_DEVICES=0 python whole_evaluate.py --validation-only
CUDA_VISIBLE_DEVICES=0 python whole_evaluate.py
CUDA_VISIBLE_DEVICES=0 python whole_fidelity.py
CUDA_VISIBLE_DEVICES=0 python whole_benchmark.py
CUDA_VISIBLE_DEVICES=0 python block_latency.py
python model_storage.py
```

The teacher MLP is actually replaced in downstream evaluation, with a guard against accidentally calling it. Deployment folds normalization into affine weights. Validation-selected EML, SwiGLU and factorized-linear students receive the same tests. `utility-results.json` reports the predeclared retention and storage criteria. `latency.json` reports actual prefill and fixed-token decode timings; `block-latency.json` is a separate standalone measurement. A fast but inaccurate replacement does not meet the utility criteria.

Do not run this reproduction into a directory containing different experiments. Existing matching artifacts are preserved; completed files are not a substitute for checking process exit status, source hashes, and the final study audit.

## Final audit and report

After all scalar and whole-block stages have completed:

```bash
CUDA_VISIBLE_DEVICES=0 python audit.py
python report.py
```

The audit checks exact problem/edit identities, duplicate conditions, required controls, finite values, and restoration traces, and recomputes saved validation objectives on CUDA. The trace checker was verified against real Qwen3-4B validation traces and deliberately corrupted copies (`trace-audit-validation.json`). Generation checks on all three models verify exact restoration and execution during decoding; they are implementation checks, not held-out performance results.

The report refuses to render without the complete audit. It creates `results/REPORT.md`, a figure in PNG/PDF, and an interactive explorer with every model/operation cell, per-format results, controls, and seeds. Reporting code does not alter checkpoint selection or the primary criteria.

After committing the completed report, `python bundle.py create --path /absolute/path/to/new-bundle` packages the tracked source, raw test traces, all candidate checkpoints, and validation tensors. It creates a manifest and ZIP with a SHA-256 delivery record. Large training activations and pretrained model weights are regenerated from the pinned inputs. The bundle includes instructions to verify, unpack, and run the GPU audit independently. Raw traces and large checkpoint collections need not be added to Git.
