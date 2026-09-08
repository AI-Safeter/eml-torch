# Component robustness study

Status: experiments in progress. Training/validation results are not held-out findings. The earlier exploratory release remains under `../research/`; its manifest is unchanged. The small `emltorch` package is unchanged by this study.

The user requested **Gemma 4 E2B IT in place of SmolLM2**. See [the amendment](GEMMA_AMENDMENT.md). Completed SmolLM2 artifacts are preserved as an incomplete, superseded arm. Gemma uses a separate pinned backend; the original Qwen environment and running evaluations continue unchanged.

Read [the scalar protocol](PROTOCOL.md), [the whole-block protocol](WHOLE_BLOCK_PROTOCOL.md), [evaluation details](EVALUATION_DETAILS.md), and [related work](RELATED_WORK.md). Append-only source snapshots record the original implementation, normalization-folding repair, and Gemma adapter/execution amendment. Run `python verify_sources.py` to verify them together.

## Environment and data

The recorded environment uses Python 3.10, PyTorch 2.5.0a0 (NVIDIA CUDA 12.6 build), Transformers 4.51.3, NumPy 1.24.4, SciPy 1.14.0, safetensors 0.4.5, huggingface-hub 0.36.2, and PyArrow. CUDA validation uses H100 GPUs. Install the core package from this checkout. Research additionally needs Transformers, SciPy, PyArrow, safetensors and huggingface-hub. Model snapshots are pinned in `models.json`; the language corpus revision is pinned in `WHOLE_BLOCK_PROTOCOL.md` and `language-freeze.json`.

From this directory:

```bash
python make_data.py --output /tmp/eml-fresh-data
python prepare_models.py qwen17b
python prepare_models.py qwen4b
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

For this checkout layout, pass the absolute sibling run directory to `--output` (or `../../emltorch-robustness-runs/qwen17b` when starting inside `robustness/`). `run_fits.py` and `run_evaluation.py` use the location defined in `runtime.py`. Repeat for `qwen4b` on another GPU. The `smollm` option remains for reproduction of the superseded arm; use the separate Gemma entry points below for its replacement. The evaluator checks that all ten candidates per operation/method have finished, creates missing sparse/linear controls, validates restoration and all-token hooks, then evaluates every selected head and every seed. It does not choose a model on test performance.

Qwen evaluation stages now cache unchanged layers before the scalar intervention when token IDs and attention masks match. See [the execution amendment](PREFIX_CACHE_AMENDMENT.md). Gemma, generation, and whole-block timing retain their native execution paths. Both Qwen models passed bitwise validation checks, and a full Qwen3-1.7B primary intervention replay reproduced the original file hash. The direct-script evaluator also reproduced four original validation cohorts exactly (`prefix-dispatch-validation.json`).

To reproduce the cache checks, use the original Qwen environment:

```bash
CUDA_VISIBLE_DEVICES=0 python validate_prefix.py qwen17b
CUDA_VISIBLE_DEVICES=0 python prefix_replay.py qwen17b add --output /tmp/eml-prefix-replay.json
```

The validation command writes a separate reproduction record, preserving the frozen reference timings. Full replay requires a previously completed uncached reference cohort and a new output file outside the source/run directories. Preliminary repeated-forward timings describe this research workload, not a general LLM throughput improvement.

The primary result is `MODEL/OPERATION/robust-results.json`. Original dense MLPs still execute in scalar replacement. This path cannot demonstrate full-block compression or a model speedup.

Gemma's adapter and collection entry points, run inside a separate environment with `gemma/requirements.txt` and the core package installed:

```bash
CUDA_VISIBLE_DEVICES=0 python gemma/validate.py
python gemma/freeze.py --verify
CUDA_VISIBLE_DEVICES=0 python gemma/collect.py
CUDA_VISIBLE_DEVICES=0 python run_fits.py gemma
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py gemma
```

The original-neuron control uses Gemma's native GELU. The learned SiLU comparator remains unchanged. Gemma's exact revision, float32 precision, and backend versions are recorded in `gemma/model.json`; adapter validation records the loaded parameter count, including embeddings and modality modules.

An independent feature-pipeline replay can be run after collection:

```bash
CUDA_VISIBLE_DEVICES=0 python replay_collection.py qwen17b --output /tmp/eml-qwen-replay
# In the separate Gemma environment:
CUDA_VISIBLE_DEVICES=0 python gemma/replay.py --output /tmp/eml-gemma-replay
```

This recollects training/validation activations and recomputes the projections and derivatives. It compares all 24 tensor archives on GPU and records file-byte equality separately. Use a different output directory from the original run. Gemma replay requires a new empty location and copies the frozen source into an isolated checkout layout before recollecting all features.

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
GEMMA_PYTHON=/path/to/gemma-env/bin/python CUDA_VISIBLE_DEVICES=0 python audit.py
python report.py
```

Run the main audit in the original Qwen environment; `GEMMA_PYTHON` selects the separate pinned environment for Gemma checkpoint validation. Checkpoint audit caches record the PyTorch version as well as source and input hashes. The audit checks exact problem/edit identities, duplicate conditions, required controls, finite values, and restoration traces, and recomputes saved validation objectives on CUDA. The trace checker was verified against real Qwen3-4B validation traces and deliberately corrupted copies (`trace-audit-validation.json`). The original generation checks and Gemma adapter validation verify exact restoration and execution during decoding; they are implementation checks, not held-out performance results.

The scalar audit also checks component and control checkpoint hashes, exact sparse/linear control grids, validation-selected control choices, and completion before selection (`selection-audit-validation.json`). Ordinary traces must agree with the frozen strict numeric parser; deliberately altered correctness/agreement labels were rejected (`parser-audit-validation.json`). Gemma independent collection reproduced all 24 tensor archives exactly on GPU and on disk (`replay-gemma.json`).

The whole-block audit also verifies the complete candidate grid, validation-only selection, localized layer, language-file hashes, and arithmetic cohorts. It recomputes every saved utility metric from the raw records on CUDA, requiring exact agreement (`whole-evidence-audit.json`).

The amended roster in `study.json` defines the exact nine primary cells. The audit separately accounts for all 90 superseded SmolLM2 fits and recomputes its completed primary ordinary metrics without requiring canceled inference stages. The report refuses to render without the complete audit for the amended roster. It creates `results/REPORT.md`, a figure in PNG/PDF, and an interactive explorer with every model/operation cell, per-format results, controls, and seeds. Reporting code does not alter checkpoint selection or the primary criteria.

After committing the completed report, `python bundle.py create --path /absolute/path/to/new-bundle` packages the tracked source, raw test traces, all candidate checkpoints, and validation tensors. It creates a manifest and ZIP with a SHA-256 delivery record. Large training activations and pretrained model weights are regenerated from the pinned inputs. The bundle includes instructions to verify, unpack, and run the GPU audit independently. Raw traces and large checkpoint collections need not be added to Git.
