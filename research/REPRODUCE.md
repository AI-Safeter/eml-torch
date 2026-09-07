# Reproducing the component-replacement study

The research branch contains the core library plus this separate `research/` directory. Open `REPORT.md` for results or `explorer.html` for the offline interactive demonstration. No server or model download is needed to use the explorer.

## Environment

Use Python 3.10 and a CUDA-enabled PyTorch installation for the recorded dependency versions. The measured environment used Python 3.10.12, NVIDIA PyTorch 2.5.0a0+e000cf0ad9.nv24.10, CUDA 12.6, Transformers 4.51.3, and H100 80 GB GPUs. Exact package and driver information is in `environment.json`. Other software builds can introduce small numerical differences; they are not bitwise reproductions of the measured container.

From the repository root:

```bash
python -m pip install build==1.6.0
python -m build --wheel
python -m pip install --no-deps --force-reinstall dist/emltorch-0.7.1-py3-none-any.whl
python -m pip install -r research/requirements.txt
cd research
python reproduce.py verify
```

The manifest checks published file sizes and SHA-256 hashes. Large raw JSON records are stored as `.json.gz`; the reproduction command expands them into a new output directory. It refuses to overwrite a nonempty output directory. Pretrained model weights and large raw activation caches are omitted. Their revisions and omitted-file hashes remain recorded. Small feature matrices, learned heads, sparse controls, model directions, and normalization statistics are included.

The recorded NVIDIA image's old `pip` loaded an incompatible system `setuptools` during a direct wheel build. The verified commands above use the isolated `build` frontend and the repository's declared `setuptools>=77` dependency. The resulting wheel contains only the core library.

## Recompute statistics from frozen observations

This needs CUDA, but no pretrained model or network connection:

```bash
python reproduce.py analyze --model 1.7b --gpu 0 --output /tmp/eml-analysis-large
python reproduce.py analyze --model 0.6b --gpu 1 --output /tmp/eml-analysis-small
```

The analysis uses 5,000 paired operand-group bootstrap samples. It keeps styles, intervention strengths, and coordinate edits grouped. It also reports per-style and per-coordinate results. Reproducing the same run order matters for the exact Monte Carlo interval endpoints.

## Repeat model interventions with frozen checkpoints

```bash
python reproduce.py evaluate --model 1.7b --gpu 0 --compact --output /tmp/eml-eval-large
python reproduce.py evaluate --model 0.6b --gpu 1 --compact --output /tmp/eml-eval-small
```

These commands load pinned Qwen snapshots, preserve checkpoint selection, and rerun ordinary generation, unseen interpolation strengths, shifted operands, a new prompt format, and local coordinate edits. The compact flag also evaluates the additional 96-pair cohorts. Each command uses one GPU; independent commands can run on different GPUs. The original four-GPU session also ran independent operations, model replications, and seed fits concurrently. A local pinned Hugging Face snapshot can be supplied with `--model-path /path/to/snapshot`.

The optional seed replication is separate: its frozen protocol uses seeds 59, 71, 83, and 97, fixed active-32 features, and the 1.7B addition component. Run `replicate_seeds.py SEED` with `EMLTORCH_RESEARCH_ROOT` pointing to a new 1.7B training output, then `evaluate_seeds.py` and `analyze_heads.py add`. Copy the published `add/seed-replication/protocol.json` first. Report every seed; do not choose a winner from evaluation results.

## Repeat training and localization

```bash
python reproduce.py train --model 1.7b --gpu 0 --compact --output /tmp/eml-training-large
python reproduce.py train --model 0.6b --gpu 1 --compact --output /tmp/eml-training-small
```

This is the expensive path: it recollects activations, localizes components using training/validation, builds feature encoders, fits all primary candidate grids and controls, freezes validation choices, and evaluates held-out data. The frozen operand datasets are reused, so this reproduces the computational pipeline rather than providing a fresh statistical replication. An independent replication should freeze new operands and model choices before looking at results.

Use `--stop-after features` or `--stop-after fit` to prepare intermediate artifacts without launching the later stages. Individual scripts accept `EMLTORCH_RESEARCH_ROOT` and `CUDA_VISIBLE_DEVICES`, allowing operation-specific fits to run on separate GPUs after collection.

The public training commands use explicit runtime limits. They do not inherit the original four-hour session's now-expired deadline. `train_heads.py --resume` and `minimal_heads.py --resume` preserve completed candidates; an optional `--deadline-epoch` limits new candidates when deliberately scheduling a timed run. Existing published checkpoint and selection files must remain unchanged.

## Airfoil export and replay

`airfoil-replay/airfoil_equation.py` is a standalone standard-library Python predictor:

```python
from airfoil_equation import predict

noise_db = predict(500, 0, 0.3048, 55.5, 0.00283081)  # approximately 125.709 dB
```

Input units are frequency in Hz, angle in degrees, chord in metres, velocity in m/s, and displacement thickness in metres. The Python API checks training bounds. C functions are unchecked and require callers to enforce appropriate input limits:

```bash
gcc -O3 -shared -fPIC airfoil-replay/airfoil_equation.c -lm -o /tmp/airfoil_equation.so
python reproduce.py airfoil --gpu 0 --output /tmp/eml-airfoil-replay
```

The replay command repeats the grid against the included frozen teacher and split, then exports and validates Python/C against CUDA predictions. Its old test split was previously examined, so it is not a fresh confirmation. The included UCI data are CC BY 4.0; see `ATTRIBUTION.md`.

## Validation and performance

For the lower-level scripts, first expand the complete bundle and work in that copy:

```bash
python reproduce.py unpack --output /tmp/eml-unpacked
cd /tmp/eml-unpacked
CUDA_VISIBLE_DEVICES=0 python audit_evidence.py
```

`audit_evidence.py` checks disjoint operands, frozen hashes, validation-only winner selection, restored text, and checkpoint objectives on CUDA. `validate_core_head.py` checks the public module, including first- and second-order derivatives. Both use the included feature matrices. `validate_evaluation.py` additionally needs the pinned model. Published validation JSON files record the original outcomes.

`benchmark_native.py` and `compare_compiled.py` benchmark standalone scalar predictors, including the input projection. They require the validation activation cache generated by the training/collection path and `nvcc` for the experimental H100-specific kernel. They do not measure end-to-end LLM acceleration. Delete or redirect a copied benchmark result file to request new measurements; existing completed compiled measurements are preserved. The core library does not depend on this kernel.

To validate the offline explorer, install Playwright and Chromium, run `build_explorer.py`, then `validate_explorer.py` with CUDA available. The validator checks desktop/mobile interactions and compares independent JavaScript equation outputs with saved GPU measurements. Scientific figures can be regenerated with `make_report.py`; this changes only derived report artifacts.
