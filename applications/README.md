# Three emltorch applications

All three workflows run. The pendulum example meets its in-range surrogate target; the airfoil distillation and LLM causal-replacement targets are not met. These are application demonstrations, not evidence that EML outperforms established alternatives.

[Open the interactive demo](index.html) · [Scientific figure](results/comparison.png)

## 1. Small-model distillation

A 5 → 32 → 32 → 1 SiLU network learns the UCI Airfoil Self-Noise measurements. EML and the regression controls fit the teacher's outputs. Entire operating configurations stay together across splits; maximum free-stream velocity is held out for shift testing. Training has 614 rows, validation 215, test 209, and shift 465.

NASA measurements: Brooks, Pope, and Marcolini (1989), [UCI Airfoil Self-Noise](https://doi.org/10.24432/C5VW2C), CC BY 4.0. Inputs are frequency, attack angle, chord, free-stream velocity, and displacement thickness. Target units are dB.

| Model | Test RMSE | Shift RMSE |
|---|---:|---:|
| eml | 4.9883 | 213223.1092 |
| network | 3.1932 | 5.0718 |
| polynomial3 | 4.1359 | 33.2641 |
| spline | 4.8190 | 9.4955 |
| PySR | 4.0485 | 5.4576 |

**Distillation acceptance: failed.** The EML student exceeds the allowed 5% increase in teacher test MSE. Its much larger shifted error also rules out extrapolation. The runnable export demonstrates deployment, but this student should not replace the teacher for this task.

## 2. Scientific surrogate

The simulator integrates `theta'' + damping*theta' + sin(theta) = 0` to time 6 with zero initial angular velocity. The target is final energy. Training uses 2,048 parameter draws, validation 512, test 512, and amplitude-shift testing 512. Initial angle is 0.1–2.0 radians and damping 0.02–0.3 in the evaluated range; shifted angles are 2.0–2.7.

An initial direct-energy fit had test RMSE 0.02646 and very poor extrapolation. That attempt is retained in `results/surrogate-unstructured`. The follow-up fits `log(final_energy / initial_energy)` and reconstructs energy with the known initial energy `(1 - cos(initial_angle))`. All learned controls use the same transformation. This is an explicitly supplied physical prior, not a discovered conservation law. The follow-up uses newly generated parameter draws, with the method fixed before their evaluation.

| Model | Test energy RMSE |
|---|---:|
| eml | 0.002662 |
| network | 0.000851 |
| polynomial3 | 0.001452 |
| spline | 0.006908 |
| small_angle_envelope | 0.008592 |
| PySR | 0.002121 |

**In-range surrogate acceptance: passed.** EML test R² is 0.999874, normalized RMSE is 1.12%, and maximum test error is 0.02787. The acceptance rule was normalized RMSE ≤5% plus faster scalar deployment than the numerical reference. The neural, cubic, and PySR controls have lower test error; this is not a comparative accuracy win.

**Extrapolation: failed.** 29.3% of shifted EML predictions are nonfinite; the remaining shifted predictions are not certified accurate. Exported models reject coordinates outside their training bounds. A bounding box cannot detect every distribution shift or guarantee accuracy inside it.

## Deployment and validation

Each regression application exports `equation.py`, which uses only Python's standard library, with named input order, finite-input checks, normalization, and range rejection. It is also included in the browser demo. Raw equation JSON, search candidates, fitted neural checkpoints, regression controls, predictions, and data provenance are retained.

| Single-input CPU evaluation | Pendulum median |
|---|---:|
| Exported equation | 9.82 µs |
| Neural model, eager PyTorch | 45.54 µs |
| Neural model, NumPy implementation | 10.99 µs |
| SciPy adaptive RK45 | 2723.46 µs |
| Reference 600-step PyTorch CPU RK4 | 46489.47 µs |

These are warm, single-input, local CPU microbenchmarks. The two solver implementations are not optimized native solvers; timings do not establish GPU or batched speedups. Export source bytes and teacher raw parameter bytes are reported separately, not equated as identical storage formats.

Training, simulation generation, equation scoring, paired bootstrap analysis, and LLM checks ran on H100 GPUs with float32 learned models and TF32 disabled. Simulation generation and convergence checks used float64. The 600-step GPU RK4 reference passed a 1,200-step convergence check; an independently implemented SciPy solver also agreed with the GPU reference. Scalar Python and NumPy deployment checks were compared with GPU outputs. Test RMSE intervals use 5,000 GPU bootstrap samples: operating groups for airfoil and independent parameter draws for the pendulum.

EML searched depths 2/3/4, two seeds per residual stage, populations of 1,024, 80 generations, and up to four stages, with 200 L-BFGS polishing iterations. Training fits candidates; validation chooses candidates and stops stages. Neural candidates use three seeds and validation early stopping. Polynomial degrees 1–3 and additive cubic regression splines select ridge strength on validation. PySR used one serial CPU seed, 40 iterations, eight populations of 32, and a 120-second search timeout; equations were selected and scored on CUDA using the same splits and targets. This is not a compute-matched benchmark or a broad multi-dataset comparison.

## 3. LLM component analysis

The runnable pipeline performs arithmetic activation patching, selects an MLP contribution, fits upstream-only equations, replaces that contribution, and measures clean answers and intervention responses. The model is Qwen3-1.7B at pinned revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. It operates on block 22, one output direction, and three supervised input coordinates.

The included replay reran all 308 prompt-format cases, including 1,848 intervention points per method, with exact-restoration, finite-logit, and alpha-zero assertions. It reuses the frozen prior test cohort and is a replication, not a new untouched test set. The additional four-node shallow-search control was selected solely by validation error. Prior held-out carry results are retained; no new eligible carry set was available.

| Predictor | Fresh-cohort replay response correlation |
|---|---:|
| Prior three-node EML | 0.583 |
| Five-node deeper EML | 0.588 |
| Four-node shallow search winner | 0.582 |
| Small neural control | 0.917 |

**Causal-replacement acceptance: failed.** EML does not meet the preceding study's correlation ≥0.9 and normalized response error ≤0.2 criteria. The workflow is useful for testing and rejecting explanatory hypotheses. These latent coordinates are not human-readable arithmetic concepts; no recovered addition algorithm, model compression, Goodfire integration, or inference speedup is claimed. Multiplication and division were not tested.

[LLM commands and artifact guide](llm/README.md) · [GPU validation record](results/validation.json)

## Reproduce

Run from the repository root on the `applications` branch:

```bash
python -m pip install -e . -r applications/requirements.txt
CUDA_VISIBLE_DEVICES=0 python -m applications.regression distillation
CUDA_VISIBLE_DEVICES=0 python -m applications.regression surrogate --energy-prior
CUDA_VISIBLE_DEVICES=0 python applications/llm/run.py
CUDA_VISIBLE_DEVICES=0 python applications/llm/validate_pipeline.py
CUDA_VISIBLE_DEVICES=0 python -m applications.validate
# Optional PySR comparison; requires a Julia-capable environment:
python -m pip install -r applications/requirements-pysr.txt
CUDA_VISIBLE_DEVICES=0 python -m applications.pysr_baseline distillation
CUDA_VISIBLE_DEVICES=0 python -m applications.pysr_baseline surrogate
python -m applications.build_report
```

The report builder consumes the included comparison results when PySR is not rerun. Model weights download from Hugging Face at the pinned revision, or set `EMLTORCH_MODEL_PATH` to a local snapshot. Commands overwrite results; use a separate clone to preserve this run. Random search, timeout budgets, and floating-point differences can change results. The recorded environment used NVIDIA's custom Torch 2.5 build. PySR 2.2.1 declares SymPy ≥1.14 whereas that custom Torch declares 1.13.1; both ran here with 1.13.1, but that combination is outside PySR's declared dependency range. Use a separate environment with a compatible modern Torch/SymPy combination for PySR reproduction.

The `core-cleanup` branch retains the core-only package. Applications, reports, and validation scripts live only on this branch. See `manifest.json` for artifact hashes.
