# Gemma complete-MLP replacement and arithmetic interventions

This study installs a compact EML network in place of an entire native Gemma MLP. The original block is removed from the registered model and its forward is poisoned during evaluation and timing. Teacher activations supervise offline training; they are not inputs to replacement inference.

Read [the results report](results/REPORT.md), [protocol](PROTOCOL.md), [BOS correction](BOS_AMENDMENT.md), [evaluation criteria](EVALUATION_PLAN.md), and [causal confirmation plan](CAUSAL_CONFIRMATION.md). [COMMANDS.md](COMMANDS.md) provides reproduction and export commands.

The selected EML block removes 0.992% of total model parameters, but fails the frozen quality gate in all three seeds. Final multiplication accuracy falls from 41.82% to 31.47–31.96%, and ARC falls from 67.99% to 64.62–65.22%. Multi-block expansion stopped. The report includes paired uncertainty, matched-depth SiLU and linear controls, full-model timing, and two failure diagnoses: the affine decoder's rank constraint and missing KV inputs in cross-layer arithmetic equations. No recovered arithmetic algorithm was validated.

The exact checkpoint is **google/gemma-4-E2B-it**, revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`. It has **5,104,297,504 unique parameters**, including embeddings and multimodal components. E2B is an effective-size designation. The layer-26 MLP has 56,623,104 parameters. A single compact replacement removes about 1% of total parameters, so it cannot meet the 20% deployment target on its own.

The study compares EML and SiLU at depths 1/2/4, coefficient ceilings 3M/6M, bottlenecks 256/512, and three seeds, plus linear controls: 42 fits. Every encoder, nonlinear projection, decoder, bias, norm, and statistic counts. Deployment folds affine statistics and collapses linear factors when that is cheaper. The surrounding native norms and all other model components remain counted.

The deployed EML block composes the following learned stages. `E`, `D`, `A`,
`B`, and `R` are affine projections, `LN` is a learned LayerNorm, and `L` is
the number of stages. Every coefficient in these operations is included in
the budget. The implementation uses the repository's stabilized `safe_eml`.

```text
h₀ = E(x)
uₗ = LNₗ(hₗ)
hₗ₊₁ = hₗ + L⁻¹ᐟ² Rₗ(exp(clamp(Aₗ(uₗ), −12, 12)) − log(1 + Bₗ(uₗ)²) − 1)
MLP replacement(x) = D(h_L)
```

SiLU uses the same stage depth, norms, encoder and decoder structure, with
its inner width adjusted to match the coefficient budget. A linear control
has no nonlinear stages. This is a learned compositional network; its weights
are not a recovered symbolic arithmetic algorithm.

The runtime enforces torch 2.9.0+cu128 and transformers 5.16.1; [environment.json](environment.json) records supporting packages. Model weights must already be available at the pinned Hugging Face snapshot. Dataset IDs, revisions, seeds, splits, training budgets and acceptance criteria are fixed in `protocol.json`. `prepare.py` now includes native BOS for each document.

The local diagnostic root is `/home/ubuntu/samuel/emltorch-gemma-replacement-runs`; the corrected replacement root appends `-bos`. Use a fresh directory for a new reproduction. The original 42 fits are retained as diagnostics after the BOS correction; they are not the accepted replacement experiment. `prepare_corrected.py` reproduces that amendment from original 256-token files. Do not apply it to already corrected 257-token inputs.

Quality evaluation uses native BF16 scoring with the PLE table on CPU, identically for all methods and validated against a fully resident model. Timing uses the fully GPU-resident model, native eager SDPA, 10 warm-ups and 50 alternating paired repetitions. Prefill, decoding, throughput, allocated/reserved memory, CUDA timing and model end-to-end latency are reported separately. Loading, tokenization and service queuing are outside this declared workload. Other users share the H100s; CPU offloading is memory placement, not parameter compression.

The causal branch tests carry and digit hypotheses under controlled residual/KV interventions. The equations consume decoded internal quantities, never operand labels or downstream teacher states. Probe accuracy and observational fits do not establish mediation. Fixed-prefix interventions remain distinct from unconditional generated-answer quality; addition evidence does not establish multiplication or division mechanisms.

[Related work](RELATED_WORK.md) connects these tests to causal abstraction,
Goodfire's feature interventions and parameter decomposition, and arithmetic
heuristic research. Those connections do not imply a validated mechanism here.

Source, protocols, compact results and hashes are committed. Raw traces, checkpoints and activation tensors remain in the external run roots. A compact export stores only the folded BF16 student and verifies a bitwise CUDA reload. It still requires the pinned base model, but no teacher activations at inference. Consult held-out quality before using an experimental export in an application.

Parameter reduction refers to the installed model after the original MLP is
removed. The small export is a replacement-module artifact; the supplied loader
still reads the full pinned base checkpoint. It is not a standalone compressed
full-model download.
