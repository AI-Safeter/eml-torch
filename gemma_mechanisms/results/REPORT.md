# Gemma complete-MLP replacement and causal arithmetic tests

This study physically replaces Gemma's entire layer-26 MLP. The removed block cannot execute in replacement evaluation, benchmarks, or the exported inference path. Whether the replacement preserves quality is a separate, held-out question.

Exact checkpoint: `google/gemma-4-E2B-it`, revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`. Native model: **5,104,297,504 unique parameters**. Target block: **56,623,104**. E2B refers to effective size, not the total stored parameter count.

The frozen one-block expansion gate **failed**. The ≥20% total-parameter, ≥10% latency, and ≤1pp accuracy-loss deployment targets are **not established**. No multi-block expansion followed a failed gate.

The selected EML replacement removes 0.992% of total model parameters. Across three seeds, final multiplication accuracy is 31.47–31.96% versus 41.82% originally; ARC is 64.62–65.22% versus 67.99%. The strongest supported result is a working complete-block replacement path with specific, experimentally supported failure diagnoses: an affine-decoder capacity constraint and missing KV inputs in the cross-layer equations. No recovered arithmetic algorithm was validated.

## Held-out quality

Every row below is a training seed, not a selected favorable run. Accuracy uses the full original-format final cohort. The linked JSON includes paired 95% loss intervals, conservative multiplicity-adjusted regression bounds, new formats, longer operands, and carry diagnostics.

| Replacement | Addition % | Multiplication % | Division % | ARC % | Language Δ nats | Gate |
|---|---:|---:|---:|---:|---:|---|
| Original | 98.51 | 41.82 | 0.95 | 67.99 | 0 | Reference |
| eml-b6000000-d1-s1103 | 98.49 | 31.64 | 0.90 | 64.71 | -0.1015 | Fail |
| eml-b6000000-d1-s2207 | 98.61 | 31.47 | 0.93 | 65.22 | -0.1021 | Fail |
| eml-b6000000-d1-s3301 | 98.68 | 31.96 | 0.88 | 64.62 | -0.1034 | Fail |
| linear-b3000000-d0-s1103 | 61.96 | 4.88 | 0.12 | 62.46 | -0.0817 | Fail |
| linear-b3000000-d0-s2207 | 52.22 | 3.96 | 0.15 | 61.94 | -0.0766 | Fail |
| linear-b3000000-d0-s3301 | 62.70 | 4.00 | 0.17 | 62.46 | -0.0796 | Fail |
| linear-b6000000-d0-s1103 | 62.16 | 4.98 | 0.12 | 62.28 | -0.0797 | Fail |
| linear-b6000000-d0-s2207 | 60.94 | 4.47 | 0.15 | 62.72 | -0.0748 | Fail |
| linear-b6000000-d0-s3301 | 62.43 | 3.88 | 0.17 | 62.63 | -0.0783 | Fail |
| silu-b6000000-d1-s1103 | 98.61 | 32.20 | 0.93 | 64.27 | -0.1064 | Fail |
| silu-b6000000-d1-s2207 | 98.56 | 30.93 | 0.95 | 65.22 | -0.1034 | Fail |
| silu-b6000000-d1-s3301 | 98.58 | 31.47 | 0.98 | 64.97 | -0.0996 | Fail |
| silu-b6000000-d2-s1103 | 98.66 | 30.91 | 0.95 | 64.88 | -0.1055 | Fail |
| silu-b6000000-d2-s2207 | 98.63 | 30.35 | 0.90 | 64.88 | -0.1032 | Fail |
| silu-b6000000-d2-s3301 | 98.58 | 31.42 | 0.83 | 65.57 | -0.1053 | Fail |

Division uses the preregistered 24-token integer-only output contract. The original model often emits prose and has very low accuracy under this contract, providing weak evidence about preservation of division competence. No parser or generation-budget change was made after observing that problem.

For the selected EML checkpoints, paired net accuracy losses versus the original follow. Positive loss is worse. These descriptive 95% intervals resample operand groups or ARC questions; the expansion decision uses the separately reported, more conservative bounds.

| Seed | Addition loss pp [95% CI] | Multiplication loss pp [95% CI] | Division loss pp [95% CI] | ARC loss pp [95% CI] |
|---|---|---|---|---|
| 1103 | +0.02 [-0.22, +0.27] | +10.18 [+9.18, +11.21] | +0.05 [-0.07, +0.17] | +3.29 [+1.38, +5.19] |
| 2207 | -0.10 [-0.32, +0.12] | +10.35 [+9.35, +11.38] | +0.02 [-0.10, +0.15] | +2.77 [+0.78, +4.76] |
| 3301 | -0.17 [-0.39, +0.05] | +9.86 [+8.84, +10.86] | +0.07 [-0.07, +0.24] | +3.37 [+1.47, +5.28] |

The multiplication losses are mainly wrong integer outputs on cases the original answered correctly. This decomposition retains the unconditional score: regressions minus gains equal the reported net loss. It identifies the observed failure type, not the internal algorithm responsible.

| EML seed | Correct original → unparseable | Correct original → wrong integer | Incorrect original → correct |
|---|---:|---:|---:|
| 1103 | 29 | 441 | 53 |
| 2207 | 22 | 457 | 55 |
| 3301 | 23 | 438 | 57 |

The initial grid omitted BOS from raw documents. A selection-only native check found 9.764632 versus 4.795804 nats/token without/with BOS. Those 42 fits were retained as diagnostics. The reported grid recollects language activations with BOS and retrains all 42 candidates; no replacement gate/final outputs were opened before the correction. Arithmetic chat tokenization already included BOS. See [BOS amendment](../BOS_AMENDMENT.md).

Matched-depth EML versus SiLU on the final original-format tasks follows. Positive differences favor EML. Intervals resample operand groups (or ARC questions); they are descriptive unadjusted 95% intervals, not a post hoc selection rule.

| Seed | Addition difference pp [95% CI] | Multiplication difference pp [95% CI] | Division difference pp [95% CI] | ARC difference pp [95% CI] |
|---|---|---|---|---|
| 1103 | -0.12 [-0.32, +0.05] | -0.56 [-1.44, +0.27] | -0.02 [-0.15, +0.10] | +0.43 [-0.95, +1.82] |
| 2207 | +0.05 [-0.07, +0.20] | +0.54 [-0.24, +1.32] | -0.02 [-0.17, +0.12] | -0.00 [-1.21, +1.21] |
| 3301 | +0.10 [-0.02, +0.24] | +0.49 [-0.29, +1.29] | -0.10 [-0.20, -0.02] | -0.35 [-1.56, +0.87] |

Generalization below keeps every generated answer, including malformed outputs and native errors. Each student entry is the mean and range over the three selected EML seeds. These summaries do not replace the paired per-seed intervals in the JSON.

| Operand split / prompts | Task | Original % | EML %, mean [seed range] | Groups per seed |
|---|---|---:|---|---:|
| final / known | add | 98.51 | 98.59 [98.49, 98.68] | 2048 |
| final / known | multiply | 41.82 | 31.69 [31.47, 31.96] | 2048 |
| final / known | divide | 0.95 | 0.90 [0.88, 0.93] | 2048 |
| final / new | add | 97.92 | 97.55 [97.36, 97.68] | 2048 |
| final / new | multiply | 51.59 | 39.52 [39.33, 39.65] | 2048 |
| final / new | divide | 42.50 | 38.78 [38.43, 39.14] | 2048 |
| shift / known | add | 1.27 | 0.80 [0.68, 0.88] | 1024 |
| shift / known | multiply | 1.46 | 0.98 [0.98, 0.98] | 1024 |
| shift / known | divide | 0.29 | 0.20 [0.15, 0.24] | 1024 |
| shift / new | add | 53.08 | 49.56 [49.02, 50.10] | 1024 |
| shift / new | multiply | 4.74 | 2.51 [2.34, 2.78] | 1024 |
| shift / new | divide | 33.89 | 28.12 [27.83, 28.37] | 1024 |

Larger operands also change output-format compliance. The next table separates integer-only parsing from unconditional correctness for every shifted prompt style. Prose and worked calculations remain errors under the frozen contract; no parser or generation budget was changed. A weak original baseline cannot establish arithmetic retention. Conditional accuracy among parseable outputs is available in the JSON, but uses a method-dependent subset and is not an acceptance metric.

| Shift task / style | Original parseable % | Original correct % | EML parseable %, mean of 3 seeds | EML correct %, mean [seed range] |
|---|---:|---:|---:|---|
| add/symbolic | 0.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] |
| add/prose | 2.73 | 2.54 | 1.79 | 1.60 [1.37, 1.76] |
| multiply/symbolic | 41.21 | 2.93 | 39.13 | 1.95 [1.95, 1.95] |
| multiply/prose | 0.00 | 0.00 | 0.00 | 0.00 [0.00, 0.00] |
| divide/symbolic | 0.20 | 0.20 | 0.07 | 0.07 [0.00, 0.10] |
| divide/prose | 0.39 | 0.39 | 0.33 | 0.33 [0.20, 0.39] |
| add/reversed | 52.73 | 22.27 | 53.19 | 21.71 [21.29, 21.97] |
| add/code | 100.00 | 83.89 | 99.87 | 77.41 [76.76, 78.22] |
| multiply/reversed | 44.34 | 1.17 | 47.20 | 0.46 [0.39, 0.59] |
| multiply/code | 98.63 | 8.30 | 98.86 | 4.56 [4.30, 4.98] |
| divide/reversed | 8.79 | 1.46 | 8.27 | 1.63 [1.56, 1.66] |
| divide/code | 99.22 | 66.31 | 99.19 | 54.62 [54.10, 55.08] |

## Depth, capacity, and training cost

Budgets count every encoder, nonlinear argument/readout projection, decoder, bias, norm, and statistic. EML/SiLU comparisons match sequential nonlinear depth and coefficient ceilings. Deployment folds affine statistics; the cheapest linear factorization is collapsed to one affine map. Native surrounding norms and all other model components remain counted.

The linear controls are trained low-rank affine surrogates of the whole MLP, including a full-rank member at the larger training budget. They do not retain the native gated-GELU computation. This study therefore makes no superiority claim over methods that factorize the native MLP's individual weight matrices.

The deployment ledger below covers every topology in held-out evaluation; counts are identical across its three seeds. Training coefficients include normalization statistics. Deployed module bytes include all parameters and buffers. The full model retains 2,289 buffer values (6,178 bytes), including native components outside the replaced MLP. The surrounding feed-forward norms contribute 3,072 parameters and remain included in the full-model count.

| Module | Training coefficients | Deployed parameters | Module buffers | BF16 module bytes | Full model parameters | Total parameter reduction % |
|---|---:|---:|---:|---:|---:|---:|
| Original MLP | 56,623,104 | 56,623,104 | 0 | 113,246,208 | 5,104,297,504 | 0 |
| eml-b6000000-d1 | 5,999,731 | 5,995,122 | 0 | 11,990,244 | 5,053,669,522 | 0.992 |
| linear-b3000000-d0 | 2,999,247 | 2,360,832 | 0 | 4,721,664 | 5,050,035,232 | 1.063 |
| linear-b6000000-d0 | 4,726,273 | 2,360,832 | 0 | 4,721,664 | 5,050,035,232 | 1.063 |
| silu-b6000000-d1 | 5,999,832 | 5,995,223 | 0 | 11,990,446 | 5,053,669,623 | 0.992 |
| silu-b6000000-d2 | 5,998,293 | 5,993,684 | 0 | 11,987,368 | 5,053,668,084 | 0.992 |

| Family | Budget | Depth | Output rank | Selection objective, mean of 3 seeds | Train raw MSE | Rank floor | Fit minutes, 3 seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| eml | 3M | 1 | 256 | 0.61860 | 0.28888 | 0.23762 | 7.0 |
| eml | 3M | 2 | 256 | 0.61696 | 0.28592 | 0.23762 | 12.1 |
| eml | 3M | 4 | 256 | 0.61702 | 0.28537 | 0.23762 | 13.7 |
| eml | 6M | 1 | 512 | 0.52114 | 0.22526 | 0.14137 | 10.5 |
| eml | 6M | 2 | 512 | 0.52214 | 0.22201 | 0.14137 | 11.4 |
| eml | 6M | 4 | 512 | 0.52327 | 0.22273 | 0.14137 | 11.7 |
| linear | 3M | 0 | 974 | 0.71825 | 0.35281 | 0.04813 | 2.1 |
| linear | 6M | 0 | 1536 | 0.72109 | 0.35379 | 0.00000 | 1.9 |
| silu | 3M | 1 | 256 | 0.61989 | 0.28807 | 0.23762 | 8.6 |
| silu | 3M | 2 | 256 | 0.61807 | 0.28545 | 0.23762 | 9.5 |
| silu | 3M | 4 | 256 | 0.61789 | 0.28470 | 0.23762 | 12.0 |
| silu | 6M | 1 | 512 | 0.52605 | 0.22648 | 0.14137 | 9.8 |
| silu | 6M | 2 | 512 | 0.52366 | 0.22108 | 0.14137 | 10.7 |
| silu | 6M | 4 | 512 | 0.52468 | 0.22070 | 0.14137 | 9.2 |

Corrected-grid fit time totals **2.17 device-hours of elapsed training time**. This includes validation and shared-device delays, and is not a hardware active-compute counter. It excludes teacher collection, initialization, the superseded BOS diagnostic grid, and final evaluation. Their individual timings remain in the run records.

The covariance-tail floor applies to raw MLP outputs confined to an affine decoder subspace of the stated rank. It bounds any such decoder on this training distribution, regardless of nonlinear depth; it does not bound EML architectures generally or the normalized residual contribution. The JSON also reports clipping, sampled exponent clamps, precision conversion, and actual CUDA memory peaks.

For the selected EML configuration, the raw training MSE is 0.22526 and its rank floor is 0.14137. The floor is 62.8% of the observed raw error. Deeper nonlinear stages cannot remove that affine-output-rank constraint. Residual error above the floor, domain-specific errors, clipping and seed spread remain separate capacity/optimization diagnostics; they do not identify a universal EML limitation.

Across the EML primary grid, gradient clipping occurred on 0.0269% of updates, and telemetry found 4 clamped exponent arguments among 3,170,672,640 sampled arguments. These observations do not support widespread exponential saturation as the main measured failure. They do not establish convergence or rule out difficult optimization geometry.

| Selected EML domain | Train raw MSE | Selection raw MSE | Train contribution MSE | Selection contribution MSE |
|---|---:|---:|---:|---:|
| arithmetic | 0.06986 | 0.07671 | 0.05291 | 0.05816 |
| language | 0.38067 | 0.43354 | 0.40576 | 0.47388 |

Domain errors use the training scales of the fitted objective; differences can reflect both target variation and approximation quality. Teacher-forced activation fit also differs from free generation after the replacement changes a token. The final/new-format/shift tables test behavior under distribution changes; this study does not isolate every source of rollout error.

All 36 nonlinear primary fits reached the 12,000-update cap and selected their best checkpoint within the last 1,000 updates. Optimization was therefore not demonstrated to converge. A separate diagnostic restarted the selected EML architecture and matched-depth SiLU for 24,000 updates, with the same initialization and minibatch stream. These weights were never substituted into the frozen held-out roster.

| Diagnostic | Primary 12k objective | Best 24k objective | Best update | Extra run minutes |
|---|---:|---:|---:|---:|
| eml-b6000000-d1-s1103 | 0.52138 | 0.51221 | 22900 | 6.1 |
| eml-b6000000-d1-s2207 | 0.52073 | 0.51222 | 23700 | 14.6 |
| eml-b6000000-d1-s3301 | 0.52132 | 0.51282 | 23300 | 4.8 |
| silu-b6000000-d1-s1103 | 0.52644 | 0.51785 | 22900 | 10.0 |
| silu-b6000000-d1-s2207 | 0.52568 | 0.51728 | 23700 | 11.4 |
| silu-b6000000-d1-s3301 | 0.52601 | 0.51736 | 23900 | 14.5 |

The withdrawn BOS grid cost 1.33 additional device-hours of elapsed fit time. The six doubled-budget diagnostic runs cost 1.02 device-hours. They replayed the first 12k objective before continuing; their saved final objectives were independently replayed on CUDA. Improved activation fit on selection data does not establish improved held-out answer quality.

Selected EML gate failures: `eml-b6000000-d1-s1103`: add, multiply, divide, arc; `eml-b6000000-d1-s2207`: add, multiply, divide, arc; `eml-b6000000-d1-s3301`: add, multiply, divide, arc. The detailed bounds distinguish measured net degradation from failure to certify a one-percentage-point limit.

An exact error decomposition sharpens the capacity diagnosis. In normalized target coordinates, QR factorization of the learned decoder separates the target component outside its affine range from prediction error inside that range. The terms sum to the observed raw MSE; closure and agreement with the earlier fit audit are checked on CUDA. The release audit independently verifies full column rank and records singular values and conditioning. Each row averages three seeds at the selected depth and budget.

| Family / updates | Train total | Train outside decoder | Train within decoder | Selection total | Selection outside | Selection within |
|---|---:|---:|---:|---:|---:|---:|
| eml / 12k | 0.22526 | 0.14809 | 0.07717 | 0.25512 | 0.15374 | 0.10139 |
| silu / 12k | 0.22648 | 0.14838 | 0.07811 | 0.25744 | 0.15405 | 0.10339 |
| eml / 24k | 0.21936 | 0.14744 | 0.07192 | 0.25054 | 0.15310 | 0.09744 |
| silu / 24k | 0.22117 | 0.14769 | 0.07348 | 0.25306 | 0.15334 | 0.09973 |

The learned decoder's irreducible error is close to the best rank-512 training floor. Longer training primarily reduces error within the representable subspace. That remaining term can include encoder information loss, limited nonlinear capacity, optimization and objective tradeoffs; this decomposition does not separate those causes. The outside term could change if the decoder learns a different subspace, but cannot cross the stated rank floor on these targets. Neither term alone determines answer accuracy.

## Actual replacement inference cost

Fully GPU-resident BF16, native eager SDPA, identical optimization of baselines. Each workload uses 10 warm-ups and 50 alternating paired runs with 32 forced decode steps. Model loading, comparison transfers, tokenization, and warm-up are excluded; prefill, decoding, and phase synchronization are included. Other users share these H100s. Intervals describe variability within these runs, not exclusive-serving or between-run uncertainty.

| Method | Batch/prefill | Prefill ms | Decode ms | End-to-end ms | Paired original E2E ms | Output tokens/s | Peak allocated GiB | E2E speedup %, block-bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|---|
| eml-b6000000-d1-s1103 | 1/128 | 53.1 | 1265.1 | 1324.1 | 1315.7 | 24.2 | 9.48 | -0.97 [-2.06, +0.03] |
| eml-b6000000-d1-s1103 | 1/512 | 42.3 | 1295.8 | 1336.5 | 1320.2 | 23.9 | 9.53 | -1.18 [-2.22, -0.27] |
| eml-b6000000-d1-s1103 | 8/128 | 75.1 | 1338.6 | 1414.3 | 1405.5 | 181.0 | 9.60 | -0.76 [-2.13, +0.72] |
| eml-b6000000-d1-s1103 | 8/512 | 157.5 | 1237.2 | 1394.3 | 1389.5 | 183.6 | 10.01 | -0.82 [-2.29, +0.72] |
| linear-b3000000-d0-s1103 | 1/128 | 59.8 | 1314.6 | 1372.1 | 1380.9 | 23.3 | 9.47 | +0.56 [-0.13, +1.26] |
| linear-b3000000-d0-s1103 | 1/512 | 64.0 | 1276.6 | 1340.3 | 1354.2 | 23.9 | 9.53 | +1.54 [+0.73, +2.36] |
| linear-b3000000-d0-s1103 | 8/128 | 74.7 | 1327.0 | 1401.0 | 1410.3 | 182.7 | 9.59 | +0.66 [+0.05, +1.33] |
| linear-b3000000-d0-s1103 | 8/512 | 151.8 | 1274.7 | 1429.4 | 1460.4 | 179.1 | 10.00 | +2.17 [+1.39, +3.06] |
| linear-b6000000-d0-s1103 | 1/128 | 40.1 | 1224.0 | 1262.2 | 1256.0 | 25.4 | 9.47 | -1.00 [-2.23, +0.26] |
| linear-b6000000-d0-s1103 | 1/512 | 42.7 | 1224.0 | 1283.2 | 1292.0 | 24.9 | 9.53 | +0.96 [-0.27, +2.32] |
| linear-b6000000-d0-s1103 | 8/128 | 40.2 | 1160.3 | 1200.2 | 1199.3 | 213.3 | 9.59 | +0.32 [-0.30, +0.91] |
| linear-b6000000-d0-s1103 | 8/512 | 72.0 | 1249.3 | 1321.3 | 1323.3 | 193.7 | 10.00 | +0.39 [-0.05, +0.80] |
| silu-b6000000-d1-s1103 | 1/128 | 60.8 | 1327.9 | 1388.6 | 1378.1 | 23.0 | 9.48 | -0.31 [-1.08, +0.39] |
| silu-b6000000-d1-s1103 | 1/512 | 64.3 | 1266.8 | 1328.7 | 1340.7 | 24.1 | 9.53 | +0.85 [+0.18, +1.53] |
| silu-b6000000-d1-s1103 | 8/128 | 74.4 | 1328.5 | 1403.1 | 1399.4 | 182.5 | 9.60 | -0.00 [-0.96, +0.94] |
| silu-b6000000-d1-s1103 | 8/512 | 72.4 | 1224.6 | 1297.0 | 1287.8 | 197.4 | 10.01 | -0.39 [-1.04, +0.26] |
| silu-b6000000-d2-s1103 | 1/128 | 50.0 | 1297.4 | 1355.8 | 1342.2 | 23.6 | 9.48 | -0.42 [-1.48, +0.57] |
| silu-b6000000-d2-s1103 | 1/512 | 50.7 | 1315.9 | 1377.6 | 1367.6 | 23.2 | 9.53 | +0.16 [-0.78, +1.02] |
| silu-b6000000-d2-s1103 | 8/128 | 43.4 | 1254.1 | 1305.6 | 1275.8 | 196.1 | 9.60 | -1.15 [-2.15, -0.24] |
| silu-b6000000-d2-s1103 | 8/512 | 85.9 | 1346.2 | 1446.0 | 1436.3 | 177.0 | 10.01 | +0.04 [-0.47, +0.45] |

The JSON reports each paired original baseline, separate GPU event timings, prefill/decode throughput, peak allocated/reserved memory, device UUID, and counted student state. Here speedup means fractional latency reduction: the mean over pairs of 1 − replacement time / original time. A negative value means higher replacement latency. Parameter reduction alone is not evidence of speedup.

Alternating comparisons retain an extra **113,246,208-byte CPU reference MLP** outside the registered replacement model. That apparatus is explicitly separate from deployed parameter counts and is absent from the export. Its transfers are outside timing; GPU execution and peak working memory are measured on the installed replacement path.

The export contains replacement-module weights. Parameter reduction describes the installed runtime model after removing the original MLP; the supplied loader still reads the full pinned base checkpoint. The export is not a standalone compressed full-model download.

Isolated module timings below diagnose token-count dependence. They omit the rest of Gemma and cannot establish an end-to-end speedup. All five tested token counts and both CUDA-event/wall measurements are in the JSON.

| Method | Tokens | Original module GPU ms | Replacement module GPU ms | Original wall ms | Replacement wall ms |
|---|---:|---:|---:|---:|---:|
| eml-b6000000-d1-s1103 | 1 | 0.0754 | 0.2077 | 0.0767 | 0.2091 |
| eml-b6000000-d1-s1103 | 4096 | 0.7957 | 0.5488 | 0.8001 | 0.5529 |
| linear-b3000000-d0-s1103 | 1 | 0.0748 | 0.0228 | 0.0759 | 0.0239 |
| linear-b3000000-d0-s1103 | 4096 | 0.8461 | 0.0433 | 0.8505 | 0.0470 |
| linear-b6000000-d0-s1103 | 1 | 0.0736 | 0.0223 | 0.0747 | 0.0233 |
| linear-b6000000-d0-s1103 | 4096 | 0.8523 | 0.0425 | 0.8562 | 0.0461 |
| silu-b6000000-d1-s1103 | 1 | 0.0729 | 0.1140 | 0.0740 | 0.1151 |
| silu-b6000000-d1-s1103 | 4096 | 0.7993 | 0.5392 | 0.8035 | 0.5431 |
| silu-b6000000-d2-s1103 | 1 | 0.0743 | 0.1874 | 0.0754 | 0.1886 |
| silu-b6000000-d2-s1103 | 4096 | 0.8093 | 0.3619 | 0.8134 | 0.3657 |

An additional module diagnostic applies identical `torch.compile` Inductor/default/fullgraph settings to every native/student pair. The table includes every configuration. Setup times and precision differences are recorded; compilation can change BF16 rounding. These results do not replace the eager full-model benchmark or establish compiled-model answer quality. CUDA-event intervals include any gaps between submitted kernels, not just kernel execution time.

| Method | Tokens | Original eager / compiled GPU ms | Student eager / compiled GPU ms | Student compiled-vs-eager NRMSE |
|---|---:|---|---|---:|
| eml-b6000000-d1-s1103 | 1 | 0.0789 / 0.1615 | 0.2171 / 0.2107 | 0.00934 |
| eml-b6000000-d1-s1103 | 4096 | 0.7992 / 0.7541 | 0.5570 / 0.4077 | 0.00946 |
| linear-b3000000-d0-s1103 | 1 | 0.0754 / 0.1574 | 0.0231 / 0.0987 | 0.00000 |
| linear-b3000000-d0-s1103 | 4096 | 0.8403 / 0.7845 | 0.0443 / 0.1180 | 0.00000 |
| linear-b6000000-d0-s1103 | 1 | 0.0754 / 0.1575 | 0.0231 / 0.0991 | 0.00000 |
| linear-b6000000-d0-s1103 | 4096 | 0.8410 / 0.7863 | 0.0446 / 0.1221 | 0.00000 |
| silu-b6000000-d1-s1103 | 1 | 0.0764 / 0.1584 | 0.1210 / 0.2038 | 0.00495 |
| silu-b6000000-d1-s1103 | 4096 | 0.8058 / 0.7581 | 0.5519 / 0.2864 | 0.00517 |
| silu-b6000000-d2-s1103 | 1 | 0.0751 / 0.1611 | 0.1926 / 0.2543 | 0.00671 |
| silu-b6000000-d2-s1103 | 4096 | 0.8042 / 0.7587 | 0.3588 / 0.3842 | 0.00539 |

## Causal result

Carry and digit probes are hypotheses. High decoding accuracy did not establish the operand-orthogonal carry direction as a sufficient or dominant mediator. The confirmation tests transplant native residual/KV state, use same-carry and matched random controls, and block the downstream carry readout. All rates include native errors and fixed prefixes inconsistent with the donor's true answer; that prefix-consistent subset is reported separately.

The target digit is the specified digit of the donor's true sum. Native donor accuracy is the reference for that score; exact-logit matching instead compares the donor model output even when its arithmetic is wrong.

| Cohort | Eligible / original groups | Both residual+KV: exact logits / cases | Native donor target digit % | Residual only % | Earlier sliding V % | Random KV % | Carry direction % |
|---|---:|---:|---:|---:|---:|---:|---:|
| gate-256-known | 896/1024 | 512/512 | 95.51 | 4.30 | 89.06 | 3.12 | 0.39 |
| gate-256-new | 896/1024 | 512/512 | 93.16 | 35.35 | 55.47 | 3.32 | 1.17 |
| shift-256-known | 860/1024 | 512/512 | 88.28 | 17.58 | 68.75 | 5.47 | 3.52 |
| shift-256-new | 860/1024 | 512/512 | 83.59 | 24.80 | 52.73 | 3.32 | 2.73 |

The four cohorts contain 2,048 prompt cases but 512 distinct operand groups: known/new formats reuse each split's 256 groups. Exact-logit failure bounds therefore group formats by operand pair. With zero failures in a 256-group cohort, the one-sided 95% binomial upper bound is about 1.16%, rather than treating every logit or prompt as independent evidence.

The edit-efficacy check separates the probe readout from behavior. The upstream edit nearly matches the donor's carry readout; the downstream carry block restores the recipient's thresholded readout in every case. The next table reports native decoding accuracy and the paired behavioral effect of blocking, including uncertainty clustered by operand group. On shifted operands, upstream threshold accuracy falls below the 75% majority-class baseline. The JSON also reports balanced accuracy; threshold miscalibration and representational failure are not separated by ordinary accuracy alone.

| Cohort | Majority baseline % | Native recipient carry accuracy at 26 / 32 % | Upstream edit: donor readout class match % | Downstream block: recipient readout class match % | Blocking effect on donor-digit accuracy pp [95% CI] |
|---|---:|---|---:|---:|---|
| gate-256-known | 50.78 | 94.34 / 99.61 | 98.44 | 100.00 | -0.59 [-1.37, +0.00] |
| gate-256-new | 50.78 | 86.52 / 99.02 | 99.22 | 100.00 | -0.39 [-0.98, +0.00] |
| shift-256-known | 75.00 | 55.86 / 95.31 | 99.22 | 100.00 | -1.37 [-2.54, -0.39] |
| shift-256-new | 75.00 | 51.17 / 98.44 | 99.61 | 100.00 | -0.20 [-0.59, +0.00] |

An exact match of the source residual can coexist with different downstream states and logits when reused KV differs. A deterministic equation of that residual alone therefore lacks a required input in these interventions. Each equation-response file quantifies the best possible average squared error on these conflicting-input pairs and tests the actual trained EML/SiLU, linear, symbolic, and identity equations. Cache-only interventions provide a direct test: unchanged equation inputs imply zero predicted change despite a nonzero native response.

Equation response NRMSE below uses the full gate/known cohort. The three-seed mean is reported for every tested depth; no intervention result selects a depth. The complete JSON includes all four cohorts, seeds, simpler controls, absolute response energy, MSE intervals and input-collision bounds. Large normalized errors for a small native response should be read alongside its absolute energy.

| Equation | Coefficients | Selection observational MSE | Natural donor response NRMSE | Residual-only response NRMSE | V-only response NRMSE |
|---|---:|---:|---:|---:|---:|
| linear | 72 | 0.1865 | 0.4743 | 3.7893 | 1.0000 |
| symbolic | 0 | 0.1365 | 0.4980 | 4.4660 | 1.0000 |
| identity | 0 | 0.1802 | 0.4831 | 3.5944 | 1.0000 |
| eml depth 1 | 1968 | 0.0669 | 0.3698 | 4.1549 | 1.0000 |
| eml depth 2 | 1992 | 0.0649 | 0.3685 | 4.2002 | 1.0000 |
| eml depth 4 | 2040 | 0.0639 | 0.3631 | 4.2083 | 1.0000 |
| silu depth 1 | 1968 | 0.0770 | 0.3876 | 4.0900 | 1.0000 |
| silu depth 2 | 1992 | 0.0847 | 0.4068 | 4.1194 | 1.0000 |
| silu depth 4 | 2040 | 0.0744 | 0.3962 | 4.1895 | 1.0000 |

Each algebraically collapsed quantity readout has **12,296 coefficients**, for input and downstream measurement separately. The research harness retains the original **95,294-coefficient probe checkpoint at each layer** and constructs those projections from it; their tensor and file bytes are reported in the JSON. The linear equation has 72 coefficients plus a shared eight-value metric scale retained for evaluation. Inputs include estimates of carry and result digits already present in the upstream state. This is a propagation fit, not derivation of arithmetic from operand labels.

This is evidence about conditional causal routing and input sufficiency. It is not a recovered addition algorithm, an explanation of multiplication/division, or proof that EML cannot succeed with a different state representation. The small blocking effects can include a partial causal contribution; they do not validate the proposed carry algorithm or justify a mechanism-based replacement.

The omitted-KV limitation applies to the cross-layer equation. The native MLP itself is a deterministic function of its complete input, so missing KV does not explain the compact MLP's reconstruction error. See [the bound derivations and their scope](../THEORY_LIMITS.md).

Shared-KV dependence is expected from Gemma's architecture. [Related work](../RELATED_WORK.md) connects this study to Goodfire's interventions and parameter decomposition, causal abstraction, and arithmetic heuristic research without claiming a new algorithm or an advantage over those methods.

## Reproduction and evidence

See [commands](../COMMANDS.md), the [prospective protocol](../PROTOCOL.md), [evaluation details](../EVALUATION_PLAN.md), and [causal confirmation plan](../CAUSAL_CONFIRMATION.md). `summary.json` contains all seed results and diagnostics. `provenance.json` binds source and scientific inputs by SHA-256. Large activation tensors, raw evaluation traces, and checkpoints are retained in the external run roots; the compact export needs the pinned base checkpoint and replacement weights, with no teacher activations at inference.
