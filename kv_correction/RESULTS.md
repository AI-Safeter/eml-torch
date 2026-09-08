# One-layer correction: gate failed

EML did not demonstrate the required benefit over SiLU. The direction stops here: no K compression, full-model answer evaluation or serving expansion was run.

Gemma 4 E2B IT, layer 4, H100, native BF16 cache and BF16 deployed corrections. Five training seeds used 48 fresh articles for training, 12 for checkpoint selection and 24 untouched articles for the gate. Two questions and 16 selected query positions per article define the local attention metric.

| Correction | V MSE | Projected attention MSE | Shared correction or extra residual bytes, 128 prefixes |
|---|---:|---:|---:|
| none | 0.05935775 | 0.38743646 | 0 |
| rms | 0.05501818 | 0.37062921 | 0 |
| stored_residual | 0.05928491 | 0.38121743 | 49,920 |
| linear | 0.05670939 | 0.00447443 | 50,320 |
| silu | 0.05642113 | 0.00411077 | 50,368 |
| eml | 0.05650423 | 0.00415931 | 50,368 |

The uncorrected packed-V bank occupies 34,078,720 bytes. EML and SiLU each add 50,368 bytes of shared BF16 weights, for an identical total of 34,129,088 bytes. Linear uses 48 fewer bytes. The stored-residual control adds 49,920 bytes of values and indices; RMS normalization has no learned weights. K and all other layers remain unchanged. No unused budget is padded with dummy tensors.

Relative to SiLU, EML has **0.147% higher V MSE** (95% interval: 0.103% to 0.184%) and **1.181% higher projected attention MSE** (interval: -0.090% to 2.602%). EML wins on both metrics in **0 of 5 seeds**. The intervals use paired resampling of seeds and articles; repeating articles across seeds does not create additional independent documents.

All learned corrections sharply reduce this local projected attention error, including the linear control. That is not evidence of an EML-specific benefit or preserved downstream answer quality. RMS normalization gives the lowest V reconstruction MSE among these candidates. This remains a single layer and fixed prompt format with native Q and K held fixed.

| Gate condition | Passed |
|---|---|
| every seed both metrics | no |
| at least one percent both metrics | no |
| positive bootstrap lower both metrics | no |
| beats all other controls | no |
| matched eml silu weight bytes | yes |
| comparable decode correct cost | no |

GPU runtime includes packed-V reconstruction, feature construction, correction and the BF16 cast. There are 50 paired repetitions after 10 warmups for each seed. Median times are 2.501 ms for SiLU and 2.518 ms for EML. The mean paired EML/SiLU ratio is 1.443, with a wide 95% interval of [1.118, 1.844]; this statistic is sensitive to tail timings on the shared GPU. It fails the predeclared upper bound of 1.10. These measurements do not establish an intrinsic 44% slowdown or describe whole-model request latency.

GPU validation reloaded all 15 selected checkpoints and reproduced their selection objectives; identity initialization was bitwise equal to uncorrected reconstruction. The float32 attention replay differed from independent float32 SDPA by 1.04e-6 relative RMS and from native BF16 SDPA by 0.177%. Reported attention MSE uses the float32 mathematical replay. After removing dependencies on the retired KV pilot, a native GPU recollection reproduced the saved V, attention weights and output projection bit for bit.

Raw errors, fit histories and runtime samples are retained as compressed JSON in `evidence/`, together with source, split and checkpoint hashes. The published split permits reproduction without the retired experiment runners. See [the protocol and commands](README.md).
