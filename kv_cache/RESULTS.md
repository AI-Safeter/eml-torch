# Gemma E2B V-cache compression pilot

EML did not meet the proposed quality target in this pilot. Packed 4-bit V quantization was the strongest observed compression baseline. The small evaluation does not establish a quality improvement or certify a 1 percentage point loss bound.

Gemma 4 E2B IT, BF16, H100: 24 training articles, 8 validation articles, 24 held-out articles with 2 questions each. Every cached document prefix has 1,024 tokens. All 15 independently stored V caches are compressed; K remains unchanged.

| Method | Exact match (%) | Answer F1 (%) | F1 loss (pp), 95% interval | Total KV reduction |
|---|---:|---:|---:|---:|
| original | 70.83 | 84.99 | 0.00 [0.00, 0.00] | 1.000× |
| uint4 | 72.92 | 86.75 | -1.77 [-4.58, 0.00] | 1.593× |
| lowrank | 62.50 | 74.04 | 10.95 [1.63, 20.49] | 1.595× |
| silu | 66.67 | 79.53 | 5.46 [-3.47, 14.95] | 1.594× |
| eml | 62.50 | 76.92 | 8.06 [-1.50, 18.33] | 1.594× |

Intervals resample articles, preserving the two questions per article. They are exploratory percentile bootstrap intervals, without correction across methods. In particular, an observed zero-regression bootstrap bound cannot certify the absence of rare regressions. Even zero regressions in 24 independent articles would give a one-sided 95% event-probability bound of 11.73%, far above 1%.

Storage includes K, packed values or latent codes, range metadata, and shared encoder/decoder weights amortized over 128 prefixes. All compressed methods fit under the same byte ceiling; integer ranks leave a little unused capacity. SiLU and EML have identical ranks and parameter counts. Their shared codec weights occupy 2,049,408 bytes each.

The measured V reduction is about 3.9×; the total KV reduction is about 1.59×. With equally sized K and V and K unchanged, total KV compression cannot exceed 2×. Achieving 4× total storage reduction requires compressing K as well.

| Method | Compression (ms) | Restoration (ms) | Cache-hit request (ms) | Paired request ratio, 95% interval |
|---|---:|---:|---:|---:|
| original | 0.03 | 22.99 | 698.10 | 1.000 [1.000, 1.000] |
| uint4 | 3.51 | 23.96 | 687.52 | 0.991 [0.975, 1.007] |
| lowrank | 3.10 | 22.08 | 698.04 | 1.006 [0.989, 1.021] |
| silu | 1.94 | 25.34 | 697.86 | 1.005 [0.986, 1.029] |
| eml | 2.20 | 28.08 | 693.74 | 0.996 [0.986, 1.007] |

Wall-clock measurements synchronize CUDA. Twelve repetitions rotate method order after a warmup; generation always runs 16 tokens for timing. Cache-hit requests include restoration, question processing and generation. The compression column above starts from an already compacted cache. The separate store-phase measurement below includes sliding-window compaction as well. Native prefill is shared by every method and measured separately in the uncached-request reference in latency.json. Measurements use a shared GPU and eager PyTorch, with allocator cleanup between cases. Every compressed method's latency interval includes a slowdown. No speedup or absence of latency regression is established.

| Method | Complete store phase, median (ms) | Store peak allocation (GiB) |
|---|---:|---:|
| original | 2.06 | 9.569 |
| uint4 | 4.93 | 9.574 |
| lowrank | 2.80 | 9.572 |
| silu | 3.53 | 9.573 |
| eml | 2.37 | 9.573 |

| Method | 128-prefix bank + codec weights (MiB) | Resident GPU allocation (GiB) | Request peak allocation (GiB) | Peak reserved (GiB) |
|---|---:|---:|---:|---:|
| original | 1534.50 | 11.038 | 11.052 | 11.119 |
| uint4 | 963.56 | 10.480 | 10.494 | 10.562 |
| lowrank | 962.08 | 10.479 | 10.493 | 10.586 |
| silu | 962.51 | 10.480 | 10.494 | 10.588 |
| eml | 962.51 | 10.480 | 10.494 | 10.588 |

The bank measurement allocates 128 distinct copies of one validation prefix payload and checks their actual backing storage. It measures physical memory, not distinct-document hit rate, request concurrency or throughput. Resident and peak figures include the full model and codec weights; peak allocation includes restoration and generation temporaries. The auxiliary bank and store-phase scripts were added after QA evaluation solely to complete memory and compaction-cost accounting, with no refitting or choice changes.

Native sliding-window tensor views retained 18 MiB after prefill. Compacting their backing storage reduced the uncompressed cache to 11.99 MiB while preserving logits bit for bit. All reported compression gains use this already compacted baseline.

This is a single-seed reconstruction pilot on 1,024-token document prefixes. It does not cover long-context scaling, multiple datasets, cache eviction, production kernels, or whole-KV compression. The proposed 4× storage / ≤1 pp loss / no latency regression target is not met. A useful next experiment would test a small EML correction of quantization residuals against an equally sized SiLU correction; replacing the whole V representation with this learned low-rank decoder is not supported by these results.
