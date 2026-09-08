# KV restoration experiment

An isolated Gemma 4 E2B IT cache-compression pilot. [Results](RESULTS.md) contain the complete comparison and its limitations. The production `emltorch` package is unchanged.

`protocol.json` fixes the model revision, precision, splits, fit budget, storage ceiling and evaluation settings. Documents and questions are disjoint by SQuAD article. Fitting reads only training and validation V tensors; questions and answer quality do not select the decoder. The test set here is a held-out subset of SQuAD's public development data, not the hidden SQuAD test set. Pretraining contamination is not assessed.

The independent cache consists of 12 sliding-window layers (511 retained tokens, 256 channels) and 3 global layers (1,024 tokens, 512 channels). The other 20 Gemma layers share KV state. Restoration preserves native sequence lengths, sliding offsets and shared-state reconstruction during forward passes. Only cached prefix V is compressed: newly generated K/V remains native BF16.

The baselines are packed affine uint4 per token, PCA projection/reconstruction, a PCA encoder with a gated SiLU decoder, and the same encoder with the repository's `EMLHead`. Both nonlinear decoders use the same two affine branches, linear skip, width and parameter count. EML features compute `exp(left(z)) - log(1 + right(z)^2)` through the core numerical guards. Training uses float32, while validation and deployment use BF16 weights and codes. The PCA encoder is fixed. All methods fit a byte ceiling defined by uint4 V storage for 128 prefixes; there is no padding to manufacture an exact tie. Encoder weights, means, decoder weights and quantization ranges count toward the ceiling.

The uint4 implementation is a transparent PyTorch baseline with actual nibble packing, not an integration with a serving engine or a reproduction of KIVI. [KIVI](https://arxiv.org/abs/2402.02750) motivates per-token V quantization. [vLLM prefix caching](https://docs.vllm.ai/en/v0.9.0/features/automatic_prefix_caching.html) already skips prefill for exact shared prefixes. [Transformers cache documentation](https://huggingface.co/docs/transformers/main/kv_cache) describes memory/latency tradeoffs and model-specific cache behavior.

Run from the repository root with a fresh output directory, a locally cached model at the pinned revision, and the Gemma environment in `../emltorch-gemma-env`. GPU execution is required for collection, fitting and validation.

```bash
export EML_KV_RUN=/path/to/new-kv-run
export CUDA_VISIBLE_DEVICES=3
KV_PY=../emltorch-gemma-env/bin/python
$KV_PY kv_cache/data.py
$KV_PY kv_cache/collect.py
$KV_PY kv_cache/fit.py
$KV_PY kv_cache/validate.py
$KV_PY kv_cache/evaluate.py
$KV_PY kv_cache/bank_memory.py
$KV_PY kv_cache/store_benchmark.py
$KV_PY kv_cache/report.py
```

The pinned environment is recorded in `protocol.json` and `../robustness/gemma/requirements.txt`. `validate.py` freezes source, split and checkpoint hashes before held-out inference. Existing completed outputs are protected against replacement. `evidence/` contains raw answer records, fit-selection metrics, measurements, GPU checks and hashes; local checkpoints and sampled activations stay outside Git in `../emltorch-kv-runs`. Reproduction should verify the published data-source and document hashes. The final reporting and bank-allocation scripts are auxiliary to the frozen fit/QA code.

SQuAD data, questions and answer references are from [Rajpurkar et al.'s SQuAD release](https://rajpurkar.github.io/SQuAD-explorer/), distributed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The evidence contains selected references and model-generated predictions; those dataset excerpts retain their original attribution and license.
