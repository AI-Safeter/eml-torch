# One-layer quantization-error correction

The five-seed gate **failed**: EML did not consistently improve V reconstruction and attention-output error over SiLU at the same storage budget. This direction stops before K compression, fresh-answer evaluation, longer contexts or serving work. See [the measured results](RESULTS.md).

The experiment uses Gemma 4 E2B IT's first global-attention layer (zero-based layer 4), selected before fitting. Forty-eight fresh SQuAD articles train the corrections; twelve select checkpoints; twenty-four untouched articles determine the gate. All 84 articles are excluded from the earlier pilot's 56 articles. Each document prefix has 1,024 tokens and two questions. Eight query positions per question sample the native attention pattern.

Correction receives only decoded uint4 V and its stored BF16 minimum and quantization step. It predicts a step-scaled residual and returns BF16 V. EML and gated SiLU use identical 25,184-parameter layouts; a rank-24 linear correction fits under the same ceiling. Shared BF16 correction weights count toward a bank of 128 prefixes. Controls include no correction, parameter-free RMS normalization, and 65 explicitly stored residuals per prefix, including their int32 indices and BF16 values. The stored-residual control spends almost the entire weight allowance on additional cache information.

Training minimizes an equal-weight combination of V MSE and attention-output MSE, each divided by the corresponding uncorrected error for that training document. Attention error uses native Q and K, float32 softmax replay, and the model's output projection. Only cached-prefix V changes. Native BF16 SDPA validates the replay numerically; the reported error is a local component metric, not downstream answer quality. Five seeds share data order across correction families. Checkpoint selection includes the identity initialization and never reads gate articles.

`protocol.json` records the stop rule before collection: EML must beat SiLU on both metrics in all five seeds, improve both pooled metrics by at least 1%, have positive paired-bootstrap lower bounds, outperform the other controls on both mean errors, match SiLU's weight storage, and have an upper latency-ratio bound no greater than 1.10. GPU cost includes unpacking, feature construction, correction and the final BF16 cast. The protocol was not relaxed after the result.

Run from the repository root using the pinned Gemma environment (PyTorch 2.9.0+cu128, Transformers 5.16.1) and the locally cached model revision from `protocol.json`. Reuse the published split to reproduce without the retired pilot runners:

```bash
export EML_CORRECTION_RUN=/path/to/new-correction-run
export CUDA_VISIBLE_DEVICES=3
KV_PY=../emltorch-gemma-env/bin/python
mkdir -p "$EML_CORRECTION_RUN"
cp kv_correction/evidence/design.json "$EML_CORRECTION_RUN/design.json"
gzip -dc kv_correction/evidence/documents.json.gz > "$EML_CORRECTION_RUN/documents.json"
$KV_PY kv_correction/collect.py
$KV_PY kv_correction/fit.py
$KV_PY kv_correction/replay.py
$KV_PY kv_correction/gate.py
```

Collection preserves the native SDPA backend and causal masks. GPU replay requires bitwise agreement for V, attention weights and the output projection after removing dependencies on the retired pilot. The gate reloads all 15 deployed checkpoints, recomputes selection objectives, checks byte budgets, and freezes source/checkpoint hashes before evaluating the untouched articles. Existing completed runs are protected against replacement. Evidence includes raw errors and timings, fit histories, hashes and the published split. Large activation tensors and trained weights stay in `../emltorch-correction-runs`.

This is a fixed-format, one-layer experiment. Q and K are held at native values; it does not test feedback through later layers or quantify a total-cache or model-speed benefit. [KIVI](https://arxiv.org/abs/2402.02750) motivates separate per-channel K and per-token V quantization experiments, but this EML gate did not justify advancing to them.

The selected [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) Wikipedia excerpts and questions retain their [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) attribution to the SQuAD authors and source articles, identified by title in the split manifest.
