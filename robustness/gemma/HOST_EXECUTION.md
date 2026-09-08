# Gemma execution with CPU embedding storage

The frozen FP32 Gemma model has 19.015 GiB of parameters. Its per-layer embedding
lookup table occupies 8.75 GiB. Keeping that table on CPU leaves **10.265 GiB of
parameters on GPU**, plus runtime buffers. The registered parameter count is
still 5,104,297,504.

`HostEmbedding` gathers the requested rows on CPU, transfers them to CUDA, and
applies the native embedding scale on CUDA. This is an execution change for the
fixed text study. The table stays in FP32 and is not quantized or trained.
The special `_apply` behavior keeps it on CPU when the parent model moves to
CUDA; the helper is not a general mixed-precision model adapter.

Model and learned-head construction must occur outside `torch.inference_mode`.
The validation harness initially constructed both inside that context, causing
small differences from historical results. Matching the frozen evaluator's
construction removed every difference, including auxiliary KL summaries. The
final checker requires exact equality, with no relaxed numeric tolerance.

[GPU validation](host-validation.json) covers 1,152 clean/corrupted internal
feature cases, the saved training generations for all three operations,
9,437,184 full-vocabulary logits against native CUDA embedding lookup on the
needed token rows, and all 2,880 records in four existing fitted-head validation
cohorts. Native validation allocated at most 10.506 GiB.

The execution driver also uses the previously tested `GemmaPrefixCache`. It
reuses unchanged layers before the scalar contribution and restores the shared
KV dictionary entries those layers write. Generation runs the native layer
forwards. A complete saved addition-intervention replay is required before the
driver can run; [its record](host-prefix-replay.json) and the validation/source
hashes are checked on every process entry.

The full replay reproduced **221,184 records and the original JSON file hash**
exactly. It cached 26 unchanged layers and allocated at most 11.583 GiB. Its
542-second runtime is a completion record on shared hardware, not a controlled
performance comparison.

The separate driver leaves the older frozen entry points available for native
reproduction. It routes each child stage through the same validated setup:

```bash
# From robustness/, inside the pinned Gemma environment:
CUDA_VISIBLE_DEVICES=3 python gemma/host_execute.py run_evaluation gemma
```

To repeat the execution audits into fresh locations:

```bash
CUDA_VISIBLE_DEVICES=3 python gemma/validate_host_embeddings.py --output /absolute/path/to/new-validation.json
CUDA_VISIBLE_DEVICES=3 python gemma/replay_host_prefix.py --output /absolute/path/to/new-replay-directory
```

The full replay uses the already recorded native primary cohort and never changes
head selection. The amendment follows partial held-out results and addresses
execution capacity on shared GPUs. It leaves model precision, problem identities,
interventions, learned weights and statistical criteria fixed. Reusing work in
an experiment is not evidence that replacing a component with EML speeds up LLM
inference.
