# Reusing unchanged Gemma prefill layers

The scalar study repeatedly generates from the same input with different late
MLP hooks. `GenerationPrefix` reuses the unchanged layers before that hook during
prefill. The native generation loop still handles every decode step, stopping
rule, score, and hook call.

In this Gemma model, layers after the cached prefix all reuse KV produced by
earlier layers. The helper stores their native prefill outputs and snapshots
both the dynamic KV cache and the shared-KV dictionary. On a hit, it restores
fresh copies inside the first cached layer, after native attention masks and
positions have been constructed with an empty cache. Decode cannot mutate the
saved snapshot. Input IDs, masks, and positions must match exactly.
The driver releases the generation snapshot before non-generative intervention
forwards so that an unused prefill does not increase their memory requirements.

This path requires a fixed evaluation-mode model, interventions after the
prefix, the pinned backend, and the study's greedy generation settings. It
rejects unsupported generation arguments. It is restricted to this experiment;
it does not provide a general prompt-cache implementation. The dense original
MLP still executes at the intervention layer.

## GPU evidence

[Validation](generation-validation.json) compares 72 native/cached combinations:
three operations, all 12 selected heads/controls, and both prefill-only and
all-token patches. Each combination has 32 validation inputs. Generated tokens,
every generation score, final KV tensors/state, and patch records match exactly.
Changing inputs invalidates the cache, and the existing non-generative prefix
cache continues to match native logits. The compact tensor fingerprints are in
`generation-native-validation.json.gz`.

A full saved addition cohort also reproduces all **36,864 method records** and
the original JSON file hash exactly. Those records cover 3,072 formatted inputs
and 12 methods; they are not 36,864 independent problems. No tolerance was added
to make either comparison pass.

The [driver check](generation-dispatch-validation.json) reproduces four saved
validation files containing another 2,880 method/condition records. It confirms
22 generation prefill hits, two misses, and release of the generation snapshot
before intervention forwards. The larger native comparison and replay allocated
at most 11.321 GiB; runtime reservations and other GPU processes are additional.

Six alternating timing comparisons use batches of 32 validation inputs and all
12 methods, starting with an empty prefill cache for every cached batch. The
per-comparison timings are included in the evidence. They describe experiment
execution on shared hardware, not controlled serving throughput or a benefit
caused by EML. The median paired time reduction was 32.0%; individual reductions
ranged from -1.2% to 34.0%, so one comparison was slightly slower with reuse.
No learned weights, problem identities, precision settings,
statistical criteria, or head-selection rules change.

## Running and reproducing

From `robustness/`, using the pinned Gemma environment:

```bash
CUDA_VISIBLE_DEVICES=3 python gemma/generation_execute.py run_evaluation gemma
CUDA_VISIBLE_DEVICES=0 python gemma/validate_generation_prefix.py --output /absolute/new-validation-directory
```

The checker requires the saved native addition ordinary cohort. It freezes any
as-yet-unwritten operation selection using the existing validation-only rule,
exactly as the standard evaluator does. It never selects using test results.

The driver verifies the existing host execution snapshot and the new
`generation-execution-freeze.json` before each child process. The study audit
and report also verify these execution snapshots. The older host driver remains
available for native-generation reproduction.

This amendment follows partial held-out results and addresses evaluation time.
An in-progress stage finishes through its original driver before the remaining
stages resume with prefill reuse. The first checker attempts rejected the
backend's unset beam-count field; inspecting the pinned implementation confirmed
that its native default is one beam. The final guard accepts that default and
the explicit value one, and all comparisons use unchanged native generation.
