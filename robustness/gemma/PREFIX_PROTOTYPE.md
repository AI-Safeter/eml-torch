# Gemma shared-state prefix-cache prototype

The current live Gemma evaluator remains uncached. This prototype is not enabled by `prefix_execution.py`; a full saved primary-cohort replay is required before an execution amendment enables it.

Native Gemma 4 attention reuses key/value tensors stored by earlier layers in a shared dictionary. Caching hidden states alone would omit these side effects. `GemmaPrefixCache` also snapshots dictionary entries written by prefix layers and restores cloned entries during reuse. It requires `past_key_values=None` during cached logits calls; generation delegates to the original forwards. Token content and mask invalidation remain inherited from the validated base cache.

Validation on H100 used the pinned Gemma backend. Forty-eight manual output interventions across addition, multiplication, and division produced bitwise-identical logits; generation was unchanged. Repeated-forward timing was preliminary and on shared hardware. Four additional validation cohorts with the actual fitted heads, native GELU controls, all-token generation, and coordinate edits exactly matched the uncached validation files.

Reproduce these checks inside the Gemma environment with `python gemma/validate_prefix.py` and `python gemma/validate_prefix_heads.py`. They write separate reproduction files, preserving the frozen reference records. Neither command changes a running study job. These checks establish computational equivalence for the tested cases, not improved arithmetic or general model throughput.
