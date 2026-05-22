# Pre-Registration — H31 Black-Box Behavioral Cert (Cross-Vendor)

**Status**: locked. Filed before any model run.
**Date**: 2026-05-22
**Author**: Samuel Hong + Claude Opus 4.7
**Scope decision (locked with user)**: N=2 vendors, FINAL — not a pilot.
**Framing decision (locked with user)**: exploratory across 5 circuit
classes; classify post-hoc. User's wording: "자유를 줘봐. 일단 결과를
보고 그 다음에 분류를 해."

## Background

CLAUDE.md Live State headlines H14–H29 build a cert-discovery track on
**white-box** signals: attention scores via hooks, gated-DeltaNet
effective weights via direct kernel extraction, residual-stream
projection via hidden_states. The library exports primitives
(`emit_raw_weight_concentration_cert`, `eml_tree_to_smt2_intervals`,
`extract_gated_effective_weights`, ...) that *all* assume internal
access.

H31 tests whether the same EML → `.smt2` cert pipeline survives a
**strictly black-box** access pattern: forward-only API, no hooks, no
attention output, no hidden state. The motivating use case: applying
EML interpretability to closed-source LLMs (OpenAI, Anthropic, Google)
where only `prompt → top-K logprobs` is available.

This is the "LLM black-box interpreter" claim's load-bearing experiment.

## Black-box access protocol (locked)

The runner script MUST satisfy ALL of the following:

1. Model loaded via `transformers.AutoModelForCausalLM.from_pretrained`
   only; NO `transformer_lens`, NO `sae_lens`, NO direct `model.layers[i]`
   access in measurement loop.
2. Forward call uses ONLY `model(input_ids=..., attention_mask=...)`.
   `output_attentions=False` and `output_hidden_states=False`
   enforced (verified by `assert outputs.attentions is None` and
   `assert outputs.hidden_states is None`).
3. No hooks registered; assertion before measurement:
   `assert len(model._forward_hooks) == 0 and len(model._forward_pre_hooks) == 0`
   for every submodule reachable via `model.modules()`.
4. Single accessed quantity per prompt: `outputs.logits[:, -1, :]` →
   `topk(K=50, dim=-1)` → store (token_ids, logprobs).
5. Import-time guard: `import emltorch` and `from transformers import ...`
   are the only DL imports. Script raises ImportError if
   `transformer_lens` appears in `sys.modules`.

**Audit script** `h31_blackbox_audit.py` runs the import + access checks
on a smoke prompt and fails CI if any constraint is violated. This is
the load-bearing discipline rule for the claim.

## Vendors (locked, N=2 FINAL)

| vendor | HF path | size | white-box cert track record |
|---|---|---|---|
| Qwen3.6 (hybrid) | `Qwen/Qwen3-Next-80B-A3B-Instruct` if available, else `Qwen/Qwen3.6-27B` | ~27B / 80B-A3B | H22j: 6 COPY+FACTUAL specialists (full-attn only); H23l: 48 specialists when gated layers probed; **0 INDUCTION-PURE** |
| Gemma-4 | `google/gemma-4-31b-it` if available, else `google/gemma-4-26b-a4b` | ~31B / 26B-A4B | H20: **0 INDUCTION-PURE across all scales** (E4B/26B-A4B/31B); induction distributed across heads |

Fallback rule: if the primary HF path 404s or OOMs on a single H100, fall
back to the listed alternate. Document which was actually used in the
results JSON.

## Probe (5 circuit classes, broad exploration)

Per user "자유를 줘봐" — broad probe, classify post-hoc.

| circuit | n_prompts | source pattern | target |
|---|---:|---|---|
| induction | 50 | `[A B C ... A]→B` repeat-completion; 8 token categories from H19c | next token = B |
| copy_oneshot | 50 | `[X] then again [X]:` pattern from H21 | `X` |
| factual | 50 | `The capital of <Country> is` from H22e | known city/lang token |
| ioi | 50 | Wang 2022 `When [A] and [B] went to the store, [B] gave the milk to` | `A` |
| syntactic | 50 | subject-verb agreement, gender-pronoun completion | function token |

**Variable features extracted per prompt (BLACK-BOX safe — derived from
prompt string alone, NOT from model internals)**:
- `T` — prompt length in tokens (via tokenizer)
- `L` — induction lag (where applicable) = `last_q − first_prior_occurrence`
- `log_freq_target` — log unigram frequency of target token (from
  pre-computed Wikipedia n-gram table; not from model)
- `n_repeats` — number of repeated tokens (copy circuits only)
- `prompt_class_id` — one-hot of {0..4} for the 5 circuit classes

**Measurement per prompt**:
- `P_target = exp(logprob_topk[target_token_id])` if target ∈ top-50, else 0
- `top1_correct = (argmax_logprob == target_token_id)`
- `entropy_top50 = -Σ p_i log p_i` on top-50

Total probes: 5 circuits × 50 prompts = 250 per vendor × 2 vendors = 500
forward calls. Trivially cheap on 4×H100.

## Fit (with all 11 filters from CLAUDE.md §Methodology)

Per vendor × per circuit class:

1. **Filter #1 tautology check**: confirm
   `max|P_target − f_deterministic(prompt_features)| > 0.1` over the
   probe set (i.e., behavior is NOT trivially decoded from prompt string).
2. **Filter #2 poly K=2 preflight**: fit OLS on `[T, L, log_freq, T², L²,
   T·L, T·log_freq, L·log_freq]`. If HELDOUT R² ≥ 0.95, **mark "trivial
   subset, EML cannot beat"** and report so.
3. **Filter #3 seed variance**: 5 EML evolution seeds, report mean ± std
   R² and best-of-5.
4. **Filter #4 PC1 manifold OOD**: hold out 20% along PC1 of (T, L,
   log_freq); evaluate EML + poly on held-out.
5. EML evolution: depth-3 and depth-4, population=4096, generations=30.
6. Baseline: poly K=2, K=5 (OLS).

## Cert obligations (locked)

Per vendor × per circuit class with HELDOUT R² ≥ 0.5:

- **Working-region cert**: lower-bound `f_EML(features) > 0.5` over a
  pre-defined "operating box" (the convex hull of the train data
  features). Expected verdict: UNSAT (proven). Dual-verify z3 + cvc5.
- **Failure-region cert**: upper-bound `f_EML(features) > 0.5` over a
  pre-defined "boundary box" (features near the model's known weak
  region, e.g. extreme L or extreme T). Expected verdict: SAT (provides
  concrete counterexample input).

Both verdicts are *informative*: UNSAT proves the working region; SAT
identifies the failure boundary with an explicit feature-vector
counterexample.

## Success criteria (descriptive, not threshold-locked)

User's "결과 보고 분류" framing means we lock the *protocol* not the
thresholds. Post-hoc we will report:

- **D1**: per-vendor × per-circuit HELDOUT R² for EML and poly K=5 +
  Δ_(EML − poly K=5) sign and magnitude.
- **D2**: vendor-differentiation — for each circuit class, do Qwen3.6
  and Gemma-4 produce *different* EML formula structures (different
  topology, different leaf set), or same? Same formula across vendors
  = generic behavior. Different = architectural fingerprint.
- **D3**: cert pass rate — number of dual-UNSAT `.smt2` per vendor ×
  circuit.

Findings will be reported honestly even if null. Three possible
post-hoc verdicts:

- **V-architectural-fingerprint**: ≥1 circuit class shows
  vendor-different EML topology AND HELDOUT R² > poly K=5 by ≥0.05.
  Strongest claim: "behavior alone (no hooks) discriminates
  architectural property."
- **V-generic-behavior-cert**: same EML formula structure across
  vendors, but certs discharge dual-UNSAT. Weaker claim: "EML →
  `.smt2` pipeline works black-box on frontier LLMs (existence
  demonstration)."
- **V-null**: EML ties or loses poly K=5 on ≥4/5 circuit classes;
  cert pass rate < 50%. Claim: "black-box behavioral level does not
  admit cert-discoverable mechanism formulas at this scope;
  white-box remains primary."

V-null is acceptable and pre-committed; no walk-back.

## Anti-tautology guards specific to behavior-level

The advisor noted that behavioral fit can degenerate into "P(correct) ≈
threshold(prompt_features)" without mechanistic content. Three
prophylactic checks:

1. **Inter-circuit cross-fit**: take EML formula fit on COPY data, try
   to predict FACTUAL behavior. If R² stays > 0.5 across circuits,
   formula is non-mechanistic (just fitting "easy prompt"
   noise-floor).
2. **Random-token control**: substitute target token with a random
   token from same vocab; if P_target_random fits EML at similar R²,
   the formula isn't actually about the *circuit-specific* behavior.
3. **Vendor cross-fit baseline**: Qwen-fit formula evaluated on Gemma
   data and vice versa. Cross-vendor R² > 0.5 (per circuit) =
   "vendor-generic behavior"; < 0.3 = "vendor-specific signature."

## What this does NOT claim

- Does NOT claim mechanism interpretability on closed-source LLMs (we
  test on local models that *could* be hooked; the discipline rule is
  refusing to hook).
- Does NOT claim "first-principles physics" — the EML operator
  universality (Odrzywolek 2603.21852) is pure math.
- Does NOT replace white-box. White-box H14–H29 atlas remains the
  primary substantiation; H31 is an extension to a strictly weaker
  access model.
- N=2 final scope means we cannot claim "vendor-agnostic." Closest
  defensible: "demonstrated on Qwen3.6 + Gemma-4 frontier
  architectures."

## Files (locked before run)

- `docs/superpowers/specs/2026-05-22-h31-blackbox-behavioral-cert-prereg.md`
  (this file)
- `sae-eml/scripts/h31_probe_generator.py`
- `sae-eml/scripts/h31_blackbox_runner.py` (with audit assertions)
- `sae-eml/scripts/h31_blackbox_audit.py` (smoke-test the access pattern)
- `sae-eml/scripts/h31_fit_and_baseline.py`
- `sae-eml/scripts/h31_emit_certs.py`
- `outputs/h31_blackbox_cert/probes.jsonl`
- `outputs/h31_blackbox_cert/measurements_{qwen36,gemma4}.jsonl`
- `outputs/h31_blackbox_cert/fit_results.json`
- `outputs/h31_blackbox_cert/certs/*.smt2`

## Commit discipline

This pre-reg file is committed to `eml_interpretability` workspace
before any model is loaded. Diff git log to verify pre-reg-commit
timestamp < first measurement timestamp.

---

**END OF PRE-REGISTRATION. LOCKED.**
