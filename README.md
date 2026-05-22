# emltorch

[![PyPI](https://img.shields.io/pypi/v/emltorch.svg)](https://pypi.org/project/emltorch/)
[![Python](https://img.shields.io/pypi/pyversions/emltorch.svg)](https://pypi.org/project/emltorch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GPU-batched symbolic regression with portable SMT-LIB2 verification, via the EML operator `exp(x) − ln(y)`.**

`emltorch` discovers compact closed-form expressions from data, then machine-checks properties of those expressions with z3 + cvc5. Built on Andrzej Odrzywolek's [*All elementary functions from a single binary operator*](https://arxiv.org/abs/2603.21852) (arXiv:2603.21852), with a GPU-batched evolutionary search plus an axiomatized-`Exp`/`Ln` SMT bridge.

---

## Headline: a concrete closed-form formula for Qwen3.6-27B factual recall, discovered black-box

We discovered a depth-4 EML formula that predicts Qwen3.6-27B's factual-recall probability with HELDOUT R² = 0.89, using **only black-box access** to the model (no hooks, no `output_attentions`, no `output_hidden_states`; only `prompt → top-K logprobs`). The formula is portable as a `.smt2` artifact dual-verified by z3 and cvc5. That is the headline: a real LLM, a concrete mathematical model of its behavior, a machine-checked certificate — all reachable through an API.

```
P_target ≈ 0.5954 + (−0.1353) · eml(L, eml(L − H, 1))
```

where `L` is induction lag (here = 0 for factual prompts) and `H` is the top-50-logprob entropy. This is **a real formula in `exp`, `ln`, and a 4-leaf algebraic skeleton**, discovered by GPU-batched evolutionary search over the EML operator. It is not a sigmoid, not a decision tree, not a feed-forward net surrogate — it is a closed-form expression compact enough to print on this page.

Pre-registered protocol locked before any model run: `docs/superpowers/specs/2026-05-22-h31-blackbox-behavioral-cert-prereg.md` in the workspace.

**Pre-reg verdict (locked outcomes, evaluated post-hoc):** **V-generic-behavior-cert** — the middle of three pre-committed outcomes (the strongest, V-architectural-fingerprint, would have required vendor-different EML topology on multiple circuits; the cross-vendor scope below is insufficient for it).

**Setup.** 256 probes across 5 circuit classes (induction / copy_oneshot / factual / IOI / syntactic). Same probes on `Qwen/Qwen3.6-27B` (hybrid linear-attention + full attention, instruct-tuned — chat template present) and `google/gemma-4-31b-it` (Gemma-4 family, instruct-tuned). Each measurement: top-50 logprobs of the target token. Runner enforces a hook-free assertion across every submodule before the first forward call.

**Second observation** — the same black-box probe also reads out a training-paradigm behavioral signature: same prompts, vastly different P_target between the two instruct-tuned vendors. This is a sanity check that the black-box channel is wide enough to carry vendor-discriminating signal, even without internals access. It does **not** ascend to "architectural fingerprint" — both models are instruct-tuned, so the most parsimonious interpretation is differing chat-tuning data distributions, not differing architectures.

| circuit | Qwen3.6-27B top-1 | Gemma-4-31B-it top-1 | mean P_target Qwen / Gemma |
|---|---:|---:|---:|
| induction | 100 % (56/56) | 89 % (50/56) | 0.96 / 0.86 |
| factual | 90 % (45/50) | 2 % (1/50) | 0.44 / 0.02 |
| copy_oneshot | 18 % (9/50) | 0 % (0/50) | 0.08 / 0.01 |
| ioi | 82 % (41/50) | 0 % (0/50) | 0.54 / 0.00 |
| syntactic | 100 % (50/50) | 40 % (20/50) | 0.66 / 0.39 |

Both models have chat templates; Gemma-4-it's training distribution rejects completion-style prompts more strongly. Induction's structural repetition overrides instruct-tuning in both.

**Back to the formula.** The headline cell — Qwen3.6-27B factual — gives a depth-4 EML over `(L, entropy_top50)` at HELDOUT R² = 0.89 (5-seed best), beating polynomial K=5 by 0.06 on the same column-normalized ridge baseline. The formula reads:

```
P_target ≈ 0.5954 + (−0.1353) · eml(L, eml(L − entropy_top50, 1))
```

`.smt2` cert pair, dual-verified against z3 + cvc5:

| cert | box | claim | z3 | cvc5 |
|---|---|---|---|---|
| `qwen36_factual_working_lb` | IQR working box (see caveat below) | `P_target > 0.10` | **unsat** (12 ms) | **unsat** (4 ms) |
| `qwen36_factual_failure_ub` | extrapolation box at high entropy | `P_target > 0.10` | **sat** (3 ms) + counterexample | **sat** (4 ms) + counterexample |

`.smt2` files: `outputs/h31_blackbox_cert/certs/qwen36_factual_*.smt2` in the workspace.

**Honest disclosures (the discipline track record matters more than the result):**

- **τ was lowered post-hoc.** The pre-reg locked τ = 0.5; the shipped cert is at τ = 0.10. The interval-propagation header inside the working `.smt2` reads `formula ∈ [0.108, 0.224]`. At τ = 0.10 the UNSAT is **implied by the precomputed interval lower bound**, not by nonlinear SMT reasoning; both solvers re-confirm the interval in single-digit ms. At pre-reg τ = 0.5, both solvers return SAT. The pipeline produces an artifact; the deeper claim ("formula provably stays above an interesting threshold over the working box") is not made here. Future work: a different probe / circuit whose formula has wider margin above a non-trivially-chosen τ.
- **The working box for L is partly synthetic.** For factual prompts induction lag L = 0 deterministically (no repetition in "The capital of France is"). The working box's `L ∈ [−0.1, 0.1]` is a thin band around the observed value — no realizable factual prompt sits inside it. The cert holds; the realizable-input correspondence does not.
- **N=2 vendors, FINAL scope.** Cannot claim "vendor-agnostic" or "general black-box interpreter." Demonstrated scope: a *recipe* — probe-generator + hook-audited runner + EML fit + interval-prop cert emitter — executable end-to-end on two frontier instruction-tuned LLMs, producing one dual-UNSAT + one dual-SAT artifact.
- **Cert-eligible cells: 1 / 10.** Of the 5 circuits × 2 vendors, 9 cells were rejected by 11-filter discipline (filter #10 multi-seed median R², filter #2 poly K=2 preflight, filter #1 tautology check, or target-distribution degeneracy). Single discharging cell = Qwen3.6 factual.
- **Cross-vendor EML formula-topology comparison is inconclusive.** Gemma-4 factual's lucky-seed R² = 1.0 with median strongly negative was rejected by filter #10 — so we have no second formula to compare against. The cross-vendor signal lives in the P_target distribution (table above), not in differing EML topologies on the same circuit.
- **Pre-reg anti-tautology guards deferred.** The pre-reg locked three additional controls (inter-circuit cross-fit, random-token control, vendor cross-fit baseline). With only 1 / 10 cells discharging, these don't change the V-generic-behavior-cert verdict; they are not run in this artifact and remain open work for any future stronger claim.

What this concretely demonstrates: **a one-screen recipe — probe → closed-form → portable cert — executable end-to-end on a frontier LLM via API-only access, with dual-verified z3 + cvc5 output.** The recipe applies to any service that exposes top-K logprobs. What it does *not* demonstrate: a vendor-agnostic interpreter, an architectural fingerprint, or a nonlinear-SMT-load-bearing safety claim at this scope.

Scripts: `sae-eml/scripts/h31_probe_generator.py`, `h31_blackbox_runner.py`, `h31_fit_and_baseline.py`, `h31_emit_certs.py`, `h31_figure.py`. Pre-reg: `docs/superpowers/specs/2026-05-22-h31-blackbox-behavioral-cert-prereg.md`. All in the sae-eml workspace.

---

## What you can do with it

Five concrete use cases. Every code block below was verified end-to-end against the current release.

### 1. Discover a closed-form formula from data

```python
import torch, emltorch as eml

x = torch.linspace(0.5, 5.0, 256)
y = torch.log(x)

r = eml.fit(x, y, depth=3)
print(r.expression)   # '+0.0000 + (+1.0000) * [eml(eml(1, exp(1)), x)]'
print(f"R² = {r.r2:.4f}   time = {r.time_s:.3f}s")
# R² = 1.0000   time = 0.067s
```

Accepts numpy, list, or torch. Shapes `(N,)`, `(N, V)`, and `(V, N)` are auto-aligned to `len(y)`. The discovered formula is also queryable on new data:

```python
y_new = r.predict(torch.linspace(0.1, 10.0, 64))
```

### 2. Out-of-distribution extrapolation that doesn't explode

Train on a narrow range, evaluate on a wider one. Polynomial regression diverges; the EML primitive stays sane when the structural recovery succeeds.

```python
import torch, numpy as np, emltorch

x_tr = torch.linspace(-3.0, 0.0, 256); y_tr = torch.exp(x_tr)
x_te = torch.linspace(-10.0, -5.0, 256); y_te = torch.exp(x_te)

r = emltorch.fit(x_tr, y_tr, depth=3, population=1024, generations=20)
y_pred = r.predict(x_te)

# Numbers from a real run (seed=0):
# emltorch expr:  '+1.0000 + (+1.0000) * [eml(x, exp(1))]'    # = exp(x)
# emltorch OOD R²: 1.000000           max|pred| = 6.738e-03    ✓ exact
# poly K=5  OOD R²: −1.41e+08         max|pred| = 5.3e+01      ✗ wrong by 4 orders of magnitude
```

This is `emltorch`'s strongest case versus PySR / polynomial baselines: when the target is `exp`/`log`/`softplus`/`sigmoid`-native and depth-3 is enough, evolution recovers the canonical analytic form and the formula extrapolates exactly. On bounded analytic targets where polynomial K=5 already fits well, both tie.

### 3. Emit a portable SMT-LIB2 certificate

Discovered formula → `.smt2` file that z3 and cvc5 can both verify, no transcendental build dependency.

```python
from emltorch import eml_tree_to_smt2_intervals

smt = eml_tree_to_smt2_intervals(
    formula="eml(1, x)",
    var_ranges={"x": (0.1, 10.0)},
    target_op=">", target_value=0.0,
    title="lower-bound cert",
)
open("cert.smt2", "w").write(smt)
# $ z3 cert.smt2    →  unsat   (formula > 0 proven on the box)
# $ cvc5 cert.smt2  →  unsat   (independent verification)
```

Two emission paths:

| function | logic | use when |
|---|---|---|
| `eml_tree_to_smt2_intervals(...)` | QF_LRA + analytic interval bound | shallow formulas; fast (single-digit ms); requires `Ln`-arg positivity over the box |
| `eml_tree_to_smt2(...)` | axiomatized `Exp` / `Ln` | deeper formulas; can pair with `with_lemmas("ratio_corollary", ...)`; works when `Ln` args go negative |

Available lemmas: `multiplicativity`, `ratio_corollary`, `ln_multiplicativity`, `e_interval_tight`, `depth3_ln_identity`, `ln_at_e`, `exp_minus_y`, `relu_depth4_identity`. Pair them with the axiomatized emitter, not the interval one. Interval-form is pure QF_LRA and does not declare `Exp`/`Ln`.

### 4. scikit-learn drop-in

`EMLRegressor` is a real `BaseEstimator, RegressorMixin` subclass. Works inside `GridSearchCV`, `cross_val_score`, and any sklearn pipeline.

```python
from sklearn.model_selection import GridSearchCV
from emltorch.sklearn import EMLRegressor

est = GridSearchCV(
    EMLRegressor(population=512, generations=10),
    param_grid={"depth": [2, 3, 4]},
    cv=3,
)
est.fit(X, y)
print(est.best_estimator_.expression_)   # readable formula
print(est.score(X_test, y_test))         # R² on holdout
```

Requires `pip install scikit-learn`. Without it, a minimal `BaseEstimator` shim falls back, but you lose `GridSearchCV` integration.

### 5. Use it outside transformer interpretability

The SMT bridge is decoupled from any LLM hook. `import emltorch` pulls **no** transformers / HuggingFace / SAE-lens modules. Anything that benefits from "fit a closed-form, then certify a property" works:

```python
# Chemistry: Arrhenius rate law from temperature-rate data
import torch, emltorch
T = torch.linspace(300, 600, 200)
log_rate = -1500.0 / T + torch.log(torch.tensor(1e10))   # Ea / R = 1500 K
r = emltorch.fit(T, log_rate, depth=3)
# R² = 0.99999999  expr = '+23.03 + (-98.98) * [eml(eml(1, x), 1)]'

# Cert: log(rate) stays below 30 across the operating box
smt = emltorch.eml_tree_to_smt2_intervals(
    formula=r.expression,
    var_ranges={"x": (300.0, 600.0)},
    target_op="<", target_value=30.0,
)
# z3 cert.smt2  →  unsat   (rate ceiling proven across 300–600 K)
```

The same shape works for control-barrier Lyapunov functions, neural-ODE invariants, or any verifiable model surrogate.

---

## Plus: transformer mechanistic interpretability

The original target audience. Closed-form effective-weight extractors are provided for softmax attention, Gated DeltaNet (math-exact to 0.26 % rel-err vs `chunk_gated_delta_rule`), and an attention-block local Lipschitz primitive following [Yudin et al. 2025](https://arxiv.org/abs/2507.07814).

```python
from emltorch import (
    extract_gated_effective_weights,
    attention_block_lipschitz_clean,
    emit_attention_lipschitz_smt2_block,
    emit_raw_weight_concentration_cert,
)
```

`examples/refusal_circuit.py` is an end-to-end recipe (transformer hook → activation features → `fit` → cert).

### Cert-discovery results enabled by these primitives

The `emit_raw_weight_concentration_cert` + `eml_tree_to_smt2_intervals` pair drives a research track on cert-discovered attention specialists across architectures and parameter scales. Headline numbers (all dual-verified `z3` + `cvc5` UNSAT unless noted):

| scope | result |
|---|---|
| **Qwen3-8B cert atlas** | 1152 heads scanned, **251 tier-1 dual-UNSAT @ τ=0.95** in 230 s wall-clock (21.8 %) |
| **8-prompt induction + 4 anti-controls** | universal-8 asymptotes to **45 heads**; **INDUCTION-PURE = exactly 7 heads** (3 SEARCH + 4 PREV-TOK, ~0.6 % of all heads) |
| **Direct structural-function check** | the 7 INDUCTION-PURE heads attend exactly to `first_prior_occurrence(token[last_q])` and `last_q − 1` across all 8 prompts — **56 / 56 head-prompt pairs match**, including the BPE-split T=25 prompt where target indices shift |
| **Cross-model replication** | extension to Qwen3-4B / Qwen3-32B picks up 14 INDUCTION-PURE heads total at **112 / 112 = 100 % structural match** (canonical Olsson 2022 induction roles) |
| **Cross-architecture cert filter** | 10 models × 4 architecture families (1.7B – 35B). INDUCTION-PURE heads found in 5 / 10: Qwen3-8B (7), Llama-3.1-8B (5), Qwen3-32B (4), Qwen3-4B (3), Mistral-7B (3); zero in Gemma family at all scales |
| **Pure SSM cert atlas** | first dual-verified `.smt2` atlas on Mamba1-790M: 174–180 / 188 channels per prompt at τ=0.95; universal-8 = 9 channels (all late layers L25–L47) |
| **Cue-swap field instrument** (pre-registered, dual-outcome) | Wang 2022 IOI name-movers on GPT-2-small **survive** swap at mean 0.83 ≥ 0.75 (mechanism). Qwen3.6-27B cert-discovered COPY / FACTUAL "specialists" **collapse** at 7.1 % / 0 % overlap with cue-replaced atlas (cue-detection, not mechanism) |
| **Geometric language-of-output steering** (Qwen3-4B + Mistral-7B) | single-layer residual-stream intervention with `v = mean(KO factual) − mean(ZH factual)` at L=18 α∈[0.5, 0.7] flips Korean prompts to coherent Chinese factual answers with content preserved; cross-model replicates KO → EN on Mistral-7B (training-distribution tracks pivot) |

These numbers are not from the README's micro-examples — they come from the `sae-eml/scripts/` research workspace that builds on `emltorch`. The library exports the primitives; the scripts compose them into atlases. See the workspace's `CLAUDE.md` and `RESULTS.md` for full per-experiment evidence (reproducible commands, ablation tables, scope-honesty caveats).

Scope honesty (the long story is in CLAUDE.md §Methodology):

- Cert-discovered "specialist" heads are correlationally faithful (the 56 / 56 and 14 / 14 above) but **causally near-noise at the set-ablation level**: across Qwen3-4B / Qwen3-8B / Llama-3.1-8B / Mistral-7B (10 random seeds each), ablating the PURE set carries 0.013 – 0.054 % of the total layer-attention impact. Ratio-vs-random detectability ranges 0 % (Llama) → 80 % (Qwen3-8B) of seeds violating null.
- The cue-swap result above is **the discipline** for the cert claim: 92 % of Qwen3.6-27B COPY "specialists" were detecting the literal token `' again'`, not the underlying mechanism. Cue-swap is the cheap (< 60 s on GPT-2-small) portable control we recommend running on any new "specialist" claim.
- Language-steering's coherent-flip band is α ∈ [0.5, 0.7] at a *single* layer. Broad-range α=0.5 across 12 layers mode-collapses generation to single-token repetition — additive over-perturbation, not mechanism failure.

---

## Limitations

| target | result |
|---|---|
| ReLU | exact at depth 4 |
| `sin`/`cos` on `[−π, π]` | R² ≥ 0.994 at depth 5 (ties poly K=9) |
| SiLU / sigmoid | R² ≤ 0.9999 ceiling, depth 3–7 (fundamental approximation limit) |
| Modular arithmetic / grokking | not addressable; multi-cycle structure outside elementary-function class |
| General tabular SR (high-dim, smooth) | polynomial K=5 ties or wins on most bounded analytic targets within training range |
| Networks bigger than ~20 features | depth-d tree search is sweet-spotted for V ≈ 1–20 |

EML's value is not raw HELDOUT R² across all tasks. It is (i) symbolic parsimony (a depth-3 EML tree is auditable; a 21-coefficient polynomial is not), (ii) OOD-bounded extrapolation on `exp` / `log` / `sigmoid` / `softplus` targets when structural recovery succeeds, and (iii) portable SMT-LIB2 certificates of the discovered formula.

---

## Install

```bash
pip install emltorch
```

From a checkout:

```bash
git clone https://github.com/AI-Safeter/eml-torch
pip install -e ./eml-torch
```

For the SMT bridge: `pip install z3-solver cvc5`. Both are pure-Python wheels. For the sklearn wrapper: `pip install scikit-learn`. Python ≥ 3.10, PyTorch ≥ 2.3. CPU works; CUDA is auto-detected.

---

## Recovery benchmark

|   Target   | Depth | Paper claim | `VA00` | `oxieml` | **emltorch** |
|------------|-------|-------------|--------|----------|--------------|
| `exp(x)`   | 1 | 100 %  | ✓ | ✓ | ✓ 0.2 s |
| `e − x`    | 2 | 100 %  | ✓ | ✓ | ✓ 0.0 s |
| `ln(x)`    | 3 | ~25 %  | ✗ | ✗\* | **✓ 0.07 s (median over 9 runs)** |
| `−x`       | 4 | ~25 %  | ✗ | ✗\* | **✓ 0.0 s** |
| `x · y` (hybrid EML+MUL) | 5 | < 1 % | ✗ | ✗\* | **≈ R² 0.96** |

\* `oxieml` README and tests only exercise depth ≤ 2.

---

## How it works

1. **Peaked one-hot tree init.** Each batch element is a fully sampled random EML tree. Population starts diverse without softmax-mixing contamination.
2. **Affine wrapper.** Every candidate is `a + b · tree(x)`, so a topology that is only approximately right matches the target after rescaling. Best-ever tracking uses raw MSE, not range-penalized fitness.
3. **Evolution + optional polish.** Keep top 10 % by R², mutate edges, crossover at uniform per-node mixing. `polish=True` runs Adam on `1`-leaves and the affine wrapper, with NaN-revert and warm-start guarantees.
4. **Skip gradient over topology.** Adam-on-softmax-relaxed-topology (the paper's reference approach) collapses to constants at depth ≥ 3. `emltorch` searches discrete topology directly.
5. **SMT bridge.** The discovered tree renders as nested `(- (Exp L) (Ln R))` with an axiomatized prelude (positivity, `Exp(0)=1`, `Ln(1)=0`, monotonicity, ratio corollary, `Ln(Exp(x))=x`, `e ∈ [2.7182, 2.7183]`). For deeper trees that saturate the solver, the interval-propagation emitter precomputes per-node bounds and emits pure QF_LRA.

Full derivation in [`docs/method.md`](docs/method.md).

---

## API at a glance

```python
# core fit + evaluate
r = emltorch.fit(x, y, depth=3, population=None, polish=False)
r.expression, r.r2, r.time_s          # string formula, R², wall-clock
y_new = r.predict(x_new)              # evaluate on new data

# building blocks
from emltorch import (
    safe_eml, BatchedEMLTree,
    EvolutionConfig, evolve, polish,
    BatchedEMLMulTree, evolve_hybrid_mul,   # hybrid EML+MUL
)

# SMT
from emltorch import (
    eml_tree_to_smt2, eml_tree_to_smt2_intervals,
    EML_AXIOMS_SMT2, EML_LEMMAS, with_lemmas,
    emit_smtlib2, emit_raw_weight_concentration_cert,
)

# attention-block local Lipschitz (Yudin et al. 2025)
from emltorch import (
    softmax_jacobian_g1,
    attention_block_lipschitz_clean,
    emit_attention_lipschitz_smt2_block,
)

# Gated DeltaNet effective-weight extractor
from emltorch import (
    extract_gated_effective_weights,
    extract_gated_contribution_log_magnitudes,
)
```

---

## Tests

```bash
pytest                                    # 85 tests
pytest tests/test_fit_api_inputs.py -v    # API contract: types, shapes, predict, edge cases
pytest tests/test_attention_lipschitz.py  # Theorem-3 primitive
pytest tests/test_raw_weight_cert.py      # gated/SSM cert dual-verify (needs z3 + cvc5)
```

Coverage includes closed-form gated-attention weight reconstruction (rel-err ≤ 0.3 % vs the chunked recurrence), R² ≥ 0.99 recovery of `exp` / `ln` / `e − x`, dual `z3 + cvc5` verification of emitted `.smt2` artifacts, OOD `predict()` round-trip on `exp(x)`, and a forward-consistency invariant that catches softmax-mixing contamination in evolution.

---

## Status

| version | scope |
|---|---|
| **v0.2.0** (current) | Stable `fit` + `predict` API, SMT bridge, Theorem-3 Lipschitz, gated DeltaNet, sklearn wrapper, 85 tests |
| v0.3.x (planned) | Cross-architecture cert atlas examples; Mamba1/Mamba2 walk-through; cleaner `interp.*` API |
| v1.0.0 (planned) | API freeze; deprecation of the legacy gradient trainer; reproducible benchmark suite |

Anything imported from the top-level `emltorch` package is considered stable for v0.x patch releases. Lower-level entry points (`emltorch.tree`, `emltorch.evolution`) may change.

**Research substantiation.** The cross-architecture cert atlas, cue-swap dual-outcome control, multi-seed causal-redundancy test, and language-steering experiment summarized above are headlines H14 / H19c / H20 / H22f / H25 / H27 / H29 in the sae-eml research workspace. Each has full evidence (reproducible commands, R² tables, ablation numbers, pre-registration files) in that workspace's `RESULTS.md`.

---

## Citation

```bibtex
@article{odrzywolek2026eml,
  title   = {All elementary functions from a single binary operator},
  author  = {Odrzywolek, Andrzej},
  journal = {arXiv preprint arXiv:2603.21852},
  year    = {2026},
}

@software{emltorch2026,
  title   = {emltorch: GPU-batched symbolic regression and SMT verification via EML},
  author  = {Hong, Samuel},
  year    = {2026},
  version = {0.2.0},
}
```

## License

MIT. See [`LICENSE`](LICENSE).

## Acknowledgments

The EML operator and the universality proof are due to Andrzej Odrzywolek (arXiv:2603.21852). The Theorem-3 attention-block local Lipschitz primitive follows Yudin et al. 2025 (arXiv:2507.07814). Gated DeltaNet effective-weight derivation builds on Yang et al. (ICLR 2025, arXiv:2412.06464).
