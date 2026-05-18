# emltorch

**GPU-batched symbolic regression with portable SMT-LIB2 verification, via the EML operator `exp(x) − ln(y)`.**

`emltorch` discovers compact closed-form expressions from data — and machine-checks properties of those expressions with z3 + cvc5. Built on Andrzej Odrzywolek's [*All elementary functions from a single binary operator*](https://arxiv.org/abs/2603.21852) (arXiv:2603.21852), with a GPU-batched evolutionary search plus an axiomatized-`Exp`/`Ln` SMT bridge.

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
# poly K=5  OOD R²: −1.41e+08         max|pred| = 5.3e+01      ✗ wrong by 4 OOM
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

Available lemmas: `multiplicativity`, `ratio_corollary`, `ln_multiplicativity`, `e_interval_tight`, `depth3_ln_identity`, `ln_at_e`, `exp_minus_y`, `relu_depth4_identity`. Pair them with the axiomatized emitter, not the interval one — interval-form is pure QF_LRA and does not declare `Exp`/`Ln`.

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

The original target audience. Closed-form effective-weight extractors are provided for softmax attention, Gated DeltaNet (math-exact to 0.26% rel-err vs `chunk_gated_delta_rule`), and an attention-block local Lipschitz primitive following [Yudin et al. 2025](https://arxiv.org/abs/2507.07814).

```python
from emltorch import (
    extract_gated_effective_weights,
    attention_block_lipschitz_clean,
    emit_attention_lipschitz_smt2_block,
    emit_raw_weight_concentration_cert,
)
```

`examples/refusal_circuit.py` is an end-to-end recipe (transformer hook → activation features → `fit` → cert).

---

## Honest limitations

| target | result |
|---|---|
| ReLU | exact at depth 4 |
| `sin`/`cos` on `[−π, π]` | R² ≥ 0.994 at depth 5 (ties poly K=9) |
| SiLU / sigmoid | R² ≤ 0.9999 ceiling, depth 3–7 (fundamental approximation limit) |
| Modular arithmetic / grokking | not addressable — multi-cycle structure outside elementary-function class |
| General tabular SR (high-dim, smooth) | polynomial K=5 ties or wins on most bounded analytic targets within training range |
| Networks bigger than ~20 features | depth-d tree search is sweet-spotted for V ≈ 1–20 |

EML's value is not raw HELDOUT R² across all tasks. It is (i) **symbolic parsimony** (a depth-3 EML tree is auditable; a 21-coefficient polynomial is not), (ii) **OOD-bounded extrapolation** on `exp` / `log` / `sigmoid` / `softplus` targets when structural recovery succeeds, and (iii) **portable SMT-LIB2** certificates of the discovered formula.

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
