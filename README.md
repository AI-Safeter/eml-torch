# emltorch

**GPU-batched symbolic regression with portable SMT-LIB2 verification, via the EML operator `exp(x) − ln(y)`.**

`emltorch` discovers compact closed-form expressions from data — and machine-checks properties of those expressions with z3 + cvc5. It's a research library aimed at **mechanistic interpretability**: pull activations out of a transformer, regress a symbolic formula, then emit a formal certificate of the resulting circuit.

Built on Andrzej Odrzywolek's [*All elementary functions from a single binary operator*](https://arxiv.org/abs/2603.21852) (arXiv:2603.21852), with a novel GPU-batched evolutionary search plus an axiomatized-`Exp`/`Ln` SMT bridge that scales to 8B-class transformer interpretability.

---

## What you get

| Capability | Function |
|---|---|
| Symbolic regression on (numpy / torch / list) inputs | `emltorch.fit(x, y, depth=3)` |
| Discovered tree → portable SMT-LIB2 (`Exp`/`Ln` axiomatized) | `emltorch.eml_tree_to_smt2` |
| Same, with QF_LRA interval propagation for big assertions | `emltorch.eml_tree_to_smt2_intervals` |
| Transformer activation extractor | `emltorch.interp.from_transformer_hook` |
| scikit-learn drop-in | `emltorch.sklearn.EMLRegressor` |
| Theorem-3 attention-block local Lipschitz | `emltorch.attention_block_lipschitz_clean` |
| Closed-form Gated DeltaNet effective weights | `emltorch.extract_gated_effective_weights` |

Every public function has a docstring; the SMT artifacts dual-verify on z3 ≥ 4.16 and cvc5 ≥ 1.3 with no transcendental build dependency.

---

## Why this exists

- **The EML primitive is universal.** `eml(x, y) = exp(x) − ln(y)` recovers every elementary function (`exp`, `ln`, `e − x`, `ReLU`, `sigmoid`, `GELU`, multiplication via depth ≥ 5 EML+MUL hybrid). Odrzywolek's paper proves the math; this library makes it tractable on GPUs.
- **Discovery isn't enough — you also want a certificate.** Other symbolic-regression libraries return a SymPy expression and stop. `emltorch` translates the discovered EML tree into a portable `.smt2` file with axiomatized `Exp`/`Ln`, ratio corollary, and interval-propagation forms, so you can prove statements like *"this formula stays above τ everywhere in this L∞ box"* with two off-the-shelf SMT solvers.
- **Designed for transformers from day one.** Gated DeltaNet effective-weight extractor (math-exact 0.26% reconstruction error against `chunk_gated_delta_rule`), attention-block local Lipschitz primitive following [Yudin et al. 2025](https://arxiv.org/abs/2507.07814), and an `interp.from_transformer_hook` plumbing utility.
- **Works where the paper's reference trainer collapses.** Adam-on-softmax-relaxed-topology (Odrzywolek §4.3) saturates to constants at depth ≥ 3. GPU-batched evolutionary search at population 4k recovers `ln(x)` at depth 3 in well under a second.

---

## Install

```bash
pip install emltorch
```

Or from a checkout:

```bash
git clone https://github.com/SabaPivot/emltorch
pip install -e ./emltorch
```

For the SMT bridge: `pip install z3-solver cvc5`. Both are pure-Python wheels; no transcendental build flags needed.

Python ≥ 3.10, PyTorch ≥ 2.3. CPU works; CUDA is auto-detected.

---

## Quick start

### 1. Discover `ln(x)` at depth 3

```python
import torch, emltorch as eml

x = torch.linspace(0.5, 5.0, 256)
y = torch.log(x)

result = eml.fit(x, y, depth=3)
print(result.expression)   # e.g. '+4.5749 + (-1.0000) * [eml(eml(1, exp(1)), x)]'
print(f"R² = {result.r2:.4f}  |  time = {result.time_s:.2f}s")
# R² = 1.0000  |  time = 0.0s
```

Accepts `numpy`, `list`, or `torch.Tensor`. Shapes `(N,)`, `(N, V)`, and `(V, N)` are auto-aligned to `len(y)`.

### 2. Emit a portable SMT-LIB2 certificate

```python
from emltorch import eml_tree_to_smt2_intervals

smt = eml_tree_to_smt2_intervals(
    formula="eml(1, x)",                # discovered formula (string form)
    var_ranges={"x": (0.1, 10.0)},      # L∞ box
    target_op=">",                      # prove formula > 0 on the box
    target_value=0.0,
    title="EML lower-bound cert",
)
open("cert.smt2", "w").write(smt)
# z3 cert.smt2  →  unsat   (formula is positive on the box)
# cvc5 cert.smt2 →  unsat  (independent verification)
```

The generated `.smt2` is fully portable: ASCII, axiomatized `Exp`/`Ln`, no `(declare-fun Real ...)` quirks, dual-verifies on z3 + cvc5 in single-digit milliseconds for typical depths.

### 3. Symbolic regression on transformer activations

```python
import torch, emltorch
from sklearn.decomposition import PCA

acts, tgt = emltorch.interp.from_transformer_hook(
    model, layer=22, inputs=prompts, target="logits",
)
feats = torch.from_numpy(PCA(n_components=4).fit_transform(acts.numpy())).float()
result = emltorch.fit(feats, tgt.float(), depth=4)
print(result.expression)
```

End-to-end recipe (refusal-direction circuit on a real LM): `examples/refusal_circuit.py`.

### 4. scikit-learn pipeline

```python
from emltorch.sklearn import EMLRegressor
from sklearn.model_selection import GridSearchCV

est = GridSearchCV(EMLRegressor(), {"depth": [2, 3, 4]}, cv=5)
est.fit(X, y)
print(est.best_estimator_.expression_)
```

---

## Recovery benchmark

| Target | Depth | Paper claim | `VA00/SymbolicRegressionPackage` | `cool-japan/oxieml` | **emltorch** |
|---|---|---|---|---|---|
| `exp(x)` | 1 | 100 % | ✓ | ✓ | ✓ 0.2 s |
| `e − x` | 2 | 100 % | ✓ | ✓ | ✓ 0.0 s |
| `ln(x)` | 3 | ~25 % | ✗ | ✗\* | **✓ 0.0 s** |
| `−x` | 4 | ~25 % | ✗ | ✗\* | **✓ 0.0 s** |
| `x · y` (hybrid EML+MUL) | 5 | < 1 % | ✗ | ✗\* | **≈ R² 0.96** |

\* `oxieml` README and tests only exercise depth ≤ 2.

---

## How it works

1. **Peaked one-hot tree init.** Each batch element is a fully sampled random EML tree (rather than a uniform softmax over operators). Population starts diverse without noise contamination.
2. **Affine wrapper.** Every candidate is `a + b · tree(x)`, so a topology that is only *approximately* right matches the target after rescaling. Best-ever tracking uses raw MSE, not range-penalized fitness, to avoid d=5 regression vs d=4.
3. **Evolution + polish.** Keep top 10 % by R², mutate edges (single random flip per child at mutation logit 150 — necessary for forward-consistency), crossover at uniform per-node mixing. Optional `polish=True` step runs Adam on `1`-leaves + affine, with NaN-revert and warm-start guarantees.
4. **Skip gradient over topology.** Adam-on-softmax-relaxed-topology (the paper's approach) collapses to constants at depth ≥ 3. `emltorch` searches discrete topology directly.
5. **SMT bridge.** The discovered tree is rendered as nested `(- (Exp L) (Ln R))` calls with an axiomatized prelude (positivity, `Exp(0)=1`, `Ln(1)=0`, monotonicity, ratio corollary, `Ln(Exp(x))=x`, `e ∈ [2.7182, 2.7183]`). Two emission paths: the **direct** form for shallow trees, and the **interval-propagation** form (QF_LRA) for trees where naive emission saturates a solver.

Full derivation, EML identities, and the d=5 saturation diagnostic: [`docs/method.md`](docs/method.md).

---

## API at a glance

```python
# core fit
result = emltorch.fit(x, y, depth=3, population=None, polish=False)
result.expression  # human-readable symbolic form
result.r2          # coefficient of determination
result.time_s      # wall-clock

# building blocks
from emltorch import (
    safe_eml, BatchedEMLTree,             # primitive + batched tree
    EvolutionConfig, evolve, polish,      # search + refinement
    BatchedEMLMulTree, evolve_hybrid_mul, # hybrid EML + MUL (multiplicative gates)
)

# SMT
from emltorch import (
    eml_tree_to_smt2,                    # direct axiomatized translation
    eml_tree_to_smt2_intervals,          # interval-propagation QF_LRA
    EML_AXIOMS_SMT2, EML_LEMMAS,         # named axiom + lemma library
    emit_smtlib2,                        # L∞-box cert helper (768-dim safe)
    emit_raw_weight_concentration_cert,  # gated/SSM kernel concentration cert
)

# attention block local Lipschitz (Yudin et al. 2025, Theorem 3)
from emltorch import (
    softmax_jacobian_g1,
    attention_block_lipschitz_clean,
    attention_block_lipschitz_interval,
    emit_attention_lipschitz_smt2_block,
)

# Gated DeltaNet effective-weight extractor
from emltorch import (
    extract_gated_effective_weights,
    extract_gated_contribution_log_magnitudes,
    compute_delta_rule_deltas,
)
```

---

## Tests

```bash
pytest                                       # 83 tests
pytest tests/test_attention_lipschitz.py -v  # Theorem-3 primitive
pytest tests/test_raw_weight_cert.py -v      # gated/SSM cert dual-verify
```

Tests include closed-form gated-attention weight reconstruction (rel-err ≤ 0.3 % vs the chunked recurrence), R² ≥ 0.99 recovery of `exp` / `ln` / `e − x`, dual `z3 + cvc5` verification of emitted `.smt2` artifacts, and a forward-consistency invariant (`forward_batched(tree)[i] == forward_argmax_walk(tree[i])`) that catches softmax-mixing contamination in evolution.

---

## Used in research

`emltorch` is the engine behind a cross-architecture mechanistic-interpretability certificate atlas on Qwen3-8B / Llama-3.1-8B / Qwen3.6-27B (gated DeltaNet) / Mamba1-790M (state-space). Companion paper and experiment-script release in preparation.

If you want to use the SMT bridge for a different application (control barriers, neural ODE invariants, physics-informed regression), it's deliberately not coupled to the transformer hooks — `emltorch.eml_tree_to_smt2` works on any tree string the polish step emits.

---

## Status

| version | scope |
|---|---|
| **v0.2.0** (current) | Stable `fit` API, SMT bridge, Theorem-3 Lipschitz, gated DeltaNet, sklearn wrapper, 83 tests |
| v0.3.x (planned) | Cross-architecture cert atlas examples; Mamba1/Mamba2 walk-through; cleaner `interp.*` API |
| v1.0.0 (planned) | API freeze; deprecation of legacy gradient-trainer entry points; reproducible benchmark suite |

API guarantee: anything imported from the top-level `emltorch` package is considered stable for v0.x patch releases; lower-level entry points (`emltorch.tree`, `emltorch.evolution`) may change.

---

## Citation

If you use `emltorch`, please cite both:

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
  url     = {https://github.com/SabaPivot/emltorch},
  version = {0.2.0},
}
```

---

## Contributing & issues

Bug reports and feature requests welcome at [github.com/SabaPivot/emltorch/issues](https://github.com/SabaPivot/emltorch/issues). For substantive PRs, please open an issue first to discuss.

## License

MIT. See [`LICENSE`](LICENSE).

## Acknowledgments

The EML operator and the universality proof are due to Andrzej Odrzywolek (arXiv:2603.21852). The Theorem-3 attention-block local Lipschitz primitive follows Yudin et al. 2025 (arXiv:2507.07814). Gated DeltaNet effective-weight derivation builds on Yang et al. (ICLR 2025, arXiv:2412.06464).
