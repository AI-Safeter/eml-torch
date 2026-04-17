# emltorch

**GPU-batched symbolic regression via the EML operator `exp(x) − ln(y)`.**

A PyTorch library for discovering compact closed-form expressions from data.
Particularly suited for mechanistic interpretability of neural networks.

Based on Andrzej Odrzywolek's work
[*All elementary functions from a single binary operator*](https://arxiv.org/abs/2603.21852)
(arXiv:2603.21852, 2026), with a novel search algorithm that recovers
formulas the reference implementations could not reach.

## Why emltorch?

- **Works at depth 3-4** — where gradient-based EML trainers (including
  the paper's own reference implementation at `VA00/SymbolicRegressionPackage`
  and `cool-japan/oxieml`) collapse to constants
- **Seconds, not hours** — recovers `ln(x)` at depth 3 in 0.0 seconds
  via GPU-batched random + evolutionary search
- **One-line API** — `emltorch.fit(x, y, depth=3)` returns a human-readable
  formula
- **Designed for interp** — `emltorch.interp.from_transformer_hook` plugs
  into transformer activations directly (just PCA-reduce first)
- **sklearn-compatible** — `emltorch.sklearn.EMLRegressor` drops into any
  scikit-learn pipeline with `fit` / `predict` / `score`

## Install

```bash
# From source (v0.1.0 — dev install)
pip install -e /path/to/emltorch
```

## Quick start

```python
import torch
import emltorch as eml

x = torch.linspace(0.5, 5.0, 512)
y = torch.log(x)

result = eml.fit(x, y, depth=3)
print(result.expression)
# '+0.0000 + (+1.0000) * [eml(1, eml(eml(1, x), 1))]'
print(f"R² = {result.r2:.4f}, time = {result.time_s:.2f}s")
# R² = 1.0000, time = 0.53s
```

## Interpretability: transformer activations in one call

```python
import emltorch

# 1. Pull (N, D_model) activations + (N,) scalar targets from an HF model.
acts, tgt = emltorch.interp.from_transformer_hook(
    model, layer=22, inputs=prompts, target="logits"
)

# 2. Reduce to a handful of features (PCA, refusal direction, etc.),
#    then fit a symbolic formula.
from sklearn.decomposition import PCA
import torch
feats = torch.from_numpy(PCA(n_components=4).fit_transform(acts.numpy())).float()
result = emltorch.fit(feats.T, tgt.float(), depth=3)
print(result.expression)
```

See `examples/refusal_circuit.py` for a full end-to-end recipe.

## scikit-learn wrapper

```python
from emltorch.sklearn import EMLRegressor

model = EMLRegressor(depth=3, device="cuda:0")
model.fit(X, y)             # X: (N, V) numpy / tensor
print(model.expression_)    # discovered formula
print(model.score(X, y))    # R^2
```

`EMLRegressor` drops into `GridSearchCV`, `cross_val_score`, etc.
scikit-learn is an optional dependency; the wrapper falls back to a
minimal `BaseEstimator` shim if it isn't installed.

## Benchmark vs reference implementations

|   Target   | Depth | Paper claim | `VA00` | `oxieml` | **emltorch** |
|------------|-------|-------------|--------|----------|--------------|
| `exp(x)`   | 1     | 100%        | ✓      | ✓        | ✓ 0.2s       |
| `e − x`    | 2     | 100%        | ✓      | ✓        | ✓ 0.0s       |
| `ln(x)`    | 3     | ~25%        | ✗      | ✗*       | **✓ 0.0s**   |
| `−x`       | 4     | ~25%        | ✗      | ✗*       | **✓ 0.0s**   |
| `x · y`    | 5     | <1%         | ✗      | ✗*       | ≈ R²=0.96    |

\* oxieml README and tests only exercise depth ≤ 2

## How it works

1. **Peaked one-hot init** — each restart initialized to a specific random
   tree structure (not uniform softmax). Diversity without noise.
2. **Affine wrapper** — every expression is `a + b · EML(x)`, letting
   topology that's only *close* to correct match the target after rescaling.
3. **Evolutionary refinement** — keep top 10% of population by R²,
   mutate their edges, repeat. GPU-batched over thousands of trees in
   parallel.
4. **Skip gradient over topology** — our key diagnostic showed that
   Adam over softmax-relaxed topology (the paper's approach) always
   collapses to the constant `e`. We sidestep this entirely.

Read the full derivation in
[`docs/method.md`](docs/method.md).

## Status

- **v0.1.0**: research-grade, tested on known elementary functions
- **v0.2.0** (current): transformer hook plumbing (`emltorch.interp`),
  sklearn-compat `EMLRegressor`
- **v1.0.0** (planned): verification bridges to `uninum` / `emlvm`,
  SRBench benchmarks, paper submission

## Citation

If you use emltorch in research, please cite both:

```bibtex
@article{odrzywolek2026eml,
  title   = {All elementary functions from a single binary operator},
  author  = {Odrzywolek, Andrzej},
  journal = {arXiv preprint arXiv:2603.21852},
  year    = {2026}
}

@software{emltorch2026,
  title  = {emltorch: GPU-batched symbolic regression via EML},
  author = {Hong, Samuel},
  year   = {2026},
  url    = {https://github.com/samuelhong-newmes/emltorch}
}
```

## License

MIT.
