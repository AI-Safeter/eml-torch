# emltorch

GPU-batched symbolic regression with `eml(x, y) = exp(x) - log(y)`.

```bash
pip install -e .
# Optional formula solvers:
pip install -e '.[smt]'
```

```python
import torch
import emltorch as eml

x = torch.linspace(-1, 1, 128)
result = eml.fit(x, torch.exp(x), depth=1, device="cuda")
print(result.expression)
prediction = result.predict(x)
```

The core provides batched tree search, constant polishing, repeated-seed fits,
residual fitting, Pareto selection, and formula export. Inputs accept NumPy
arrays, lists, or tensors; predictions use raw input coordinates, including
when fitting with normalization. CUDA is selected automatically when available.

For a trainable neural head with learned affine arguments:

```python
head = eml.EMLHead(32, 16, out_features=1, device="cuda")
prediction = head(torch.randn(64, 32, device="cuda"))
```

Each hidden unit uses `eml(left(x), 1 + right(x)**2)`, followed by a learned
readout and an optional linear residual. The head supports ordinary PyTorch
optimizers and `torch.compile`; it does not run symbolic tree discovery.

Numerical EML clips exponential and logarithm arguments. SMT export describes
real formula obligations; it does not bound floating-point or approximation
error.

MIT. The EML operator and universality construction are due to
[Andrzej Odrzywołek](https://arxiv.org/abs/2603.21852).

The [one-layer quantization-correction experiment](kv_correction/README.md) compares EML with linear and SiLU controls under a fixed stop rule. Earlier [arithmetic findings](research/README.md) and [KV compression results](kv_cache/RESULTS.md) remain as evidence archives; superseded runners are available through Git history.

The [Gemma Jacobian-lens study](jspace/README.md) provides a GPU-validated
selected-token estimator and a failed semantic-swap feasibility gate. It has
not established an EML benefit.
