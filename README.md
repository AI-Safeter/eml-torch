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

Numerical EML clips exponential and logarithm arguments. SMT export describes
real formula obligations; it does not bound floating-point or approximation
error.

MIT. The EML operator and universality construction are due to
[Andrzej Odrzywołek](https://arxiv.org/abs/2603.21852).
