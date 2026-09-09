# emltorch

GPU-batched symbolic regression with `eml(x, y) = exp(x) - log(y)`.

Keep checkouts, environments, and experiment outputs inside this repository.
Run outputs default to the ignored `.artifacts/` directory; the local Gemma
environment is `.venv-gemma/`. Use `.artifacts/<experiment>` for explicit output
paths too. Historical results retain their original paths and source hashes.

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

The [complete Gemma MLP replacement study](gemma_mechanisms/README.md)
removes the original block from inference and compares compositional EML,
matched-depth SiLU, and linear replacements. It also tests arithmetic
feature hypotheses with residual and KV interventions. See its frozen
acceptance criteria and [held-out results](gemma_mechanisms/results/REPORT.md).
The selected EML replacement failed the quality gate; expansion to several
blocks stopped. The report explains the measured capacity and input-sufficiency
limits without claiming a consistent EML advantage or a recovered arithmetic algorithm.

The [full-width architecture screen](gemma_architecture/README.md) compares
bottleneck, affine-shortcut, and structured replacements with matched EML and
SiLU budgets. Removing the decoder restriction modestly improves reconstruction
for both activations, but no candidate passes the development quality screen.
Fresh confirmation remains unopened; the report includes installed-path timing,
capacity bounds, and explicit limits on the conclusion.
