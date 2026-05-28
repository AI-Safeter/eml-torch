# emltorch

GPU-batched symbolic regression via the EML operator `eml(x, y) = exp(x) − ln(y)`. Built on Odrzywolek, [arXiv:2603.21852](https://arxiv.org/abs/2603.21852) (March 2026) — universal for elementary functions.

![EML formula vs Qwen3.6-27B factual-recall data](examples/h31_blackbox_cert/outputs/headline_figure.png)

## A closed-form formula for Qwen3.6-27B factual recall — found black-box

```
P_target ≈ 0.5954 − 0.1353 · eml(L, eml(L − H, 1))
```

`L` = induction lag, `H` = top-50 logprob entropy. Discovered by depth-4 evolutionary search using only `prompt → top-K logprobs` (no hooks, no hidden states). HELDOUT R² = 0.89 on a 75-25 split, with a portable `.smt2` cert that z3 and cvc5 dual-verify in single-digit milliseconds. Reproduction: `examples/h31_blackbox_cert/`.

## Where EML wins vs loses

| Benchmark | EML | Best baseline | Verdict |
|---|---|---|---|
| `exp(a·b)`, n=300, use_mul=True | depth-1 finds `eml((a*b), 1)` exactly, R²=**1.0000**, 1 node | poly K=5: R²=0.9997, 21 terms | **EML wins R² AND parsimony** |
| Qwen3.6-27B factual recall (top-K only) | depth-4 formula, R²=0.89, **dual-verified `.smt2` cert** | no other SR tool ships portable certs | **only EML produces a verifiable formula** |
| Gemma-4-31B induction probe, n=432, 10 seeds | 5 nodes, R²=0.937, **9/10 seeds identical** | PySR: 10 nodes, R²=0.953, 1/10 identical | **parity on R², EML wins reproducibility** |
| Feynman 8-equation subset (analytic targets) | mean R²=0.920, 0.47 s | poly K=5: 0.989 (<1 ms), PySR: 0.979 (23 s) | **EML loses R² on every equation**; ~49× faster than PySR |

![EML structural recovery on exp(a·b)](examples/srbench_feynman/figure_eml_wins_v2.png)

EML is **not a general-purpose accuracy-first SR engine** — for analytic targets in a polynomial's expressive class, low-degree OLS wins on R² *and* speed. EML's niche is **closed-form, reproducible, formally-verifiable symbolic predictions** of LLM/probe behavior. Full benchmark tables and figures in `examples/srbench_feynman/`.

## Library

```python
import emltorch as eml

result = eml.fit(x, y, depth=3)
print(result.expression)   # "+0.0000 + (+1.0000) * [eml((a * b), 1)]"
print(result.r2)
```

- `eml.fit_multi_seed(x, y, n_seeds=10)` — N independent fits + identical-expression-rate (the reproducibility axis).
- `eml.fit_pareto(x, y, depths=(1,2,3,4,5))` — accuracy/complexity Pareto front; `.best()`, `.select(max_complexity=k)`, `.predict(x)`.
- `eml.fit_residual_boost(x, y, n_stages=3)` — gradient-boosting-style additive EML stages; per-stage tree is still SMT-translatable.
- `eml.fit(..., polish=True, polish_optimizer="lbfgs")` — quasi-Newton constant refinement.

## When NOT to use this

- Raw HELDOUT R² on smooth analytic targets → polynomial OLS or PySR.
- Categorical or modular targets → a different SR engine.
- You don't need a portable formal certificate of the discovered formula → PySR.

## License

MIT. EML operator: Odrzywolek, [arXiv:2603.21852](https://arxiv.org/abs/2603.21852).
