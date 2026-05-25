# emltorch

GPU-batched symbolic regression via the EML operator `eml(x, y) = exp(x) − ln(y)`. Built on Andrzej Odrzywolek, [arXiv:2603.21852](https://arxiv.org/abs/2603.21852) (March 2026), which proves EML is universal for elementary functions.

![EML formula vs Qwen3.6-27B factual-recall data](examples/h31_blackbox_cert/outputs/headline_figure.png)

## A closed-form formula for Qwen3.6-27B factual recall, found black-box

```
P_target ≈ 0.5954 + (−0.1353) · eml(L, eml(L − H, 1))
```

`L` = induction lag (0 on factual prompts).
`H` = entropy of the model's top-50 logprob distribution.

*Factual* probes here are completion prompts asking for a single canonical fact the model is expected to know — 50 prompts of the form `"The capital of {country} is"`, target = the canonical capital. The formula was discovered by depth-4 evolutionary search over the EML operator on this set. The pipeline uses only `prompt → top-K logprobs`: no hooks, no attention output, no hidden states. HELDOUT R² = 0.89 on a 75-25 random split. The formula renders to a portable `.smt2` certificate that z3 and cvc5 both verify in single-digit milliseconds. Reproduction package (scripts + shipped artifacts + pre-registration): `examples/h31_blackbox_cert/`.

## EML vs polynomial and PySR baselines

On a separate Gemma-4-31B-it induction probe (n = 432, identical 75-25 split, 10 random seeds for EML and PySR alike):

| | EML d=3 | PySR | poly K=2 | poly K=5 |
|---|---:|---:|---:|---:|
| Best HELDOUT R² | 0.937 | **0.953** | 0.933 | 0.878 |
| Median HELDOUT R² | 0.937 | 0.934 | — | — |
| Nodes / coefficients | **5** | 10 | 21 | 252 |
| Seeds → identical expression | **9 / 10** | 1 / 10 | n/a | n/a |
| SMT-LIB2 cert | direct (`eml_tree_to_smt2_intervals`) | per-operator axioms | QF_NRA solver | QF_NRA solver |

PySR matches the R² tier with about 2× longer expressions and a different form on every seed. EML loses ≈ 0.016 R² (best-of-10) but stays compact, stable across seeds, and translates to portable SMT-LIB2 in one library call. The `emltorch.fit_multi_seed(x, y, n_seeds=10)` API operationalizes this: it runs N seeds, byte-equality-counts the resulting expressions, and reports a `topology_stability` fraction — the empirical axis on which EML separates from polynomial-degree symbolic regression.

When single-stage EML hits a local optimum that misses informative features (e.g. Gemma-4 induction at pool size L ∈ [21, 25], where stage-1 converges to `eml(n_rep, n_rep)` and drops the rest), gradient-boosting-style residual fitting recovers them. `emltorch.fit_residual_boost(x, y, n_stages=3)` fits a sequence of EML trees on running residuals and returns their additive sum; on the Gemma L_large slice this lifts HELDOUT R² from 0.78 to 0.82 (3 stages) — second and third stages discover the L and entropy structure that stage 1 dropped, in symbolic form, still SMT-translatable per stage.

## Limitations

`L = 0` deterministically across all factual prompts in this probe set, so the formula's value algebraically collapses to an affine function of `H`; linear regression on `H` matches the EML formula's HELDOUT R² to four decimals on this slice. The cert's `working_lb` UNSAT at τ = 0.10 is implied by the precomputed interval bound and re-confirmed by the solvers, not load-bearing on nonlinear SMT reasoning; at the pre-registered τ = 0.5, both solvers return SAT.

## License

MIT. EML operator and universality proof: Odrzywolek, [arXiv:2603.21852](https://arxiv.org/abs/2603.21852).
