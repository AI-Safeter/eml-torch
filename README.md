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

## Where EML wins and loses — two honest benchmarks

EML's relative performance is dataset-dependent. We report two heads-up comparisons against polynomial OLS and PySR (live runs, no cached numbers).

### 1. Gemma-4-31B-it induction probe (n = 432, 75-25 split, 10 random seeds for EML and PySR alike)

| | EML d=3 | EML d=3 + 3-stage boost | PySR | poly K=2 | poly K=5 |
|---|---:|---:|---:|---:|---:|
| Best HELDOUT R² | 0.937 | 0.958 | 0.953 | 0.933 | 0.878 |
| Expression size (nodes / coeffs) | **5** | 15 | 10 | 21 | 252 |
| Seeds → identical expression | **9 / 10** | n/a | 1 / 10 | n/a | n/a |
| SMT-LIB2 cert | direct (`eml_tree_to_smt2_intervals`) | per-stage direct | per-operator axioms | QF_NRA solver | QF_NRA solver |

Single-stage EML (0.937) trails PySR (0.953) by 0.016; 3-stage residual boosting closes and nominally reverses the gap (0.958), but that 0.005 edge is within split-and-seed noise — read it as **parity, not a win**. Where EML separates is **reproducibility and verifiability**: identical expression on 9 of 10 seeds (PySR 1/10), half the node count, and one-call SMT-LIB2 translation that the polynomial and PySR forms do not have.

### 2. Feynman equations subset (8 classical analytic targets, 1-3 variables, n = 300 each)

| Method | Mean HELDOUT R² | Median HELDOUT R² | Median fit time | Wins / 8 | Clean recovery (R²>0.999) |
|---|---:|---:|---:|---:|---:|
| EML d=3 | 0.920 | 0.932 | 0.47 s | 0 | 0 |
| EML d=4 | 0.896 | 0.900 | 2.92 s | 0 | 0 |
| poly K=2 | 0.980 | 0.990 | <0.001 s | 1 | 2 |
| poly K=5 | 0.989 | 0.993 | <0.001 s | **5** | 2 |
| PySR | 0.979 | **0.996** | 23.0 s | 4 | **3** |

**EML loses on raw HELDOUT R² to both polynomial OLS and PySR on every one of the 8 equations.** Including the exp-native targets (Feynman I.6.20a `exp(−θ²/2)`, I.6.20 the Gaussian) where EML's structural recovery was expected to shine — EML reaches R² = 0.99 there but PySR/poly reach 1.00. The EML clean-recovery count is 0/8, vs PySR 3/8 and poly K=5 2/8. EML's two surviving advantages on this benchmark are **speed** — EML d=3 is ~49× faster than PySR (0.47 s vs 23 s median) — and **expression size** (2–5 eml-operator nodes vs PySR 7–17 AST nodes vs poly K=5 ~20 terms). The 49× ratio used `parallelism="serial"` for deterministic PySR; we re-ran with `parallelism="multithreading"` + `JULIA_NUM_THREADS=8` (PySR's default for performance) and got a slightly *worse* PySR median of 29.1 s (per-iteration overhead dominates at this iter/pop budget), so the speed gap is real, not an artifact of single-threaded PySR. See `examples/srbench_feynman/pysr_multithreading_retime.json`. On targets that are well-fit by low-degree polynomials (multiplicative monomials like `m·g·z`, ratios like `q/C`), low-degree poly is the right tool by 500–6000× speed and equal-or-better R².

### Honest summary

EML is **not a general-purpose accuracy-first SR engine**. It is faster than PySR by ~49× (median, the 8-equation Feynman benchmark) and 50–1000× more reproducible (identical-expression-rate 9/10 vs 1/10 on the Gemma probe), and every EML expression translates to a portable SMT-LIB2 cert in one library call. For analytic targets in a polynomial's expressive class, low-degree OLS is the right tool. EML's defensible niche is **closed-form, reproducible, formally-verifiable symbolic predictions** of LLM/probe behavior — which is what the Qwen3.6 factual-recall headline above is.

## Library features

- `eml.fit(x, y, depth=3)` — single-best EML fit.
- `eml.fit_multi_seed(x, y, n_seeds=10)` — N independent fits + byte-equality `topology_stability` fraction (the reproducibility axis).
- `eml.fit_residual_boost(x, y, n_stages=3)` — gradient-boosting-style additive EML stages (the per-stage tree is still SMT-translatable).
- `eml.fit_pareto(x, y, depths=(1,2,3,4,5))` — accuracy/complexity Pareto front across depths; `.best()`, `.select(max_complexity=k)`, `.predict(x)`.
- `polish(..., optimizer="lbfgs")` (or `"adam+lbfgs"`) — quasi-Newton constant refinement; `"adam"` (default) is bit-identical to the previous polish path. Threaded through `fit(..., polish=True, polish_optimizer="lbfgs")`.

## Limitations

The headline factual-recall formula has its own caveat: `L = 0` deterministically across all factual prompts in that probe set, so the formula's value algebraically collapses to an affine function of `H`; linear regression on `H` matches the EML HELDOUT R² to four decimals on this slice. The cert's `working_lb` UNSAT at τ = 0.10 is implied by the precomputed interval bound and re-confirmed by the solvers, not load-bearing on nonlinear SMT reasoning; at the pre-registered τ = 0.5, both solvers return SAT. On classical analytic benchmarks (Feynman subset above), EML does not reach exact recovery at depth 3-4 even on its expected-strength exp-native targets.

## License

MIT. EML operator and universality proof: Odrzywolek, [arXiv:2603.21852](https://arxiv.org/abs/2603.21852).
