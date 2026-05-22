# H31 — Black-box derivation of an elementary-function formula on Qwen3.6-27B

This example is the self-contained reproduction package for the H31 headline result described in the main `emltorch` README. Pre-registered protocol locked before any model run: `PREREG.md` in this directory.

## What's here

- `h31_probe_generator.py` — 256 black-box probes across 5 circuit classes.
- `h31_blackbox_runner.py` — hook-audited forward-only runner. Asserts no hooks, `output_attentions=False`, `output_hidden_states=False`.
- `h31_fit_and_baseline.py` — EML d=3/4 fit with 5 seeds, poly K=1/2/5 OLS baselines, full 11-filter discipline.
- `h31_emit_certs.py` — interval-propagation `.smt2` cert emit and dual-verify (z3 + cvc5).
- `h31_figure.py` — headline figure.
- `h31_random_target_runner.py` — anti-tautology guard: re-runs Qwen3.6 on the same 50 factual prompts with target replaced by a randomly sampled non-self capital.
- `h31_anti_tautology_guards.py` — runs the 3 pre-registered controls (inter-circuit cross-fit, random-target, vendor cross-fit) against the headline formula.
- `h31_h_only_refit.py` — H-only EML refit walk-back audit. Compares EML d=4 (10 seeds) to linear regression `a + b·H` and poly K=2/3/5 on `H` alone.
- `outputs/` — shipped artifacts from the original 2026-05-22 run: probes, measurements, fit results, cert verdicts, figure.
- `PREREG.md` — pre-registration locked before any model run.

## Reproducing

Requirements:

```bash
pip install emltorch z3-solver cvc5
# Or, from the eml-torch repo root:
pip install -e .. && pip install z3-solver cvc5
```

For Qwen3.6-27B inference, `transformers >= 5.7` and a CUDA-capable GPU with ~60 GB free.

Run from this directory:

```bash
# 1. Generate probes (no GPU)
python h31_probe_generator.py

# 2. Run black-box probe on Qwen3.6-27B
python h31_blackbox_runner.py Qwen/Qwen3.6-27B 0 qwen36

# 3. Fit EML + polynomial baselines
python h31_fit_and_baseline.py

# 4. Emit certs and dual-verify
python h31_emit_certs.py

# 5. Headline figure
python h31_figure.py

# 6. Anti-tautology guards (Qwen3.6 only for guards 1 and 2)
python h31_random_target_runner.py
python h31_anti_tautology_guards.py

# 7. Walk-back audit
python h31_h_only_refit.py
```

Each script writes to `outputs/` under this directory. The shipped artifacts there were produced by the original 2026-05-22 run.

## Headline formula

```
P_target ≈ 0.5954 + (−0.1353) · eml(L, eml(L − H, 1))
```

`L` = induction lag (0 for factual prompts).
`H` = entropy of the top-50 logprob distribution.
`eml(x, y) = exp(x) − ln(y)`.

Discovered via depth-4 EML evolutionary search on 50 prompts of the form `"The capital of {country} is"`, fit on a 75-25 random split. Best HELDOUT R² = 0.89 (random split, 5 seeds).

## Cert pair (dual-verified)

| cert | claim | z3 | cvc5 |
|---|---|---|---|
| `working_lb` | `P_target > 0.10` over IQR working box | unsat (12 ms) | unsat (4 ms) |
| `failure_ub` | `P_target > 0.10` over extrapolation box at high entropy | sat (3 ms) + counterexample | sat (4 ms) + counterexample |

`.smt2` files at `outputs/certs/qwen36_factual_{working_lb,failure_ub}.smt2`.

## Honest disclosures (from the main README)

`L = 0` deterministically for all 50 factual prompts. Substituting `L = 0` and simplifying, the formula reduces to `P ≈ 0.4601 − 0.1353·H` on the training manifold. Linear regression `P = 1.23 − 0.36·H` matches the EML formula's HELDOUT R² to 4 decimals (Δ = 0.0000). Poly K=2 in `H` alone gives HELDOUT R² = 0.897, slightly above EML. The contribution is the elementary-function symbolic form, not an R² advantage.

τ was lowered post-hoc from pre-reg 0.5 to 0.10 to demonstrate the cert pipeline. At τ=0.10 the `working_lb` UNSAT is implied by the precomputed interval bound (`formula ∈ [0.108, 0.224]` per the `.smt2` header). At pre-reg τ=0.5, both solvers return SAT.

Anti-tautology guard summary (from `outputs/anti_tautology_results.json`):

- Inter-circuit cross-fit: 3/4 catastrophic fails; syntactic R²=0.75 partial transfer due to shared L=0 + entropy distribution.
- Random-target re-measurement: R² = −1.1×10¹³ on true `P(random_target) ≈ 0`. The formula is target-canonical (predicts confidence assuming canonical target).
- Vendor cross-fit (not load-bearing per project scope): R² = −65 on Gemma-4-31B-it factual; formula is Qwen-specific.

## License and citation

MIT. EML operator and universality proof: Andrzej Odrzywolek, [arXiv:2603.21852](https://arxiv.org/abs/2603.21852).
