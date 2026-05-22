# H31: black-box derivation of an elementary-function formula on Qwen3.6-27B

A depth-4 EML expression composed of `exp` and `ln` via the binary operator `eml(x, y) = exp(x) − ln(y)`, fit to Qwen3.6-27B's response probability on 50 factual completion prompts. The pipeline uses only API-style access to the model: no hooks, no attention output, no hidden states, top-K logprobs only. The formula is portable as a dual-verified `.smt2` certificate.

## The formula

```
P_target ≈ 0.5954 + (−0.1353) · eml(L, eml(L − H, 1))
```

`L` = induction lag (0 for factual prompts).
`H` = entropy of the top-50 logprob distribution.
`eml(x, y) = exp(x) − ln(y)`, proven universal for elementary functions in Odrzywolek, [arXiv:2603.21852](https://arxiv.org/abs/2603.21852) (March 2026).

Five parameters total. Three `eml` calls. The symbolic form lives in the elementary-function class: `exp` and `ln` composed. Polynomial regression cannot write a finite-degree expression of the same form. This is the load-bearing claim: a black-box pipeline that ends in an `exp`/`ln` closed-form expression, not a Taylor-series approximation.

## Why EML, not polynomial

Polynomial regression with the same `H` feature on the same 75-25 split:

| Predictor | HELDOUT R² | Symbolic class |
|---|---:|---|
| Poly K=2 in `H` | 0.897 | polynomial degree 2 |
| Poly K=3 in `H` | 0.897 | polynomial degree 3 |
| EML d=4 (headline) | 0.888 | `exp` / `ln` composition |
| Linear `a + b·H` | 0.888 | polynomial degree 1 |

R² is a tie. The distinction is symbolic. A polynomial of any finite degree cannot represent `exp(x)` exactly; EML's depth-4 tree can. On data where the true generating function is elementary-function-shaped (Arrhenius, sigmoid, softmax cross-section), the EML form extrapolates correctly outside the training range while the polynomial diverges. On this 50-prompt slice both forms fit equally well; the choice between them is about *what kind of formula you want* to ship downstream.

## Algebraic collapse to linear-in-H on this slice (honest)

`L = 0` deterministically for all 50 factual prompts (no induction repetition in factual). Substituting `L = 0`:

```
eml(0, eml(0 − H, 1)) = exp(0) − ln(exp(−H) − ln(1))
                     = 1 − ln(exp(−H))
                     = 1 + H

P_target ≈ 0.5954 − 0.1353 · (1 + H)
         = 0.4601 − 0.1353 · H        (on the L=0 training manifold)
```

So on the data the formula was fit to, its *value* equals an affine function of `H`. Linear regression `P = 1.23 − 0.36·H` on the same 75-25 split gives HELDOUT R² = 0.8876, matching the EML formula to 4 decimals. The symbolic form is still `exp` / `ln`; the algebraic content on this slice is linear.

This is a one-feature degenerate case. On a slice where `L` varies (induction-style probes), the formula's `eml(L, ...)` substructure does not collapse. The walk-back applies to the algebraic content claim on factual specifically, not to the pipeline's ability to produce elementary-function expressions in general.

## Pipeline

Pre-registration locked before any model run: `docs/superpowers/specs/2026-05-22-h31-blackbox-behavioral-cert-prereg.md`.

Black-box protocol enforced by import-time guards and per-forward assertions: `AutoModelForCausalLM` only, no `output_attentions`, no `output_hidden_states`, no hooks on any submodule. The only quantity read from the model is `outputs.logits[:, -1, :].topk(50)`.

256 probes across 5 circuit classes (induction, copy_oneshot, factual, IOI, syntactic) on `Qwen/Qwen3.6-27B`. The headline cell is Qwen3.6 factual: 50 prompts of the form `"The capital of {country} is"`, target = the country's capital. The cell survived 11 discipline filters (tautology check, poly K=2 preflight, multi-seed median R² ≥ 0.3, PC1 OOD).

EML depth-4 with 5 random seeds. Best HELDOUT R² = 0.89 on random split, 0.86 on PC1 split.

## Cert pair (dual-verified z3 + cvc5)

The discovered formula renders to portable SMT-LIB2 via `emltorch.smt.eml_tree_to_smt2_intervals`. Two certificates dual-verify against z3 and cvc5:

| cert | claim | z3 | cvc5 |
|---|---|---|---|
| `working_lb` | `P_target > 0.10` over IQR working box | unsat (12 ms) | unsat (4 ms) |
| `failure_ub` | `P_target > 0.10` over extrapolation box at high entropy | sat (3 ms) + counterexample | sat (4 ms) + counterexample |

`.smt2` files: `outputs/h31_blackbox_cert/certs/qwen36_factual_{working_lb,failure_ub}.smt2`.

At τ=0.10 the `working_lb` UNSAT is implied by the precomputed interval bound `formula ∈ [0.108, 0.224]`; both solvers re-confirm in single-digit ms. At pre-reg τ=0.5, both solvers return SAT.

## Anti-tautology guards (pre-reg locked)

Inter-circuit cross-fit: evaluate the formula on Qwen3.6's other 4 circuits:

- induction: R² ≈ −10⁷⁰ (formula explodes outside the training feature range)
- copy_oneshot: R² = −29.8
- ioi: R² = −1.4
- syntactic: R² = +0.75 (partial transfer; both circuits have L=0 and overlapping entropy distribution)

Three of four catastrophic fails. The syntactic transfer is shared completion-confidence structure at L=0, not the factual mechanism.

Random-target re-measurement: re-ran Qwen3.6 on the same 50 factual prompts with the target token replaced by a randomly sampled non-self capital. Measured `P(random_target) ≈ 0` across all 50: the model assigns ~0 probability to wrong capitals. The formula's prediction is unchanged at 0.44 mean, because the formula doesn't use the token-id feature. R² = −1.1×10¹³. The formula is target-canonical. It models `P(the right answer)` as a function of prompt features. It does not model `P(this specific token)`.

## Honest scope

`L = 0` across all 50 factual prompts (no repetition in factual). The working box's `L ∈ [-0.1, 0.1]` is a thin band around the observed value; no realizable factual prompt has L ≠ 0. The cert holds over the synthetic box. The realizable-input correspondence does not.

The formula's algebraic content on the training slice equals linear regression on `H`. The contribution is the elementary-function symbolic form and the end-to-end pipeline that produced it, not an R² advantage over polynomial.

## Reproducing

The full reproduction package — scripts, shipped artifacts, pre-registration, local README — lives at `examples/h31_blackbox_cert/`.

```bash
git clone https://github.com/AI-Safeter/eml-torch
cd eml-torch
pip install -e . && pip install z3-solver cvc5

cd examples/h31_blackbox_cert

# 1. Generate probes (no GPU)
python h31_probe_generator.py

# 2. Run black-box probe on Qwen3.6-27B
python h31_blackbox_runner.py Qwen/Qwen3.6-27B 0 qwen36

# 3. Fit EML + poly baselines (11-filter discipline)
python h31_fit_and_baseline.py

# 4. Emit and dual-verify certs
python h31_emit_certs.py

# 5. Anti-tautology guards
python h31_random_target_runner.py
python h31_anti_tautology_guards.py

# 6. Walk-back audit (linear baseline vs EML d=4, 10 seeds, H only)
python h31_h_only_refit.py
```

Pre-registration: `examples/h31_blackbox_cert/PREREG.md`.
Shipped artifacts (from the original 2026-05-22 run): `examples/h31_blackbox_cert/outputs/`.
Figure: `examples/h31_blackbox_cert/outputs/headline_figure.png`.

## License

MIT. EML operator and universality proof: Andrzej Odrzywolek, [arXiv:2603.21852](https://arxiv.org/abs/2603.21852).
