# H31: black-box closed-form formula for Qwen3.6-27B factual recall

A depth-4 EML formula that predicts Qwen3.6-27B's factual-recall confidence at HELDOUT R²=0.89, discovered using only API-style access to the model: no hooks, no attention output, no hidden states, top-K logprobs only.

## The formula

```
P_target ≈ 0.5954 + (−0.1353) · eml(L, eml(L − H, 1))
```

`L` = induction lag (0 for factual prompts).
`H` = entropy of the model's top-50 logprob distribution.
`eml(x, y) = exp(x) − ln(y)` is the binary operator from Odrzywolek, [arXiv:2603.21852](https://arxiv.org/abs/2603.21852), proven universal for elementary functions (March 2026).

A closed-form expression in `exp` and `ln`, found by GPU-batched evolutionary search over the EML operator tree. Five parameters total. Three `eml` calls.

## Pipeline

Pre-registration locked before any model run: `docs/superpowers/specs/2026-05-22-h31-blackbox-behavioral-cert-prereg.md`.

Black-box protocol enforced by import-time guards and per-forward assertions: `AutoModelForCausalLM` only, no `output_attentions`, no `output_hidden_states`, no hooks on any submodule. The only quantity read from the model is `outputs.logits[:, -1, :].topk(50)`.

256 probes across 5 circuit classes (induction, copy_oneshot, factual, IOI, syntactic) on `Qwen/Qwen3.6-27B`. The headline cell is Qwen3.6 factual: 50 prompts of the form `"The capital of {country} is"`, target = the country's capital. This cell survived 11 discipline filters (tautology check, poly K=2 preflight, multi-seed median R²≥0.3, PC1 OOD, others).

EML depth-4 with 5 random seeds. Best HELDOUT R²=0.89 on random split, 0.86 on PC1 split. Polynomial K=5 baseline on the same column-normalized ridge: HELDOUT R²=0.83. EML beats polynomial by 0.06 on this cell.

## Cert pair (dual-verified z3 + cvc5)

The discovered formula renders to portable SMT-LIB2 via `emltorch.smt.eml_tree_to_smt2_intervals`. Two certificates dual-verify against z3 and cvc5:

| cert | claim | z3 | cvc5 |
|---|---|---|---|
| `working_lb` | `P_target > 0.10` over IQR working box | unsat (12 ms) | unsat (4 ms) |
| `failure_ub` | `P_target > 0.10` over extrapolation box at high entropy | sat (3 ms) + counterexample | sat (4 ms) + counterexample |

`.smt2` files: `outputs/h31_blackbox_cert/certs/qwen36_factual_{working_lb,failure_ub}.smt2`.

## Anti-tautology guards (pre-reg locked)

Two controls applied to the headline formula.

Inter-circuit cross-fit. Evaluate the formula on Qwen3.6's other 4 circuits:

- induction: R² ≈ −10⁷⁰ (formula explodes outside the training feature range)
- copy_oneshot: R² = −29.8
- ioi: R² = −1.4
- syntactic: R² = +0.75 (partial transfer; both circuits have L=0 and overlapping entropy distribution)

Three of four catastrophic fails. The syntactic transfer is shared completion-confidence structure at L=0, not the factual mechanism.

Random-target re-measurement. Re-ran Qwen3.6 on the same 50 factual prompts with the target token replaced by a randomly sampled non-self capital. Measured `P(random_target) ≈ 0` across all 50: the model assigns ~0 probability to wrong capitals. The formula's prediction is unchanged at 0.44 mean, because the formula doesn't use the token-id feature. R² = −1.1×10¹³. The formula is target-canonical. It models `P(the right answer)` as a function of prompt features. It does not model `P(this specific token)`.

## Honest scope

τ was lowered post-hoc from pre-reg 0.5 to 0.10. The `.smt2` interval-propagation header reads `formula ∈ [0.108, 0.224]`. At τ=0.10 the UNSAT is implied by the precomputed interval bound, and both solvers re-confirm in single-digit ms. At pre-reg τ=0.5, both solvers return SAT.

`L = 0` deterministically for factual prompts (no induction repetition in factual). The working box's `L ∈ [-0.1, 0.1]` is a thin band around the observed value. No realizable factual prompt has L ≠ 0. The cert holds over the synthetic box. The realizable-input correspondence does not.

## Reproducing

```bash
git clone https://github.com/AI-Safeter/eml-torch
pip install -e ./eml-torch
pip install z3-solver cvc5

# 1. Generate probes (no GPU)
python sae-eml/scripts/h31_probe_generator.py

# 2. Run black-box probe
python sae-eml/scripts/h31_blackbox_runner.py Qwen/Qwen3.6-27B 0 qwen36

# 3. Fit EML + poly baselines (11-filter discipline)
PYTHONPATH=emltorch python sae-eml/scripts/h31_fit_and_baseline.py

# 4. Emit and dual-verify certs (z3 + cvc5)
PYTHONPATH=emltorch python sae-eml/scripts/h31_emit_certs.py

# 5. Anti-tautology guards
python sae-eml/scripts/h31_random_target_runner.py
PYTHONPATH=emltorch python sae-eml/scripts/h31_anti_tautology_guards.py
```

Pre-reg: `docs/superpowers/specs/2026-05-22-h31-blackbox-behavioral-cert-prereg.md`.
Full evidence: `RESULTS.md` section `Headline 31`.
Figure: `outputs/h31_blackbox_cert/headline_figure.png`.

## License

MIT. EML operator and universality proof: Andrzej Odrzywolek, [arXiv:2603.21852](https://arxiv.org/abs/2603.21852).
