# emltorch

GPU-batched symbolic regression via the EML operator `eml(x, y) = exp(x) − ln(y)`. Built on Andrzej Odrzywolek, [arXiv:2603.21852](https://arxiv.org/abs/2603.21852) (March 2026), which proves EML is universal for elementary functions.

![H31 figure](examples/h31_blackbox_cert/outputs/headline_figure.png)

## H31: a closed-form formula for Qwen3.6-27B factual recall, found black-box

```
P_target ≈ 0.5954 + (−0.1353) · eml(L, eml(L − H, 1))
```

`L` = induction lag (0 on factual prompts).
`H` = entropy of the model's top-50 logprob distribution.

The formula was discovered by depth-4 evolutionary search over the EML operator on 50 prompts of the form `"The capital of {country} is"`. The pipeline uses only `prompt → top-K logprobs`: no hooks, no attention output, no hidden states. HELDOUT R² = 0.89 on a 75-25 random split. The formula renders to a portable `.smt2` certificate that z3 and cvc5 both verify in single-digit milliseconds. Reproduction package (scripts + shipped artifacts + pre-registration): `examples/h31_blackbox_cert/`.

## Limitations

`L = 0` deterministically across all factual prompts in this probe set, so the formula's value algebraically collapses to an affine function of `H`; linear regression on `H` matches the EML formula's HELDOUT R² to four decimals on this slice. The cert's `working_lb` UNSAT at τ = 0.10 is implied by the precomputed interval bound and re-confirmed by the solvers, not load-bearing on nonlinear SMT reasoning; at the pre-registered τ = 0.5, both solvers return SAT.

## License

MIT. EML operator and universality proof: Odrzywolek, [arXiv:2603.21852](https://arxiv.org/abs/2603.21852).
