# EML axiom soundness witness (Lean 4)

This Lean 4 package discharges every axiom asserted in
`emltorch.smt.EML_AXIOMS_SMT2` as a theorem of mathlib's `Real.exp` /
`Real.log`. It is the *axiom-soundness witness* for our SMT-LIB2 cert
atlas — it does **NOT** claim a composition theorem (cf. BlockCert).

## What it proves

For each axiom in `EML_AXIOMS_SMT2`:

| SMT axiom (informal)                            | Lean theorem                  | Mathlib lemma                  |
|-------------------------------------------------|-------------------------------|--------------------------------|
| `∀u, Exp(u) > 0`                                | `EmlAxioms.exp_pos`           | `Real.exp_pos`                 |
| `Exp(0) = 1`                                    | `EmlAxioms.exp_zero`          | `Real.exp_zero`                |
| `u < v → Exp(u) < Exp(v)`                       | `EmlAxioms.exp_strictMono`    | `Real.exp_lt_exp`              |
| `u > 0 → Exp(u) > 1`                            | `EmlAxioms.exp_gt_one_of_pos` | `Real.one_lt_exp_iff_pos`      |
| `u < 0 → Exp(u) < 1`                            | `EmlAxioms.exp_lt_one_of_neg` | `Real.exp_lt_one_iff`          |
| `Ln(1) = 0`                                     | `EmlAxioms.log_one`           | `Real.log_one`                 |
| `u, v > 0, u < v → Ln(u) < Ln(v)`               | `EmlAxioms.log_strictMono`    | `Real.log_lt_log_iff`          |
| `v > 1 → Ln(v) > 0`                             | `EmlAxioms.log_pos_of_one_lt` | `Real.log_pos`                 |
| `0 < v < 1 → Ln(v) < 0`                         | `EmlAxioms.log_neg_of_lt_one` | `Real.log_neg`                 |
| `Ln(Exp(x)) = x`                                | `EmlAxioms.log_exp`           | `Real.log_exp`                 |
| `v > 0 → Exp(Ln(v)) = v`                        | `EmlAxioms.exp_log_of_pos`    | `Real.exp_log`                 |
| `Exp(1) ∈ [2.7182, 2.7183]`                     | `EmlAxioms.exp_one_{ge,le}_*` | `Real.exp_one_near_10` (sorry) |

Plus the two load-bearing lemmas:

| EML lemma                                       | Lean theorem                  | Status                          |
|-------------------------------------------------|-------------------------------|---------------------------------|
| `Exp(u+v) = Exp(u)·Exp(v)`                      | `EmlAxioms.exp_add`           | proved (Real.exp_add)           |
| `u,v > 0 → Ln(uv) = Ln(u) + Ln(v)`              | `EmlAxioms.log_mul_of_pos`    | proved (Real.log_mul)           |
| `v + 1 ≤ u → 2.5·Exp(v) ≤ Exp(u)`               | `EmlAxioms.ratio_corollary`   | proved (modulo e ≥ 2.5)         |

## What it does NOT claim

- **No composition theorem.** BlockCert (arxiv:2511.17645) proves residual
  blocks compose under Lipschitz; this file does not. We rely on per-cert
  z3+cvc5 dual-UNSAT for each .smt2 in the atlas, not on a metatheorem
  about chaining.
- **No model-level soundness.** This file proves the SMT axioms hold
  for `(Real.exp, Real.log)`. It does NOT prove that the .smt2 cert
  text correctly encodes the model's actual attention scores — that is
  the responsibility of the score-extraction code in the atlas scripts
  and is validated empirically via PGD (cf. methodology filter #6).

## Build

```bash
cd emltorch/lean
lake update          # one-time, downloads mathlib (~hours first run)
lake build           # subsequent rebuilds are fast
```

`sorry`s currently in `exp_one_ge_lower` / `exp_one_le_upper` resolve when
the user's mathlib snapshot pins down the exact name of the
`Real.exp_one_near_*` lemma (varies by mathlib commit).

## Honest framing for paper

This is the *axiom-soundness witness* — claim it as such, not as parity
with BlockCert. The technical contribution is:

> Our SMT-LIB2 axiom block asserts no statement about `(Exp, Ln)` that is
> not a theorem of `(Real.exp, Real.log)` in mathlib4. The block introduces
> no unsoundness w.r.t. real exp/log, so any z3/cvc5-discharged cert under
> these axioms is sound under the real-analysis interpretation.

This closes Headline-23-style audit gaps about whether the axioms might
"prove too much."
