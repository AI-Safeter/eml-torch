# Method

## The EML operator

The Elementary Mathematical Library (EML) operator is the single binary
primitive

```
eml(x, y) = exp(x) − ln(y)
```

introduced by Andrzej Odrzywolek in
[*All elementary functions from a single binary operator*](https://arxiv.org/abs/2603.21852)
(arXiv:2603.21852, 2026). Odrzywolek proves that the elementary
functions (polynomials, exp, log, trig, hyperbolic, etc.) can be
represented as finite compositions of `eml(·, ·)` applied to a single
free variable `x` and the constant `1`. Two corollaries used
throughout this library:

```
eml(x, 1)               = exp(x)
eml(1, 1)               = e
eml(1, eml(eml(1,z),1)) = e − ln(z)         # depth 3 ln
ReLU(z)                 = exact depth-4 EML
```

## `safe_eml`

The naive operator over-/underflows for moderate inputs. `safe_eml`
clamps both arguments to keep `exp(x)` and `ln(y)` in a numerically
sane range and returns finite values on all inputs. This is the
function used inside every tree forward pass.

## Search pipeline

Symbolic regression over EML trees runs in two stages:

1. **Evolutionary search** (`emltorch.evolve`). A population of B
   one-hot depth-`d` EML trees is evaluated in a single batched GPU
   forward via `BatchedEMLTree`. Fitness is MSE plus a range penalty.
   Each generation: select top `elite_fraction`, mutate one node per
   child, optionally crossover. A `best_ever` snapshot tracks the
   lowest raw MSE seen (not fitness — they differ when the range
   penalty is non-zero).
2. **Polish** (`emltorch.polish`). Adam refines the leaf `1`-constants
   and the affine wrapper `a + b · tree(x)` over the topology fixed by
   evolution. Two safeguards: polish never returns a worse tree than
   the warm-start, and any NaN/Inf reverts to the warm-start.

Depth escalation `d3 → d_max` plants the best lower-depth tree into
the left subtree slot of a higher-depth template — required at d ≥ 5.

## SMT-LIB2 bridge

`emltorch.smt.eml_tree_to_smt2` translates a polished tree to portable
SMT-LIB2 with `eml(L, R) → (- (Exp L) (Ln R))` over axiomatized
function symbols `Exp` and `Ln`. The axiom block declares positivity,
`Exp(0) = 1`, `Ln(1) = 0`, monotonicity, the inverse axioms, signed
corollaries, and the interval `e ∈ [2.7182, 2.7183]`. Named lemmas
(`EML_LEMMAS`, `with_lemmas`) include multiplicativity of `Exp`, the
ratio corollary `Exp(u + 1) ≥ 2.5 · Exp(u)`, and `Ln`
multiplicativity.

The emitted `.smt2` is portable — no transcendental build dependency
is required. Each cert is dual-verified by z3 (≥ 4.16) and cvc5
(≥ 1.3); discharged cases UNSAT in single-digit milliseconds.

For 768-dim L_∞ ball certs (e.g. SAE features), use the QF_LRA
interval-propagation variant `eml_tree_to_smt2_intervals` which
pre-computes per-node bounds and emits linear assertions only.
