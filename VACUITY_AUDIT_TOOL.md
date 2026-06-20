# `emltorch.certify.vacuity_audit` — does this concentration cert certify anything?

A drop-in, dual-solver test-suite that takes an attention-concentration /
softmax-mass / routing-cert claim and tells you whether its `UNSAT` is a
genuine concentration guarantee (`SOUND`) or one of three failure modes:
`VACUOUS`, `RELATIVE-ONLY`, or `UNDER-PRECISION` (plus the honest non-result
`NOT-CERTIFIED` when the cert simply does not discharge).

It operationalizes two 2026-06-19 audits
(`VACUITY_AUDIT_2026-06-19.md`, `CERT_SOUNDNESS_SWEEP_2026-06-19.md`) into four
tested checks so that **anyone** — not just the original authors — can re-run the
soundness test on a published cert. Every check runs z3 **and** cvc5
(`full-saturate-quant` baked in) and only trusts a definitive dual agreement.

## The four checks

1. **Shift-invariance.** A softmax-mass claim is invariant to adding a constant
   to every score (softmax is shift-invariant). The sound claim is
   `s_target − Ln(Σ exp s_j) > log τ` (⇔ `softmax_target > τ`). A
   *shift-variant* body such as the `v3` form `Exp(s_target) − Ln(sumE) > log τ`
   collapses on **log-prob** inputs: `sumE = Σ exp(log prob) = 1`, so
   `Ln(1) = 0` and the body becomes `Exp(s_target) > log τ < 0` — true for
   essentially any head. Detected structurally (form metadata) **and**
   empirically (verdict on `scores` vs `scores + c`). Failure → **VACUOUS**.

2. **Non-vacuity control.** A *known* non-concentrated control row must NOT
   discharge at the same τ. The audit runs the cert on (a) a uniform `1/T` row
   and (b) a low-mass row (target holds 5% of the mass). If **either**
   discharges, the cert would pass with no concentration. Failure → **VACUOUS**.
   (Generalizes the uniform-control guard already in `atlas.certified_radius`.)

3. **Relative-vs-absolute / mass floor.** If the cert excludes keys (e.g. BOS,
   self) from its denominator without an absolute floor, a target holding 9% —
   or 1e-9 — of *total* mass can still discharge at τ=0.95 because the real mass
   sits on the excluded keys (the H23 gated/SSM raw-weight defect). The audit
   recomputes the target's share of **total** mass (excluded keys **in** the
   denominator) and, when the cert discharges, requires it to meet τ. Failure →
   **RELATIVE-ONLY**.

4. **Numerical-precision floor.** A cert finer than the model's own numerical
   noise certifies nothing physical. The audit finds the certified L∞ radius
   (via the sound `softmax_interval` ladder) and requires it to exceed
   `precision_floor` (e.g. `2**-8 ≈ 0.0039` for bf16 round-off). Radius
   at/below the floor → **UNDER-PRECISION**.

**Verdict precedence (worst wins):**
`VACUOUS` > `RELATIVE-ONLY` > `UNDER-PRECISION` > `SOUND`.
A cert that does not discharge at all is `NOT-CERTIFIED` (an honest non-result,
not a vacuous pass). A check that is not applicable (no `precision_floor`, no
excluded keys, or a cert that did not discharge) is recorded as
passed-and-skipped and never downgrades the verdict.

## Python API

```python
from emltorch.certify import vacuity_audit, audit_attention_atlas
import math

# A head whose log-prob attention row puts 10% mass on the "target" key
# (the median of the H19 atlas corpus). BOS holds 45%, self 30%.
scores = [math.log(p) for p in (0.45, 0.05, 0.10, 0.05, 0.05, 0.30)]

report = vacuity_audit(scores, target_idx=2, tau=0.95, form="v3")
print(report.verdict)        # 'VACUOUS'
print(report.explanation)    # per-check PASS/FAIL with reasons

# The sound form refuses the same head (it isn't 95%-concentrated):
report = vacuity_audit(scores, target_idx=2, tau=0.95, form="softmax_interval")
print(report.verdict)        # 'NOT-CERTIFIED'

# A genuinely concentrated head clears all four checks:
sound = [math.log(p) for p in (0.005, 0.005, 0.99)]
report = vacuity_audit(sound, target_idx=2, tau=0.95,
                       form="softmax_interval", precision_floor=0.001)
print(report.verdict)        # 'SOUND'
```

`vacuity_audit(scores, target_idx, tau, *, form="softmax_interval",
exclude_from_sum=None, precision_floor=None, control="uniform", rho=0.005,
rhos=..., timeout_ms=8000) -> AuditReport`. The `AuditReport` exposes one
`CheckResult` per check (`shift_invariance`, `non_vacuity_control`,
`mass_floor`, `precision_floor`), an overall `verdict`, a human-readable
`explanation`, and `.to_json()`.

`audit_attention_atlas(rows, tau, ...)` audits a list of head dicts
(`{"scores", "target_idx"[, "exclude_from_sum", "layer", "head"]}`) and returns
one `AuditReport` per head, with `layer`/`head` provenance filled in — so you
can tally how many heads in an atlas are `SOUND` vs `VACUOUS` / `RELATIVE-ONLY`.

### Auditing a raw-weight (gated/SSM) cert

```python
# BOS (idx0) holds 91% of magnitude, target (idx2) holds 9%; the cert excludes
# BOS+self from its comparison sum.
report = vacuity_audit([50.0, 0.02, 5.0, 0.02, 0.01, 0.05],
                       target_idx=2, tau=0.95,
                       form="raw_weight", exclude_from_sum=(0, 5))
print(report.verdict)        # 'RELATIVE-ONLY'
```

## CLI: `eml-vacuity-audit`

Registered as a `console_scripts` entry point (`pip install -e .`), or run via
`python -m emltorch.certify.vacuity_cli`.

```bash
# From a scores JSON ({scores, target_idx[, exclude_from_sum, layer, head]},
# or a list of such rows for a whole atlas):
eml-vacuity-audit --scores-json head.json --tau 0.95 --form v3

# Straight from a HuggingFace model (extracts the head's log-prob row):
eml-vacuity-audit --model gpt2 --prompt "..." --layer 5 --head 5 \
    --tau 0.5 --form softmax_interval --precision-floor 0.0039

# Machine-readable:
eml-vacuity-audit --scores-json atlas.json --tau 0.95 --json
```

Exit code is `0` iff every audited head is `SOUND`, else nonzero (so it composes
into CI).

### Worked example — our own H19-style log-prob cert → `VACUOUS`

```
$ eml-vacuity-audit --scores-json h19_head.json --tau 0.95 --form v3
L19.H27  tau=0.95 form=v3
VERDICT: VACUOUS

[FAIL] shift-invariance: form 'v3' is shift-VARIANT (body exponentiates the raw
  score). On log-prob inputs sumE=1 => Ln(1)=0 and the body collapses to a
  near-tautology. discharge@shift0=True, discharge@shift7=True; a sound
  softmax-mass claim would be invariant to the shift.
[FAIL] non-vacuity control: the uniform 1/T control row (target holds 0.167 of
  total mass) ALSO discharges UNSAT at tau=0.95 -- the cert passes with no
  concentration, so its UNSAT is vacuous.
[FAIL] mass floor (relative-vs-absolute): cert DISCHARGES UNSAT at tau=0.95 but
  the target holds only 0.1 of TOTAL mass (denominator includes excluded keys
  none). That is a RELATIVE ranking against the surviving keys, not
  tau-concentration.
[skip] numerical-precision floor: no precision_floor supplied; numerical-noise
  check skipped.

The cert would discharge without real concentration (shift-variant body and/or a
non-concentrated control also passes). Do NOT treat its UNSAT as a concentration
certificate.
```

This is exactly the H19/H20/H21 atlas defect: a head with 10% target mass under
the `v3` log-prob form returns dual-UNSAT at τ=0.95 yet certifies nothing. Switch
`--form softmax_interval` and the same head is honestly `NOT-CERTIFIED`.

## Tests

```bash
cd emltorch/
CUDA_VISIBLE_DEVICES="" /home/ubuntu/samuel/eval_venv/bin/python3 \
    -m pytest tests/test_vacuity_audit.py tests/test_vacuity_cli.py -q
```

21 tests (15 module + 6 CLI), all dual-solver, CPU-only.
```
```
