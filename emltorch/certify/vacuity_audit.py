"""Vacuity audit: a drop-in test-suite that tells you whether an
attention-concentration, softmax-mass, or routing cert claim is SOUND or merely
VACUOUS, RELATIVE-ONLY, or UNDER-PRECISION.

Motivation
==========
Two soundness audits found that a "dual-UNSAT @ tau=0.95 concentration cert"
verdict can be decoupled from real concentration in four distinct ways. This
module turns each failure mode into a single tested check, so that a reviewer
can run one function on a cert claim and get a structured, human-readable
verdict, the test others can re-run on a published cert.

The four checks
===============
1. **Shift-invariance.** A concentration claim about softmax mass is
   shift-invariant in the scores (softmax is invariant to adding a constant to
   every logit). The SOUND claim is ``s_target - Ln(Sum_j exp s_j) > log tau``
   (<=> softmax_target > tau). A SHIFT-VARIANT body such as the ``v3`` form
   ``Exp(s_target) - Ln(sumE) > log tau`` collapses on log-prob inputs (where
   ``sumE = Sum exp(log prob) = 1`` so ``Ln(1)=0`` and the body is
   ``Exp(s_target) > log tau < 0`` -- always true). We detect shift-variance
   both structurally (form metadata) AND empirically (certify the scores, then
   certify the scores shifted by a constant; a shift-invariant cert returns the
   same verdict). Failure => VACUOUS.

2. **Non-vacuity control.** A KNOWN non-concentrated control row must NOT
   discharge at the same tau. We try (a) a uniform 1/T row, and (b) a low-mass
   row (target holds ``lowmass_frac`` of the mass). If EITHER discharges, the
   cert would pass with no concentration at all. Failure => VACUOUS.

3. **Relative-vs-absolute / mass floor.** If the cert excludes some keys
   (e.g. BOS, self) from the denominator without an absolute floor, a target
   holding 9% -- or 1e-9 -- of *total* mass can discharge at tau=0.95 because
   the real mass sits on the excluded keys. We recompute the target's mass
   fraction over ALL keys (excluded keys IN the denominator) and require it to
   meet tau when the cert discharges. Failure => RELATIVE-ONLY.

4. **Numerical-precision floor.** A cert finer than the model's own numerical
   noise certifies nothing physical. We find the certified L_inf radius and
   require it to exceed ``precision_floor`` (e.g. bf16 round-off ~ 2^-8). A
   radius at/below the floor => UNDER-PRECISION.

Verdict precedence (worst wins): VACUOUS > RELATIVE-ONLY > UNDER-PRECISION >
SOUND. A check that is not applicable to the inputs (e.g. no precision_floor
supplied, or no excluded keys) is recorded as passed-and-skipped and does not
downgrade the verdict.

If the cert does NOT discharge at the audit box radius (the head simply is not
tau-concentrated), there is no UNSAT claim to validate; the verdict is
``NOT-CERTIFIED`` -- distinct from both SOUND (a real, validated cert) and
VACUOUS (a cert that discharges for the wrong reason).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .concentration import (
    _SOUND_FORMS,
    attention_concentration_cert,
)
from .atlas import certified_radius, _DEFAULT_RHOS
from .solvers import dual_verify

# Verdict constants (public).
SOUND = "SOUND"
VACUOUS = "VACUOUS"
RELATIVE_ONLY = "RELATIVE-ONLY"
UNDER_PRECISION = "UNDER-PRECISION"
NOT_CERTIFIED = "NOT-CERTIFIED"

# Forms whose body is SHIFT-INVARIANT in the scores (the property that makes a
# softmax-mass claim sound on log-prob inputs). softmax_interval subtracts the
# raw score s_target (not Exp(s_target)) from Ln(sumE); both move by +c under a
# constant shift, so the claim is invariant. v3/v2/interval exponentiate the
# score first, so a constant shift changes the body.
_SHIFT_INVARIANT_FORMS = {"softmax_interval"}

# Raw-magnitude forms (H23 gated/SSM): a separate cert family keyed off
# emit_raw_weight_concentration_cert. These are NOT log-prob softmax forms; their
# vacuity mode is the mass-floor (relative-vs-absolute) one, not Ln-collapse.
_RAW_WEIGHT_FORMS = {"raw_weight"}

_DEFAULT_LOWMASS_FRAC = 0.05  # control: target holds 5% of the mass
_DEFAULT_SHIFT = 7.0  # constant added to every score in the empirical shift test


@dataclass
class CheckResult:
    """One audit check outcome."""

    name: str
    passed: bool
    detail: str
    skipped: bool = False  # not applicable to these inputs (does not downgrade)


@dataclass
class AuditReport:
    """Structured verdict from the four-check vacuity audit."""

    verdict: str
    shift_invariance: CheckResult
    non_vacuity_control: CheckResult
    mass_floor: CheckResult
    precision_floor: CheckResult
    explanation: str = ""
    # optional provenance for atlas mode
    layer: Optional[int] = None
    head: Optional[int] = None
    target_prob_total: Optional[float] = None
    certified_radius: Optional[float] = None

    def to_json(self) -> dict:
        def _c(cr: CheckResult) -> dict:
            return {
                "name": cr.name,
                "passed": bool(cr.passed),
                "skipped": bool(cr.skipped),
                "detail": cr.detail,
            }

        return {
            "verdict": self.verdict,
            "layer": self.layer,
            "head": self.head,
            "target_prob_total": self.target_prob_total,
            "certified_radius": self.certified_radius,
            "checks": {
                "shift_invariance": _c(self.shift_invariance),
                "non_vacuity_control": _c(self.non_vacuity_control),
                "mass_floor": _c(self.mass_floor),
                "precision_floor": _c(self.precision_floor),
            },
            "explanation": self.explanation,
        }


# --------------------------------------------------------------------------
# Cert emission dispatch (softmax forms vs raw-weight form)
# --------------------------------------------------------------------------


def _emit_cert(
    scores: Sequence[float],
    target_idx: int,
    tau: float,
    rho: float,
    form: str,
    exclude_from_sum=None,
) -> str:
    """Emit the cert text for either a softmax concentration form or the
    raw-weight (gated/SSM) form, so the audit can drive both families."""
    if form in _RAW_WEIGHT_FORMS:
        from emltorch.smt import emit_raw_weight_concentration_cert

        return emit_raw_weight_concentration_cert(
            list(scores),
            target_idx,
            tau=tau,
            rho_log=rho,
            exclude_from_sum=exclude_from_sum,
        )
    return attention_concentration_cert(
        scores, target_idx, tau=tau, rho_box=rho, form=form
    )


def _discharges(
    scores: Sequence[float],
    target_idx: int,
    tau: float,
    rho: float,
    form: str,
    timeout_ms: int,
    exclude_from_sum=None,
) -> bool:
    """True iff both solvers definitively agree UNSAT (the cert discharges)."""
    cert = _emit_cert(scores, target_idx, tau, rho, form, exclude_from_sum)
    dual = dual_verify(cert, timeout_ms=timeout_ms)
    return dual.verdict == "unsat" and dual.agree


# --------------------------------------------------------------------------
# mass-fraction helpers (denominator INCLUDES excluded keys)
# --------------------------------------------------------------------------


def _total_mass_fraction(scores: Sequence[float], target_idx: int, form: str) -> float:
    """Target's share of TOTAL mass, counting every key (including any that the
    cert excludes from its comparison Sum). For softmax forms ``scores`` are
    log-probs/logits -> softmax. For raw_weight ``scores`` are |a_j| magnitudes
    -> normalize by their sum.
    """
    if form in _RAW_WEIGHT_FORMS:
        total = float(sum(abs(s) for s in scores))
        return abs(scores[target_idx]) / total if total > 0 else 0.0
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    z = sum(exps)
    return exps[target_idx] / z if z > 0 else 0.0


def _control_scores(scores: Sequence[float], target_idx: int, control: str, form: str):
    """Build a KNOWN non-concentrated control row matching the cert's input
    domain (log-probs for softmax forms; magnitudes for raw_weight)."""
    n = len(scores)
    if control == "uniform":
        fracs = [1.0 / n] * n
    elif control == "lowmass":
        rest = (1.0 - _DEFAULT_LOWMASS_FRAC) / (n - 1) if n > 1 else 0.0
        fracs = [rest] * n
        fracs[target_idx] = _DEFAULT_LOWMASS_FRAC
    else:
        raise ValueError(f"unknown control {control!r}; expected 'uniform'|'lowmass'")
    if form in _RAW_WEIGHT_FORMS:
        return fracs  # magnitudes proportional to the mass fractions
    return [math.log(f) for f in fracs]


# --------------------------------------------------------------------------
# the four checks
# --------------------------------------------------------------------------


def _check_shift_invariance(
    scores, target_idx, tau, rho, form, timeout_ms
) -> CheckResult:
    """A sound softmax-mass cert is invariant to adding a constant to every
    score. We assert this both structurally (form metadata) and empirically
    (verdict on ``scores`` == verdict on ``scores + c``).

    The raw_weight form is shift-INVARIANT in a different sense (scale-invariant
    on magnitudes, not additive shift), so additive shift does not apply; we
    skip the empirical leg and pass it (its real defect is the mass floor).
    """
    if form in _RAW_WEIGHT_FORMS:
        return CheckResult(
            name="shift_invariance",
            passed=True,
            skipped=True,
            detail="raw-weight magnitude form: additive shift N/A "
            "(its vacuity mode is the mass floor, check 3)",
        )

    structurally_invariant = form in _SHIFT_INVARIANT_FORMS
    if not structurally_invariant:
        # Empirically demonstrate the collapse: certify scores and scores+c and
        # show the verdict is shift-DEPENDENT (or vacuously always-UNSAT on
        # log-probs). We report the shifted verdict as the witness.
        base = _discharges(scores, target_idx, tau, rho, form, timeout_ms)
        shifted = [s + _DEFAULT_SHIFT for s in scores]
        shifted_v = _discharges(shifted, target_idx, tau, rho, form, timeout_ms)
        detail = (
            f"form {form!r} is shift-VARIANT (body exponentiates the raw score). "
            f"On log-prob inputs sumE=1 => Ln(1)=0 and the body collapses to a "
            f"near-tautology. discharge@shift0={base}, discharge@shift{+_DEFAULT_SHIFT:g}"
            f"={shifted_v}; a sound softmax-mass claim would be invariant to the shift."
        )
        return CheckResult("shift_invariance", passed=False, detail=detail)

    # Sound form: confirm empirically that shifting does not change the verdict.
    base = _discharges(scores, target_idx, tau, rho, form, timeout_ms)
    shifted = [s + _DEFAULT_SHIFT for s in scores]
    shifted_v = _discharges(shifted, target_idx, tau, rho, form, timeout_ms)
    invariant = base == shifted_v
    detail = (
        f"form {form!r} subtracts the raw score (not Exp(score)) from Ln(sumE), "
        f"so the claim is shift-invariant. discharge@shift0={base}, "
        f"discharge@shift{+_DEFAULT_SHIFT:g}={shifted_v} (equal => invariant)."
    )
    return CheckResult("shift_invariance", passed=invariant, detail=detail)


def _check_non_vacuity_control(
    scores, target_idx, tau, rho, form, control, timeout_ms, exclude_from_sum
) -> CheckResult:
    """A KNOWN non-concentrated control row must NOT discharge. If it does, the
    cert would pass with no concentration -> vacuous."""
    ctrl = _control_scores(scores, target_idx, control, form)
    ctrl_unsat = _discharges(
        ctrl, target_idx, tau, rho, form, timeout_ms, exclude_from_sum
    )
    label = (
        "uniform 1/T" if control == "uniform" else f"low-mass {_DEFAULT_LOWMASS_FRAC:g}"
    )
    if ctrl_unsat:
        detail = (
            f"the {label} control row (target holds "
            f"{_total_mass_fraction(ctrl, target_idx, form):.3g} of total mass) "
            f"ALSO discharges UNSAT at tau={tau:g} -- the cert passes with no "
            f"concentration, so its UNSAT is vacuous."
        )
        return CheckResult("non_vacuity_control", passed=False, detail=detail)
    detail = (
        f"the {label} control row is correctly REFUSED (SAT) at tau={tau:g}; the "
        f"cert cannot be discharged by a non-concentrated head."
    )
    return CheckResult("non_vacuity_control", passed=True, detail=detail)


def _check_mass_floor(
    scores, target_idx, tau, rho, form, timeout_ms, exclude_from_sum
) -> CheckResult:
    """If the cert discharges, the target's share of TOTAL mass (excluded keys
    IN the denominator) must meet tau. Otherwise the UNSAT is RELATIVE-ranking,
    not concentration."""
    discharges = _discharges(
        scores, target_idx, tau, rho, form, timeout_ms, exclude_from_sum
    )
    frac = _total_mass_fraction(scores, target_idx, form)
    excluded = sorted(set(exclude_from_sum)) if exclude_from_sum else []
    if not discharges:
        # Cert does not even discharge -> the mass-floor failure mode cannot
        # arise (nothing is being over-claimed). Pass-and-skip.
        return CheckResult(
            name="mass_floor",
            passed=True,
            skipped=True,
            detail=f"cert does not discharge at rho={rho:g} (no over-claim to "
            f"audit); target total-mass fraction={frac:.3g}.",
        )
    if frac + 1e-9 < tau:
        detail = (
            f"cert DISCHARGES UNSAT at tau={tau:g} but the target holds only "
            f"{frac:.3g} of TOTAL mass (denominator includes excluded keys "
            f"{excluded or 'none'}). That is a RELATIVE ranking against the "
            f"surviving keys, not tau-concentration."
        )
        return CheckResult("mass_floor", passed=False, detail=detail)
    detail = (
        f"cert discharges and the target holds {frac:.3g} >= tau={tau:g} of TOTAL "
        f"mass (excluded keys {excluded or 'none'} counted in the denominator) -- "
        f"a genuine absolute-mass concentration."
    )
    return CheckResult("mass_floor", passed=True, detail=detail)


def _check_precision_floor(
    scores, target_idx, tau, rhos, form, precision_floor, timeout_ms
) -> tuple[CheckResult, Optional[float]]:
    """The certified L_inf radius must exceed the model's numerical-noise floor.
    Only meaningful for the softmax forms (the radius search uses the sound
    softmax_interval form internally). Returns ``(CheckResult, radius_or_None)``.
    """
    if precision_floor is None:
        return (
            CheckResult(
                name="precision_floor",
                passed=True,
                skipped=True,
                detail="no precision_floor supplied; numerical-noise check skipped.",
            ),
            None,
        )
    if form in _RAW_WEIGHT_FORMS:
        return (
            CheckResult(
                name="precision_floor",
                passed=True,
                skipped=True,
                detail="raw-weight form: certified_radius search is softmax-specific; "
                "precision floor not applied.",
            ),
            None,
        )
    cr = certified_radius(scores, target_idx, tau=tau, rhos=rhos, timeout_ms=timeout_ms)
    radius = cr.radius
    if radius <= precision_floor:
        detail = (
            f"certified radius {radius:.4g} <= precision_floor {precision_floor:.4g} "
            f"(e.g. bf16 round-off ~ {2 ** -8:.4g}): the cert is finer than the "
            f"model's numerical noise -- UNDER-PRECISION."
        )
        return CheckResult("precision_floor", passed=False, detail=detail), radius
    detail = (
        f"certified radius {radius:.4g} > precision_floor {precision_floor:.4g}: the "
        f"robustness margin is above the model's numerical noise."
    )
    return CheckResult("precision_floor", passed=True, detail=detail), radius


# --------------------------------------------------------------------------
# top-level audit
# --------------------------------------------------------------------------


def vacuity_audit(
    scores: Sequence[float],
    target_idx: int,
    tau: float,
    *,
    form: str = "softmax_interval",
    exclude_from_sum: Optional[Sequence[int]] = None,
    precision_floor: Optional[float] = None,
    control: str = "uniform",
    rho: float = 0.005,
    rhos: Sequence[float] = _DEFAULT_RHOS,
    timeout_ms: int = 8000,
) -> AuditReport:
    """Run the four-check vacuity audit on a single concentration-cert claim.

    Args:
        scores: per-key scores at the query position. For softmax forms these are
            log-probs (or logits); for ``form="raw_weight"`` they are the
            non-negative magnitudes ``|a_j|``.
        target_idx: index of the key the cert claims dominates.
        tau: concentration threshold in (0, 1).
        form: cert form to audit -- ``"softmax_interval"`` (sound), ``"v3"`` /
            ``"v2"`` / ``"interval"`` (diagnostic softmax forms), or
            ``"raw_weight"`` (the gated/SSM magnitude cert).
        exclude_from_sum: keys the cert drops from its comparison Sum (e.g.
            BOS=0, self=last). Counted in the mass-floor denominator regardless.
        precision_floor: if given, the certified radius must exceed this (e.g.
            ``2**-8`` for bf16). Omit to skip check 4.
        control: ``"uniform"`` or ``"lowmass"`` -- the non-vacuity control row.
        rho: L_inf box radius used for the discharge-based checks (1-3).
        rhos: rho ladder for the certified-radius search (check 4).
        timeout_ms: per-solver timeout.

    Returns:
        An ``AuditReport`` with one ``CheckResult`` per check, an overall
        ``verdict`` (SOUND / VACUOUS / RELATIVE-ONLY / UNDER-PRECISION), and a
        human-readable ``explanation``.
    """
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be in (0, 1); got {tau!r}")
    n = len(scores)
    if not (0 <= target_idx < n):
        raise ValueError(f"target_idx {target_idx} out of range for {n} scores")
    if not all(math.isfinite(s) for s in scores):
        raise ValueError("scores must all be finite")

    primary_discharges = _discharges(
        scores, target_idx, tau, rho, form, timeout_ms, exclude_from_sum
    )

    shift = _check_shift_invariance(scores, target_idx, tau, rho, form, timeout_ms)
    control_chk = _check_non_vacuity_control(
        scores, target_idx, tau, rho, form, control, timeout_ms, exclude_from_sum
    )
    mass = _check_mass_floor(
        scores, target_idx, tau, rho, form, timeout_ms, exclude_from_sum
    )
    prec, radius = _check_precision_floor(
        scores, target_idx, tau, rhos, form, precision_floor, timeout_ms
    )

    total_frac = _total_mass_fraction(scores, target_idx, form)

    # The cert is considered "certified" if it discharges at the audit box OR (when
    # the radius ladder was searched for check 4) it certifies any positive radius
    # below the audit box. This keeps the NOT-CERTIFIED gate consistent with the
    # certified-radius the precision check reports.
    certified = primary_discharges or (radius is not None and radius > 0.0)

    # Worst-wins precedence. A skipped check never downgrades. A shift-VARIANT or
    # control-failing form is VACUOUS even if it didn't happen to discharge on
    # THIS row (the form itself is unsound). Otherwise, a cert that simply did
    # not discharge is NOT-CERTIFIED -- there is no UNSAT claim to validate.
    if not shift.passed or not control_chk.passed:
        verdict = VACUOUS
    elif not certified:
        verdict = NOT_CERTIFIED
    elif not mass.passed:
        verdict = RELATIVE_ONLY
    elif not prec.passed:
        verdict = UNDER_PRECISION
    else:
        verdict = SOUND

    explanation = _explain(verdict, shift, control_chk, mass, prec)

    return AuditReport(
        verdict=verdict,
        shift_invariance=shift,
        non_vacuity_control=control_chk,
        mass_floor=mass,
        precision_floor=prec,
        explanation=explanation,
        target_prob_total=total_frac,
        certified_radius=radius,
    )


_CHECK_TITLES = {
    "shift_invariance": "shift-invariance",
    "non_vacuity_control": "non-vacuity control",
    "mass_floor": "mass floor (relative-vs-absolute)",
    "precision_floor": "numerical-precision floor",
}


def _explain(verdict, shift, control_chk, mass, prec) -> str:
    lines = [f"VERDICT: {verdict}", ""]
    for cr in (shift, control_chk, mass, prec):
        if cr.skipped:
            mark = "skip"
        elif cr.passed:
            mark = "PASS"
        else:
            mark = "FAIL"
        title = _CHECK_TITLES.get(cr.name, cr.name)
        lines.append(f"[{mark}] {title}: {cr.detail}")
    lines.append("")
    if verdict == SOUND:
        lines.append(
            "All applicable checks pass: the UNSAT is a genuine, shift-invariant, "
            "non-vacuous, absolute-mass concentration above the numerical floor."
        )
    elif verdict == VACUOUS:
        lines.append(
            "The cert would discharge without real concentration (shift-variant "
            "body and/or a non-concentrated control also passes). Do NOT treat its "
            "UNSAT as a concentration certificate."
        )
    elif verdict == RELATIVE_ONLY:
        lines.append(
            "The cert proves a RELATIVE ranking against the surviving (non-excluded) "
            "keys, not absolute tau-mass concentration. Add a total-mass floor "
            "before claiming concentration."
        )
    elif verdict == UNDER_PRECISION:
        lines.append(
            "The certified robustness radius is at/below the model's numerical "
            "noise, so the cert certifies nothing the model can physically "
            "represent. Re-run at a coarser tau or report radius honestly."
        )
    elif verdict == NOT_CERTIFIED:
        lines.append(
            "The cert does NOT discharge at this box radius: the head is simply not "
            "tau-concentrated, so there is no UNSAT claim to validate. This is an "
            "honest non-result, not a vacuous pass."
        )
    return "\n".join(lines)


def audit_attention_atlas(
    rows: Sequence[dict],
    tau: float,
    *,
    form: str = "softmax_interval",
    exclude_from_sum: Optional[Sequence[int]] = None,
    precision_floor: Optional[float] = None,
    control: str = "uniform",
    rho: float = 0.005,
    rhos: Sequence[float] = _DEFAULT_RHOS,
    timeout_ms: int = 8000,
) -> list[AuditReport]:
    """Audit a whole head atlas. Each ``row`` is a dict with at least
    ``scores`` and ``target_idx`` (and optionally ``layer``, ``head``,
    ``exclude_from_sum``). Returns one ``AuditReport`` per row, with ``layer`` /
    ``head`` provenance filled in so a reviewer can tally how many heads are
    SOUND vs VACUOUS / RELATIVE-ONLY / UNDER-PRECISION.
    """
    reports: list[AuditReport] = []
    for row in rows:
        report = vacuity_audit(
            row["scores"],
            row["target_idx"],
            tau,
            form=form,
            exclude_from_sum=row.get("exclude_from_sum", exclude_from_sum),
            precision_floor=precision_floor,
            control=control,
            rho=rho,
            rhos=rhos,
            timeout_ms=timeout_ms,
        )
        report.layer = row.get("layer")
        report.head = row.get("head")
        reports.append(report)
    return reports
