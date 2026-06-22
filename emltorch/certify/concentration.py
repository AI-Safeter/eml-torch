"""Attention-concentration certificate builder.

Absorbs the script-trapped ``_cert_v3.py`` and the 30x-duplicated ``_num`` into
the library. Emits portable SMT-LIB2 proving that one attention key holds a
``tau`` share of the softmax mass, robust to an L_inf(rho) box on the scores.

SOUNDNESS NOTE
==============
Exactly ONE form is a soundness-guaranteed concentration certificate:
``softmax_interval`` (the public DEFAULT). It is the only member of
``_SOUND_FORMS``. The ``v3`` / ``v2`` / ``interval`` forms are kept for
backward compatibility and as EML-in-body illustrations, but they are
**diagnostic only, NOT soundness-guaranteed** and can produce vacuous or
false UNSAT on log-prob inputs (what ``emltorch.certify.extract`` produces).

Forms:

- ``softmax_interval`` (DEFAULT, SOUND): QF_LRA relaxation of the TRUE softmax
  claim ``s_target - Ln(sumE) > log(tau)`` <=> ``softmax_target > tau``.
  Shift-invariant, so a non-concentrated head is correctly refused (SAT). This
  is the only form whose UNSAT is a non-vacuous concentration guarantee.

- ``v3`` (diagnostic / EML-in-body illustration ONLY, NOT soundness-guaranteed):
  ``eml(s_target, Sum_j Exp(s_j)) > log(tau)``. Two distinct false-UNSAT modes:
  (a) the gap precondition ``s_target >= s_j + 1`` is jointly infeasible with the
  box when the real gap < ~1 nat -> vacuously UNSAT; (b) on ``log(prob)`` inputs
  ``sumE = Sum exp(log prob) = 1`` and ``Ln(1) = 0`` collapses v3 to Exp-only,
  so it discharges for nearly any head. Do NOT treat v3 UNSAT as a concentration
  certificate.

- ``v2`` (diagnostic ONLY, NOT soundness-guaranteed): ``eml(s_target, 1) > tau *
  Sum_j eml(s_j, 1)``. Since ``Ln(1) = 0``, ``eml(s, 1) = Exp(s)``, this is
  Exp-only (H23 audit: v1 == v2) and inherits the same gap-precondition
  vacuity as v3.

- ``interval`` (diagnostic ONLY, NOT soundness-guaranteed): QF_LRA relaxation
  of the WRONG inequality ``Exp(s_target) - Ln(sumE) > log(tau)`` (NOT
  log-softmax). On log-prob inputs this discharges for nearly any head; it is
  NOT a concentration certificate. Use ``softmax_interval`` for soundness.
"""

from __future__ import annotations

import math
from typing import Sequence

from emltorch.smt import EML_AXIOMS_SMT2, with_lemmas

# The ONLY soundness-guaranteed concentration cert form. v3/v2/interval are
# diagnostic / EML-in-body illustrations and can produce vacuous or false UNSAT
# on log-prob inputs -- never treat their UNSAT as a concentration guarantee.
_SOUND_FORMS = {"softmax_interval"}


def num_to_smt(v: float) -> str:
    """Render a Python float as an SMT-LIB2 numeral, negatives as ``(- x)``."""
    s = f"{v:.18f}"
    return f"(- {s[1:]})" if v < 0 else s


# Backwards-compatible alias for the 30 scripts that call the private name.
_num = num_to_smt


def _box_decls(score_obs: Sequence[float], rho_box: float) -> list[str]:
    lines: list[str] = []
    for j, s in enumerate(score_obs):
        lines.append(f"(declare-const s{j} Real)")
        lines.append(f"(assert (>= s{j} {num_to_smt(s - rho_box)}))")
        lines.append(f"(assert (<= s{j} {num_to_smt(s + rho_box)}))")
    return lines


def _gap_preconditions(target_idx: int, n: int) -> list[str]:
    return [
        f"(assert (>= s{target_idx} (+ s{j} 1.0)))" for j in range(n) if j != target_idx
    ]


def _build_v3(score_obs, target_idx, tau, rho_box, head_label) -> str:
    """DIAGNOSTIC / EML-in-body illustration ONLY -- NOT a soundness-guaranteed
    concentration certificate. Can produce vacuous or false UNSAT: (a) the gap
    precondition ``s_target >= s_j + 1`` is infeasible with the box when the real
    gap < ~1 nat -> vacuously UNSAT; (b) on log-prob inputs ``Ln(1)=0`` collapses
    it to Exp-only and it discharges for nearly any head. Use ``softmax_interval``
    for a sound cert.
    """
    n = len(score_obs)
    decls = _box_decls(score_obs, rho_box)
    decls.append("(declare-const sumE Real)")
    decls.append(
        "(assert (= sumE (+ " + " ".join(f"(Exp s{j})" for j in range(n)) + ")))"
    )
    decls.append("(assert (> sumE 0.0))")
    log_tau = math.log(tau)
    neg_safe = (
        f"(assert (not (> (- (Exp s{target_idx}) (Ln sumE)) {num_to_smt(log_tau)})))"
    )
    title = (
        "V3 (load-bearing EML)"
        + (f" {head_label}: " if head_label else ": ")
        + f"eml(s_target={target_idx}, sumE) > log({tau:.4f})"
    )
    text = (
        f"; {title}\n"
        f";   <=> log(softmax[s_target]) concentration; Ln(sumE) load-bearing on raw logits\n"
        "(set-logic ALL)\n"
        + EML_AXIOMS_SMT2
        + "\n".join(decls)
        + "\n"
        + "\n".join(_gap_preconditions(target_idx, n))
        + "\n"
        + neg_safe
        + "\n(check-sat)\n"
    )
    return with_lemmas(text, "ratio_corollary", "ln_multiplicativity")


def _build_v2(score_obs, target_idx, tau, rho_box, head_label) -> str:
    """DIAGNOSTIC ONLY -- NOT a soundness-guaranteed concentration certificate.
    ``eml(s,1)=Exp(s)`` (Ln(1)=0), so the EML operator is decorative (H23: v1==v2)
    and it inherits the same gap-precondition vacuity as v3. Use
    ``softmax_interval`` for a sound cert.
    """
    n = len(score_obs)
    decls = _box_decls(score_obs, rho_box)
    eml_each = [f"(- (Exp s{j}) (Ln 1.0))" for j in range(n)]
    eml_target = f"(- (Exp s{target_idx}) (Ln 1.0))"
    sum_eml = "(+ " + " ".join(eml_each) + ")"
    neg_safe = f"(assert (not (> {eml_target} (* {num_to_smt(tau)} {sum_eml}))))"
    title = (
        "V2 (Exp-only, EML decorative)"
        + (f" {head_label}: " if head_label else ": ")
        + f"eml(s_target={target_idx}, 1) > {tau:.4f} * Sum eml(s_j, 1)"
    )
    text = (
        f"; {title}\n"
        "(set-logic ALL)\n"
        + EML_AXIOMS_SMT2
        + "\n".join(decls)
        + "\n"
        + "\n".join(_gap_preconditions(target_idx, n))
        + "\n"
        + neg_safe
        + "\n(check-sat)\n"
    )
    return with_lemmas(text, "ratio_corollary")


def _build_softmax_interval(score_obs, target_idx, tau, rho_box, head_label) -> str:
    """QF_LRA sound relaxation of the TRUE softmax claim s_t - Ln(sumE) > log(tau)
    <=> softmax_target > tau. Shift-invariant => NON-VACUOUS (a non-concentrated
    head is correctly refused), unlike v3.

    This is the HONEST primary cert form. It discharges (dual-UNSAT, both solvers,
    instant) iff the head is genuinely tau-concentrated with enough margin to
    absorb the rho box; otherwise SAT. The certifiable rho therefore reflects the
    head's concentration margin -- exactly what a sound robustness radius means.

    NOTE: the independent-interval relaxation drops cross-variable correlation, so
    a many-key head needs a SMALL rho (e.g. <= 0.005 at tau=0.95) to discharge;
    rho=0.1 is too loose for the tau=0.95 claim. The axiomatized true-softmax form
    is NOT offered: it returns 'unknown' on every input (the solver cannot prove
    s_t - Ln(sumE) > log(tau) from the axioms). Use the atlas certified-radius
    search to find the max discharging rho per head.
    """
    los = [s - rho_box for s in score_obs]
    his = [s + rho_box for s in score_obs]
    sume_max = sum(math.exp(hi) for hi in his)
    lsum_max = math.log(sume_max)
    log_tau = math.log(tau)
    title = (
        "SOFTMAX-INTERVAL (QF_LRA, sound, loose)"
        + (f" {head_label}: " if head_label else ": ")
        + f"s_target={target_idx} - Ln(sumE) > log({tau:.4f})"
    )
    return (
        f"; {title}\n"
        "(set-logic QF_LRA)\n"
        "(declare-const st Real)\n"
        f"(assert (>= st {num_to_smt(los[target_idx])}))\n"
        f"(assert (<= st {num_to_smt(his[target_idx])}))\n"
        "(declare-const LsumE Real)\n"
        f"(assert (<= LsumE {num_to_smt(lsum_max)}))\n"
        f"(assert (not (> (- st LsumE) {num_to_smt(log_tau)})))\n"
        "(check-sat)\n"
    )


def _build_interval(score_obs, target_idx, tau, rho_box, head_label) -> str:
    """DIAGNOSTIC ONLY -- NOT a soundness-guaranteed concentration certificate.

    Encodes the WRONG inequality ``Exp(s_target) - Ln(sumE) > log(tau)`` (this is
    NOT log-softmax). The interval relaxation over the box is internally sound for
    *that* inequality, but the inequality itself is not the concentration claim:
    on log-prob inputs (what ``extract.py`` produces) it discharges for nearly any
    head, i.e. false UNSAT for "concentration". Use ``softmax_interval`` for a
    sound concentration cert.
    """
    los = [s - rho_box for s in score_obs]
    his = [s + rho_box for s in score_obs]
    et_min = math.exp(los[target_idx])
    et_max = math.exp(his[target_idx])
    sume_min = sum(math.exp(lo) for lo in los)
    sume_max = sum(math.exp(hi) for hi in his)
    lsum_min = math.log(sume_min)
    lsum_max = math.log(sume_max)
    log_tau = math.log(tau)
    title = (
        "INTERVAL (QF_LRA, sound relaxation)"
        + (f" {head_label}: " if head_label else ": ")
        + f"Exp(s_target={target_idx}) - Ln(sumE) > log({tau:.4f})"
    )
    return (
        f"; {title}\n"
        f";   Et in [exp(lo_t), exp(hi_t)], Ln(sumE) in [ln(sumE_min), ln(sumE_max)]\n"
        "(set-logic QF_LRA)\n"
        "(declare-const Et Real)\n"
        f"(assert (>= Et {num_to_smt(et_min)}))\n"
        f"(assert (<= Et {num_to_smt(et_max)}))\n"
        "(declare-const LsumE Real)\n"
        f"(assert (>= LsumE {num_to_smt(lsum_min)}))\n"
        f"(assert (<= LsumE {num_to_smt(lsum_max)}))\n"
        f"(assert (not (> (- Et LsumE) {num_to_smt(log_tau)})))\n"
        "(check-sat)\n"
    )


def _validate_inputs(scores: Sequence[float], target_idx: int, tau: float) -> None:
    """Shared precondition checks for the concentration cert + radius search.

    Guards the math.log domain (tau in (0,1)), rejects NaN/inf scores, and
    bounds target_idx. Raised as ValueError so a bad call fails loudly rather
    than emitting a meaningless or undefined-behavior cert.
    """
    if not (0.0 < tau < 1.0):
        raise ValueError(
            f"tau must be in the open interval (0, 1); got {tau!r}. "
            "tau>=1 is meaningless (no single key can exceed the whole mass) and "
            "tau<=0 is outside the log(tau) domain."
        )
    if not all(math.isfinite(s) for s in scores):
        raise ValueError(f"scores must all be finite; got {list(scores)!r}")
    n = len(scores)
    if not (0 <= target_idx < n):
        raise ValueError(
            f"target_idx {target_idx} out of range for {n} scores (expected 0..{n - 1})"
        )


def attention_concentration_cert(
    scores: Sequence[float],
    target_idx: int,
    tau: float,
    rho_box: float = 0.10,
    form: str = "softmax_interval",
    head_label: str = "",
) -> str:
    """Emit a portable SMT-LIB2 attention-concentration certificate.

    Args:
        scores: per-key attention scores at the query position. Must be finite.
            For the sound default the scores may be log-probs (concentration is
            shift-invariant).
        target_idx: index of the key the cert claims dominates. Must be in range.
        tau: concentration threshold (e.g. 0.95). Must be in the open (0, 1).
        rho_box: L_inf perturbation budget applied to every score.
        form: "softmax_interval" (DEFAULT, the ONLY soundness-guaranteed form;
            QF_LRA, shift-invariant, non-vacuous). The remaining forms are
            DIAGNOSTIC ONLY -- NOT soundness-guaranteed, can produce vacuous or
            false UNSAT on log-prob inputs: "v3"/"v2" (EML-in-body illustrations,
            gap-precondition vacuity) and "interval" (wrong inequality). See
            ``_SOUND_FORMS`` and the module docstring.
        head_label: optional label embedded in the cert title.

    Returns:
        SMT-LIB2 text. For the sound default, dual-UNSAT means the concentration
        property is proven over the box; for the diagnostic forms UNSAT is NOT a
        soundness guarantee.
    """
    _validate_inputs(scores, target_idx, tau)
    if form == "v3":
        return _build_v3(scores, target_idx, tau, rho_box, head_label)
    if form == "v2":
        return _build_v2(scores, target_idx, tau, rho_box, head_label)
    if form == "interval":
        return _build_interval(scores, target_idx, tau, rho_box, head_label)
    if form == "softmax_interval":
        return _build_softmax_interval(scores, target_idx, tau, rho_box, head_label)
    raise ValueError(
        f"unknown cert form {form!r}; expected one of "
        "'v3', 'v2', 'interval', 'softmax_interval'"
    )
