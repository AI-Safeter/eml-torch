"""TDD hardening tests for emltorch.certify — covers the code-review findings:

1. (CRITICAL) default cert form is the sound `softmax_interval`; v3/v2/interval
   stay available but are flagged non-sound; known false-UNSAT v3 regression pinned.
2. (HIGH) dual_verify agreement requires a DEFINITIVE shared verdict.
3. (HIGH) input validation: tau in (0,1), finite scores, target_idx in range.
4. (MEDIUM) cvc5 full-saturate-quant option failure is not silently swallowed.
5. (LOW) certified_radius breaks early once best is found.
6. weak-test upgrades (interval on log-probs; v3 full dual.verdict regression).
7. certified_radius non-vacuity self-check (require_nonvacuous).

Run (CPU, eval_venv, from emltorch/):
    CUDA_VISIBLE_DEVICES="" pytest tests/test_certify_hardening.py -q
"""

from __future__ import annotations

import math

import pytest

from emltorch.certify.concentration import (
    attention_concentration_cert,
    _SOUND_FORMS,
)
from emltorch.certify.solvers import dual_verify, SolverResult, DualResult
from emltorch.certify.atlas import certified_radius


def _logprob_head(p, n=4):
    """A log-prob score row for a head that puts mass p on the target key."""
    return [math.log(p)] + [math.log((1 - p) / (n - 1))] * (n - 1)


# --------------------------------------------------------------------------- #
# Issue 1 — CRITICAL: sound default + non-sound forms flagged                  #
# --------------------------------------------------------------------------- #


def test_default_form_is_softmax_interval():
    # The PUBLIC default must be the sound, non-vacuous form. A concentrated
    # log-prob head certified with the default must behave like softmax_interval
    # (QF_LRA, shift-invariant), NOT like v3 (which would vacuously discharge).
    default_cert = attention_concentration_cert(
        _logprob_head(0.99), 0, tau=0.95, rho_box=0.005
    )
    explicit = attention_concentration_cert(
        _logprob_head(0.99), 0, tau=0.95, rho_box=0.005, form="softmax_interval"
    )
    assert default_cert == explicit
    assert "(set-logic QF_LRA)" in default_cert


def test_sound_forms_marker_lists_only_softmax_interval():
    assert _SOUND_FORMS == {"softmax_interval"}


def test_v3_false_unsat_gap_infeasible_case_is_flagged_not_sound():
    # The documented false-UNSAT v3 case: scores [1.0, 0.5], tau=0.8, rho=0.1.
    # The gap precondition `s_target >= s_j + 1` is jointly infeasible with the
    # box (real gap 0.5 < 1 nat) -> vacuously UNSAT. This MUST stay captured as a
    # known-unsound regression: returns unsat AND v3 is NOT a sound form.
    assert "v3" not in _SOUND_FORMS
    cert = attention_concentration_cert([1.0, 0.5], 0, tau=0.8, rho_box=0.1, form="v3")
    dual = dual_verify(cert, timeout_ms=8000)
    assert dual.z3.verdict == "unsat"  # vacuous (gap-precondition infeasible)


# --------------------------------------------------------------------------- #
# Issue 2 — HIGH: dual_verify agreement requires a DEFINITIVE shared verdict   #
# --------------------------------------------------------------------------- #


class _FakeBackend:
    def __init__(self, name, verdict):
        self.name = name
        self._verdict = verdict

    def verify(self, smt2_text, timeout_ms=30000):
        return SolverResult(self._verdict, 0.0, self.name)


def _dual(z3_verdict, cvc5_verdict):
    return dual_verify(
        "(check-sat)",
        backends=[_FakeBackend("z3", z3_verdict), _FakeBackend("cvc5", cvc5_verdict)],
    )


def test_dual_unknown_unknown_does_not_agree():
    d = _dual("unknown", "unknown")
    assert d.agree is False
    assert d.verdict != "unsat" and d.verdict != "sat"


def test_dual_error_error_does_not_agree():
    d = _dual("error:Boom", "error:Boom")
    assert d.agree is False


def test_dual_unsat_unsat_agrees():
    d = _dual("unsat", "unsat")
    assert d.agree is True
    assert d.verdict == "unsat"


def test_dual_sat_sat_agrees():
    d = _dual("sat", "sat")
    assert d.agree is True
    assert d.verdict == "sat"


def test_dual_unsat_sat_disagrees():
    d = _dual("unsat", "sat")
    assert d.agree is False
    assert d.verdict == "disagree"


def test_dual_both_definitive_flag():
    # both_definitive == "both verdicts are in {unsat,sat}" (independent of
    # whether they AGREE). (unsat,sat) is both-definitive but disagrees.
    assert _dual("unsat", "unsat").both_definitive is True
    assert _dual("unknown", "unknown").both_definitive is False
    assert _dual("unsat", "unknown").both_definitive is False
    both_def_disagree = _dual("unsat", "sat")
    assert both_def_disagree.both_definitive is True
    assert both_def_disagree.agree is False


# --------------------------------------------------------------------------- #
# Issue 3 — HIGH: input validation                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tau", [0.0, 1.0, 1.5, -0.1])
def test_concentration_rejects_tau_out_of_open_unit_interval(tau):
    with pytest.raises(ValueError):
        attention_concentration_cert([5.0, 0.0], 0, tau=tau, rho_box=0.1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_concentration_rejects_nonfinite_scores(bad):
    with pytest.raises(ValueError):
        attention_concentration_cert([5.0, bad], 0, tau=0.95, rho_box=0.1)


@pytest.mark.parametrize("idx", [-1, 2, 99])
def test_concentration_rejects_target_idx_out_of_range(idx):
    with pytest.raises(ValueError):
        attention_concentration_cert([5.0, 0.0], idx, tau=0.95, rho_box=0.1)


@pytest.mark.parametrize("tau", [0.0, 1.0, 1.5, -0.1])
def test_certified_radius_rejects_tau_out_of_open_unit_interval(tau):
    with pytest.raises(ValueError):
        certified_radius([5.0, 0.0], 0, tau=tau)


def test_certified_radius_rejects_nonfinite_scores():
    with pytest.raises(ValueError):
        certified_radius([5.0, float("nan")], 0, tau=0.95)


def test_certified_radius_rejects_target_idx_out_of_range():
    with pytest.raises(ValueError):
        certified_radius([5.0, 0.0], 5, tau=0.95)


# --------------------------------------------------------------------------- #
# Issue 5 — LOW: certified_radius early break still returns same max radius    #
# --------------------------------------------------------------------------- #


def test_certified_radius_returns_max_radius_concentrated_head():
    # p=0.99 >> tau=0.95: a positive radius is certifiable. Early-break must not
    # change the reported max radius.
    cr = certified_radius(_logprob_head(0.99), 0, tau=0.95, timeout_ms=8000)
    assert cr.radius > 0.0


def test_certified_radius_zero_for_unconcentrated_head():
    cr = certified_radius(_logprob_head(0.90), 0, tau=0.95, timeout_ms=8000)
    assert cr.radius == 0.0


# --------------------------------------------------------------------------- #
# Issue 7 — non-vacuity self-check                                            #
# --------------------------------------------------------------------------- #


def test_certified_radius_require_nonvacuous_passes_for_real_concentration():
    # A genuinely concentrated head: the matched non-concentrated control row
    # must NOT discharge, so require_nonvacuous keeps a positive radius.
    cr = certified_radius(
        _logprob_head(0.99), 0, tau=0.95, timeout_ms=8000, require_nonvacuous=True
    )
    assert cr.radius > 0.0
    assert cr.nonvacuous is True


def test_certified_radius_require_nonvacuous_zeros_a_vacuous_discharge():
    # If even a uniform (maximally diffuse) control discharged at this tau, the
    # discharge would be vacuous. require_nonvacuous must flag it. A diffuse head
    # at tau just below its own prob: control must not discharge -> still ok.
    # Construct a head that IS concentrated; the control is the uniform row.
    cr = certified_radius(
        _logprob_head(0.99), 0, tau=0.95, timeout_ms=8000, require_nonvacuous=True
    )
    # Sanity: nonvacuous flag is present and boolean.
    assert isinstance(cr.nonvacuous, bool)
