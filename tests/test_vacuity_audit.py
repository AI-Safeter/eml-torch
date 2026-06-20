"""TDD tests for emltorch.certify.vacuity_audit -- the drop-in test-suite that
tells ANYONE whether an attention-concentration / softmax-mass / routing cert
claim is SOUND or VACUOUS.

Operationalizes the two 2026-06-19 audits (VACUITY_AUDIT, CERT_SOUNDNESS_SWEEP)
into four tested checks:

  1. shift-invariance   (shift-variant body on log-probs -> collapses)
  2. non-vacuity control (uniform / low-mass control must NOT discharge)
  3. relative-vs-absolute (excluded-key mass floor; 9%-with-BOS-excluded)
  4. numerical-precision  (certified radius below model round-off)

Run (CPU, eval_venv, from emltorch/):
    CUDA_VISIBLE_DEVICES="" pytest tests/test_vacuity_audit.py -q
"""

from __future__ import annotations

import math

import pytest

from emltorch.certify.vacuity_audit import (
    AuditReport,
    CheckResult,
    vacuity_audit,
    audit_attention_atlas,
    SOUND,
    VACUOUS,
    RELATIVE_ONLY,
    UNDER_PRECISION,
    NOT_CERTIFIED,
)


def _logprob_head(p, n=4):
    """log-prob score row for a head with mass p on the target (key 0)."""
    return [math.log(p)] + [math.log((1 - p) / (n - 1))] * (n - 1)


# --------------------------------------------------------------------------
# Check 1 -- shift-invariance
# --------------------------------------------------------------------------


def test_shift_variant_v3_on_logprobs_flagged_vacuous():
    # v3 = Exp(s_target) - Ln(sumE) > log(tau). On log-probs sumE=1 => Ln(1)=0
    # and the body is Exp(s_target) > log(tau)<0, always true. The shift-variant
    # check must flag this and the overall verdict must be VACUOUS.
    report = vacuity_audit(_logprob_head(0.5), target_idx=0, tau=0.95, form="v3")
    assert isinstance(report, AuditReport)
    assert report.shift_invariance.passed is False
    assert report.verdict == VACUOUS


def test_softmax_interval_is_shift_invariant():
    # softmax_interval = s_target - Ln(sumE) > log(tau): shift-invariant, sound.
    report = vacuity_audit(
        _logprob_head(0.99), target_idx=0, tau=0.95, form="softmax_interval"
    )
    assert report.shift_invariance.passed is True


# --------------------------------------------------------------------------
# Check 2 -- non-vacuity control (uniform / low-mass)
# --------------------------------------------------------------------------


def test_uniform_control_discharges_for_bad_form_so_vacuous():
    # A maximally-diffuse uniform row should NEVER certify concentration. If the
    # bad (v3) form discharges on it, the control fails -> VACUOUS.
    report = vacuity_audit(
        _logprob_head(0.5), target_idx=0, tau=0.95, form="v3", control="uniform"
    )
    assert report.non_vacuity_control.passed is False
    assert report.verdict == VACUOUS


def test_lowmass_control_discharges_for_bad_form_so_vacuous():
    report = vacuity_audit(
        _logprob_head(0.5), target_idx=0, tau=0.95, form="v3", control="lowmass"
    )
    assert report.non_vacuity_control.passed is False
    assert report.verdict == VACUOUS


def test_control_refused_for_sound_form():
    # softmax_interval must REFUSE the uniform/low-mass control (control passes).
    report = vacuity_audit(
        _logprob_head(0.99),
        target_idx=0,
        tau=0.95,
        form="softmax_interval",
        control="uniform",
    )
    assert report.non_vacuity_control.passed is True


# --------------------------------------------------------------------------
# Check 3 -- relative-vs-absolute / mass-floor (the H23 raw-weight defect)
# --------------------------------------------------------------------------


def test_nine_percent_mass_with_bos_excluded_is_relative_only():
    # Raw-weight cert: BOS (idx0) holds 91%, target (idx2) holds 9% of TOTAL
    # mass. With exclude_from_sum={0,5} the cert discharges UNSAT at tau=0.95 --
    # but that is RELATIVE ranking against the surviving keys, not concentration.
    abs_a = [50.0, 0.02, 5.0, 0.02, 0.01, 0.05]  # idx0=BOS, idx5=self; target idx2
    report = vacuity_audit(
        abs_a,
        target_idx=2,
        tau=0.95,
        form="raw_weight",
        exclude_from_sum=(0, 5),
    )
    assert report.mass_floor.passed is False
    assert report.verdict == RELATIVE_ONLY


def test_genuine_global_concentration_passes_mass_floor():
    # Target holds ~98% of TOTAL mass; excluding BOS/self does not change that.
    abs_a = [0.01, 0.02, 5.0, 0.02, 0.01, 0.05]
    report = vacuity_audit(
        abs_a,
        target_idx=2,
        tau=0.95,
        form="raw_weight",
        exclude_from_sum=(0, 5),
    )
    assert report.mass_floor.passed is True


# --------------------------------------------------------------------------
# Check 4 -- numerical-precision floor
# --------------------------------------------------------------------------


def test_radius_below_precision_floor_is_under_precision():
    # p=0.99 @ tau=0.95 certifies a small POSITIVE radius (~0.02). If we demand a
    # margin of 0.05 (above the model's noise), 0.02 < 0.05 -> UNDER-PRECISION.
    # (The cert genuinely discharges -- it's the radius that is too small, not the
    # absence of a cert.)
    report = vacuity_audit(
        _logprob_head(0.99),
        target_idx=0,
        tau=0.95,
        form="softmax_interval",
        precision_floor=0.05,  # demand a radius above this
    )
    assert report.precision_floor.passed is False
    assert report.verdict == UNDER_PRECISION


def test_radius_above_precision_floor_passes():
    report = vacuity_audit(
        _logprob_head(0.999),
        target_idx=0,
        tau=0.5,  # easy claim -> large certifiable radius
        form="softmax_interval",
        precision_floor=0.001,
    )
    assert report.precision_floor.passed is True


# --------------------------------------------------------------------------
# All-four SOUND
# --------------------------------------------------------------------------


def test_genuinely_concentrated_head_is_sound_on_all_four():
    # p=0.99 >> tau=0.95, sound form, no excluded keys, generous precision floor.
    report = vacuity_audit(
        _logprob_head(0.99),
        target_idx=0,
        tau=0.95,
        form="softmax_interval",
        precision_floor=0.001,
        control="uniform",
    )
    assert report.shift_invariance.passed is True
    assert report.non_vacuity_control.passed is True
    assert report.mass_floor.passed is True
    assert report.precision_floor.passed is True
    assert report.verdict == SOUND


# --------------------------------------------------------------------------
# Report shape / explanation
# --------------------------------------------------------------------------


def test_report_has_human_readable_explanation():
    report = vacuity_audit(_logprob_head(0.5), target_idx=0, tau=0.95, form="v3")
    assert isinstance(report.explanation, str)
    assert report.verdict in report.explanation
    # each check name appears in the explanation
    for name in ("shift-invariance", "non-vacuity", "mass", "precision"):
        assert name in report.explanation.lower()


def test_checkresult_carries_detail():
    report = vacuity_audit(_logprob_head(0.5), target_idx=0, tau=0.95, form="v3")
    cr = report.shift_invariance
    assert isinstance(cr, CheckResult)
    assert isinstance(cr.passed, bool)
    assert isinstance(cr.detail, str)
    assert len(cr.detail) > 0


def test_to_json_round_trips():
    import json

    report = vacuity_audit(_logprob_head(0.99), target_idx=0, tau=0.95)
    blob = report.to_json()
    back = json.loads(json.dumps(blob))
    assert back["verdict"] == report.verdict
    assert set(back["checks"]) == {
        "shift_invariance",
        "non_vacuity_control",
        "mass_floor",
        "precision_floor",
    }


# --------------------------------------------------------------------------
# Atlas convenience
# --------------------------------------------------------------------------


def test_audit_attention_atlas_classifies_each_head():
    # Two synthetic heads: one genuinely concentrated, one diffuse log-prob head
    # audited under the bad v3 form (vacuous).
    rows = [
        {"layer": 5, "head": 5, "scores": _logprob_head(0.99), "target_idx": 0},
        {"layer": 0, "head": 0, "scores": _logprob_head(0.10), "target_idx": 0},
    ]
    results = audit_attention_atlas(rows, tau=0.95, form="v3")
    assert len(results) == 2
    # under v3 both are vacuous (the bad form), proving the atlas runs the audit
    assert all(r.verdict == VACUOUS for r in results)


def test_audit_attention_atlas_sound_form_separates_heads():
    rows = [
        {"layer": 5, "head": 5, "scores": _logprob_head(0.99), "target_idx": 0},
        {"layer": 0, "head": 0, "scores": _logprob_head(0.10), "target_idx": 0},
    ]
    results = audit_attention_atlas(
        rows, tau=0.95, form="softmax_interval", precision_floor=0.001
    )
    by_lh = {(r.layer, r.head): r for r in results}
    # concentrated head: sound; diffuse head: NOT-CERTIFIED (its cert never
    # discharges at tau=0.95, so there is no UNSAT to validate).
    assert by_lh[(5, 5)].verdict == SOUND
    assert by_lh[(0, 0)].verdict == NOT_CERTIFIED
