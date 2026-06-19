"""TDD tests for emltorch.certify.concentration — the attention-concentration
cert builder absorbed from the script-trapped _cert_v3.py + 30x-duplicated _num.

Run (CPU, eval_venv, from emltorch/):
    CUDA_VISIBLE_DEVICES="" pytest tests/test_certify_concentration.py -q
"""

from __future__ import annotations

import math

import pytest

from emltorch.certify.concentration import attention_concentration_cert, num_to_smt
from emltorch.certify.solvers import dual_verify

# A sharply concentrated head in RAW-LOGIT space: target dominates by ~10 nats.
# Here sumE = Σ exp(s_j) is a large non-trivial partition function, so the v3
# Ln(sumE) branch does real work (the load-bearing regime).
CONCENTRATED = [10.0, 0.0, 0.0, 0.0]


def test_num_to_smt_renders_negative_as_smt_unary_minus():
    assert num_to_smt(-0.5) == "(- 0.500000000000000000)"
    assert num_to_smt(0.5) == "0.500000000000000000"


def test_v3_cert_contains_load_bearing_ln_of_sum():
    cert = attention_concentration_cert(
        CONCENTRATED, 0, tau=0.95, rho_box=0.1, form="v3"
    )
    assert "(set-logic ALL)" in cert
    assert "(declare-const sumE Real)" in cert  # the partition function is materialized
    assert "(Ln sumE)" in cert  # Ln of a non-trivial sum => load-bearing
    assert "(check-sat)" in cert


def test_v2_cert_is_exp_only_decorative():
    cert = attention_concentration_cert(
        CONCENTRATED, 0, tau=0.95, rho_box=0.1, form="v2"
    )
    # v2 uses eml(s, 1) = Exp(s) - Ln(1); Ln(1)=0 => EML decorative, no sumE.
    assert "(Ln 1.0)" in cert
    assert "sumE" not in cert


def test_unknown_form_raises():
    with pytest.raises(ValueError):
        attention_concentration_cert(
            CONCENTRATED, 0, tau=0.95, rho_box=0.1, form="bogus"
        )


def test_v3_two_key_discharges_dual_unsat():
    # [5, 0] @ tau=0.95: target dominates by 5 nats; both z3 and cvc5
    # (the latter only with full-saturate-quant baked into CVC5Backend) discharge.
    cert = attention_concentration_cert([5.0, 0.0], 0, tau=0.95, rho_box=0.1, form="v3")
    dual = dual_verify(cert, timeout_ms=15000)
    assert dual.agree is True
    assert dual.verdict == "unsat"


def test_v3_multi_key_discharges_dual_unsat():
    # 4-key concentrated head discharges on both solvers (depth > 2 works).
    cert = attention_concentration_cert(
        [4.0, 0.0, 0.0, 0.0], 0, tau=0.9, rho_box=0.1, form="v3"
    )
    dual = dual_verify(cert, timeout_ms=15000)
    assert dual.agree is True
    assert dual.verdict == "unsat"


def test_interval_form_is_qf_lra_with_no_transcendentals():
    # The robust form precomputes Exp/Ln of box endpoints in Python and emits
    # pure linear arithmetic: decidable, both solvers, no quantifiers.
    cert = attention_concentration_cert(
        [5.0, 0.0], 0, tau=0.95, rho_box=0.1, form="interval"
    )
    assert "(set-logic QF_LRA)" in cert
    assert "(Exp " not in cert and "(Ln " not in cert
    assert "forall" not in cert


def test_interval_discharges_where_axiomatized_was_fragile():
    # [3,0]@0.9 and [6,0,0,0]@0.95 gave z3/cvc5 DISAGREEMENT under the
    # axiomatized form; the interval form must dual-UNSAT them robustly.
    for scores, tau in ([3.0, 0.0], 0.9), ([6.0, 0.0, 0.0, 0.0], 0.95):
        cert = attention_concentration_cert(
            scores, 0, tau=tau, rho_box=0.1, form="interval"
        )
        dual = dual_verify(cert, timeout_ms=10000)
        assert dual.agree is True, (scores, tau, dual)
        assert dual.verdict == "unsat", (scores, tau, dual)


def test_interval_does_not_falsely_discharge_diffuse_head():
    # A diffuse head (target barely ahead) must NOT prove >=0.95 concentration:
    # soundness guard — interval relaxation stays SAT, never a false UNSAT.
    cert = attention_concentration_cert(
        [0.1, 0.0, 0.0, 0.0], 0, tau=0.95, rho_box=0.1, form="interval"
    )
    dual = dual_verify(cert, timeout_ms=10000)
    assert dual.verdict == "sat"


def _logprob_head(p, n=4):
    """A log-prob score row for a head that puts mass p on the target key."""
    return [math.log(p)] + [math.log((1 - p) / (n - 1))] * (n - 1)


def test_softmax_interval_is_qf_lra_shift_invariant_form():
    cert = attention_concentration_cert(
        _logprob_head(0.99), 0, tau=0.95, rho_box=0.005, form="softmax_interval"
    )
    assert "(set-logic QF_LRA)" in cert
    assert "(Exp " not in cert and "(Ln " not in cert and "forall" not in cert


def test_softmax_interval_certifies_genuinely_concentrated_head():
    # prob_target = 0.99 >> tau=0.95: dual-UNSAT at a margin-appropriate rho.
    cert = attention_concentration_cert(
        _logprob_head(0.99), 0, tau=0.95, rho_box=0.005, form="softmax_interval"
    )
    dual = dual_verify(cert, timeout_ms=8000)
    assert dual.agree is True
    assert dual.verdict == "unsat"


def test_softmax_interval_is_non_vacuous_refuses_unconcentrated_head():
    # THE honesty test: prob_target = 0.90 < tau=0.95 must NOT discharge --
    # the shift-invariant softmax form cannot be fooled the way v3 is.
    cert = attention_concentration_cert(
        _logprob_head(0.90), 0, tau=0.95, rho_box=0.005, form="softmax_interval"
    )
    dual = dual_verify(cert, timeout_ms=8000)
    assert dual.verdict == "sat"


def test_v3_is_vacuous_on_logprobs_documented_regression():
    # Pins the finding: v3 on a NON-concentrated log-prob head (p=0.5) still
    # discharges -> vacuous. This regression guards the documented limitation so
    # the tool's docs and the non-vacuity guard can never silently drift.
    cert = attention_concentration_cert(
        _logprob_head(0.5), 0, tau=0.95, rho_box=0.0, form="v3"
    )
    dual = dual_verify(cert, timeout_ms=8000)
    assert dual.z3.verdict == "unsat"  # vacuous discharge -- NOT a real cert
