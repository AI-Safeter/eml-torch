"""Unit tests for emltorch.smt.emit_raw_weight_concentration_cert.

Cert form (QF_LRA, decidable in linear arithmetic):
    Variables:  abs_a_j in [|a_j_obs| * exp(-rho_log), |a_j_obs| * exp(+rho_log)]
                with abs_a_j >= 0
    Claim:      abs_a_target > tau * sum_j abs_a_j
    Negation:   not(abs_a_target > tau * sum_j abs_a_j)  -> UNSAT means SAFE
"""

from __future__ import annotations

import z3

from emltorch.smt import emit_raw_weight_concentration_cert


def _solve(text: str, timeout_ms: int = 2000) -> str:
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    s.add(z3.parse_smt2_string(text))
    return str(s.check())


def test_dominant_target_unsat():
    """|a_target| = 0.95, all others = 0.01 -> SAFE at tau=0.5."""
    abs_a_obs = [0.01, 0.01, 0.01, 0.01, 0.95, 0.01, 0.01]
    text = emit_raw_weight_concentration_cert(
        abs_a_obs, target_idx=4, tau=0.5, rho_log=0.10, head_label="test_dominant"
    )
    assert _solve(text) == "unsat"


def test_uniform_weights_sat():
    """All weights equal -> claim 'target dominates' is FALSE -> SAT."""
    abs_a_obs = [0.20, 0.20, 0.20, 0.20, 0.20]
    text = emit_raw_weight_concentration_cert(
        abs_a_obs, target_idx=2, tau=0.5, rho_log=0.10, head_label="test_uniform"
    )
    assert _solve(text) == "sat"


def test_rho_log_perturbation_makes_unsafe():
    """Borderline target -> at large rho_log the perturbation can flip dominance."""
    abs_a_obs = [0.098, 0.098, 0.098, 0.098, 0.098, 0.51]
    text = emit_raw_weight_concentration_cert(
        abs_a_obs, target_idx=5, tau=0.5, rho_log=0.50, head_label="test_borderline"
    )
    assert _solve(text) == "sat"


def test_qf_lra_logic_set():
    """Cert text should declare QF_LRA logic (no Exp/Ln tokens needed)."""
    abs_a_obs = [0.1, 0.1, 0.8]
    text = emit_raw_weight_concentration_cert(
        abs_a_obs, target_idx=2, tau=0.5, rho_log=0.10
    )
    assert "(set-logic QF_LRA)" in text
    assert "(Exp" not in text
    assert "(Ln" not in text


def test_dual_verify_cvc5():
    """The dominant-target cert should also pass cvc5 (dual-UNSAT)."""
    import cvc5
    import os
    import tempfile

    abs_a_obs = [0.01, 0.01, 0.01, 0.01, 0.95, 0.01, 0.01]
    text = emit_raw_weight_concentration_cert(
        abs_a_obs, target_idx=4, tau=0.5, rho_log=0.10, head_label="dual_verify"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".smt2", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        s = cvc5.Solver()
        try:
            s.setOption("tlimit-per", "5000")
        except Exception:
            pass
        ip = cvc5.InputParser(s)
        ip.setFileInput(cvc5.InputLanguage.SMT_LIB_2_6, path)
        sm = ip.getSymbolManager()
        result = "unknown"
        while True:
            cmd = ip.nextCommand()
            if cmd.isNull():
                break
            out = cmd.invoke(s, sm).strip()
            if out in ("sat", "unsat", "unknown"):
                result = out
                break
        assert result == "unsat", f"cvc5 returned {result!r}"
    finally:
        os.unlink(path)
