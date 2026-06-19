"""TDD tests for emltorch.certify.solvers — the unified SMT backend that
replaces 31 copy-pasted verify_z3/verify_cvc5 pairs across sae-eml scripts.

Run (CPU, eval_venv): CUDA_VISIBLE_DEVICES="" pytest emltorch/tests/test_certify_solvers.py -q
"""

from __future__ import annotations

import pytest

from emltorch.certify.solvers import (
    Z3Backend,
    CVC5Backend,
    dual_verify,
    SolverResult,
    DualResult,
)

# Trivially decidable QF_LRA certs — no transcendentals, instant for any solver.
UNSAT_SMT2 = (
    "(set-logic QF_LRA)\n(declare-const x Real)\n"
    "(assert (> x 0.0))\n(assert (< x 0.0))\n(check-sat)\n"
)
SAT_SMT2 = (
    "(set-logic QF_LRA)\n(declare-const x Real)\n" "(assert (> x 0.0))\n(check-sat)\n"
)


def test_z3_backend_reports_unsat():
    res = Z3Backend().verify(UNSAT_SMT2)
    assert isinstance(res, SolverResult)
    assert res.verdict == "unsat"
    assert res.solver == "z3"
    assert res.elapsed_s >= 0.0


def test_z3_backend_reports_sat():
    assert Z3Backend().verify(SAT_SMT2).verdict == "sat"


def test_cvc5_backend_reports_unsat_via_tempfile():
    # cvc5 has no parse_smt2_string; the backend must handle the file-only
    # InputParser quirk internally so callers pass a plain string.
    res = CVC5Backend().verify(UNSAT_SMT2)
    assert res.verdict == "unsat"
    assert res.solver == "cvc5"


def test_cvc5_backend_reports_sat():
    assert CVC5Backend().verify(SAT_SMT2).verdict == "sat"


def test_dual_verify_agrees_on_unsat():
    dual = dual_verify(UNSAT_SMT2)
    assert isinstance(dual, DualResult)
    assert dual.z3.verdict == "unsat"
    assert dual.cvc5.verdict == "unsat"
    assert dual.agree is True
    assert dual.verdict == "unsat"


def test_dual_verify_agrees_on_sat():
    dual = dual_verify(SAT_SMT2)
    assert dual.agree is True
    assert dual.verdict == "sat"


def test_solver_verdict_is_normalized_lowercase():
    # Verdicts must be normalized to lowercase 'unsat'/'sat'/'unknown' so
    # downstream tier logic never has to care about solver-specific casing.
    assert Z3Backend().verify(UNSAT_SMT2).verdict in {"unsat", "sat", "unknown"}
