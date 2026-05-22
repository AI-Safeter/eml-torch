"""End-to-end smoke test for the SMT cert emitter.

Catches the failure mode where eml_tree_to_smt2_intervals depends on an
internal parser module (currently emltorch._ast) — if that module is
removed or renamed without updating the smt.py imports, this test fires.

The test also exercises eml_tree_to_smt2 (the axiomatized-Exp emitter).
"""

from __future__ import annotations

from emltorch.smt import (
    eml_tree_to_smt2,
    eml_tree_to_smt2_intervals,
)


def test_eml_tree_to_smt2_intervals_emits_valid_qf_lra():
    """Smoke test: the interval emitter renders the H31 headline formula
    over a non-empty box and emits a QF_LRA assertion block."""
    out = eml_tree_to_smt2_intervals(
        formula="+0.5954 + (-0.1353) * [eml(L, eml((L - H), 1))]",
        var_ranges={"L": (-0.1, 0.1), "H": (1.94, 2.40)},
        target_op=">",
        target_value=0.10,
        title="cert emit smoke",
    )
    assert "QF_LRA" in out
    assert "(declare-const L Real)" in out
    assert "(declare-const H Real)" in out
    assert "(check-sat)" in out
    # Interval-propagation header line names the analytic bound — present
    # only when the parser + numeric evaluator both succeed.
    assert "Interval-propagation analytic bound" in out


def test_eml_tree_to_smt2_axiomatized_emits_exp_ln():
    """The axiomatized-Exp emitter must declare Exp/Ln and ratio lemmas."""
    out = eml_tree_to_smt2(
        formula="eml(1, x)",
        var_ranges={"x": (0.1, 10.0)},
        target_op=">",
        target_value=0.0,
        title="axiom smoke",
    )
    assert "Exp" in out
    assert "Ln" in out
    assert "(check-sat)" in out
