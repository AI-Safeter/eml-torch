"""Generic EML formula translation and real-arithmetic interval obligations."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import z3 as _z3


def _smt_num(value) -> str:
    """Encode finite decimal values without dropping tiny nonzero constants."""
    number = value if isinstance(value, Decimal) else Decimal(str(float(value)))
    if not number.is_finite():
        raise ValueError(f"SMT numeric values must be finite, got {value!r}")
    text = format(number.copy_abs(), "f")
    return f"(- {text})" if number < 0 else text


def _comment(text: str) -> str:
    return " ".join(text.splitlines())


def _validate_var_ranges(var_ranges):
    for name, interval in var_ranges.items():
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) is None:
            raise ValueError(f"Invalid variable name: {name!r}")
        if name in {"Exp", "Ln", "true", "false", "Real", "Int", "Bool"}:
            raise ValueError(f"Reserved SMT variable name: {name!r}")
        if len(interval) != 2:
            raise ValueError(f"Range for {name!r} must contain (lower, upper)")
        lo, hi = interval
        _smt_num(lo)
        _smt_num(hi)
        if lo > hi:
            raise ValueError(f"Range for {name!r} is reversed: {interval!r}")


def _safe_claim(body, target_op, target_value):
    operators = {">": ">", ">=": ">=", "<": "<", "<=": "<=", "==": "=", "!=": "distinct"}
    if target_op not in operators:
        raise ValueError(f"target_op must be one of {list(operators)}, got {target_op!r}")
    return f"({operators[target_op]} {body} {_smt_num(target_value)})"


def _lazy_import_z3():
    import z3

    return z3


def eml_formula_to_z3(formula: str, z3_vars: dict[str, "_z3.ArithRef"]):
    """
    Convert an EML formula string (bare or polish format) into a Z3 expression
    evaluated over the given Z3 variable bindings.

    Returns a Z3 ArithRef for the formula value `a + b * inner(z3_vars)`.

    Notes:
        - eml(L, R) requires custom Z3 bindings exposing Exp/Ln. Standard
          z3-solver wheels do not provide these functions. For portable
          EML export, use eml_tree_to_smt2 or eml_tree_to_smt2_intervals.
    """
    z3 = _lazy_import_z3()
    from emltorch._ast import (
        _EML,
        _Add,
        _Const,
        _Div,
        _Exp,
        _Mul,
        _parse_inner,
        _strip_affine,
        _Sub,
        _Var,
    )

    a, b, inner = _strip_affine(formula)
    node = _parse_inner(inner)

    def emit(n):
        if isinstance(n, _Const):
            return z3.RealVal(n.value)
        if isinstance(n, _Var):
            if n.name not in z3_vars:
                raise KeyError(f"No Z3 binding for variable '{n.name}'")
            return z3_vars[n.name]
        if isinstance(n, _EML):
            if not hasattr(z3, "Exp") or not hasattr(z3, "Ln"):
                raise RuntimeError(
                    "Your Z3 build lacks transcendentals (Exp/Ln). "
                    "Use eml_tree_to_smt2, eml_tree_to_smt2_intervals, "
                    "for portable formula obligations."
                )
            L = emit(n.left)
            R = emit(n.right)
            return z3.Exp(L) - z3.Ln(R)
        if isinstance(n, _Add):
            return emit(n.left) + emit(n.right)
        if isinstance(n, _Sub):
            return emit(n.left) - emit(n.right)
        if isinstance(n, _Mul):
            return emit(n.left) * emit(n.right)
        if isinstance(n, _Div):
            return emit(n.left) / emit(n.right)
        if isinstance(n, _Exp):
            if not hasattr(z3, "Exp"):
                raise RuntimeError("Z3 missing Exp().")
            return z3.Exp(emit(n.arg))
        raise TypeError(f"Unknown AST node {type(n)}")

    body = emit(node)
    return z3.RealVal(a) + z3.RealVal(b) * body


EML_AXIOMS_SMT2 = """\
; ─── Axiomatized Exp + Ln (no transcendentals required) ───
(declare-fun Exp (Real) Real)
(declare-fun Ln  (Real) Real)
; --- Exp axioms ---
(assert (forall ((u Real)) (! (> (Exp u) 0.0)        :pattern ((Exp u)))))
(assert (= (Exp 0.0) 1.0))
(assert (forall ((u Real) (v Real))
    (! (=> (< u v) (< (Exp u) (Exp v)))               :pattern ((Exp u) (Exp v)))))
(assert (forall ((u Real)) (! (=> (> u 0.0) (> (Exp u) 1.0)) :pattern ((Exp u)))))
(assert (forall ((u Real)) (! (=> (< u 0.0) (< (Exp u) 1.0)) :pattern ((Exp u)))))
; --- Ln axioms (domain: v > 0) ---
(assert (= (Ln 1.0) 0.0))
(assert (forall ((u Real) (v Real))
    (! (=> (and (> u 0.0) (> v 0.0) (< u v)) (< (Ln u) (Ln v)))
       :pattern ((Ln u) (Ln v)))))
(assert (forall ((v Real))
    (! (=> (> v 1.0) (> (Ln v) 0.0))                  :pattern ((Ln v)))))
(assert (forall ((v Real))
    (! (=> (and (> v 0.0) (< v 1.0)) (< (Ln v) 0.0))  :pattern ((Ln v)))))
; --- inverse axioms ---
(assert (forall ((x Real)) (! (= (Ln (Exp x)) x)      :pattern ((Ln (Exp x))))))
(assert (forall ((v Real)) (! (=> (> v 0.0) (= (Exp (Ln v)) v))
                                                       :pattern ((Exp (Ln v))))))
; --- numeric anchor:  e ∈ [2.7182, 2.7183] ---
(assert (>= (Exp 1.0) 2.7182))
(assert (<= (Exp 1.0) 2.7183))
"""


def eml_tree_to_smt2(
    formula: str,
    var_ranges: dict[str, tuple[float, float]],
    target_op: str,
    target_value: float,
    title: str = "EML-tree cert",
) -> str:
    """
    Translate an EML formula string (polish or bare) into a portable
    SMT-LIB2 proof obligation:

        ∀ vars within ``var_ranges``:    formula  {target_op}  {target_value}

    Returns a ``.smt2`` text in which `eml(L, R)` is encoded as
    `(- (Exp L) (Ln R))`, with `Exp` and `Ln` declared as uninterpreted
    functions and constrained by the axiom block ``EML_AXIOMS_SMT2``.

    This emitter uses mathematical, unclipped exp/log. It also requires
    every logarithm argument to be positive and every divisor nonzero.
    It does not certify PyTorch roundoff or a surrogate's prediction error.

    The proof obligation is the NEGATION of the SAFE claim, so UNSAT means
    the SAFE claim holds for all variable assignments in the ranges given.

    Args:
        formula:     EML formula string, e.g. ``"3.0 + 1.5 * eml(eml(1, x), 1)"``
                     (or bare ``"eml(...)"``, affine wrapper optional).
        var_ranges:  e.g. ``{"x": (-1.0, 2.0), "gap": (-4.0, -1.0)}``
        target_op:   one of ``">", ">=", "<", "<=", "==", "!="``.
        target_value: numeric RHS of the SAFE claim.
        title:       descriptive header comment.

    The body lists every leaf variable in ``var_ranges`` with its bound;
    every `eml(L, R)` node is rendered as ``(- (Exp L) (Ln R))``; constants
    pass through.  Combos like `x_i+x_j`, `x_i-x_j`, `x_i*x_j` translate to
    `(+ ...)`, `(- ...)`, `(* ...)` literally.

    """
    from emltorch._ast import (
        _EML,
        _Add,
        _Const,
        _Div,
        _Exp,
        _Mul,
        _parse_inner,
        _strip_affine,
        _Sub,
        _Var,
    )

    _validate_var_ranges(var_ranges)
    a, b, inner = _strip_affine(formula)
    node = _parse_inner(inner)

    # collect leaf variable names that appear (cross-check against ranges)
    seen_vars: set[str] = set()
    domains: list[str] = []

    _num = _smt_num

    def emit(n) -> str:
        if isinstance(n, _Const):
            return _num(n.value)
        if isinstance(n, _Var):
            seen_vars.add(n.name)
            return n.name
        if isinstance(n, _EML):
            L = emit(n.left)
            R = emit(n.right)
            domains.append(f"(> {R} 0)")
            return f"(- (Exp {L}) (Ln {R}))"
        if isinstance(n, _Add):
            return f"(+ {emit(n.left)} {emit(n.right)})"
        if isinstance(n, _Sub):
            return f"(- {emit(n.left)} {emit(n.right)})"
        if isinstance(n, _Mul):
            return f"(* {emit(n.left)} {emit(n.right)})"
        if isinstance(n, _Div):
            left, right = emit(n.left), emit(n.right)
            domains.append(f"(distinct {right} 0)")
            return f"(/ {left} {right})"
        if isinstance(n, _Exp):
            return f"(Exp {emit(n.arg)})"
        raise TypeError(f"Unknown AST node {type(n)}")

    body_inner = emit(node)
    body_value = f"(+ {_num(a)} (* {_num(b)} {body_inner}))"

    op_map = {">": ">", ">=": ">=", "<": "<", "<=": "<=", "==": "=", "!=": "distinct"}
    if target_op not in op_map:
        raise ValueError(f"target_op must be one of {list(op_map)}, got {target_op!r}")
    safe_z3 = f"({op_map[target_op]} {body_value} {_num(target_value)})"

    # Negation of SAFE → UNSAT means SAFE holds.
    if domains:
        safe_z3 = "(and " + " ".join(domains + [safe_z3]) + ")"
    neg_safe = f"(not {safe_z3})"

    # Variable declarations + range constraints.
    decl_lines: list[str] = []
    for v in sorted(seen_vars):
        lo, hi = var_ranges.get(v, (None, None))
        if lo is None:
            raise ValueError(f"variable {v!r} appeared in formula but has no range in var_ranges")
        decl_lines.append(f"(declare-const {v} Real)")
        decl_lines.append(f"(assert (>= {v} {_num(lo)}))")
        decl_lines.append(f"(assert (<= {v} {_num(hi)}))")

    header = (
        f"; {_comment(title)}\n"
        f"; Formula: {_comment(formula)}\n"
        f"; Var ranges: {var_ranges}\n"
        f"; Claim (SAFE):  formula {target_op} {target_value}\n"
        f"; UNSAT below proves SAFE for all variable assignments in the ranges.\n"
        f"(set-logic ALL)\n"
    )
    return (
        header
        + EML_AXIOMS_SMT2
        + "\n".join(decl_lines)
        + "\n; Negation of SAFE\n"
        + f"(assert {neg_safe})\n"
        + "(check-sat)\n"
    )


def eml_tree_to_smt2_intervals(
    formula: str,
    var_ranges: dict[str, tuple[float, float]],
    target_op: str,
    target_value: float,
    title: str = "EML-tree interval-propagation cert",
    eps: float = 1e-9,
    clamp_log_eps: float = 1e-6,
) -> str:
    """Emit a QF_LRA obligation using outward-rounded real intervals.

    Exp, EML, variable products, and variable quotients are abstracted by
    fresh reals with enclosing bounds. Addition, subtraction, and scaling by
    constants retain their structure. UNSAT proves the requested bound for
    the real formula; SAT may be an artifact of the interval relaxation and
    does not establish an actual counterexample.

    By default EML uses exp arguments clipped to [-80, 80] and log arguments
    clipped to [clamp_log_eps, 1e30]. Set clamp_log_eps=0 for mathematical
    exp/log without clipping (non-positive log domains raise ValueError).
    Explicit exp(...) always denotes the mathematical exponential.

    eps is optional extra padding; rounding is outward even when eps=0.
    PyTorch roundoff and surrogate prediction error are not certified.
    """
    from ._ast import _EML, _Add, _Const, _Div, _Exp, _Mul, _parse_inner, _Sub, _Var
    from ._intervals import decimal_intervals

    _validate_var_ranges(var_ranges)
    node = _parse_inner(formula)
    bounds = decimal_intervals(node, var_ranges, eps, clamp_log_eps)
    lines = []
    seen_vars = set()
    used_names = set(var_ranges)
    fresh_id = 0

    def fresh(lo, hi):
        nonlocal fresh_id
        while True:
            fresh_id += 1
            name = f"eml_interval_{fresh_id}"
            if name not in used_names:
                used_names.add(name)
                break
        lines.extend(
            [
                f"(declare-const {name} Real)",
                f"(assert (>= {name} {_smt_num(lo)}))",
                f"(assert (<= {name} {_smt_num(hi)}))",
            ]
        )
        return name

    def emit(n):
        if isinstance(n, _Const):
            return _smt_num(n.value)
        if isinstance(n, _Var):
            seen_vars.add(n.name)
            return n.name
        if isinstance(n, _Exp):
            emit(n.arg)
            return fresh(*bounds[id(n)])
        left, right = emit(n.left), emit(n.right)
        if isinstance(n, _Add):
            return f"(+ {left} {right})"
        if isinstance(n, _Sub):
            return f"(- {left} {right})"
        if isinstance(n, _Mul) and isinstance(n.left, _Const):
            return f"(* {left} {right})"
        if isinstance(n, _Mul) and isinstance(n.right, _Const):
            return f"(* {left} {right})"
        if isinstance(n, _Div) and isinstance(n.right, _Const):
            return f"(/ {left} {right})"
        if isinstance(n, (_EML, _Mul, _Div)):
            return fresh(*bounds[id(n)])
        raise TypeError(f"Unknown AST node {type(n)}")

    body = emit(node)
    safe = _safe_claim(body, target_op, target_value)
    var_lines = []
    for name in sorted(seen_vars):
        lo, hi = var_ranges[name]
        var_lines.extend(
            [
                f"(declare-const {name} Real)",
                f"(assert (>= {name} {_smt_num(lo)}))",
                f"(assert (<= {name} {_smt_num(hi)}))",
            ]
        )
    lo, hi = bounds[id(node)]
    return (
        f"; {_comment(title)}\n"
        f"; Formula: {_comment(formula)}\n"
        f"; Var ranges: {var_ranges}\n"
        f"; Claim: formula {target_op} {target_value}\n"
        f"; Interval-propagation analytic bound: formula ∈ [{lo:.6e}, {hi:.6e}]\n"
        "; UNSAT certifies the real interval abstraction. SAT may be spurious.\n"
        "; Floating-point execution and surrogate error are outside this obligation.\n"
        "(set-logic QF_LRA)\n"
        + "\n".join(var_lines + lines)
        + f"\n(assert (not {safe}))\n(check-sat)\n"
    )
