"""
emltorch/smt.py — Export EML formulas to Z3 / SMT-LIB for formal verification.

The EML operator `eml(x, y) = exp(x) - ln(y)` and its compositions can be
translated to SMT formulas over the theory of real arithmetic with
transcendentals (using Z3's Python bindings). This enables:

  1. **Bounded safety proofs**: prove that for all r in a ball of radius rho,
     a safety feature does NOT activate — no perturbation within budget
     rho can bypass it. A machine-checkable certificate.

  2. **Exact adversarial witness search**: minimize ||d|| subject to the
     activation condition. For linear-threshold features (SAE + ReLU),
     Z3 recovers the Cauchy-Schwarz optimum exactly.

  3. **SMT-LIB2 export**: produce a .smt2 file that any SMT solver
     (Z3, CVC5, Yices) can verify — portable formal proof.

Key identity for the safety audit:
    For a jbloom SAE ReLU-gated feature k, the activation condition
    simplifies to a linear threshold:
        active_k(r) iff W_enc[k] . r  >  b_enc[k] + W_enc[k] . b_dec
                                       = threshold_k

Z3 proves "for all ||r - r_0|| <= eps : W_enc[k].r <= threshold_k" by
trying to find a counter-example (a violating r). UNSAT = proved safe.

Usage
-----
    from emltorch.smt import (
        certify_linear_threshold_safe,
        find_min_norm_witness,
        eml_formula_to_z3,
        emit_smtlib2,
    )

    proof = certify_linear_threshold_safe(
        W_enc_k=W, threshold_k=4.52, r_center=benign_resid, radius=0.5,
    )
    # proof.verdict in {"SAFE", "UNSAFE"}
    # proof.witness : residual that activates feature (if UNSAFE)
    # proof.smt2    : SMT-LIB2 proof obligation text
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import z3 as _z3  # type: ignore


def _lazy_import_z3():
    import z3

    return z3


def _extract_model_value(model, z3_var) -> float:
    """Z3 model lookup via indexing syntax (avoids .eval naming collision)."""
    v = model[z3_var]
    if v is None:
        return 0.0
    if hasattr(v, "numerator_as_long"):
        num = v.numerator_as_long()
        den = v.denominator_as_long() if hasattr(v, "denominator_as_long") else 1
        return num / max(den, 1)
    if hasattr(v, "as_long"):
        return float(v.as_long())
    if hasattr(v, "as_decimal"):
        s = v.as_decimal(20).rstrip("?")
        return float(s)
    return float(v.__repr__())


# ─── AST bridge (reuses gradient.py parser) ──────────────────────────────────


def eml_formula_to_z3(formula: str, z3_vars: dict[str, "_z3.ArithRef"]):
    """
    Convert an EML formula string (bare or polish format) into a Z3 expression
    evaluated over the given Z3 variable bindings.

    Returns a Z3 ArithRef for the formula value `a + b * inner(z3_vars)`.

    Notes:
        - eml(L, R) uses Z3's Exp/Ln transcendentals (requires Z3 >= 4.11 with
          nlsat enabled).
        - For linear-threshold safety certificates, prefer
          `certify_linear_threshold_safe` which avoids transcendentals entirely
          (decidable via linear real arithmetic).
    """
    z3 = _lazy_import_z3()
    from emltorch.gradient import _parse_inner, _strip_affine
    from emltorch.gradient import (
        _Const,
        _Var,
        _Combo,
        _EML,
        _Add,
        _Sub,
        _Mul,
        _Div,
        _Exp,
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
        if isinstance(n, _Combo):
            lv = z3_vars.get(n.left, z3.RealVal(0))
            rv = z3_vars.get(n.right, z3.RealVal(0))
            return (lv + rv) if n.op == "+" else (lv - rv)
        if isinstance(n, _EML):
            if not hasattr(z3, "Exp") or not hasattr(z3, "Ln"):
                raise RuntimeError(
                    "Your Z3 build lacks transcendentals (Exp/Ln). "
                    "Upgrade Z3 or use linear-threshold certification."
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


# ─── High-level safety certification ─────────────────────────────────────────


@dataclass
class SafetyCertificate:
    verdict: str  # "SAFE" | "UNSAFE" | "UNKNOWN"
    radius: float
    threshold: float
    margin: float  # threshold - W.r_center
    witness: np.ndarray | None  # residual that activates (if UNSAFE)
    witness_norm: float | None  # ||witness - r_center||
    solver_time_s: float
    smt2: str  # SMT-LIB2 proof obligation text


def certify_linear_threshold_safe(
    W_enc_k: np.ndarray,
    threshold_k: float,
    r_center: np.ndarray,
    radius: float,
    norm: str = "l2",
    timeout_ms: int = 30000,
) -> SafetyCertificate:
    """
    Prove or refute: "for all r with ||r - r_center|| <= radius, the safety
    feature with weight W_enc[k] and threshold threshold_k does NOT activate".

    This IS the mathematical safety certificate demanded by EU AI Act 13 and
    NIST AI RMF Measure.1.1 — a deterministic, court-testable statement.

    SAFE iff Z3 returns UNSAT on the negation
        (exists d : ||d||<=radius AND W.(r_center+d) > threshold_k).

    Closed-form (Cauchy-Schwarz L2): min-norm witness = margin / ||W_enc||.
    Z3 provides an independent SMT-verified proof distinct from the analytic bound.
    """
    z3 = _lazy_import_z3()
    import time

    W = np.asarray(W_enc_k, dtype=np.float64)
    r_c = np.asarray(r_center, dtype=np.float64)
    hidden = W.shape[0]
    assert r_c.shape == (hidden,)

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    delta = [z3.Real(f"d{i}") for i in range(hidden)]

    w_dot_delta = z3.Sum([z3.RealVal(float(W[i])) * delta[i] for i in range(hidden)])
    w_dot_rc = float(np.dot(W, r_c))
    margin = threshold_k - w_dot_rc

    if norm == "l2":
        norm_sq = z3.Sum([delta[i] * delta[i] for i in range(hidden)])
        solver.add(norm_sq <= float(radius) ** 2)
    elif norm == "linf":
        for i in range(hidden):
            solver.add(delta[i] <= float(radius))
            solver.add(delta[i] >= -float(radius))
    elif norm == "l1":
        abs_terms = []
        for i in range(hidden):
            ai = z3.Real(f"a{i}")
            solver.add(ai >= delta[i], ai >= -delta[i])
            abs_terms.append(ai)
        solver.add(z3.Sum(abs_terms) <= float(radius))
    else:
        raise ValueError(f"norm must be l2/linf/l1, got {norm!r}")

    # Negation of safety: try to find a δ that activates the feature.
    solver.add(w_dot_delta > float(margin))

    # SMT-LIB only accepts decimal-style numeric literals (no exponential notation).
    # Use ~18 significant digits of fixed-point decimal to preserve float64 precision.
    def _num(x: float) -> str:
        s = f"{x:.18f}"
        if x < 0:
            return f"(- {s[1:]})"
        return s

    # Pick the tightest decidable logic for the norm being used
    logic = "QF_LRA" if norm in ("linf", "l1") else "QF_NRA"
    smt2_lines = [f"(set-logic {logic})"]
    for i in range(hidden):
        smt2_lines.append(f"(declare-const d{i} Real)")
    if norm == "l2":
        sum_terms = " ".join(f"(* d{i} d{i})" for i in range(hidden))
        smt2_lines.append(f"(assert (<= (+ {sum_terms}) {_num(float(radius) ** 2)}))")
    elif norm == "linf":
        for i in range(hidden):
            smt2_lines.append(f"(assert (<= d{i} {_num(float(radius))}))")
            smt2_lines.append(f"(assert (>= d{i} {_num(-float(radius))}))")
    elif norm == "l1":
        for i in range(hidden):
            smt2_lines.append(f"(declare-const a{i} Real)")
            smt2_lines.append(f"(assert (>= a{i} d{i}))")
            smt2_lines.append(f"(assert (>= a{i} (- 0 d{i})))")
        sum_a = " ".join(f"a{i}" for i in range(hidden))
        smt2_lines.append(f"(assert (<= (+ {sum_a}) {_num(float(radius))}))")
    coeff_terms = " ".join(f"(* {_num(float(W[i]))} d{i})" for i in range(hidden))
    smt2_lines.append(f"(assert (> (+ {coeff_terms}) {_num(float(margin))}))")
    smt2_lines.append("(check-sat)")
    smt2_lines.append("(get-model)")
    smt2 = "\n".join(smt2_lines)

    t0 = time.time()
    result = solver.check()
    t1 = time.time() - t0

    witness = None
    witness_norm = None
    if result == z3.sat:
        model = solver.model()
        delta_vals = np.zeros(hidden, dtype=np.float64)
        for i in range(hidden):
            delta_vals[i] = _extract_model_value(model, delta[i])
        witness = r_c + delta_vals
        witness_norm = float(np.linalg.norm(delta_vals))
        verdict = "UNSAFE"
    elif result == z3.unsat:
        verdict = "SAFE"
    else:
        verdict = "UNKNOWN"

    return SafetyCertificate(
        verdict=verdict,
        radius=float(radius),
        threshold=float(threshold_k),
        margin=float(margin),
        witness=witness,
        witness_norm=witness_norm,
        solver_time_s=t1,
        smt2=smt2,
    )


def find_min_norm_witness(
    W_enc_k: np.ndarray,
    threshold_k: float,
    r_center: np.ndarray,
    tol: float = 1e-3,
    max_iters: int = 40,
    timeout_ms: int = 10000,
) -> SafetyCertificate:
    """
    Bisection-search the minimum L2 radius rho* such that the feature becomes
    activatable. By Cauchy-Schwarz this equals margin/||W|| exactly.
    """
    W = np.asarray(W_enc_k, dtype=np.float64)
    analytic = (threshold_k - float(np.dot(W, r_center))) / max(
        float(np.linalg.norm(W)), 1e-30
    )
    assert analytic > 0, "r_center already activates feature (margin <= 0)"

    lo, hi = 0.0, analytic * 1.5
    last_safe: SafetyCertificate | None = None
    last_unsafe: SafetyCertificate | None = None

    for _ in range(max_iters):
        mid = (lo + hi) / 2
        cert = certify_linear_threshold_safe(
            W, threshold_k, r_center, mid, norm="l2", timeout_ms=timeout_ms
        )
        if cert.verdict == "SAFE":
            lo = mid
            last_safe = cert
        elif cert.verdict == "UNSAFE":
            hi = mid
            last_unsafe = cert
        else:
            break
        if hi - lo < tol:
            break

    return last_unsafe or last_safe  # type: ignore


def optimize_min_linf_witness(
    W_enc_k: np.ndarray,
    threshold_k: float,
    r_center: np.ndarray,
    timeout_ms: int = 60000,
) -> dict:
    """
    Use Z3's Optimize tactic to find the EXACT minimum L_inf perturbation
    that activates the feature. This is a linear program:
        minimize  t
        subject to
          |d_i|  <=  t  for all i
          W . d  >   threshold_k - W . r_center   (margin)

    For linear threshold features, Z3 proves optimality (not just finds a
    witness). The closed-form optimum is t* = margin / ||W||_1 attained
    at d_i = t* * sign(W_i) — Z3 recovers this exactly.
    """
    z3 = _lazy_import_z3()
    import time

    W = np.asarray(W_enc_k, dtype=np.float64)
    r_c = np.asarray(r_center, dtype=np.float64)
    hidden = W.shape[0]
    margin = threshold_k - float(np.dot(W, r_c))

    opt = z3.Optimize()
    opt.set("timeout", timeout_ms)

    delta = [z3.Real(f"d{i}") for i in range(hidden)]
    t = z3.Real("t")
    opt.add(t >= 0)
    for i in range(hidden):
        opt.add(delta[i] <= t)
        opt.add(delta[i] >= -t)

    w_dot_delta = z3.Sum([z3.RealVal(float(W[i])) * delta[i] for i in range(hidden)])
    opt.add(w_dot_delta > float(margin))

    h = opt.minimize(t)
    t0 = time.time()
    result = opt.check()
    t1 = time.time() - t0

    out = {
        "verdict": str(result),
        "solver_time_s": t1,
        "margin": margin,
        "min_t": None,
        "witness": None,
        "analytic_linf": margin / max(float(np.linalg.norm(W, ord=1)), 1e-30),
    }
    if result == z3.sat:
        delta_vals = np.zeros(hidden, dtype=np.float64)
        model = opt.model()
        for i in range(hidden):
            delta_vals[i] = _extract_model_value(model, delta[i])
        t_val = _extract_model_value(model, t)
        out["min_t"] = t_val
        out["witness"] = r_c + delta_vals
    return out


def emit_smtlib2(
    W_enc_k: np.ndarray,
    threshold_k: float,
    r_center: np.ndarray,
    radius: float,
    feature_name: str = "feature_k",
) -> str:
    """
    Produce a standalone SMT-LIB2 text asserting the safety-negation.

    UNSAT when passed to any SMT solver (z3, cvc5, yices) proves the
    feature is inactive for all ||delta||_2 <= radius.
    """
    cert = certify_linear_threshold_safe(
        W_enc_k, threshold_k, r_center, radius, norm="l2", timeout_ms=1
    )
    header = (
        f"; SMT-LIB2 proof obligation for EML safety certificate\n"
        f"; Feature: {feature_name}\n"
        f"; Threshold: {threshold_k}\n"
        f"; Center norm: {np.linalg.norm(r_center):.4f}\n"
        f"; Perturbation budget: {radius}\n"
        f"; UNSAT implies feature provably inactive for all ||delta||_2 <= {radius}.\n"
    )
    return header + cert.smt2


__all__ = [
    "SafetyCertificate",
    "eml_formula_to_z3",
    "certify_linear_threshold_safe",
    "find_min_norm_witness",
    "emit_smtlib2",
]
