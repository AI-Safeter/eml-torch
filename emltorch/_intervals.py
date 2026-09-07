"""Outward-rounded real intervals for formula export.

Decimal exp/ln are correctly rounded. Adjacent Decimal values enclose their
exact results; arithmetic uses directed rounding. This bounds real formulas,
not accumulated floating-point error in a PyTorch execution.
"""

from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal, DecimalException

from ._ast import _EML, _Add, _Const, _Div, _Exp, _Mul, _Sub, _Var


def decimal_intervals(node, var_ranges, eps=0.0, clamp_log_eps=0.0):
    down = Context(prec=50, rounding=ROUND_FLOOR)
    up = Context(prec=50, rounding=ROUND_CEILING)
    padding = Decimal(str(eps))
    log_floor = Decimal(str(clamp_log_eps))
    if not padding.is_finite() or padding < 0:
        raise ValueError("eps must be finite and non-negative")
    if not log_floor.is_finite() or log_floor < 0 or log_floor > Decimal("1e30"):
        raise ValueError("clamp_log_eps must be finite and in [0, 1e30]")
    bounds = {}

    def exp_bounds(lo, hi):
        return down.next_minus(down.exp(lo)), up.next_plus(up.exp(hi))

    def visit(n):
        if id(n) in bounds:
            return bounds[id(n)]
        if isinstance(n, _Const):
            lo = hi = Decimal(str(n.value))
        elif isinstance(n, _Var):
            if n.name not in var_ranges:
                raise ValueError(f"Variable {n.name!r} has no range in var_ranges")
            lo, hi = (Decimal(str(v)) for v in var_ranges[n.name])
        elif isinstance(n, _Exp):
            lo, hi = exp_bounds(*visit(n.arg))
        elif isinstance(n, (_Add, _Sub, _Mul, _Div, _EML)):
            a, b = visit(n.left)
            c, d = visit(n.right)
            if isinstance(n, _EML):
                if log_floor > 0:
                    # Real-valued counterpart of safe_eml's finite-input clamps.
                    a, b = max(-80, min(80, a)), max(-80, min(80, b))
                    c = max(log_floor, min(Decimal("1e30"), c))
                    d = max(log_floor, min(Decimal("1e30"), d))
                elif c <= 0:
                    raise ValueError(f"Ln of non-positive interval: [{c}, {d}]")
                exp_lo, exp_hi = exp_bounds(Decimal(a), Decimal(b))
                ln_lo = down.next_minus(down.ln(c))
                ln_hi = up.next_plus(up.ln(d))
                lo, hi = down.subtract(exp_lo, ln_hi), up.subtract(exp_hi, ln_lo)
            elif isinstance(n, _Add):
                lo, hi = down.add(a, c), up.add(b, d)
            elif isinstance(n, _Sub):
                lo, hi = down.subtract(a, d), up.subtract(b, c)
            else:
                if isinstance(n, _Div) and c <= 0 <= d:
                    raise ValueError(f"Division by interval containing 0: [{c}, {d}]")
                lower_op = down.divide if isinstance(n, _Div) else down.multiply
                upper_op = up.divide if isinstance(n, _Div) else up.multiply
                lo = min(lower_op(x, y) for x in (a, b) for y in (c, d))
                hi = max(upper_op(x, y) for x in (a, b) for y in (c, d))
        else:
            raise TypeError(f"Unknown AST node {type(n)}")
        if not lo.is_finite() or not hi.is_finite() or lo > hi:
            raise ValueError(f"Invalid or unbounded interval: [{lo}, {hi}]")
        # Inputs and constants are exact decimal values, so do not perturb
        # their domains (e.g. a positive constant smaller than eps).
        if not isinstance(n, (_Const, _Var)):
            lo, hi = down.subtract(lo, padding), up.add(hi, padding)
        bounds[id(n)] = lo, hi
        return lo, hi

    try:
        visit(node)
    except DecimalException as exc:
        raise ValueError("Formula interval exceeds the supported numeric range") from exc
    return bounds
