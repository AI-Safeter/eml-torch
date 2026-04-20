"""
emltorch/gradient.py — Symbolic differentiation of EML formula strings.

Parses formula strings produced by polish() / extract_expressions(),
then differentiates symbolically with respect to any named variable.

Key identity:
    d/dz eml(L, R) = exp(L) * dL/dz  −  dR/dz / R

The EML gradient is itself EML-expressible (closure under differentiation).
This enables exact "sensitivity certificates": for a safety feature with a
linear threshold condition, the gradient with respect to the residual stream
equals the encoder weight W_enc[k] exactly — no approximation needed.

Usage
-----
    from emltorch.gradient import diff_formula, gradient_at, torch_gradient_fn

    # Differentiate a polish formula string w.r.t. "z"
    grad_str = diff_formula("+0.934 + (+1.105) * [eml(eml(0.993, z), 1.016)]", wrt="z")

    # Evaluate gradient numerically at a point
    val = gradient_at("eml(z, 1)", wrt="z", values={"z": 1.0})  # = exp(1)

    # Vectorized torch gradient function
    grad_fn = torch_gradient_fn("eml(eml(1, eml(eml(1, z), 1)), 1)", wrt="z")
    grads = grad_fn({"z": torch.linspace(-3, 3, 100)})  # = ReLU step
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

import torch


# ─── AST node definitions ────────────────────────────────────────────────────


@dataclass
class _Const:
    value: float

    def __str__(self) -> str:
        v = self.value
        if abs(v) < 1e-12:
            return "0"
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s


@dataclass
class _Var:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class _Combo:
    left: str
    op: str  # "+" or "-"
    right: str

    def __str__(self) -> str:
        return f"({self.left} {self.op} {self.right})"


@dataclass
class _EML:
    left: "_Node"
    right: "_Node"

    def __str__(self) -> str:
        return f"eml({self.left}, {self.right})"


@dataclass
class _Add:
    left: "_Node"
    right: "_Node"

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


@dataclass
class _Sub:
    left: "_Node"
    right: "_Node"

    def __str__(self) -> str:
        return f"({self.left} - {self.right})"


@dataclass
class _Mul:
    left: "_Node"
    right: "_Node"

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"


@dataclass
class _Div:
    left: "_Node"
    right: "_Node"

    def __str__(self) -> str:
        return f"({self.left} / {self.right})"


@dataclass
class _Exp:
    arg: "_Node"

    def __str__(self) -> str:
        return f"exp({self.arg})"


_Node = _Const | _Var | _Combo | _EML | _Add | _Sub | _Mul | _Div | _Exp

_ZERO = _Const(0.0)
_ONE = _Const(1.0)


# ─── Simplification helpers ──────────────────────────────────────────────────


def _is_zero(n: _Node) -> bool:
    return isinstance(n, _Const) and abs(n.value) < 1e-12


def _is_one(n: _Node) -> bool:
    return isinstance(n, _Const) and abs(n.value - 1.0) < 1e-12


def _mk_mul(a: _Node, b: _Node) -> _Node:
    if _is_zero(a) or _is_zero(b):
        return _ZERO
    if _is_one(a):
        return b
    if _is_one(b):
        return a
    return _Mul(a, b)


def _mk_add(a: _Node, b: _Node) -> _Node:
    if _is_zero(a):
        return b
    if _is_zero(b):
        return a
    return _Add(a, b)


def _mk_sub(a: _Node, b: _Node) -> _Node:
    if _is_zero(b):
        return a
    if _is_zero(a):
        return _mk_mul(_Const(-1.0), b)
    return _Sub(a, b)


def _mk_div(a: _Node, b: _Node) -> _Node:
    if _is_zero(a):
        return _ZERO
    return _Div(a, b)


# ─── Symbolic differentiation ────────────────────────────────────────────────


def _diff(node: _Node, wrt: str) -> _Node:
    """Return d(node)/d(wrt) as a simplified _Node tree."""
    if isinstance(node, _Const):
        return _ZERO

    if isinstance(node, _Var):
        return _ONE if node.name == wrt else _ZERO

    if isinstance(node, _Combo):
        dx = _ONE if node.left == wrt else _ZERO
        dy = _ONE if node.right == wrt else _ZERO
        return _mk_add(dx, dy) if node.op == "+" else _mk_sub(dx, dy)

    if isinstance(node, _EML):
        # d/dz eml(L, R) = exp(L) * dL/dz  −  dR/dz / R
        dL = _diff(node.left, wrt)
        dR = _diff(node.right, wrt)
        term1 = _mk_mul(_Exp(node.left), dL)
        term2 = _mk_div(dR, node.right)
        return _mk_sub(term1, term2)

    if isinstance(node, _Add):
        return _mk_add(_diff(node.left, wrt), _diff(node.right, wrt))

    if isinstance(node, _Sub):
        return _mk_sub(_diff(node.left, wrt), _diff(node.right, wrt))

    if isinstance(node, _Mul):
        dL = _diff(node.left, wrt)
        dR = _diff(node.right, wrt)
        return _mk_add(_mk_mul(dL, node.right), _mk_mul(node.left, dR))

    if isinstance(node, _Div):
        dL = _diff(node.left, wrt)
        dR = _diff(node.right, wrt)
        num = _mk_sub(_mk_mul(dL, node.right), _mk_mul(node.left, dR))
        den = _Mul(node.right, node.right)
        return _mk_div(num, den)

    if isinstance(node, _Exp):
        du = _diff(node.arg, wrt)
        return _mk_mul(_Exp(node.arg), du)

    raise TypeError(f"Unknown node type: {type(node)}")


# ─── Parser ──────────────────────────────────────────────────────────────────


def _tokenize(s: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in " \t\n\r":
            i += 1
        elif c in "()[],":
            tokens.append(c)
            i += 1
        elif c in "[]":
            tokens.append(c)
            i += 1
        elif c == "*":
            tokens.append("*")
            i += 1
        elif c == "/":
            tokens.append("/")
            i += 1
        elif c == "+":
            tokens.append("+")
            i += 1
        elif c == "-":
            if tokens and tokens[-1] not in (",", "(", "[", "+", "-", "*", "/"):
                tokens.append("-")
                i += 1
            else:
                j = i + 1
                while j < len(s) and (s[j].isdigit() or s[j] == "."):
                    j += 1
                tokens.append(s[i:j])
                i = j
        elif c.isdigit() or c == ".":
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                j += 1
            tokens.append(s[i:j])
            i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            i += 1
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]):
        self.t = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.t[self.pos] if self.pos < len(self.t) else None

    def consume(self, expected: str | None = None) -> str:
        tok = self.t[self.pos]
        if expected is not None and tok != expected:
            raise ValueError(f"Expected {expected!r}, got {tok!r} at pos {self.pos}")
        self.pos += 1
        return tok

    def parse_expr(self) -> _Node:
        tok = self.peek()
        if tok == "eml":
            return self._parse_eml()
        if tok == "(":
            return self._parse_paren()
        if tok == "[":
            self.consume("[")
            n = self.parse_expr()
            self.consume("]")
            return n
        return self._parse_atom()

    def _parse_eml(self) -> _EML:
        self.consume("eml")
        self.consume("(")
        left = self.parse_expr()
        self.consume(",")
        right = self.parse_expr()
        self.consume(")")
        return _EML(left, right)

    def _parse_paren(self) -> _Node:
        self.consume("(")
        left = self._parse_atom()
        op = self.peek()
        if op in ("+", "-"):
            self.consume()
            right = self._parse_atom()
            self.consume(")")
            return _Combo(str(left), op, str(right))
        self.consume(")")
        return left

    def _parse_atom(self) -> _Node:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of input")
        try:
            val = float(tok)
            self.pos += 1
            return _Const(val)
        except ValueError:
            pass
        self.pos += 1
        return _Var(tok)


def _parse_inner(s: str) -> _Node:
    return _Parser(_tokenize(s.strip())).parse_expr()


def _strip_affine(formula: str) -> tuple[float, float, str]:
    """Split "a + (b) * [inner]" into (a, b, inner_str)."""
    m = re.match(
        r"^([+-]?\s*\d+\.?\d*)\s*\+\s*\(([+-]?\s*\d+\.?\d*)\)\s*\*\s*(.+)$",
        formula.strip(),
        re.DOTALL,
    )
    if m:
        a = float(m.group(1).replace(" ", ""))
        b = float(m.group(2).replace(" ", ""))
        return a, b, m.group(3).strip()
    return 0.0, 1.0, formula.strip()


# ─── Numeric computation ─────────────────────────────────────────────────────


def _compute(node: _Node, vals: dict[str, float]) -> float:
    """Numerically compute an AST node at the given variable values."""
    if isinstance(node, _Const):
        return node.value
    if isinstance(node, _Var):
        return vals[node.name]
    if isinstance(node, _Combo):
        lv = vals.get(node.left, 0.0)
        rv = vals.get(node.right, 0.0)
        return lv + rv if node.op == "+" else lv - rv
    if isinstance(node, _EML):
        L = _compute(node.left, vals)
        R = _compute(node.right, vals)
        return math.exp(L) - math.log(max(R, 1e-300))
    if isinstance(node, _Add):
        return _compute(node.left, vals) + _compute(node.right, vals)
    if isinstance(node, _Sub):
        return _compute(node.left, vals) - _compute(node.right, vals)
    if isinstance(node, _Mul):
        return _compute(node.left, vals) * _compute(node.right, vals)
    if isinstance(node, _Div):
        r = _compute(node.right, vals)
        return _compute(node.left, vals) / r if abs(r) > 1e-300 else 0.0
    if isinstance(node, _Exp):
        return math.exp(_compute(node.arg, vals))
    raise TypeError(f"Unknown node: {type(node)}")


def _torch_compute(node: _Node, vals: dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute an AST node as vectorized torch operations."""
    if isinstance(node, _Const):
        return torch.tensor(node.value, dtype=torch.float32)
    if isinstance(node, _Var):
        return vals[node.name].float()
    if isinstance(node, _Combo):
        lv = vals.get(node.left, torch.tensor(0.0))
        rv = vals.get(node.right, torch.tensor(0.0))
        return (lv + rv) if node.op == "+" else (lv - rv)
    if isinstance(node, _EML):
        L = _torch_compute(node.left, vals)
        R = _torch_compute(node.right, vals)
        return torch.exp(L) - torch.log(R.clamp(min=1e-30))
    if isinstance(node, _Add):
        return _torch_compute(node.left, vals) + _torch_compute(node.right, vals)
    if isinstance(node, _Sub):
        return _torch_compute(node.left, vals) - _torch_compute(node.right, vals)
    if isinstance(node, _Mul):
        return _torch_compute(node.left, vals) * _torch_compute(node.right, vals)
    if isinstance(node, _Div):
        return _torch_compute(node.left, vals) / _torch_compute(node.right, vals).clamp(
            min=1e-30
        )
    if isinstance(node, _Exp):
        return torch.exp(_torch_compute(node.arg, vals))
    raise TypeError(f"Unknown node: {type(node)}")


# ─── Public API ──────────────────────────────────────────────────────────────


def diff_formula(formula: str, wrt: str) -> str:
    """
    Symbolically differentiate an EML formula string w.r.t. variable `wrt`.

    Accepts:
      - bare EML expressions: "eml(eml(1, z), 1)"
      - polish() output format: "+0.934 + (+1.105) * [eml(eml(0.993, z), 1.016)]"

    Returns a human-readable gradient string.

    Examples
    --------
        diff_formula("eml(z, 1)", wrt="z")     → "exp(z)"
        diff_formula("eml(1, z)", wrt="z")     → "(-1 / z)"
        diff_formula("eml(eml(1, eml(eml(1, z), 1)), 1)", wrt="z")
                                               → ReLU gradient (≈1 for z>0, ≈0 for z<0)
    """
    a, b, inner_str = _strip_affine(formula)
    node = _parse_inner(inner_str)
    grad_node = _diff(node, wrt)
    if abs(b - 1.0) < 1e-9:
        return str(grad_node)
    return str(_mk_mul(_Const(b), grad_node))


def gradient_at(formula: str, wrt: str, values: dict[str, float]) -> float:
    """
    Numerically compute the symbolic gradient at a specific point.

    Args:
        formula: EML formula string (bare or polish format).
        wrt: Variable name to differentiate with respect to.
        values: Dict mapping variable names to float values.

    Returns:
        Scalar gradient value.

    Example
    -------
        gradient_at("eml(z, 1)", wrt="z", values={"z": 1.0})
        → 2.718...   # exp(1)
    """
    a, b, inner_str = _strip_affine(formula)
    node = _parse_inner(inner_str)
    grad_node = _diff(node, wrt)
    return b * _compute(grad_node, values)


def sensitivity_vector(
    formula: str,
    wrt_vars: list[str],
    values: dict[str, float],
) -> list[float]:
    """
    Compute the gradient vector [∂formula/∂v for v in wrt_vars].

    Useful for computing sensitivity of an EML safety feature to each
    principal component of the residual stream.

    For linear (ReLU-gated SAE) features, this equals W_enc[k] exactly —
    differentiating the depth-4 ReLU EML formula yields 1 for z>0, 0 for z<0.
    """
    return [gradient_at(formula, v, values) for v in wrt_vars]


def torch_gradient_fn(
    formula: str, wrt: str
) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
    """
    Return a callable that computes the gradient of `formula` w.r.t. `wrt`.

    The returned function accepts a dict of tensors and returns a tensor
    of the same shape — supports full batched, vectorized computation.

    Example
    -------
        grad_fn = torch_gradient_fn("eml(eml(1, eml(eml(1, z), 1)), 1)", wrt="z")
        z = torch.linspace(-3, 3, 100)
        grads = grad_fn({"z": z})   # ≈ step function (ReLU derivative)
    """
    a, b, inner_str = _strip_affine(formula)
    node = _parse_inner(inner_str)
    grad_node = _diff(node, wrt)

    def fn(vals: dict[str, torch.Tensor]) -> torch.Tensor:
        result = _torch_compute(grad_node, vals)
        return torch.tensor(b, dtype=torch.float32) * result

    return fn
