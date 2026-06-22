"""emltorch/_ast.py, Private AST nodes + parser for EML formula strings.

Used internally by emltorch.smt to translate polished formulas to SMT-LIB2.
Not part of the public API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# AST node definitions

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
    op: str  # "+", "-", or "*" (mul combo added 2026-04-24)
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


# Tokenizer + parser

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
        """Parse a parenthesized expression. Supports:
        (a + b), (a - b), (a * b)           -> _Combo (flat pair)
        ((a * b) * c), (a * (b + c)), ...   -> _Mul/_Add/_Sub of sub-AST
        """
        self.consume("(")
        # Left operand can be an atom OR a nested paren expression
        left = self._parse_paren() if self.peek() == "(" else self._parse_atom()
        op = self.peek()
        if op in ("+", "-", "*"):
            self.consume()
            right = self._parse_paren() if self.peek() == "(" else self._parse_atom()
            self.consume(")")
            # If both operands are bare variables/constants, emit flat _Combo
            # (preserves legacy behavior for pair combos from use_mul).
            if isinstance(left, (_Var, _Const)) and isinstance(right, (_Var, _Const)):
                return _Combo(str(left), op, str(right))
            # Otherwise emit proper AST node so nested expressions work
            if op == "+":
                return _Add(left, right)
            if op == "-":
                return _Sub(left, right)
            return _Mul(left, right)
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
