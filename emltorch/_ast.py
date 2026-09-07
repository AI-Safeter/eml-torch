"""emltorch/_ast.py, Private AST nodes + parser for EML formula strings.

Used internally by emltorch.smt to translate polished formulas to SMT-LIB2.
Not part of the public API.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# AST node definitions


@dataclass
class _Const:
    value: float

    def __str__(self) -> str:
        return repr(self.value)


@dataclass
class _Var:
    name: str

    def __str__(self) -> str:
        return self.name


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


_Node = _Const | _Var | _EML | _Add | _Sub | _Mul | _Div | _Exp

_ZERO = _Const(0.0)
_ONE = _Const(1.0)


# Numbers include scientific notation; unary signs are parsed separately.
_NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_TOKEN = re.compile(rf"{_NUMBER}|[A-Za-z_]\w*|[()\[\],+*/-]")


def _tokenize(s: str) -> list[str]:
    tokens = []
    pos = 0
    while pos < len(s):
        if s[pos].isspace():
            pos += 1
            continue
        match = _TOKEN.match(s, pos)
        if match is None:
            raise ValueError(f"Unexpected character {s[pos]!r} at position {pos}")
        tokens.append(match.group())
        pos = match.end()
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]):
        self.t = tokens
        self.pos = 0

    def peek(self) -> str | None:
        return self.t[self.pos] if self.pos < len(self.t) else None

    def consume(self, expected: str | None = None) -> str:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of formula")
        if expected is not None and tok != expected:
            raise ValueError(f"Expected {expected!r}, got {tok!r} at token {self.pos}")
        self.pos += 1
        return tok

    def parse_expr(self) -> _Node:
        node = self._parse_product()
        while self.peek() in ("+", "-"):
            op = self.consume()
            right = self._parse_product()
            node = _Add(node, right) if op == "+" else _Sub(node, right)
        return node

    def _parse_product(self) -> _Node:
        node = self._parse_unary()
        while self.peek() in ("*", "/"):
            op = self.consume()
            right = self._parse_unary()
            node = _Mul(node, right) if op == "*" else _Div(node, right)
        return node

    def _parse_unary(self) -> _Node:
        if self.peek() in ("+", "-"):
            op = self.consume()
            node = self._parse_unary()
            if op == "+":
                return node
            return _Const(-node.value) if isinstance(node, _Const) else _Sub(_ZERO, node)
        return self._parse_atom()

    def _parse_atom(self) -> _Node:
        tok = self.consume()
        if tok in ("(", "["):
            node = self.parse_expr()
            self.consume(")" if tok == "(" else "]")
            return node
        if re.fullmatch(_NUMBER, tok):
            value = float(tok)
            if not math.isfinite(value):
                raise ValueError("Formula constants must be finite")
            return _Const(value)
        if re.fullmatch(r"[A-Za-z_]\w*", tok):
            if self.peek() != "(":
                return _Var(tok)
            if tok not in ("eml", "exp"):
                raise ValueError(f"Unsupported function {tok!r}")
            self.consume("(")
            left = self.parse_expr()
            if tok == "eml":
                self.consume(",")
                node = _EML(left, self.parse_expr())
            else:
                node = _Exp(left)
            self.consume(")")
            return node
        raise ValueError(f"Unexpected token {tok!r}")


def _parse_inner(s: str) -> _Node:
    parser = _Parser(_tokenize(s.strip()))
    node = parser.parse_expr()
    if parser.peek() is not None:
        raise ValueError(f"Unexpected trailing token {parser.peek()!r}")
    return node


def _strip_affine(formula: str) -> tuple[float, float, str]:
    """Split the library's affine wrapper, preserving coefficient precision."""
    signed = rf"[+-]?\s*{_NUMBER}"
    match = re.fullmatch(
        rf"({signed})\s*\+\s*\(({signed})\)\s*\*\s*(.+)",
        formula.strip(),
        re.DOTALL,
    )
    if match:
        a = float(re.sub(r"\s+", "", match.group(1)))
        b = float(re.sub(r"\s+", "", match.group(2)))
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("Formula constants must be finite")
        return a, b, match.group(3).strip()
    return 0.0, 1.0, formula.strip()
