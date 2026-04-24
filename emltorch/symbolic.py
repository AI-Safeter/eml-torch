"""
Extract symbolic expressions from snapped EML trees.

After Phase 3 (snap), every softmax distribution is one-hot. We walk
the integer choice indices to build a nested eml(...) string, then
optionally rewrite known sub-expressions into standard notation.
"""

from __future__ import annotations

import re
import torch
from typing import TYPE_CHECKING

from .tree import enumerate_combos, enumerate_triples

if TYPE_CHECKING:
    from .tree import BatchedEMLTree


def extract_expressions(
    tree: BatchedEMLTree,
    tree_indices: list[int],
    var_names: list[str],
) -> list[str]:
    """
    Build symbolic expression strings for selected trees.

    Args:
        tree: A snapped BatchedEMLTree.
        tree_indices: Batch indices of the trees to extract.
        var_names: Human-readable names for each input variable.

    Returns:
        List of expression strings, one per index.
    """
    leaf_idx, internal_idx = tree.snapped_choices()

    use_mul = getattr(tree, "use_mul", False)
    use_mul3 = getattr(tree, "use_mul3", False)
    combo_strs = _combo_strings(var_names, use_mul=use_mul, use_mul3=use_mul3)
    results = []
    for b in tree_indices:
        if tree.depth == 1:
            raw = _leaf_expr(b, 0, leaf_idx, var_names, combo_strs)
        else:
            raw = _internal_expr(
                b,
                len(internal_idx) - 1,
                0,
                leaf_idx,
                internal_idx,
                var_names,
                combo_strs,
            )
        results.append(raw)
    return results


def _combo_strings(
    var_names: list[str], use_mul: bool = False, use_mul3: bool = False
) -> list[str]:
    """Render each combo entry as a math string.

    Order: pair combos (add, sub, [mul]) then triple combos (mul3) —
    same order as `enumerate_combos` followed by `enumerate_triples`.
    """
    out = []
    for op, i, j in enumerate_combos(len(var_names), use_mul=use_mul):
        if op == "add":
            out.append(f"({var_names[i]} + {var_names[j]})")
        elif op == "sub":
            out.append(f"({var_names[i]} - {var_names[j]})")
        else:  # mul
            out.append(f"({var_names[i]} * {var_names[j]})")
    for i, j, k in enumerate_triples(len(var_names), use_mul3=use_mul3):
        # Emit as nested pair-mul so the gradient / Z3 parser (which expects
        # binary `(atom op atom)`) handles the triple without changes.
        out.append(f"(({var_names[i]} * {var_names[j]}) * {var_names[k]})")
    return out


def annotate(expr: str) -> str:
    """Rewrite known EML sub-patterns into standard math notation.

    This is cosmetic — the raw eml(...) string is always authoritative.
    """
    out = expr
    prev = None
    while out != prev:
        prev = out
        # eml(X, 1) → exp(X) for any single-token X
        out = re.sub(r"eml\((\w+), 1\)", r"exp(\1)", out)
        # eml(1, 1) → e
        out = out.replace("eml(1, 1)", "e")
    return out


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _leaf_expr(
    b: int,
    node: int,
    leaf_idx: torch.Tensor,
    var_names: list[str],
    combo_strs: list[str],
) -> str:
    left = _choice_str(leaf_idx[b, node, 0].item(), var_names, combo_strs, child=None)
    right = _choice_str(leaf_idx[b, node, 1].item(), var_names, combo_strs, child=None)
    return f"eml({left}, {right})"


def _internal_expr(
    b: int,
    level: int,
    node: int,
    leaf_idx: torch.Tensor,
    internal_idx: list[torch.Tensor],
    var_names: list[str],
    combo_strs: list[str],
) -> str:
    # Recurse to children
    if level == 0:
        child_l = _leaf_expr(b, 2 * node, leaf_idx, var_names, combo_strs)
        child_r = _leaf_expr(b, 2 * node + 1, leaf_idx, var_names, combo_strs)
    else:
        child_l = _internal_expr(
            b, level - 1, 2 * node, leaf_idx, internal_idx, var_names, combo_strs
        )
        child_r = _internal_expr(
            b, level - 1, 2 * node + 1, leaf_idx, internal_idx, var_names, combo_strs
        )

    left = _choice_str(
        internal_idx[level][b, node, 0].item(), var_names, combo_strs, child=child_l
    )
    right = _choice_str(
        internal_idx[level][b, node, 1].item(), var_names, combo_strs, child=child_r
    )
    return f"eml({left}, {right})"


def _choice_str(
    idx: int, var_names: list[str], combo_strs: list[str], child: str | None
) -> str:
    """Map integer choice index to string.

    Layout: 0='1', 1..V=vars, V+1..V+K=combos, V+K+1=f_child (only at internal).
    """
    V = len(var_names)
    K = len(combo_strs)
    if idx == 0:
        return "1"
    if idx <= V:
        return var_names[idx - 1]
    if idx <= V + K:
        return combo_strs[idx - V - 1]
    if child is not None:
        return child
    return "?"
