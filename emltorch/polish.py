"""
Post-evolution polish: learnable numeric constants at fixed topology.

After evolutionary search commits to a specific tree topology, this module
replaces every "1" leaf choice with a learnable real-valued parameter and
optimizes those constants (plus an affine wrapper a + b * tree) with Adam.

This is what oxieml does internally during search (relax `One` leaves to
R-valued params) — but they then project the learned value back to `1` in
the output, throwing away the continuous information. We keep it in the
output, so the final formula may contain arbitrary constants like `-2.718`
or `0.567` rather than just `1` and `e`.

Mathematically: turns our discrete search into a two-level optimizer:
  outer: evolutionary search over topology (discrete)
  inner: Adam on constants + affine (continuous, closed gradient)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .operator import safe_eml, safe_eml_param
from .tree import BatchedEMLTree, build_base, enumerate_combos, num_combos


@dataclass
class PolishResult:
    r2: float
    mse: float
    constants: list[float]  # learned per-leaf constants (in tree order)
    a: float  # affine intercept
    b: float  # affine scale
    formula: str  # formula with constants substituted


class _FixedTopologyTree(nn.Module):
    """Evaluate one specific EML tree topology with learnable constants.

    Topology is given as choice indices. Every position whose choice was
    originally `1` (index 0) is replaced with a learnable scalar parameter.
    Variable choices (indices 1..V) and f_child (index V+1) remain fixed.
    """

    def __init__(
        self,
        leaf_choices: torch.Tensor,  # (num_leaves, 2)  int
        internal_choices: list[torch.Tensor],  # list of (M_level, 2) int
        num_vars: int,
        dtype: torch.dtype = torch.float32,
        use_mul: bool = False,
        use_mul3: bool = False,
    ):
        super().__init__()
        self.num_vars = num_vars
        self.dtype = dtype
        self.use_mul = use_mul
        self.use_mul3 = use_mul3
        self.num_leaves = leaf_choices.shape[0]

        self.register_buffer("leaf_choices", leaf_choices.clone())
        self._internal_choices = [c.clone() for c in internal_choices]
        for i, c in enumerate(self._internal_choices):
            self.register_buffer(f"internal_choices_{i}", c)

        # One learnable scalar per (node, input) position whose choice == 0 (was "1").
        # Store them in a flat parameter; a mask tells us which positions use them.
        self.leaf_const_mask = leaf_choices == 0  # (L, 2) bool
        self.internal_const_masks = [c == 0 for c in internal_choices]

        n_leaf_consts = int(self.leaf_const_mask.sum().item())
        n_internal_consts = int(sum(m.sum().item() for m in self.internal_const_masks))
        self.n_constants = n_leaf_consts + n_internal_consts

        # Init constants at 1.0 (matches the original "1" they replace)
        self.constants = nn.Parameter(torch.ones(self.n_constants, dtype=dtype))

        # Precompute flat indices for each mask position so we can look up
        # the right constant on forward
        leaf_flat_idx = torch.full_like(leaf_choices, -1, dtype=torch.long)
        k = 0
        leaf_nodes, leaf_inputs = torch.where(self.leaf_const_mask)
        for ni, ii in zip(leaf_nodes.tolist(), leaf_inputs.tolist()):
            leaf_flat_idx[ni, ii] = k
            k += 1
        self.register_buffer("leaf_flat_idx", leaf_flat_idx)

        self._internal_flat_idxs = []
        for m_tensor, m in zip(self.internal_const_masks, internal_choices):
            flat = torch.full_like(m, -1, dtype=torch.long)
            nodes, inputs = torch.where(m_tensor)
            for ni, ii in zip(nodes.tolist(), inputs.tolist()):
                flat[ni, ii] = k
                k += 1
            self._internal_flat_idxs.append(flat)
        for i, fi in enumerate(self._internal_flat_idxs):
            self.register_buffer(f"internal_flat_idx_{i}", fi)

    @property
    def internal_flat_idxs(self):
        return [
            getattr(self, f"internal_flat_idx_{i}")
            for i in range(len(self._internal_choices))
        ]

    @property
    def internal_choices(self):
        return [
            getattr(self, f"internal_choices_{i}")
            for i in range(len(self._internal_choices))
        ]

    def _select(
        self, choice_idx, flat_idx_tensor, base_vals, child_val, node_i, input_i
    ):
        """
        Map one (node, input) choice to its actual value tensor at this forward.

        Choice layout (0-indexed):
            0                           -> learnable constant (was '1')
            1 .. V                      -> x_v
            V+1 .. V+K                  -> combo v (K = num_combos(V))
            V+K+1                       -> f_child (internal nodes only)
        """
        V = self.num_vars
        K = base_vals.shape[0] - 1 - V  # combo count from base width
        if choice_idx == 0:
            c_idx = int(flat_idx_tensor[node_i, input_i].item())
            const_val = self.constants[c_idx]
            return const_val.expand(base_vals.shape[-1])
        if choice_idx <= V + K:
            return base_vals[choice_idx]
        # f_child
        assert child_val is not None
        return child_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (V, N)
        returns: (N,)
        """
        V, N = x.shape
        assert V == self.num_vars
        # Use the shared base builder. build_base expects (B, V, N); we run
        # polish single-batch so B=1, then squeeze.
        base = build_base(
            x.to(self.dtype).unsqueeze(0),
            V,
            self.dtype,
            use_mul=self.use_mul,
            use_mul3=self.use_mul3,
        ).squeeze(0)
        # base: (C_base, N) where C_base = 1 + V + num_combos(V, use_mul, use_mul3)

        # Leaf level — evaluate all leaves
        leaf_outputs = []
        for node in range(self.num_leaves):
            left_choice = int(self.leaf_choices[node, 0].item())
            right_choice = int(self.leaf_choices[node, 1].item())
            left_val = self._select(
                left_choice, self.leaf_flat_idx, base, None, node, 0
            )
            right_val = self._select(
                right_choice, self.leaf_flat_idx, base, None, node, 1
            )
            leaf_outputs.append(safe_eml(left_val, right_val))
        outputs = torch.stack(leaf_outputs, dim=0)  # (L, N)

        # Internal levels (bottom-up)
        for lvl, (choices, flat_idx) in enumerate(
            zip(self.internal_choices, self.internal_flat_idxs)
        ):
            M = choices.shape[0]
            new_outputs = []
            for node in range(M):
                child_left = outputs[2 * node]
                child_right = outputs[2 * node + 1]
                left_choice = int(choices[node, 0].item())
                right_choice = int(choices[node, 1].item())
                left_val = self._select(
                    left_choice, flat_idx, base, child_left, node, 0
                )
                right_val = self._select(
                    right_choice, flat_idx, base, child_right, node, 1
                )
                new_outputs.append(safe_eml(left_val, right_val))
            outputs = torch.stack(new_outputs, dim=0)  # (M, N)

        return outputs[0]  # Root


def polish(
    tree: BatchedEMLTree,
    best_idx: int,
    x: torch.Tensor,  # (V, N) or (N,)
    y: torch.Tensor,  # (N,)
    var_names: list[str],
    n_iters: int = 2000,
    lr: float = 5e-2,
    device: str = "cuda:7",
    warm_a: float = 0.0,
    warm_b: float = 1.0,
    const_reg: float = 0.0,
    min_b_abs: float = 0.0,
    range_reg: float = 0.0,
) -> PolishResult:
    """
    Fit learnable numeric constants at the fixed topology of `tree[best_idx]`.

    `warm_a`/`warm_b` initialize the affine wrapper — pass the values from
    the evolution result to ensure polish starts from evolution's best R²
    (not worse).

    Result contains the learned constants, affine coefficients, and a
    formula string with all constants substituted in.
    """
    # Pull choice indices from the snapped tree
    leaf_idx_all, internal_idx_all = tree.snapped_choices()
    leaf_choices = leaf_idx_all[best_idx].to("cpu")  # (L, 2)
    internal_choices = [c[best_idx].to("cpu") for c in internal_idx_all]

    # Build the specialized tree
    dtype = torch.float32
    if x.dim() == 1:
        x = x.unsqueeze(0)
    V, N = x.shape
    fixed = _FixedTopologyTree(
        leaf_choices=leaf_choices,
        internal_choices=internal_choices,
        num_vars=V,
        dtype=dtype,
        use_mul=getattr(tree, "use_mul", False),
        use_mul3=getattr(tree, "use_mul3", False),
    ).to(device)

    x_dev = x.to(device, dtype)
    y_dev = y.to(device, dtype)

    # Warm-start affine wrapper from evolution's best values
    a = nn.Parameter(torch.full((1,), warm_a, device=device, dtype=dtype))
    b = nn.Parameter(torch.full((1,), warm_b, device=device, dtype=dtype))

    # Verify warm start matches evolution before any Adam step
    with torch.no_grad():
        initial_pred = fixed(x_dev)
        initial_fit = a + b * initial_pred
        initial_mse = (initial_fit - y_dev).pow(2).mean().item()

    opt = torch.optim.Adam(list(fixed.parameters()) + [a, b], lr=lr)
    best_mse = initial_mse
    best_state = {
        "constants": fixed.constants.detach().clone(),
        "a": warm_a,
        "b": warm_b,
    }

    for step in range(n_iters):
        opt.zero_grad()
        pred = fixed(x_dev)
        fit = a + b * pred
        mse_loss = (fit - y_dev).pow(2).mean()
        loss = mse_loss
        if const_reg > 0.0:
            loss = loss + const_reg * (fixed.constants - 1.0).pow(2).sum()
        if range_reg > 0.0 and min_b_abs > 0.0:
            # Penalize |b| dropping below min_b_abs — forces the tree to
            # carry the signal's dynamic range instead of the affine wrapper.
            b_shortfall = torch.relu(min_b_abs - b.abs())
            loss = loss + range_reg * b_shortfall.pow(2).sum()
        if not torch.isfinite(loss):
            # Perturb constants slightly and continue
            with torch.no_grad():
                fixed.constants.mul_(0.9)
                fixed.constants.add_(torch.randn_like(fixed.constants) * 0.01)
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(list(fixed.parameters()) + [a, b], 1.0)
        opt.step()

        if mse_loss.item() < best_mse:
            best_mse = mse_loss.item()
            best_state = {
                "constants": fixed.constants.detach().clone(),
                "a": float(a.detach().item()),
                "b": float(b.detach().item()),
            }

    # Restore best state
    if best_state is not None:
        with torch.no_grad():
            fixed.constants.copy_(best_state["constants"])

    # Safeguard: never return a polish that made things worse than warm-start.
    # If best_mse tracking got polluted (NaN/inf interference), revert.
    import math as _math

    if not _math.isfinite(best_mse) or best_mse > initial_mse:
        with torch.no_grad():
            fixed.constants.fill_(1.0)
        best_state = {
            "constants": fixed.constants.detach().clone(),
            "a": warm_a,
            "b": warm_b,
        }

    # Final eval
    with torch.no_grad():
        pred = fixed(x_dev)
        fit = best_state["a"] + best_state["b"] * pred
        final_mse = (fit - y_dev).pow(2).mean().item()

    # Second safeguard: post-restoration MSE can differ from tracked best_mse
    # when constants are large (numerical drift). Fall back if final > initial.
    if not _math.isfinite(final_mse) or final_mse > initial_mse:
        with torch.no_grad():
            fixed.constants.fill_(1.0)
            pred = fixed(x_dev)
            fit = warm_a + warm_b * pred
            final_mse = (fit - y_dev).pow(2).mean().item()
        best_state = {
            "constants": fixed.constants.detach().clone(),
            "a": warm_a,
            "b": warm_b,
        }

    ss_tot = ((y_dev - y_dev.mean()).pow(2).sum()).item()
    r2 = 1 - final_mse * N / max(ss_tot, 1e-12)

    # Build formula with constants substituted
    constants_list = best_state["constants"].detach().cpu().tolist()
    formula = _format_with_constants(
        leaf_choices,
        internal_choices,
        var_names,
        fixed.leaf_flat_idx.cpu(),
        [fi.cpu() for fi in fixed.internal_flat_idxs],
        constants_list,
        use_mul=getattr(tree, "use_mul", False),
        use_mul3=getattr(tree, "use_mul3", False),
    )
    full = f"{best_state['a']:+.4f} + ({best_state['b']:+.4f}) * " f"[{formula}]"

    return PolishResult(
        r2=r2,
        mse=final_mse,
        constants=constants_list,
        a=best_state["a"],
        b=best_state["b"],
        formula=full,
    )


def _format_with_constants(
    leaf_choices,
    internal_choices,
    var_names,
    leaf_flat_idx,
    internal_flat_idxs,
    constants,
    use_mul: bool = False,
    use_mul3: bool = False,
):
    """Recursively format the tree with learned constants in place of '1' slots."""
    from .tree import enumerate_triples

    V = len(var_names)
    combos = enumerate_combos(V, use_mul=use_mul)
    triples = enumerate_triples(V, use_mul3=use_mul3)
    K = len(combos)
    T = len(triples)

    def combo_str(k: int) -> str:
        op, i, j = combos[k]
        sym = {"add": "+", "sub": "-", "mul": "*"}[op]
        return f"({var_names[i]} {sym} {var_names[j]})"

    def triple_str(t: int) -> str:
        i, j, k = triples[t]
        # Nested binary form so gradient/Z3 parser handles it.
        return f"(({var_names[i]} * {var_names[j]}) * {var_names[k]})"

    def choice_str(idx, flat_idx_tensor, node, input_side, child_str):
        if idx == 0:
            k = int(flat_idx_tensor[node, input_side].item())
            return f"{constants[k]:.4f}"
        if 1 <= idx <= V:
            return var_names[idx - 1]
        if idx <= V + K:
            return combo_str(idx - V - 1)
        if idx <= V + K + T:
            return triple_str(idx - V - K - 1)
        return child_str

    def leaf_expr(leaf_idx):
        left = choice_str(
            int(leaf_choices[leaf_idx, 0]), leaf_flat_idx, leaf_idx, 0, None
        )
        right = choice_str(
            int(leaf_choices[leaf_idx, 1]), leaf_flat_idx, leaf_idx, 1, None
        )
        return f"eml({left}, {right})"

    # Build bottom-up
    level_exprs = [leaf_expr(i) for i in range(leaf_choices.shape[0])]
    for lvl, (choices, flat_idx) in enumerate(
        zip(internal_choices, internal_flat_idxs)
    ):
        new_exprs = []
        M = choices.shape[0]
        for node in range(M):
            child_l = level_exprs[2 * node]
            child_r = level_exprs[2 * node + 1]
            left = choice_str(int(choices[node, 0]), flat_idx, node, 0, child_l)
            right = choice_str(int(choices[node, 1]), flat_idx, node, 1, child_r)
            new_exprs.append(f"eml({left}, {right})")
        level_exprs = new_exprs

    return level_exprs[0]


# ─── Parameterized EML extension (H22h follow-up) ─────────────────────────


class _FixedTopologyTreeParam(_FixedTopologyTree):
    """Same fixed-topology tree as `_FixedTopologyTree` but with per-internal-
    node learnable scaling factors α, β applied at each `eml(L, R)` evaluation:

        eml_param(L, R; α, β) = α · exp(L) − β · ln(R)

    Default α = β = 1 reduces to the standard EML operator.

    α and β are learnable nn.Parameters of shape (n_internal_nodes,).
    Internal nodes are flattened bottom-up across levels (leaves are
    NOT parameterized — α=β=1 fixed there to keep parameter count low).

    Motivation (H22h finding): standard EML at depth 4 fits cross-model
    attention α coefficient at HELDOUT R² 0.78, just below poly K=5's 0.82.
    Per-node α, β add ~30 free parameters at depth 4 — comparable to poly
    K=5's 21 quintic features in 2 variables.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Count internal nodes (NOT leaves): sum of M across internal levels.
        self.n_internal_nodes = sum(c.shape[0] for c in self._internal_choices)
        # Learnable α, β per internal node, init=1.0 → reduces to standard EML.
        self.eml_alpha = nn.Parameter(
            torch.ones(self.n_internal_nodes, dtype=self.dtype)
        )
        self.eml_beta = nn.Parameter(
            torch.ones(self.n_internal_nodes, dtype=self.dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Same as parent forward but uses safe_eml_param at internal levels.
        Leaf level uses standard safe_eml (α=β=1 implicit)."""
        V, N = x.shape
        assert V == self.num_vars
        base = build_base(
            x.to(self.dtype).unsqueeze(0),
            V,
            self.dtype,
            use_mul=self.use_mul,
            use_mul3=self.use_mul3,
        ).squeeze(0)

        # Leaves — standard safe_eml.
        leaf_outputs = []
        for node in range(self.num_leaves):
            left_choice = int(self.leaf_choices[node, 0].item())
            right_choice = int(self.leaf_choices[node, 1].item())
            left_val = self._select(
                left_choice, self.leaf_flat_idx, base, None, node, 0
            )
            right_val = self._select(
                right_choice, self.leaf_flat_idx, base, None, node, 1
            )
            leaf_outputs.append(safe_eml(left_val, right_val))
        outputs = torch.stack(leaf_outputs, dim=0)

        # Internal levels — parameterized via per-node α, β.
        node_idx = 0
        for lvl, (choices, flat_idx) in enumerate(
            zip(self.internal_choices, self.internal_flat_idxs)
        ):
            M = choices.shape[0]
            new_outputs = []
            for node in range(M):
                child_left = outputs[2 * node]
                child_right = outputs[2 * node + 1]
                left_choice = int(choices[node, 0].item())
                right_choice = int(choices[node, 1].item())
                left_val = self._select(
                    left_choice, flat_idx, base, child_left, node, 0
                )
                right_val = self._select(
                    right_choice, flat_idx, base, child_right, node, 1
                )
                a = self.eml_alpha[node_idx]
                b = self.eml_beta[node_idx]
                new_outputs.append(safe_eml_param(left_val, right_val, a, b))
                node_idx += 1
            outputs = torch.stack(new_outputs, dim=0)

        return outputs[0]


def polish_param(
    tree: BatchedEMLTree,
    best_idx: int,
    x: torch.Tensor,
    y: torch.Tensor,
    var_names: list[str],
    n_iters: int = 2000,
    lr: float = 5e-2,
    device: str = "cuda:7",
    warm_a: float = 0.0,
    warm_b: float = 1.0,
    alpha_beta_reg: float = 1e-3,
) -> PolishResult:
    """Polish constants AND per-internal-node α, β scaling factors.

    Drop-in replacement for `polish()` that uses `_FixedTopologyTreeParam`.
    α, β init at 1.0 (reduces to standard EML), regularized via
    `alpha_beta_reg * Σ((α-1)^2 + (β-1)^2)` to prefer near-identity scaling
    unless the data demands otherwise.

    Returns: PolishResult with `constants` containing leaf+internal scalars
    AND a flattened `[α_0, β_0, α_1, β_1, ...]` tail (n_internal_nodes × 2).
    The `formula` string is identical to standard polish (does not yet
    surface α, β — future enhancement).
    """
    leaf_idx_all, internal_idx_all = tree.snapped_choices()
    leaf_choices = leaf_idx_all[best_idx].to("cpu")
    internal_choices = [c[best_idx].to("cpu") for c in internal_idx_all]

    dtype = torch.float32
    if x.dim() == 1:
        x = x.unsqueeze(0)
    V, N = x.shape
    fixed = _FixedTopologyTreeParam(
        leaf_choices=leaf_choices,
        internal_choices=internal_choices,
        num_vars=V,
        dtype=dtype,
        use_mul=getattr(tree, "use_mul", False),
        use_mul3=getattr(tree, "use_mul3", False),
    ).to(device)

    x_dev = x.to(device, dtype)
    y_dev = y.to(device, dtype)

    a = nn.Parameter(torch.full((1,), warm_a, device=device, dtype=dtype))
    b = nn.Parameter(torch.full((1,), warm_b, device=device, dtype=dtype))

    with torch.no_grad():
        initial_pred = fixed(x_dev)
        initial_mse = (a + b * initial_pred - y_dev).pow(2).mean().item()

    opt = torch.optim.Adam(list(fixed.parameters()) + [a, b], lr=lr)
    best_mse = initial_mse
    best_state = {
        "constants": fixed.constants.detach().clone(),
        "alpha": fixed.eml_alpha.detach().clone(),
        "beta": fixed.eml_beta.detach().clone(),
        "a": warm_a,
        "b": warm_b,
    }

    for step in range(n_iters):
        opt.zero_grad()
        pred = fixed(x_dev)
        fit = a + b * pred
        mse_loss = (fit - y_dev).pow(2).mean()
        loss = mse_loss
        # Regularize α, β toward 1 (standard EML default).
        if alpha_beta_reg > 0.0:
            loss = loss + alpha_beta_reg * (
                (fixed.eml_alpha - 1.0).pow(2).sum()
                + (fixed.eml_beta - 1.0).pow(2).sum()
            )
        if not torch.isfinite(loss):
            with torch.no_grad():
                fixed.constants.mul_(0.9)
                fixed.constants.add_(torch.randn_like(fixed.constants) * 0.01)
                fixed.eml_alpha.mul_(0.9).add_(0.1)  # pull back toward 1
                fixed.eml_beta.mul_(0.9).add_(0.1)
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(list(fixed.parameters()) + [a, b], 1.0)
        opt.step()

        if mse_loss.item() < best_mse:
            best_mse = mse_loss.item()
            best_state = {
                "constants": fixed.constants.detach().clone(),
                "alpha": fixed.eml_alpha.detach().clone(),
                "beta": fixed.eml_beta.detach().clone(),
                "a": float(a.detach().item()),
                "b": float(b.detach().item()),
            }

    # Restore best state
    with torch.no_grad():
        fixed.constants.copy_(best_state["constants"])
        fixed.eml_alpha.copy_(best_state["alpha"])
        fixed.eml_beta.copy_(best_state["beta"])

    import math as _math

    if not _math.isfinite(best_mse) or best_mse > initial_mse:
        with torch.no_grad():
            fixed.constants.fill_(1.0)
            fixed.eml_alpha.fill_(1.0)
            fixed.eml_beta.fill_(1.0)
        best_state = {
            "constants": fixed.constants.detach().clone(),
            "alpha": fixed.eml_alpha.detach().clone(),
            "beta": fixed.eml_beta.detach().clone(),
            "a": warm_a,
            "b": warm_b,
        }

    with torch.no_grad():
        pred = fixed(x_dev)
        fit = best_state["a"] + best_state["b"] * pred
        final_mse = (fit - y_dev).pow(2).mean().item()

    ss_tot = ((y_dev - y_dev.mean()).pow(2).sum()).item()
    r2 = 1 - final_mse * N / max(ss_tot, 1e-12)

    constants_list = best_state["constants"].detach().cpu().tolist()
    alpha_list = best_state["alpha"].detach().cpu().tolist()
    beta_list = best_state["beta"].detach().cpu().tolist()

    formula = _format_with_constants(
        leaf_choices,
        internal_choices,
        var_names,
        fixed.leaf_flat_idx.cpu(),
        [fi.cpu() for fi in fixed.internal_flat_idxs],
        constants_list,
        use_mul=getattr(tree, "use_mul", False),
        use_mul3=getattr(tree, "use_mul3", False),
    )
    # Append α, β summary to the formula string (for human inspection).
    n_internal = len(alpha_list)
    ab_summary = (
        f"  [α range {min(alpha_list):.3f}..{max(alpha_list):.3f}, "
        f"β range {min(beta_list):.3f}..{max(beta_list):.3f}, "
        f"n_internal={n_internal}]"
    )
    full = (
        f"{best_state['a']:+.4f} + ({best_state['b']:+.4f}) * "
        f"[{formula}]{ab_summary}"
    )

    # Pack constants + flattened (α, β) into the .constants tail so callers
    # reconstructing the tree can retrieve them.  Tail layout:
    #   constants[: n_const]                 : leaf+internal scalars
    #   constants[n_const : n_const + n_internal]      : alpha
    #   constants[n_const + n_internal : ...]          : beta
    full_constants = constants_list + alpha_list + beta_list

    return PolishResult(
        r2=r2,
        mse=final_mse,
        constants=full_constants,
        a=best_state["a"],
        b=best_state["b"],
        formula=full,
    )
