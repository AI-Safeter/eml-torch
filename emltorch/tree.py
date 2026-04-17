"""
Batched EML tree module for GPU-parallel symbolic regression.

Simultaneously evaluates B independent depth-d EML trees on GPU, where
each tree is a perfect binary tree of eml(x, y) = exp(x) - ln(y) nodes.

Node inputs are soft-selected from a base choice set via softmax over
learned logits. After training, logits are snapped to argmax for exact
symbolic expressions.

Base choice set per input slot:
    leaf:     {1, x_1, ..., x_V, <combos>}
    internal: {1, x_1, ..., x_V, <combos>, f_child}

Combos (active only when V >= 2) encode 2-variable linear pre-features:
    x_i + x_j  for each unordered pair i<j                (V*(V-1)/2 entries)
    x_i - x_j  for each ordered pair i!=j                 (V*(V-1) entries)

These unblock 2-variable targets like softmax[0] = sigmoid(x_1 - x_2),
which otherwise require an external linear stage before the EML tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .operator import safe_eml


def enumerate_combos(num_vars: int) -> list[tuple[str, int, int]]:
    """Return the ordered list of (op, i, j) combos active for V variables.

    op is 'add' or 'sub'; indices i, j are 0-based variable indices.
    Empty when V < 2. The order here defines the choice-index mapping used
    everywhere (tree forward, symbolic, polish).
    """
    combos: list[tuple[str, int, int]] = []
    if num_vars < 2:
        return combos
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            combos.append(("add", i, j))
    for i in range(num_vars):
        for j in range(num_vars):
            if i == j:
                continue
            combos.append(("sub", i, j))
    return combos


def build_base(x: torch.Tensor, num_vars: int, dtype: torch.dtype) -> torch.Tensor:
    """Build the extended base tensor from input x.

    Args:
        x: (B, V, N) input, already in `dtype`.
        num_vars: V (must match x.shape[1]).
        dtype: working dtype.

    Returns:
        (B, C_base, N) with columns [1, x_1, ..., x_V, <combos>] where
        combos follow `enumerate_combos(num_vars)` ordering.
    """
    B, V, N = x.shape
    assert V == num_vars
    ones = torch.ones(B, 1, N, dtype=dtype, device=x.device)
    parts = [ones, x]
    for op, i, j in enumerate_combos(num_vars):
        xi = x[:, i : i + 1]
        xj = x[:, j : j + 1]
        parts.append(xi + xj if op == "add" else xi - xj)
    return torch.cat(parts, dim=1)


def num_combos(num_vars: int) -> int:
    """Number of combo entries for V variables."""
    if num_vars < 2:
        return 0
    return num_vars * (num_vars - 1) // 2 + num_vars * (num_vars - 1)


class BatchedEMLTree(nn.Module):
    """
    GPU-batched EML trees for parallel symbolic regression.

    Args:
        num_trees: Number of independent trees to evaluate in parallel (B).
        depth: EML composition depth. 1 = single eml node, 2 = 3 nodes, 3 = 7 nodes.
        num_vars: Number of input variables (default 1).
        dtype: Working dtype for tree evaluation (torch.complex64 recommended).
        device: CUDA device string.
    """

    def __init__(
        self,
        num_trees: int,
        depth: int,
        num_vars: int = 1,
        dtype: torch.dtype = torch.complex64,
        device: torch.device | str = "cuda:7",
        init_scale: float = 0.1,
        init_mode: str = "uniform",
    ):
        """
        Args:
            init_scale: logit init magnitude. 0.1 → near-uniform softmax
                        (original), 3.0+ → sharply peaked random one-hot
                        (diverse structural starting points).
            init_mode:  "uniform" = all restarts drawn from same randn*scale;
                        "peaked"  = each restart initialized to a random
                                    specific tree (one-hot logits with scale).
        """
        super().__init__()
        assert depth >= 1, "Depth must be >= 1"
        self.num_trees = num_trees
        self.depth = depth
        self.num_vars = num_vars
        self.dtype = dtype

        n_combo = num_combos(num_vars)
        # Leaves pick from {1, x_1..V, <combos>}; internals add f_child.
        # f_child therefore lives at index (1 + V + n_combo) in internal choice sets.
        leaf_choices = 1 + num_vars + n_combo
        internal_choices = leaf_choices + 1
        self.n_combo = n_combo
        self.f_child_idx = leaf_choices          # == 1 + V + n_combo
        num_leaves = 2 ** (depth - 1)

        def _make_logits(shape, n_choices):
            """Build a parameter tensor according to init_mode."""
            if init_mode == "peaked":
                # Each (B, node, input) triple gets a random one-hot peak.
                idx = torch.randint(0, n_choices, shape[:-1], device=device)
                base = torch.randn(*shape, device=device) * 0.1
                base.scatter_(-1, idx.unsqueeze(-1), init_scale)
                return nn.Parameter(base)
            return nn.Parameter(
                torch.randn(*shape, device=device) * init_scale
            )

        # Leaf logits: (B, num_leaves, 2_inputs, choices)
        self.leaf_logits = _make_logits(
            (num_trees, num_leaves, 2, leaf_choices), leaf_choices
        )

        # Internal logits: one tensor per level (levels 2 .. depth)
        self.internal_logits = nn.ParameterList()
        for level in range(2, depth + 1):
            num_nodes = 2 ** (depth - level)
            self.internal_logits.append(_make_logits(
                (num_trees, num_nodes, 2, internal_choices), internal_choices
            ))

        # Temperature inverse — increased during hardening to sharpen softmax
        self.register_buffer("temp_inv", torch.tensor(1.0, device=device))

        # Selection mode:
        #   "softmax"      - plain softmax mixture (original)
        #   "gumbel_soft"  - Gumbel-softmax with tau=1/temp_inv (breaks constant
        #                    attractor via injected noise; paper Fix 1)
        #   "gumbel_hard"  - Gumbel-softmax with straight-through; forward pass
        #                    is one-hot, gradients flow continuously
        self.selection_mode = "softmax"

    # ------------------------------------------------------------------
    # Selection weights
    # ------------------------------------------------------------------

    def _weights(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute selection weights according to self.selection_mode.

        Shape in/out: (..., C). The last dim contains per-choice scores.
        """
        scaled = logits * self.temp_inv
        if self.selection_mode == "softmax":
            return torch.softmax(scaled, dim=-1)
        # Gumbel variants — tau is the temperature (1.0 default; decays during
        # hardening). Equivalent shape, but adds Gumbel noise so Adam can move
        # restarts *between* basins instead of averaging them.
        tau = 1.0 / (self.temp_inv.item() if self.temp_inv.numel() == 1
                     else self.temp_inv.mean().item())
        tau = max(tau, 1e-3)
        hard = self.selection_mode == "gumbel_hard"
        if self.training:
            return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        # At eval time, fall back to deterministic softmax so results are
        # reproducible — equivalent to (sampling without noise) + argmax.
        return torch.softmax(scaled, dim=-1) if not hard else \
            F.one_hot(scaled.argmax(dim=-1), num_classes=scaled.shape[-1]).to(scaled.dtype)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all B trees on input data.

        Args:
            x: Input data.
               Shape (B, N) for single variable, or (B, V, N) for V variables.
               B must equal num_trees. N = number of data points.

        Returns:
            (B, N) tensor of tree outputs (same dtype as self.dtype).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, N)
        B, V, N = x.shape

        device = x.device
        num_leaves = self.leaf_logits.shape[1]

        # Cast input to working dtype
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        # Base choices: [1, x_1, ..., x_V, <combos>] → (B, C_base, N)
        base = build_base(x, self.num_vars, self.dtype)
        C_base = base.shape[1]                    # == 1 + V + n_combo

        # ---- Leaf level ----
        w = self._weights(self.leaf_logits)                           # (B, L, 2, C)
        leaf_ch = base.unsqueeze(1).expand(B, num_leaves, C_base, N)  # (B, L, C, N)

        # Weighted selection: sum_c(w[c] * choice[c]) for left and right inputs
        left = (w[:, :, 0, :].unsqueeze(-1) * leaf_ch).sum(dim=2)    # (B, L, N)
        right = (w[:, :, 1, :].unsqueeze(-1) * leaf_ch).sum(dim=2)   # (B, L, N)
        outputs = safe_eml(left, right)                                # (B, L, N)

        # ---- Internal levels (bottom-up) ----
        for logits in self.internal_logits:
            M = logits.shape[1]  # nodes at this level
            w = self._weights(logits)                             # (B, M, 2, C+1)

            # Pair consecutive children from the level below
            child_left = outputs[:, 0::2, :]    # (B, M, N)
            child_right = outputs[:, 1::2, :]   # (B, M, N)

            # Build choice tensors: [1, x_1..V, <combos>, f_child]
            int_base = base.unsqueeze(1).expand(B, M, C_base, N)       # (B, M, C_base, N)
            l_ch = torch.cat([int_base, child_left.unsqueeze(2)], 2)   # (B, M, C_base+1, N)
            r_ch = torch.cat([int_base, child_right.unsqueeze(2)], 2)

            left = (w[:, :, 0, :].unsqueeze(-1) * l_ch).sum(dim=2)    # (B, M, N)
            right = (w[:, :, 1, :].unsqueeze(-1) * r_ch).sum(dim=2)   # (B, M, N)
            outputs = safe_eml(left, right)                             # (B, M, N)

        # Root is the sole remaining node → (B, 1, N) → (B, N)
        return outputs.squeeze(1)

    # ------------------------------------------------------------------
    # Auxiliary methods
    # ------------------------------------------------------------------

    def entropy(self) -> torch.Tensor:
        """Mean entropy across all softmax distributions (scalar).

        Low entropy = peaked distributions (close to one-hot).
        Used as a penalty term during the hardening phase.
        """
        total = torch.tensor(0.0, device=self.leaf_logits.device)
        count = 0

        for logits in [self.leaf_logits] + list(self.internal_logits):
            w = torch.softmax(logits * self.temp_inv, dim=-1)
            ent = -(w * torch.log(w + 1e-10)).sum(dim=-1)  # (B, nodes, 2)
            total = total + ent.sum()
            count += ent.numel()

        return total / max(count, 1)

    @torch.no_grad()
    def snap(self):
        """Snap all logits to one-hot (argmax). Call after hardening."""
        for logits in [self.leaf_logits] + list(self.internal_logits):
            idx = logits.argmax(dim=-1, keepdim=True)
            logits.zero_()
            logits.scatter_(-1, idx, 100.0)  # large → softmax ≈ 1.0

    def snapped_choices(self) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return integer choice indices (call after snap).

        Returns:
            leaf_idx:     (B, num_leaves, 2) int — per-input choice at each leaf.
            internal_idx: list of (B, M, 2) int — per-input choice at each level.
        """
        leaf_idx = self.leaf_logits.argmax(dim=-1)
        internal_idx = [lg.argmax(dim=-1) for lg in self.internal_logits]
        return leaf_idx, internal_idx

    @property
    def total_params_per_tree(self) -> int:
        """Number of learnable logit values per tree."""
        n = self.leaf_logits[0].numel()
        for lg in self.internal_logits:
            n += lg[0].numel()
        return n
