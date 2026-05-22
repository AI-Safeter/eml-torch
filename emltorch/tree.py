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
from .operator import safe_eml


def enumerate_combos(
    num_vars: int, use_mul: bool = False
) -> list[tuple[str, int, int]]:
    """Return the ordered list of (op, i, j) pair combos active for V variables.

    op is 'add', 'sub', or (optional) 'mul'; indices i, j are 0-based variable
    indices. Empty when V < 2. The order here defines the choice-index mapping
    used everywhere (tree forward, symbolic, polish).

    If `use_mul` is True, the combo list is extended with unordered pairs
    x_i * x_j for i<j. Opt-in; default preserves backward compatibility.
    Multiplicative combos unblock targets like h * SiLU(z) (multiplicative gates)
    where the EML operator's additive-only composition has documented ceilings.
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
    if use_mul:
        for i in range(num_vars):
            for j in range(i + 1, num_vars):
                combos.append(("mul", i, j))
    return combos


def enumerate_triples(
    num_vars: int, use_mul3: bool = False
) -> list[tuple[int, int, int]]:
    """Return the ordered list of unordered triple products x_i * x_j * x_k
    for i<j<k. Only populated when `use_mul3=True` and V >= 3.

    Triple combos live AFTER the pair combos in the choice index space; the
    layout is therefore [1, x_1..V, <pair_combos>, <triple_combos>, f_child].
    Backward-compat: with `use_mul3=False` (default), this returns [].
    """
    if not use_mul3 or num_vars < 3:
        return []
    triples: list[tuple[int, int, int]] = []
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            for k in range(j + 1, num_vars):
                triples.append((i, j, k))
    return triples


def build_base(
    x: torch.Tensor,
    num_vars: int,
    dtype: torch.dtype,
    use_mul: bool = False,
    use_mul3: bool = False,
) -> torch.Tensor:
    """Build the extended base tensor from input x.

    Args:
        x: (B, V, N) input, already in `dtype`.
        num_vars: V (must match x.shape[1]).
        dtype: working dtype.
        use_mul: include x_i * x_j pair combos (see enumerate_combos).
        use_mul3: include x_i * x_j * x_k triple combos (V >= 3).

    Returns:
        (B, C_base, N) with columns [1, x_1, ..., x_V, <pair_combos>,
        <triple_combos>]. Pair combos follow `enumerate_combos(num_vars, use_mul)`;
        triple combos follow `enumerate_triples(num_vars, use_mul3)`.
    """
    B, V, N = x.shape
    assert V == num_vars
    ones = torch.ones(B, 1, N, dtype=dtype, device=x.device)
    parts = [ones, x]
    for op, i, j in enumerate_combos(num_vars, use_mul=use_mul):
        xi = x[:, i : i + 1]
        xj = x[:, j : j + 1]
        if op == "add":
            parts.append(xi + xj)
        elif op == "sub":
            parts.append(xi - xj)
        else:  # mul
            parts.append(xi * xj)
    for i, j, k in enumerate_triples(num_vars, use_mul3=use_mul3):
        parts.append(x[:, i : i + 1] * x[:, j : j + 1] * x[:, k : k + 1])
    return torch.cat(parts, dim=1)


def num_combos(num_vars: int, use_mul: bool = False, use_mul3: bool = False) -> int:
    """Number of combo entries for V variables (pairs and optionally triples)."""
    if num_vars < 2:
        return 0
    base = num_vars * (num_vars - 1) // 2 + num_vars * (num_vars - 1)
    if use_mul:
        base += num_vars * (num_vars - 1) // 2
    if use_mul3 and num_vars >= 3:
        # C(V, 3)
        base += num_vars * (num_vars - 1) * (num_vars - 2) // 6
    return base


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
        device: torch.device | str | None = None,
        init_scale: float = 0.1,
        init_mode: str = "uniform",
        use_mul: bool = False,
        use_mul3: bool = False,
    ):
        """
        Args:
            init_scale: logit init magnitude. 0.1 → near-uniform softmax
                        (original), 3.0+ → sharply peaked random one-hot
                        (diverse structural starting points).
            init_mode:  "uniform" = all restarts drawn from same randn*scale;
                        "peaked"  = each restart initialized to a random
                                    specific tree (one-hot logits with scale).
            use_mul: include x_i * x_j pre-feature combos in the leaf/internal
                        choice set. Opt-in; default preserves backward behavior.
            use_mul3: include x_i * x_j * x_k triple-product combos (V >= 3).
                        Opt-in; default preserves backward behavior.
        """
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        assert depth >= 1, "Depth must be >= 1"
        self.num_trees = num_trees
        self.depth = depth
        self.num_vars = num_vars
        self.dtype = dtype
        self.use_mul = use_mul
        self.use_mul3 = use_mul3

        n_combo = num_combos(num_vars, use_mul=use_mul, use_mul3=use_mul3)
        # Leaves pick from {1, x_1..V, <combos>}; internals add f_child.
        # f_child therefore lives at index (1 + V + n_combo) in internal choice sets.
        leaf_choices = 1 + num_vars + n_combo
        internal_choices = leaf_choices + 1
        self.n_combo = n_combo
        self.f_child_idx = leaf_choices  # == 1 + V + n_combo
        num_leaves = 2 ** (depth - 1)

        def _make_logits(shape, n_choices):
            """Build a parameter tensor according to init_mode."""
            if init_mode == "peaked":
                # Each (B, node, input) triple gets a random one-hot peak.
                idx = torch.randint(0, n_choices, shape[:-1], device=device)
                base = torch.randn(*shape, device=device) * 0.1
                base.scatter_(-1, idx.unsqueeze(-1), init_scale)
                return nn.Parameter(base)
            return nn.Parameter(torch.randn(*shape, device=device) * init_scale)

        # Leaf logits: (B, num_leaves, 2_inputs, choices)
        self.leaf_logits = _make_logits(
            (num_trees, num_leaves, 2, leaf_choices), leaf_choices
        )

        # Internal logits: one tensor per level (levels 2 .. depth)
        self.internal_logits = nn.ParameterList()
        for level in range(2, depth + 1):
            num_nodes = 2 ** (depth - level)
            self.internal_logits.append(
                _make_logits(
                    (num_trees, num_nodes, 2, internal_choices), internal_choices
                )
            )

        # Temperature inverse, increased during hardening to sharpen softmax
        self.register_buffer("temp_inv", torch.tensor(1.0, device=device))

        # Buffers for scale-invariant input normalization
        self.register_buffer(
            "x_mean", torch.zeros(1, num_vars, 1, dtype=dtype, device=device)
        )
        self.register_buffer(
            "x_std", torch.ones(1, num_vars, 1, dtype=dtype, device=device)
        )
        self.register_buffer(
            "normalize_inputs", torch.tensor(False, dtype=torch.bool, device=device)
        )

    def set_normalization_stats(self, x_mean: torch.Tensor, x_std: torch.Tensor):
        """Set cached input normalization mean and standard deviation, and enable normalization."""
        self.x_mean.copy_(x_mean)
        self.x_std.copy_(x_std)
        self.normalize_inputs.copy_(
            torch.tensor(True, dtype=torch.bool, device=self.normalize_inputs.device)
        )

    # ------------------------------------------------------------------
    # Selection weights
    # ------------------------------------------------------------------

    def _weights(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax selection weights over the last dim, temperature-scaled."""
        return torch.softmax(logits * self.temp_inv, dim=-1)

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

        if self.normalize_inputs:
            mean = self.x_mean.to(device=x.device, dtype=x.dtype)
            std = self.x_std.to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std

        device = x.device
        num_leaves = self.leaf_logits.shape[1]

        # Cast input to working dtype
        if x.dtype != self.dtype:
            x = x.to(self.dtype)

        # Base choices: [1, x_1, ..., x_V, <combos>] → (B, C_base, N)
        base = build_base(
            x,
            self.num_vars,
            self.dtype,
            use_mul=self.use_mul,
            use_mul3=self.use_mul3,
        )
        C_base = base.shape[1]  # == 1 + V + n_combo

        # ---- Leaf level ----
        w = self._weights(self.leaf_logits)  # (B, L, 2, C)
        leaf_ch = base.unsqueeze(1).expand(B, num_leaves, C_base, N)  # (B, L, C, N)

        # Weighted selection: sum_c(w[c] * choice[c]) for left and right inputs
        left = (w[:, :, 0, :].unsqueeze(-1) * leaf_ch).sum(dim=2)  # (B, L, N)
        right = (w[:, :, 1, :].unsqueeze(-1) * leaf_ch).sum(dim=2)  # (B, L, N)
        outputs = safe_eml(left, right)  # (B, L, N)

        # ---- Internal levels (bottom-up) ----
        for logits in self.internal_logits:
            M = logits.shape[1]  # nodes at this level
            w = self._weights(logits)  # (B, M, 2, C+1)

            # Pair consecutive children from the level below
            child_left = outputs[:, 0::2, :]  # (B, M, N)
            child_right = outputs[:, 1::2, :]  # (B, M, N)

            # Build choice tensors: [1, x_1..V, <combos>, f_child]
            int_base = base.unsqueeze(1).expand(B, M, C_base, N)  # (B, M, C_base, N)
            l_ch = torch.cat(
                [int_base, child_left.unsqueeze(2)], 2
            )  # (B, M, C_base+1, N)
            r_ch = torch.cat([int_base, child_right.unsqueeze(2)], 2)

            left = (w[:, :, 0, :].unsqueeze(-1) * l_ch).sum(dim=2)  # (B, M, N)
            right = (w[:, :, 1, :].unsqueeze(-1) * r_ch).sum(dim=2)  # (B, M, N)
            outputs = safe_eml(left, right)  # (B, M, N)

        # Root is the sole remaining node → (B, 1, N) → (B, N)
        return outputs.squeeze(1)

    # ------------------------------------------------------------------
    # Auxiliary methods
    # ------------------------------------------------------------------

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
