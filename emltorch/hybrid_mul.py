"""
Hybrid EML + MUL tree — each internal node picks between
    eml(L, R) = exp(L) - ln(R)   and   mul(L, R) = L * R
via a learnable per-node operator choice.

Motivation. The `use_mul` flag on `BatchedEMLTree` (tree.py) adds
multiplicative pre-features `x_i * x_j` at the LEAF level. This is
sufficient for targets where multiplication is between raw inputs
(e.g. `exp(a*b)` recovers at depth 1). It is NOT sufficient for
multiplicative *gate* targets like `h * f(z)` (SwiGLU / Mamba gate)
where the multiplication is between SUBTREE outputs.

This module introduces a second binary operator (`mul`) at each tree
node, selected via per-node op_logits[2] alongside the existing
input-choice logits. The module is intentionally a *separate class*
from `BatchedEMLTree` — the latter stays pure-EML — so existing
callers are unaffected and this class can be adopted opt-in.

The "single binary operator" thesis from Odrzywolek (arXiv:2603.21852,
2026) is explicitly relaxed: EML-only is universal at unbounded depth,
but depth-7+ compositions `a*b = exp(log(a) + log(b))` are unreachable
by evolutionary search at our bounded horizon (d<=6). Adding `mul` as
a second operator collapses these to depth-3 compositions.

Documented result (sae-eml/scripts/hybrid_eml_mul.py, 2026-04-24):
    h*SiLU(z) HELDOUT R² = 0.9892 (prior documented ≤0.69 with EML-only)

Public API
----------
    BatchedEMLMulTree          — GPU-batched hybrid trees
    evolve_hybrid_mul          — evolutionary search over topology + op choices
    polish_hybrid_mul          — constant polishing at fixed topology + ops
    HybridMulResult            — evolve return dataclass
    HybridMulPolishResult      — polish return dataclass
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .tree import enumerate_combos, num_combos

__all__ = [
    "safe_eml",
    "safe_mul",
    "build_base",
    "BatchedEMLMulTree",
    "HybridMulConfig",
    "HybridMulResult",
    "HybridMulPolishResult",
    "evolve_hybrid_mul",
    "polish_hybrid_mul",
]


def safe_eml(
    left: torch.Tensor, right: torch.Tensor, clamp: float = 60.0, log_eps: float = 1e-6
) -> torch.Tensor:
    """EML operator `exp(L) - ln(R)` with clamp guards."""
    left_s = torch.nan_to_num(left, nan=0.0, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp)
    right_s = torch.nan_to_num(right, nan=1.0, posinf=1e30, neginf=-1e30).clamp(min=log_eps, max=1e30)
    out = torch.exp(left_s) - torch.log(right_s)
    return torch.nan_to_num(out, nan=0.0, posinf=1e30, neginf=-1e30)


def safe_mul(
    left: torch.Tensor, right: torch.Tensor, clamp: float = 1e6
) -> torch.Tensor:
    """Stabilized multiplication — clamps each operand then the product so
    that nested `mul` or `mul` feeding into `eml`'s log argument doesn't
    overflow.
    """
    left_s = torch.nan_to_num(left, nan=0.0, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp)
    right_s = torch.nan_to_num(right, nan=0.0, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp)
    out = left_s * right_s
    out = torch.nan_to_num(out, nan=0.0, posinf=clamp, neginf=-clamp)
    return out.clamp(-clamp, clamp)


def build_base(
    x: torch.Tensor, num_vars: int, dtype: torch.dtype, use_mul: bool = False
) -> torch.Tensor:
    """Shape-matched to `tree.build_base` — reused here so the choice-index
    layout (`[1, x_1..V, <combos>, f_child]`) is identical for leaves and
    internal nodes. `use_mul=True` extends combos with `x_i * x_j` pairs.
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
    return torch.cat(parts, dim=1)


class BatchedEMLMulTree(nn.Module):
    """
    GPU-batched hybrid EML/MUL trees of depth `depth`, with `num_trees`
    trees evaluated in parallel.

    Args:
        num_trees: Parallel population size (B).
        depth: Tree depth (1 = single node; 2 = 3 nodes; ...).
        num_vars: Number of input features V.
        dtype: Working dtype (float32 recommended).
        device: CUDA/CPU device.
        use_mul: Include leaf-level `x_i * x_j` multiplicative combos.

    Each node stores:
        *_logits: input choice logits (left, right)
        *_op_logits: operator choice logits (eml=0, mul=1)

    Forward pass computes both `eml(L, R)` and `mul(L, R)` at each node,
    softmax-mixes by op logit. After `snap()`, each node collapses to a
    single operator and single input choice per side.
    """

    def __init__(
        self,
        num_trees: int,
        depth: int,
        num_vars: int = 1,
        dtype: torch.dtype = torch.float32,
        device=None,
        use_mul: bool = False,
    ):
        super().__init__()
        assert depth >= 1
        self.num_trees = num_trees
        self.depth = depth
        self.num_vars = num_vars
        self.dtype = dtype
        self.device_ = device
        self.use_mul = use_mul

        n_combo = num_combos(num_vars, use_mul=use_mul)
        leaf_choices = 1 + num_vars + n_combo
        internal_choices = leaf_choices + 1  # + f_child
        self.n_combo = n_combo
        num_leaves = 2 ** (depth - 1)
        self.num_leaves = num_leaves

        def peaked(shape, C):
            base = torch.randn(*shape, device=device, dtype=dtype) * 0.1
            idx = torch.randint(0, C, shape[:-1], device=device)
            base.scatter_(-1, idx.unsqueeze(-1), 150.0)
            return nn.Parameter(base)

        self.leaf_logits = peaked(
            (num_trees, num_leaves, 2, leaf_choices), leaf_choices
        )
        self.leaf_op_logits = nn.Parameter(
            torch.randn(num_trees, num_leaves, 2, device=device, dtype=dtype) * 0.1
        )

        self.internal_logits = nn.ParameterList()
        self.internal_op_logits = nn.ParameterList()
        for level in range(2, depth + 1):
            num_nodes = 2 ** (depth - level)
            self.internal_logits.append(
                peaked((num_trees, num_nodes, 2, internal_choices), internal_choices)
            )
            self.internal_op_logits.append(
                nn.Parameter(
                    torch.randn(num_trees, num_nodes, 2, device=device, dtype=dtype)
                    * 0.1
                )
            )

        # Buffers for scale-invariant input normalization
        self.register_buffer("x_mean", torch.zeros(1, num_vars, 1, dtype=dtype, device=device))
        self.register_buffer("x_std", torch.ones(1, num_vars, 1, dtype=dtype, device=device))
        self.register_buffer("normalize_inputs", torch.tensor(False, dtype=torch.bool, device=device))

    def set_normalization_stats(self, x_mean: torch.Tensor, x_std: torch.Tensor):
        """Set cached input normalization mean and standard deviation, and enable normalization."""
        self.x_mean.copy_(x_mean)
        self.x_std.copy_(x_std)
        self.normalize_inputs.copy_(torch.tensor(True, dtype=torch.bool, device=self.normalize_inputs.device))

    @torch.no_grad()
    def snap(self) -> None:
        """Collapse all logits to argmax one-hot (irreversible hardening)."""
        for lg in [self.leaf_logits] + list(self.internal_logits):
            idx = lg.argmax(dim=-1, keepdim=True)
            lg.zero_()
            lg.scatter_(-1, idx, 100.0)
        for lg in [self.leaf_op_logits] + list(self.internal_op_logits):
            idx = lg.argmax(dim=-1, keepdim=True)
            lg.zero_()
            lg.scatter_(-1, idx, 100.0)

    def snapped(self):
        """Return integer choices: (leaf_input, leaf_op, internal_input, internal_op)."""
        leaf_in = self.leaf_logits.argmax(dim=-1)
        leaf_op = self.leaf_op_logits.argmax(dim=-1)
        int_in = [lg.argmax(dim=-1) for lg in self.internal_logits]
        int_op = [lg.argmax(dim=-1) for lg in self.internal_op_logits]
        return leaf_in, leaf_op, int_in, int_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate all B trees on x; returns (B, N)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, V, N = x.shape

        if self.normalize_inputs:
            mean = self.x_mean.to(device=x.device, dtype=x.dtype)
            std = self.x_std.to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std

        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        base = build_base(x, self.num_vars, self.dtype, use_mul=self.use_mul)
        C_base = base.shape[1]

        # Leaf level
        w_in = torch.softmax(self.leaf_logits, dim=-1)
        w_op = torch.softmax(self.leaf_op_logits, dim=-1)
        leaf_ch = base.unsqueeze(1).expand(B, self.num_leaves, C_base, N)
        left = (w_in[:, :, 0, :].unsqueeze(-1) * leaf_ch).sum(2)
        right = (w_in[:, :, 1, :].unsqueeze(-1) * leaf_ch).sum(2)
        outputs = w_op[:, :, 0:1] * safe_eml(left, right) + w_op[:, :, 1:2] * safe_mul(
            left, right
        )

        # Internal levels
        for logits, op_logits in zip(self.internal_logits, self.internal_op_logits):
            M = logits.shape[1]
            w_in = torch.softmax(logits, dim=-1)
            w_op = torch.softmax(op_logits, dim=-1)
            child_l = outputs[:, 0::2, :]
            child_r = outputs[:, 1::2, :]
            int_base = base.unsqueeze(1).expand(B, M, C_base, N)
            l_ch = torch.cat([int_base, child_l.unsqueeze(2)], 2)
            r_ch = torch.cat([int_base, child_r.unsqueeze(2)], 2)
            left = (w_in[:, :, 0, :].unsqueeze(-1) * l_ch).sum(2)
            right = (w_in[:, :, 1, :].unsqueeze(-1) * r_ch).sum(2)
            outputs = w_op[:, :, 0:1] * safe_eml(left, right) + w_op[
                :, :, 1:2
            ] * safe_mul(left, right)
        return outputs.squeeze(1)


# ------------------------------------------------------------------
# Evolution
# ------------------------------------------------------------------


@dataclass
class HybridMulConfig:
    depth: int
    num_vars: int
    population: int = 4096
    generations: int = 30
    elite_frac: float = 0.1
    mutations_per_child: int = 1
    r2_target: float = 0.99995
    use_mul: bool = False
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    # Normalize target to unit std before evolution; denormalize in result.
    # Avoids exp() overflow on large-magnitude targets (e.g. Chinchilla L~100).
    normalize_target: bool = True
    # Normalize input features to zero mean and unit std before evolution.
    normalize_inputs: bool = False


@dataclass
class HybridMulResult:
    r2: float
    tree: BatchedEMLMulTree
    idx: int
    a: float
    b: float
    target_mean: float = 0.0
    target_std: float = 1.0


def _best_affine_mse(pred: torch.Tensor, y: torch.Tensor):
    """Per-tree best-affine MSE. pred (B, N); y (N,)."""
    p_mean = pred.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)
    p_c = pred - p_mean
    y_c = y - y_mean
    var_p = (p_c * p_c).mean(dim=-1, keepdim=True).clamp(min=1e-12)
    b = (p_c * y_c).mean(dim=-1, keepdim=True) / var_p
    a = y_mean - b * p_mean
    return a, b, ((a + b * pred - y) ** 2).mean(dim=-1)


def evolve_hybrid_mul(
    x: torch.Tensor, y: torch.Tensor, cfg: HybridMulConfig
) -> HybridMulResult:
    """Evolutionary search over hybrid tree topology + op choices.

    Returns HybridMulResult with a/b on the *original* target scale
    (i.e. if `normalize_target=True`, denormalized).
    """
    device = cfg.device
    V = x.shape[0] if x.dim() > 1 else 1
    x_t = x.to(device)
    if x_t.dim() == 1:
        x_t = x_t.unsqueeze(0)
    y_t = y.to(device).float()

    y_mean = float(y_t.mean().item())
    y_std = float(y_t.std().clamp(min=1e-8).item()) if cfg.normalize_target else 1.0
    y_work = (y_t - y_mean) / y_std if cfg.normalize_target else y_t

    pop = cfg.population
    x_pop = x_t.unsqueeze(0).expand(pop, V, x_t.shape[-1]).contiguous()

    tree = BatchedEMLMulTree(
        num_trees=pop,
        depth=cfg.depth,
        num_vars=V,
        device=device,
        use_mul=cfg.use_mul,
    )
    tree.snap()

    if cfg.normalize_inputs:
        x_mean = x_t.mean(dim=-1, keepdim=True).unsqueeze(0) # (1, V, 1)
        x_std = x_t.std(dim=-1, keepdim=True).clamp(min=1e-8).unsqueeze(0) # (1, V, 1)
        tree.set_normalization_stats(x_mean, x_std)

    ss_tot = ((y_work - y_work.mean()) ** 2).sum().clamp(min=1e-12)
    # Default to a valid state so callers never see tree=None.
    best = {"r2": -1e9, "tree": tree, "idx": 0, "a": 0.0, "b": 1.0}

    for gen in range(cfg.generations):
        with torch.no_grad():
            pred = tree(x_pop)
            a, b, mse = _best_affine_mse(pred, y_work)
            finite_mask = torch.isfinite(pred).all(dim=-1)
            r2 = 1 - mse * x_t.shape[-1] / ss_tot
            r2 = torch.where(finite_mask, r2, torch.full_like(r2, -1e9))
        r2_cpu = r2.cpu()
        idx = int(r2_cpu.argmax().item())
        cur_r2 = float(r2_cpu[idx].item())
        if cur_r2 > best["r2"]:
            best = {
                "r2": cur_r2,
                "tree": tree,
                "idx": idx,
                "a": float(a[idx].item()),
                "b": float(b[idx].item()),
            }
        if cur_r2 >= cfg.r2_target:
            break

        # Elite selection + random mutation
        K = max(int(cfg.elite_frac * pop), 2)
        _, elite = r2_cpu.topk(K)
        elite_idx = elite.to(device)
        n_off = pop - K

        def clone_(src_idx, dst_idx):
            for p in (
                [tree.leaf_logits]
                + list(tree.internal_logits)
                + [tree.leaf_op_logits]
                + list(tree.internal_op_logits)
            ):
                p.data[dst_idx] = p.data[src_idx]

        src_choice = elite_idx[torch.randint(0, K, (n_off,), device=device)]
        dst_idx = torch.arange(K, pop, device=device)
        clone_(src_choice, dst_idx)

        for _ in range(cfg.mutations_per_child):
            for offsp in range(K, pop):
                if torch.rand(1).item() < 0.2:
                    # Mutate op choice
                    if tree.depth == 1 or torch.rand(1).item() < 0.5:
                        n = int(torch.randint(0, tree.num_leaves, (1,)).item())
                        new_op = int(torch.randint(0, 2, (1,)).item())
                        tree.leaf_op_logits.data[offsp, n] = 0
                        tree.leaf_op_logits.data[offsp, n, new_op] = 150.0
                    else:
                        lvl = int(
                            torch.randint(0, len(tree.internal_op_logits), (1,)).item()
                        )
                        M = tree.internal_op_logits[lvl].shape[1]
                        n = int(torch.randint(0, M, (1,)).item())
                        new_op = int(torch.randint(0, 2, (1,)).item())
                        tree.internal_op_logits[lvl].data[offsp, n] = 0
                        tree.internal_op_logits[lvl].data[offsp, n, new_op] = 150.0
                else:
                    # Mutate input choice
                    if torch.rand(1).item() < 0.5 or tree.depth == 1:
                        n = int(torch.randint(0, tree.num_leaves, (1,)).item())
                        side = int(torch.randint(0, 2, (1,)).item())
                        C = tree.leaf_logits.shape[-1]
                        new_c = int(torch.randint(0, C, (1,)).item())
                        tree.leaf_logits.data[offsp, n, side] = 0
                        tree.leaf_logits.data[offsp, n, side, new_c] = 150.0
                    else:
                        lvl = int(
                            torch.randint(0, len(tree.internal_logits), (1,)).item()
                        )
                        M = tree.internal_logits[lvl].shape[1]
                        n = int(torch.randint(0, M, (1,)).item())
                        side = int(torch.randint(0, 2, (1,)).item())
                        C = tree.internal_logits[lvl].shape[-1]
                        new_c = int(torch.randint(0, C, (1,)).item())
                        tree.internal_logits[lvl].data[offsp, n, side] = 0
                        tree.internal_logits[lvl].data[offsp, n, side, new_c] = 150.0

    # Denormalize a, b back to original target scale.
    # y ~ a + b * tree(x)  where y = y_work * y_std + y_mean,
    # so on the original scale: y_orig ~ (a * y_std + y_mean) + (b * y_std) * tree(x).
    a_orig = best["a"] * y_std + y_mean
    b_orig = best["b"] * y_std
    return HybridMulResult(
        r2=best["r2"],
        tree=best["tree"],
        idx=best["idx"],
        a=a_orig,
        b=b_orig,
        target_mean=y_mean,
        target_std=y_std,
    )


# ------------------------------------------------------------------
# Polishing
# ------------------------------------------------------------------


@dataclass
class HybridMulPolishResult:
    """Polish result for hybrid EML+MUL trees.

    `a`, `b` are on the ORIGINAL target scale (internal normalization is
    denormalized before return). `constants` lists the learned scalar
    values for leaves that chose the `1`-slot.
    """

    r2: float
    mse: float
    a: float
    b: float
    constants: list
    # Topology (integer choice indices per node, per side):
    leaf_in: list
    leaf_op: list
    int_in: list
    int_op: list


def polish_hybrid_mul(
    tree: BatchedEMLMulTree,
    idx: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    n_iters: int = 2500,
    lr: float = 0.05,
    warm_a: float = 0.0,
    warm_b: float = 1.0,
    normalize_target: bool = True,
    grad_clip: float = 1.0,
) -> HybridMulPolishResult:
    """Adam-polish constants at a fixed snapped topology + op-choice.

    Mirrors the semantics of `emltorch.polish.polish` for the hybrid tree:
    every leaf whose choice slot is 0 (the `1`-slot) becomes a learnable
    constant, jointly optimized with the outer affine `a + b · tree(x)`.

    Robustness features added 2026-04-24:
      - `normalize_target=True` (default): train in `(y - mean)/std` frame;
        denormalize on return. Prevents R² collapse on large-magnitude
        targets (e.g. Chinchilla L~100) where Adam/grad-clip get pinned
        at one end of the trade-off.
      - NaN-revert: non-finite `fit` OR `mse` at any iteration rolls back
        constants/a/b to the last best state + small perturbation, so the
        function never returns worse than the warm start.

    Args:
        tree: BatchedEMLMulTree (must already be .snap()-ed).
        idx:  Individual index within the batched tree.
        x_train, y_train: Training data. x shape (V, N) or (N,) if V=1.
        n_iters: Adam iterations.
        lr: Learning rate (default 0.05).
        warm_a, warm_b: Outer affine warm start (from evolve).
        normalize_target: Internally normalize y_train to unit std.
        grad_clip: Gradient clip norm for [constants, a, b].

    Returns:
        HybridMulPolishResult — a, b on the ORIGINAL target scale.
    """
    device = next(tree.parameters()).device
    dtype = torch.float32

    leaf_in, leaf_op, int_in, int_op = tree.snapped()
    leaf_in_ch = leaf_in[idx]  # (L, 2)
    leaf_op_ch = leaf_op[idx]  # (L,)
    int_in_ch = [c[idx] for c in int_in]
    int_op_ch = [c[idx] for c in int_op]

    V = tree.num_vars
    n_combo = tree.n_combo

    # Count '1' leaves (choice idx 0) in leaf + internal
    leaf_mask = leaf_in_ch == 0
    int_masks = [c == 0 for c in int_in_ch]
    n_consts = int(leaf_mask.sum() + sum(m.sum() for m in int_masks))
    constants = nn.Parameter(torch.ones(n_consts, dtype=dtype, device=device))

    # Build flat index maps: (leaf, side) -> constant slot
    leaf_flat = torch.full(leaf_in_ch.shape, -1, dtype=torch.long, device=device)
    k = 0
    for n_i in range(leaf_in_ch.shape[0]):
        for s in range(2):
            if int(leaf_in_ch[n_i, s].item()) == 0:
                leaf_flat[n_i, s] = k
                k += 1
    int_flats = []
    for ch_i in int_in_ch:
        flat = torch.full(ch_i.shape, -1, dtype=torch.long, device=device)
        for n_i in range(ch_i.shape[0]):
            for s in range(2):
                if int(ch_i[n_i, s].item()) == 0:
                    flat[n_i, s] = k
                    k += 1
        int_flats.append(flat)

    x_dev = x_train.to(device, dtype)
    if x_dev.dim() == 1:
        x_dev = x_dev.unsqueeze(0)
    y_dev = y_train.to(device, dtype)

    # Target normalization (see HybridMulConfig.normalize_target for rationale)
    if normalize_target:
        y_mu = float(y_dev.mean().item())
        y_sd = float(y_dev.std().clamp(min=1e-8).item())
    else:
        y_mu, y_sd = 0.0, 1.0
    y_work = (y_dev - y_mu) / y_sd

    warm_a_work = (warm_a - y_mu) / y_sd
    warm_b_work = warm_b / y_sd
    a = nn.Parameter(torch.tensor([warm_a_work], dtype=dtype, device=device))
    b = nn.Parameter(torch.tensor([warm_b_work], dtype=dtype, device=device))

    def forward_fixed() -> torch.Tensor:
        base = build_base(x_dev.unsqueeze(0), V, dtype, use_mul=tree.use_mul).squeeze(0)

        def sel(choice_i, flat_t, n, s, base_v, child_v):
            c = int(choice_i[n, s].item())
            if c == 0:
                return constants[int(flat_t[n, s].item())].expand(base_v.shape[-1])
            if c <= V + n_combo:
                return base_v[c]
            return child_v

        L = leaf_in_ch.shape[0]
        outs = []
        for n in range(L):
            lft = sel(leaf_in_ch, leaf_flat, n, 0, base, None)
            rgt = sel(leaf_in_ch, leaf_flat, n, 1, base, None)
            op = int(leaf_op_ch[n].item())
            outs.append(safe_eml(lft, rgt) if op == 0 else safe_mul(lft, rgt))
        cur = torch.stack(outs)
        for ch_i, op_i, flat in zip(int_in_ch, int_op_ch, int_flats):
            M = ch_i.shape[0]
            new = []
            for n in range(M):
                cL = cur[2 * n]
                cR = cur[2 * n + 1]
                lft = sel(ch_i, flat, n, 0, base, cL)
                rgt = sel(ch_i, flat, n, 1, base, cR)
                op = int(op_i[n].item())
                new.append(safe_eml(lft, rgt) if op == 0 else safe_mul(lft, rgt))
            cur = torch.stack(new)
        return cur[0]

    opt = torch.optim.Adam([constants, a, b], lr=lr)
    best_state = (constants.detach().clone(), float(a.item()), float(b.item()))
    with torch.no_grad():
        init_pred = forward_fixed()
        init_mse_t = ((a + b * init_pred - y_work) ** 2).mean()
        init_mse = (
            float(init_mse_t.item()) if torch.isfinite(init_mse_t) else float("inf")
        )
    best_mse = init_mse
    for _ in range(n_iters):
        opt.zero_grad()
        pred = forward_fixed()
        fit = a + b * pred
        if not torch.isfinite(fit).all():
            with torch.no_grad():
                a.data.fill_(best_state[1])
                b.data.fill_(best_state[2])
                constants.data.copy_(best_state[0])
                constants.add_(torch.randn_like(constants) * 0.01)
            continue
        mse = ((fit - y_work) ** 2).mean()
        if not torch.isfinite(mse):
            with torch.no_grad():
                a.data.fill_(best_state[1])
                b.data.fill_(best_state[2])
                constants.data.copy_(best_state[0])
                constants.add_(torch.randn_like(constants) * 0.01)
            continue
        mse.backward()
        nn.utils.clip_grad_norm_([constants, a, b], grad_clip)
        opt.step()
        mse_i = float(mse.item())
        if mse_i < best_mse:
            best_mse = mse_i
            best_state = (
                constants.detach().clone(),
                float(a.item()),
                float(b.item()),
            )

    with torch.no_grad():
        constants.copy_(best_state[0])

    a_orig = best_state[1] * y_sd + y_mu
    b_orig = best_state[2] * y_sd
    ss_tot = ((y_dev - y_dev.mean()) ** 2).sum().clamp(min=1e-12).item()
    final_mse = best_mse * (y_sd * y_sd)
    final_r2 = 1 - final_mse * y_dev.shape[-1] / ss_tot

    return HybridMulPolishResult(
        r2=float(final_r2),
        mse=float(final_mse),
        a=float(a_orig),
        b=float(b_orig),
        constants=best_state[0].detach().cpu().tolist(),
        leaf_in=leaf_in_ch.cpu().tolist(),
        leaf_op=leaf_op_ch.cpu().tolist(),
        int_in=[c.cpu().tolist() for c in int_in_ch],
        int_op=[c.cpu().tolist() for c in int_op_ch],
    )
