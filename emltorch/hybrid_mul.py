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


def safe_eml(
    left: torch.Tensor, right: torch.Tensor, clamp: float = 60.0, log_eps: float = 1e-6
) -> torch.Tensor:
    """EML operator `exp(L) - ln(R)` with clamp guards."""
    left_s = left.clamp(-clamp, clamp)
    right_s = right.clamp(min=log_eps)
    return torch.exp(left_s) - torch.log(right_s)


def safe_mul(
    left: torch.Tensor, right: torch.Tensor, clamp: float = 1e6
) -> torch.Tensor:
    """Stabilized multiplication — clamps each operand then the product so
    that nested `mul` or `mul` feeding into `eml`'s log argument doesn't
    overflow.
    """
    left_s = left.clamp(-clamp, clamp)
    right_s = right.clamp(-clamp, clamp)
    return (left_s * right_s).clamp(-clamp, clamp)


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
            base.scatter_(-1, idx.unsqueeze(-1), 50.0)
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
    device: str = "cuda:0"
    # Normalize target to unit std before evolution; denormalize in result.
    # Avoids exp() overflow on large-magnitude targets (e.g. Chinchilla L~100).
    normalize_target: bool = True


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
                        tree.leaf_op_logits.data[offsp, n, new_op] = 50.0
                    else:
                        lvl = int(
                            torch.randint(0, len(tree.internal_op_logits), (1,)).item()
                        )
                        M = tree.internal_op_logits[lvl].shape[1]
                        n = int(torch.randint(0, M, (1,)).item())
                        new_op = int(torch.randint(0, 2, (1,)).item())
                        tree.internal_op_logits[lvl].data[offsp, n] = 0
                        tree.internal_op_logits[lvl].data[offsp, n, new_op] = 50.0
                else:
                    # Mutate input choice
                    if torch.rand(1).item() < 0.5 or tree.depth == 1:
                        n = int(torch.randint(0, tree.num_leaves, (1,)).item())
                        side = int(torch.randint(0, 2, (1,)).item())
                        C = tree.leaf_logits.shape[-1]
                        new_c = int(torch.randint(0, C, (1,)).item())
                        tree.leaf_logits.data[offsp, n, side] = 0
                        tree.leaf_logits.data[offsp, n, side, new_c] = 50.0
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
                        tree.internal_logits[lvl].data[offsp, n, side, new_c] = 50.0

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
