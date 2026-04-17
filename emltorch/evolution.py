"""
Evolutionary search over EML tree topologies.

Random search plateaus at depth 4+. This module keeps top-K restarts,
mutates them via single-edge flips, and repeats — O(generations × population)
instead of O(search_space).

Combined with peaked one-hot init, each "tree" in the population is a
fully discrete structure; mutation flips one operand choice at one node.
All operations are GPU-batched over the population.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .tree import BatchedEMLTree
from .symbolic import extract_expressions, annotate


@dataclass
class EvolutionConfig:
    depth: int = 4
    num_vars: int = 1
    population: int = 2048
    generations: int = 20
    elite_fraction: float = 0.1     # keep top-k of each generation as parents
    mutations_per_child: int = 1    # edges flipped per offspring
    # Crossover: fraction of offspring produced by mixing two elite parents
    # (0 → pure mutation, 0.5 → half crossover / half mutation).
    crossover_fraction: float = 0.3
    device: str = "cuda:7"
    dtype: str = "float32"
    r2_target: float = 0.99          # early-exit when any tree hits this
    log_every: int = 5

    @property
    def torch_dtype(self):
        return {"float32": torch.float32, "float64": torch.float64}[self.dtype]


@dataclass
class EvolutionResult:
    best_expression: str             # "a + b * (eml_tree_formula)"
    best_r2: float                   # affine-adjusted R² on full data
    best_mse: float
    best_tree: BatchedEMLTree
    best_idx: int
    generation_r2s: list[float]      # best R² per generation (affine-adjusted)
    total_time_s: float
    best_a: float = 0.0              # affine bias (intercept)
    best_b: float = 1.0              # affine scale (slope)


def _snap_peaked(tree: BatchedEMLTree):
    """Force any soft logits into one-hot by snapping to argmax."""
    tree.snap()
    tree.temp_inv.fill_(1.0)
    tree.selection_mode = "softmax"
    tree.training = False


def _evaluate(tree: BatchedEMLTree, x: torch.Tensor, y: torch.Tensor,
              affine: bool = True) -> torch.Tensor:
    """Per-tree MSE (B,). NaN trees get +inf.

    If affine=True, MSE is after the best per-tree affine rescaling
    y ≈ a + b * tree(x). This lets evolutionary search select topologies
    that are close up to linear rescaling — much more forgiving than raw MSE.
    """
    with torch.no_grad():
        pred = tree(x)                         # (B, N)
    if affine:
        # Closed-form best affine fit per tree:
        #   b = cov(pred, y) / var(pred);  a = mean(y) - b * mean(pred)
        pred_mean = pred.mean(dim=-1, keepdim=True)
        y_mean = y.mean(dim=-1, keepdim=True)
        pred_c = pred - pred_mean
        y_c = y - y_mean
        var_pred = (pred_c * pred_c).mean(dim=-1, keepdim=True)
        # Avoid div-by-zero for constant predictions
        safe_var = torch.where(var_pred > 1e-12, var_pred,
                               torch.full_like(var_pred, 1e-12))
        b = (pred_c * y_c).mean(dim=-1, keepdim=True) / safe_var
        # If tree is ~constant (var<1e-12), force b=0 so affine fit = mean(y)
        b = torch.where(var_pred > 1e-12, b, torch.zeros_like(b))
        a = y_mean - b * pred_mean
        fit = a + b * pred
        diff = fit - y
    else:
        diff = pred - y
    mse = diff.abs().pow(2).mean(dim=-1) if diff.is_complex() else diff.pow(2).mean(dim=-1)
    return torch.where(torch.isfinite(mse), mse,
                       torch.full_like(mse, float("inf")))


def _mutate_(tree: BatchedEMLTree, indices: torch.Tensor, n_mutations: int = 1):
    """In-place mutation: for each tree in `indices`, pick random edges and
    randomize their one-hot choice. `indices` is a LongTensor of tree indices.
    """
    with torch.no_grad():
        all_logit_tensors = [tree.leaf_logits.data] + \
                            [lg.data for lg in tree.internal_logits]
        n = len(indices)
        for _ in range(n_mutations):
            which = torch.randint(0, len(all_logit_tensors), (n,),
                                  device=indices.device)
            for tensor_idx, logits in enumerate(all_logit_tensors):
                mask = which == tensor_idx
                if not mask.any():
                    continue
                sel_indices = indices[mask]
                n_sel = len(sel_indices)
                n_nodes = logits.shape[1]
                n_inputs = logits.shape[2]
                n_choices = logits.shape[3]
                node_i = torch.randint(0, n_nodes, (n_sel,), device=logits.device)
                input_i = torch.randint(0, n_inputs, (n_sel,), device=logits.device)
                new_choice = torch.randint(0, n_choices, (n_sel,), device=logits.device)
                # Build one-hot row of size (n_sel, n_choices) with 50.0 at new_choice
                row = torch.zeros(n_sel, n_choices, device=logits.device,
                                  dtype=logits.dtype)
                row.scatter_(-1, new_choice.unsqueeze(-1), 50.0)
                logits[sel_indices, node_i, input_i] = row


def _clone_(tree: BatchedEMLTree, dst_indices: torch.Tensor,
            src_indices: torch.Tensor):
    """In-place copy: dst_indices <- src_indices for each logit tensor."""
    with torch.no_grad():
        tree.leaf_logits.data[dst_indices] = tree.leaf_logits.data[src_indices]
        for lg in tree.internal_logits:
            lg.data[dst_indices] = lg.data[src_indices]


def _crossover_(tree: BatchedEMLTree, dst: torch.Tensor,
                p1: torch.Tensor, p2: torch.Tensor):
    """Uniform crossover: for each (node, input), pick p1 or p2 with 50/50.

    `dst`, `p1`, `p2` are equal-length LongTensors of tree indices. The logits
    at each dst slot are a coin-flip mix from the corresponding p1 and p2
    slots, one flip per (node, input) position.
    """
    with torch.no_grad():
        all_logit_tensors = [tree.leaf_logits.data] + \
                            [lg.data for lg in tree.internal_logits]
        for logits in all_logit_tensors:
            # shape: (B, nodes, inputs, choices)
            shape_mask = (len(dst), logits.shape[1], logits.shape[2], 1)
            mask = torch.rand(*shape_mask, device=logits.device) < 0.5
            src_1 = logits[p1]                          # (D, nodes, inputs, choices)
            src_2 = logits[p2]
            mixed = torch.where(mask, src_1, src_2)
            logits[dst] = mixed


def evolve(x: torch.Tensor, y: torch.Tensor, cfg: EvolutionConfig,
           var_names: list[str] | None = None) -> EvolutionResult:
    """
    Run evolutionary search for y ≈ EML(x).

    Args:
        x: (V, N) feature tensor OR (N,) for single variable.
        y: (N,) target tensor.
        cfg: EvolutionConfig.

    Returns:
        EvolutionResult with best tree and metrics.
    """
    t0 = time.time()
    device = cfg.device

    # Shape to (pop, V, N)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    V, N = x.shape
    x_batch = x.to(device=device, dtype=cfg.torch_dtype)
    y_batch = y.to(device=device, dtype=cfg.torch_dtype)
    x_pop = x_batch.unsqueeze(0).expand(cfg.population, V, N).contiguous()
    y_pop = y_batch.unsqueeze(0).expand(cfg.population, N).contiguous()

    # ss_tot for R² computation
    ss_tot = ((y_batch - y_batch.mean()) ** 2).sum().clamp(min=1e-12)

    # Initialize population
    tree = BatchedEMLTree(
        num_trees=cfg.population, depth=cfg.depth, num_vars=V,
        dtype=cfg.torch_dtype, device=device,
        init_scale=50.0, init_mode="peaked",
    )
    _snap_peaked(tree)

    n_elite = max(1, int(cfg.population * cfg.elite_fraction))
    generation_r2s = []
    best_ever_mse = torch.tensor(float("inf"), device=device)
    best_ever_idx = 0
    best_ever_logits = None

    for gen in range(cfg.generations):
        mse = _evaluate(tree, x_pop, y_pop)
        r2 = 1 - mse * N / ss_tot

        # Track global best
        min_mse_this_gen, argmin_this_gen = mse.min(dim=0)
        if min_mse_this_gen < best_ever_mse:
            best_ever_mse = min_mse_this_gen
            best_ever_idx = int(argmin_this_gen)
            # Snapshot best tree's logits
            best_ever_logits = (
                tree.leaf_logits[best_ever_idx].clone(),
                [lg[best_ever_idx].clone() for lg in tree.internal_logits],
            )

        best_r2 = r2.max().item()
        generation_r2s.append(best_r2)

        if gen % cfg.log_every == 0 or gen == cfg.generations - 1:
            print(f"[evo] gen {gen:>3d}  best R²={best_r2:+.4f}  "
                  f"elite R² [{r2.topk(n_elite).values.min().item():+.4f}, "
                  f"{best_r2:+.4f}]")

        if best_r2 >= cfg.r2_target:
            print(f"[evo] r2_target reached at gen {gen}")
            break

        # Selection: top-k indices
        _, elite_idx = mse.topk(n_elite, largest=False)

        # Offspring slots = all slots except elites
        n_offspring = cfg.population - n_elite
        offspring_slots = torch.arange(cfg.population, device=device)
        non_elite_mask = torch.ones(cfg.population, dtype=torch.bool, device=device)
        non_elite_mask[elite_idx] = False
        non_elite_slots = offspring_slots[non_elite_mask]

        # Split offspring between mutation and crossover
        n_cross = int(n_offspring * cfg.crossover_fraction)
        n_mut = n_offspring - n_cross
        mut_slots = non_elite_slots[:n_mut]
        cross_slots = non_elite_slots[n_mut:]

        # Mutation branch: pick a random elite parent, clone, mutate
        if n_mut > 0:
            parent_idx = elite_idx[torch.randint(0, n_elite, (n_mut,),
                                                 device=device)]
            _clone_(tree, mut_slots, parent_idx)
            _mutate_(tree, mut_slots, n_mutations=cfg.mutations_per_child)

        # Crossover branch: pick two random elites, uniform crossover
        if n_cross > 0:
            p1 = elite_idx[torch.randint(0, n_elite, (n_cross,), device=device)]
            p2 = elite_idx[torch.randint(0, n_elite, (n_cross,), device=device)]
            _crossover_(tree, cross_slots, p1, p2)
            # Small mutation on crossover children for diversity
            _mutate_(tree, cross_slots, n_mutations=1)

    # Restore best-ever tree at slot `best_ever_idx`
    if best_ever_logits is not None:
        leaf_best, internals_best = best_ever_logits
        with torch.no_grad():
            tree.leaf_logits.data[best_ever_idx] = leaf_best
            for lg, b_snap in zip(tree.internal_logits, internals_best):
                lg.data[best_ever_idx] = b_snap

    # Re-evaluate the best tree to get its affine coefficients.
    # We must evaluate the entire population (the tree is fixed batch-size),
    # then slice out the restored best slot.
    with torch.no_grad():
        all_preds = tree(x_pop)                         # (B, N)
    best_pred = all_preds[best_ever_idx]
    pred_mean = best_pred.mean()
    pred_c = best_pred - pred_mean
    y_c = y_batch - y_batch.mean()
    var_pred = (pred_c * pred_c).mean()
    if var_pred > 1e-12:
        b_coef = float(((pred_c * y_c).mean() / var_pred).item())
        a_coef = float((y_batch.mean() - b_coef * pred_mean).item())
    else:
        b_coef = 0.0
        a_coef = float(y_batch.mean().item())

    # Extract best expression
    if var_names is None:
        var_names = [f"x{i+1}" for i in range(V)] if V > 1 else ["x"]
    best_expr_raw = extract_expressions(tree, [best_ever_idx], var_names)[0]
    tree_str = annotate(best_expr_raw)
    best_expr = f"{a_coef:+.4f} + ({b_coef:+.4f}) * [{tree_str}]"

    best_mse_val = best_ever_mse.item()
    best_r2 = 1 - best_mse_val * N / ss_tot.item()

    return EvolutionResult(
        best_expression=best_expr,
        best_r2=best_r2,
        best_mse=best_mse_val,
        best_tree=tree,
        best_idx=best_ever_idx,
        generation_r2s=generation_r2s,
        total_time_s=time.time() - t0,
        best_a=a_coef,
        best_b=b_coef,
    )
