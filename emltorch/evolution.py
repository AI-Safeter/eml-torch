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
from dataclasses import dataclass, field

import torch

from .tree import BatchedEMLTree
from .symbolic import extract_expressions, annotate


@dataclass
class EvolutionConfig:
    depth: int = 4
    num_vars: int = 1
    population: int = 2048
    generations: int = 20
    elite_fraction: float = 0.1  # keep top-k of each generation as parents
    mutations_per_child: int = 1  # edges flipped per offspring
    # Crossover: fraction of offspring produced by mixing two elite parents
    # (0 → pure mutation, 0.5 → half crossover / half mutation).
    crossover_fraction: float = 0.3
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype: str = "float32"
    r2_target: float = 0.99  # early-exit when any tree hits this
    log_every: int = 5
    verbose: bool = False  # if False (default), suppress per-generation stdout
    # Range-aware fitness: penalize trees whose output scale differs from y's.
    # Adds `range_penalty * log(std(tree_out)/std(y))^2` to per-tree MSE.
    # Prevents evolution from hiding scale-collapsed trees (|b|~0.05) behind
    # the affine wrapper. 0.0 disables.
    range_penalty: float = 0.0
    # Enable multiplicative combos (x_i * x_j) in the leaf/internal choice set.
    # Opt-in; breaks multiplicative ceilings (e.g. h·SiLU(z) Mamba gate).
    use_mul: bool = False
    # Enable triple-product combos (x_i * x_j * x_k) for V >= 3. Opt-in.
    use_mul3: bool = False
    # ─── Cert-friendly evolution flags (Track 5, 2026-04-27) ─────────────
    # When > 0, biases evolution toward formulas that compose more cleanly
    # with the axiomatized Exp/Ln SMT track (Headlines 7-10).  The penalty
    # is added to fitness; selection-elite still tracked by MSE so a final
    # solution with cert_friendly_penalty > 0 may have slightly higher MSE
    # than the unpenalized run.
    #
    # `cert_friendly_const_bonus`:  reward (negative penalty) per LEAF slot
    # that snaps to the constant `1` choice (choice index 0).  Trees with
    # more `1` leaves discharge more cleanly under the inverse axioms
    # `Ln(Exp(x)) = x` and `Exp(Ln(v)) = v`, since `eml(L, 1) = exp(L)`
    # and `eml(1, R) = e − ln(R)` are the canonical EML reduction patterns.
    # Typical value: 1e-3 (small).  0.0 disables (default).
    cert_friendly_const_bonus: float = 0.0
    # Normalize input features to zero mean and unit std before evolution.
    normalize_inputs: bool = False
    # ─── Island model (multi-population) ─────────────────────────────────
    # Partition the population into `n_islands` equal sub-populations that
    # evolve independently (selection + reproduction confined within each
    # island), with periodic ring migration of each island's best
    # individual into its neighbour. This preserves diversity and lets the
    # search explore multiple basins in parallel — the targeted fix for the
    # basin-trap failure mode where every seed of a single panmictic
    # population converges to the same local optimum (e.g. Gemma L_large,
    # where 10/10 seeds collapsed to `eml(x2, x2)`).
    #
    # n_islands=1 (default) is byte-identical to the original panmictic
    # evolution — the island code path is only taken when n_islands > 1.
    # The population is split into contiguous blocks of size
    # `population // n_islands`; any remainder slots form a final ragged
    # block that participates normally.
    n_islands: int = 1
    # Migrate every `migration_interval` generations (ignored if n_islands=1).
    migration_interval: int = 5
    # Number of top individuals copied from each island into its ring
    # neighbour at each migration event.
    migration_size: int = 1

    @property
    def torch_dtype(self):
        return {"float32": torch.float32, "float64": torch.float64}[self.dtype]


@dataclass
class EvolutionResult:
    best_expression: str  # "a + b * (eml_tree_formula)"
    best_r2: float  # affine-adjusted R² on full data
    best_mse: float
    best_tree: BatchedEMLTree
    best_idx: int
    generation_r2s: list[float]  # best R² per generation (affine-adjusted)
    total_time_s: float
    best_a: float = 0.0  # affine bias (intercept)
    best_b: float = 1.0  # affine scale (slope)


def _snap_peaked(tree: BatchedEMLTree):
    """Force any soft logits into one-hot by snapping to argmax."""
    tree.snap()
    tree.temp_inv.fill_(1.0)
    tree.selection_mode = "softmax"
    tree.training = False


def _cert_friendly_bonus(tree: BatchedEMLTree, bonus: float) -> torch.Tensor:
    """Per-tree fitness bonus for using the constant-1 choice (idx 0) at leaves.

    Returns a (B,) tensor to ADD to fitness (so a NEGATIVE value is a reward
    that encourages evolution to favor that tree).  When bonus = 0, returns
    a zero tensor.

    The motivation is cert-tractability: trees whose leaves are mostly the
    constant `1` discharge more cleanly under the axiomatized Exp/Ln SMT
    track (Headlines 7-10), since `eml(L, 1) = exp(L) - ln(1) = exp(L)`
    is the canonical reduction pattern that the inverse axioms target.
    """
    leaf_logits = tree.leaf_logits.data  # (B, n_leaves, 2, n_choices)
    if bonus <= 0.0 or leaf_logits.numel() == 0:
        return torch.zeros(leaf_logits.shape[0], device=leaf_logits.device)
    chosen = leaf_logits.argmax(dim=-1)  # (B, n_leaves, 2)
    n_const_one = (chosen == 0).float().sum(dim=(-1, -2))  # (B,)
    return -bonus * n_const_one  # negative = reward


def _evaluate(
    tree: BatchedEMLTree,
    x: torch.Tensor,
    y: torch.Tensor,
    range_penalty: float = 0.0,
) -> torch.Tensor:
    """Per-tree MSE (B,) after closed-form best affine rescaling
    y ≈ a + b·tree(x). NaN trees get +inf.

    The affine wrapper lets evolutionary search select topologies that are
    close up to linear rescaling — much more forgiving than raw MSE.
    """
    with torch.no_grad():
        pred = tree(x)  # (B, N)
    # Closed-form best affine fit per tree:
    #   b = cov(pred, y) / var(pred);  a = mean(y) - b * mean(pred)
    pred_mean = pred.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)
    pred_c = pred - pred_mean
    y_c = y - y_mean
    var_pred = (pred_c * pred_c).mean(dim=-1, keepdim=True)
    safe_var = torch.where(var_pred > 1e-12, var_pred, torch.full_like(var_pred, 1e-12))
    b = (pred_c * y_c).mean(dim=-1, keepdim=True) / safe_var
    # If tree is ~constant (var<1e-12), force b=0 so affine fit = mean(y)
    b = torch.where(var_pred > 1e-12, b, torch.zeros_like(b))
    a = y_mean - b * pred_mean
    diff = (a + b * pred) - y
    mse = (
        diff.abs().pow(2).mean(dim=-1)
        if diff.is_complex()
        else diff.pow(2).mean(dim=-1)
    )
    if range_penalty > 0.0:
        y_std = y_c.pow(2).mean(dim=-1, keepdim=True).clamp(min=1e-12).sqrt()
        p_std = var_pred.clamp(min=1e-12).sqrt()
        log_ratio_sq = (p_std / y_std).log().pow(2).squeeze(-1)
        mse = mse + range_penalty * log_ratio_sq
    return torch.where(torch.isfinite(mse), mse, torch.full_like(mse, float("inf")))


def _mutate_(tree: BatchedEMLTree, indices: torch.Tensor, n_mutations: int = 1):
    """In-place mutation: for each tree in `indices`, pick random edges and
    randomize their one-hot choice. `indices` is a LongTensor of tree indices.
    """
    with torch.no_grad():
        all_logit_tensors = [tree.leaf_logits.data] + [
            lg.data for lg in tree.internal_logits
        ]
        n = len(indices)
        for _ in range(n_mutations):
            which = torch.randint(
                0, len(all_logit_tensors), (n,), device=indices.device
            )
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
                # Build one-hot row of size (n_sel, n_choices) with 150.0 at new_choice.
                # Logit 150 ensures softmax[non-argmax] = exp(-150) underflows to 0
                # in float32, preventing softmax-mixing contamination when a
                # non-selected choice holds a saturated value (e.g., safe_eml at
                # exp(60) ≈ 1.14e26): logit=50 gave weight 1.9e-22 and a 2.2e+4
                # spurious contribution; logit=150 gives exactly 0.
                row = torch.zeros(
                    n_sel, n_choices, device=logits.device, dtype=logits.dtype
                )
                row.scatter_(-1, new_choice.unsqueeze(-1), 150.0)
                logits[sel_indices, node_i, input_i] = row


def _clone_(tree: BatchedEMLTree, dst_indices: torch.Tensor, src_indices: torch.Tensor):
    """In-place copy: dst_indices <- src_indices for each logit tensor."""
    with torch.no_grad():
        tree.leaf_logits.data[dst_indices] = tree.leaf_logits.data[src_indices]
        for lg in tree.internal_logits:
            lg.data[dst_indices] = lg.data[src_indices]


def _crossover_(
    tree: BatchedEMLTree, dst: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor
):
    """Uniform crossover: for each (node, input), pick p1 or p2 with 50/50.

    `dst`, `p1`, `p2` are equal-length LongTensors of tree indices. The logits
    at each dst slot are a coin-flip mix from the corresponding p1 and p2
    slots, one flip per (node, input) position.
    """
    with torch.no_grad():
        all_logit_tensors = [tree.leaf_logits.data] + [
            lg.data for lg in tree.internal_logits
        ]
        for logits in all_logit_tensors:
            # shape: (B, nodes, inputs, choices)
            shape_mask = (len(dst), logits.shape[1], logits.shape[2], 1)
            mask = torch.rand(*shape_mask, device=logits.device) < 0.5
            src_1 = logits[p1]  # (D, nodes, inputs, choices)
            src_2 = logits[p2]
            mixed = torch.where(mask, src_1, src_2)
            logits[dst] = mixed


def _seed_from_shallower_(
    deep_tree: BatchedEMLTree,
    seed_slots: torch.Tensor,
    src_tree: BatchedEMLTree,
    src_idx: int,
):
    """Plant src_tree[src_idx] (depth d-1) as the left subtree of
    deep_tree's `seed_slots` (depth d).

    The left subtree at depth d occupies the first 2^(d-2) leaves and the
    first 2^(d-1-l) nodes of internal level l (for l = 2 .. d-1).
    The root internal node + right subtree are left as their random init.

    NOTE: this does not preserve the shallower tree's exact output — EML's
    primitive `eml(x, y) = exp(x) - ln(y)` has no identity, so any root node
    will transform the subtree's output. Seeding works by giving evolution a
    known-good building block to compose around, not by warm-starting with
    the prior answer verbatim.
    """
    with torch.no_grad():
        n_prev_leaves = src_tree.leaf_logits.shape[1]
        # Sanity check: prev leaves should equal half of deep's leaves
        assert (
            deep_tree.leaf_logits.shape[1] == 2 * n_prev_leaves
        ), f"Depth mismatch: deep {deep_tree.leaf_logits.shape[1]} vs 2*src {2*n_prev_leaves}"

        src_leaf = src_tree.leaf_logits.data[src_idx]  # (L_prev, 2, C)
        deep_tree.leaf_logits.data[seed_slots, :n_prev_leaves] = src_leaf

        for lvl_idx, src_lg in enumerate(src_tree.internal_logits):
            src_block = src_lg.data[src_idx]  # (n_nodes_src, 2, C)
            n_src_nodes = src_block.shape[0]
            deep_tree.internal_logits[lvl_idx].data[
                seed_slots, :n_src_nodes
            ] = src_block


def evolve(
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: EvolutionConfig,
    var_names: list[str] | None = None,
    seed_tree: BatchedEMLTree | None = None,
    seed_idx: int | None = None,
    seed_fraction: float = 0.2,
) -> EvolutionResult:
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

    if cfg.n_islands > 1 and cfg.population % cfg.n_islands != 0:
        raise ValueError(
            f"population ({cfg.population}) must be divisible by n_islands "
            f"({cfg.n_islands}) for the island model. The top-level "
            "`emltorch.fit` rounds population up automatically; if you build "
            "EvolutionConfig directly, choose a divisible population."
        )

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
        num_trees=cfg.population,
        depth=cfg.depth,
        num_vars=V,
        dtype=cfg.torch_dtype,
        device=device,
        init_scale=50.0,
        init_mode="peaked",
        use_mul=cfg.use_mul,
        use_mul3=cfg.use_mul3,
    )
    _snap_peaked(tree)

    if cfg.normalize_inputs:
        x_mean = x_batch.mean(dim=-1, keepdim=True).unsqueeze(0)  # (1, V, 1)
        x_std = (
            x_batch.std(dim=-1, keepdim=True).clamp(min=1e-8).unsqueeze(0)
        )  # (1, V, 1)
        tree.set_normalization_stats(x_mean, x_std)

    # Warm-start: seed a fraction of the population with `seed_tree[seed_idx]`
    # placed in the left-subtree slot (depth d-1 embedded into depth d).
    if seed_tree is not None and seed_idx is not None:
        n_seed = max(1, int(cfg.population * seed_fraction))
        seed_slots = torch.arange(n_seed, device=device)
        _seed_from_shallower_(tree, seed_slots, seed_tree, seed_idx)
        # Diversify: mutate a single edge on all but the first seed
        if n_seed > 1:
            _mutate_(tree, seed_slots[1:], n_mutations=1)
        if cfg.verbose:
            print(
                f"[evo] seeded {n_seed}/{cfg.population} from shallower tree "
                f"(depth {seed_tree.depth} → {cfg.depth})"
            )

    n_elite = max(1, int(cfg.population * cfg.elite_fraction))
    generation_r2s = []
    best_ever_mse = torch.tensor(float("inf"), device=device)
    best_ever_idx = 0
    best_ever_logits = None

    for gen in range(cfg.generations):
        # Fitness (for selection) uses range penalty; r2 report uses raw MSE.
        fitness = _evaluate(tree, x_pop, y_pop, range_penalty=cfg.range_penalty)
        mse = (
            _evaluate(tree, x_pop, y_pop, range_penalty=0.0)
            if cfg.range_penalty > 0.0
            else fitness.clone()
        )
        # Cert-friendly bias (Track 5): reward leaf-constant-1 usage.
        # Fitness is the SELECTION criterion, so the bonus only changes which
        # individuals are kept as elites — the MSE-based best_ever_idx tracker
        # is unaffected.
        if cfg.cert_friendly_const_bonus > 0.0:
            fitness = fitness + _cert_friendly_bonus(
                tree, cfg.cert_friendly_const_bonus
            )
        r2 = 1 - mse * N / ss_tot

        # Track global best by raw MSE (not fitness) so range_penalty doesn't
        # bias which tree snapshot we return — selection elites still use fitness.
        min_mse_this_gen, argmin_mse_this_gen = mse.min(dim=0)
        if min_mse_this_gen < best_ever_mse:
            best_ever_mse = min_mse_this_gen
            best_ever_idx = int(argmin_mse_this_gen)
            # Snapshot best tree's logits
            best_ever_logits = (
                tree.leaf_logits[best_ever_idx].clone(),
                [lg[best_ever_idx].clone() for lg in tree.internal_logits],
            )

        best_r2 = r2.max().item()
        generation_r2s.append(best_r2)

        if cfg.verbose and (gen % cfg.log_every == 0 or gen == cfg.generations - 1):
            print(
                f"[evo] gen {gen:>3d}  best R²={best_r2:+.4f}  "
                f"elite R² [{r2.topk(n_elite).values.min().item():+.4f}, "
                f"{best_r2:+.4f}]"
            )

        if best_r2 >= cfg.r2_target:
            if cfg.verbose:
                print(f"[evo] r2_target reached at gen {gen}")
            break

        if cfg.n_islands <= 1:
            # ---- Panmictic path (original; unchanged) ----------------------
            # Selection: top-k by fitness (MSE + range penalty)
            _, elite_idx = fitness.topk(n_elite, largest=False)

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
                parent_idx = elite_idx[
                    torch.randint(0, n_elite, (n_mut,), device=device)
                ]
                _clone_(tree, mut_slots, parent_idx)
                _mutate_(tree, mut_slots, n_mutations=cfg.mutations_per_child)

            # Crossover branch: pick two random elites, uniform crossover
            if n_cross > 0:
                p1 = elite_idx[torch.randint(0, n_elite, (n_cross,), device=device)]
                p2 = elite_idx[torch.randint(0, n_elite, (n_cross,), device=device)]
                _crossover_(tree, cross_slots, p1, p2)
                # Small mutation on crossover children for diversity
                _mutate_(tree, cross_slots, n_mutations=1)
        else:
            # ---- Island path (block-structured selection + ring migration) -
            # Selection and reproduction are confined within each island, so
            # distinct islands can settle into distinct basins. Every
            # `migration_interval` generations the best `migration_size`
            # individuals of each island replace the WORST of its ring
            # successor, sharing good genes without collapsing diversity.
            M = cfg.n_islands
            S = cfg.population // M  # population guaranteed divisible (api/assert)
            n_elite_isl = max(1, int(S * cfg.elite_fraction))
            island_base = (torch.arange(M, device=device) * S).unsqueeze(1)  # (M,1)

            fit_blk = fitness.view(M, S)
            # Best (lowest-fitness) n_elite_isl within each island.
            elite_local = fit_blk.topk(n_elite_isl, dim=1, largest=False).indices
            elite_global = elite_local + island_base  # (M, n_elite_isl)
            elite_flat = elite_global.reshape(-1)

            non_elite_mask = torch.ones(cfg.population, dtype=torch.bool, device=device)
            non_elite_mask[elite_flat] = False
            non_elite_slots = torch.arange(cfg.population, device=device)[
                non_elite_mask
            ]
            slot_island = non_elite_slots // S  # island id per offspring slot

            n_offspring = non_elite_slots.numel()
            n_cross = int(n_offspring * cfg.crossover_fraction)
            n_mut = n_offspring - n_cross
            mut_slots, mut_isl = non_elite_slots[:n_mut], slot_island[:n_mut]
            cross_slots, cross_isl = non_elite_slots[n_mut:], slot_island[n_mut:]

            # Mutation: parent is a random elite from the SAME island.
            if n_mut > 0:
                rc = torch.randint(0, n_elite_isl, (n_mut,), device=device)
                _clone_(tree, mut_slots, elite_global[mut_isl, rc])
                _mutate_(tree, mut_slots, n_mutations=cfg.mutations_per_child)

            # Crossover: both parents random elites from the SAME island.
            if n_cross > 0:
                rc1 = torch.randint(0, n_elite_isl, (n_cross,), device=device)
                rc2 = torch.randint(0, n_elite_isl, (n_cross,), device=device)
                _crossover_(
                    tree,
                    cross_slots,
                    elite_global[cross_isl, rc1],
                    elite_global[cross_isl, rc2],
                )
                _mutate_(tree, cross_slots, n_mutations=1)

            # Ring migration: island d's worst-k <- island (d-1)%M's best-k.
            if (gen + 1) % max(1, cfg.migration_interval) == 0:
                k = min(cfg.migration_size, n_elite_isl, S)
                src_best = elite_global[:, :k]  # (M, k) best-k per island
                src_ring = torch.roll(src_best, shifts=1, dims=0)  # from predecessor
                worst_local = fit_blk.topk(k, dim=1, largest=True).indices  # (M,k)
                dest_global = (worst_local + island_base).reshape(-1)
                _clone_(tree, dest_global, src_ring.reshape(-1))

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
        all_preds = tree(x_pop)  # (B, N)
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

    # Recompute raw (no-penalty) MSE for the restored best tree
    raw_mse_all = _evaluate(tree, x_pop, y_pop, range_penalty=0.0)
    best_mse_val = float(raw_mse_all[best_ever_idx].item())
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
