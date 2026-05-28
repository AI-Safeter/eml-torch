"""
Stable public API for emltorch: a single `fit()` function that dispatches
to the best search strategy for the given depth.

Routing:
    depth 1-2       → random search (peaked init, 256 restarts)
    depth 3-4       → evolution + affine wrapper
    depth 5+        → evolution + affine + more generations
"""

import warnings
from dataclasses import dataclass
from typing import Literal

import torch

from .evolution import EvolutionConfig, evolve
from .symbolic import annotate
from .polish import polish as polish_tree


def _coerce_inputs(x, y, device):
    """Accept numpy / list / torch; return torch float tensors on `device`
    shaped x=(V, N), y=(N,). Handles both (N, V) and (V, N) for x by
    aligning the sample dimension with len(y)."""
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)
    if not x.is_floating_point():
        x = x.float()
    if not y.is_floating_point():
        y = y.float()

    if y.ndim == 2 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D (or (N,1)); got shape {tuple(y.shape)}")
    N = y.shape[0]
    if N == 0:
        raise ValueError("x and y must be non-empty")

    if x.ndim == 1:
        x = x.unsqueeze(0)  # → (1, N)
    elif x.ndim == 2:
        rows, cols = x.shape
        if cols == N and rows != N:
            pass  # already (V, N)
        elif rows == N and cols != N:
            x = x.t().contiguous()  # (N, V) → (V, N)
        elif rows == N and cols == N:
            warnings.warn(
                f"x is square ({N}x{N}); assuming sklearn (N, V) convention "
                "and transposing to (V, N). Pass shape explicitly if wrong.",
                stacklevel=3,
            )
            x = x.t().contiguous()
        else:
            raise ValueError(
                f"x shape {tuple(x.shape)} incompatible with len(y)={N}; "
                "expected (N,), (N, V), or (V, N)."
            )
    else:
        raise ValueError(f"x must be 1D or 2D; got {x.ndim}D")

    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("x or y contains NaN/Inf; clean inputs before calling fit().")

    return x.to(device), y.to(device)


@dataclass
class FitResult:
    """Result of `emltorch.fit(x, y, ...)`."""

    expression: str  # "a + b * (eml-tree-formula)"
    r2: float  # coefficient of determination on x,y
    mse: float  # MSE after affine rescaling (if used)
    depth_used: int  # tree depth chosen by router
    strategy: str  # "random" | "evolution"
    a: float  # affine intercept
    b: float  # affine scale
    time_s: float
    generations: list[float] | None = None  # R² per generation (evolution only)
    # Internal handles for evaluation; set by fit().
    _tree: object = None
    _idx: int = 0
    _device: str = "cpu"

    def predict(self, x) -> torch.Tensor:
        """Evaluate the discovered formula on new data.

        Accepts the same x conventions as ``fit`` (numpy / list / torch;
        (N,), (N, V), or (V, N)). Returns a 1-D torch.Tensor of length N
        with the affine-wrapped prediction ``a + b * tree(x)``.
        """
        # Coerce x to (V, N) shape matching training
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        if not x.is_floating_point():
            x = x.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 2:
            # If V doesn't match tree's V, try transposing
            V_tree = self._tree.num_vars
            if x.shape[0] != V_tree and x.shape[1] == V_tree:
                x = x.t().contiguous()
        x = x.to(self._device)
        # Tree forward expects (B, V, N) or (B, N) for V=1; broadcast same x
        # to all B trees, then select the best one.
        with torch.no_grad():
            B = self._tree.num_trees
            V = self._tree.num_vars
            N = x.shape[-1]
            if V == 1:
                x_b = (
                    (x.squeeze(0) if x.ndim == 2 else x)
                    .unsqueeze(0)
                    .expand(B, N)
                    .contiguous()
                )
            else:
                x_b = x.unsqueeze(0).expand(B, V, N).contiguous()
            preds_all = self._tree.forward(x_b)  # (B, N)
            tree_pred = preds_all[self._idx]
        return (self.a + self.b * tree_pred).cpu()


def fit(
    x,
    y,
    depth: int = 3,
    *,
    strategy: Literal["auto", "random", "evolution"] = "auto",
    population: int | None = None,
    generations: int | None = None,
    device: str | None = None,
    r2_target: float = 0.99,
    polish: bool = False,
    polish_iters: int = 2000,
    normalize_inputs: bool = False,
    n_islands: int = 1,
    migration_interval: int = 5,
) -> FitResult:
    """
    Discover a closed-form EML expression fitting y ≈ f(x).

    Args:
        x: features. numpy array, list, or torch tensor. Accepted shapes:
           (N,) for one variable; (N, V) sklearn-style; or (V, N).
           Sample dimension is auto-aligned with len(y).
        y: target, length N. numpy array, list, or torch tensor.
        depth: maximum EML tree depth to search.
        strategy: "auto" picks the best method for the given depth.
                  "random" = peaked init + evaluate only (fast, shallow only).
                  "evolution" = population-based search with affine wrapper.
        population: for evolution/random, number of candidate trees in
                    parallel. Defaults: depth≤3 → 1024, depth 4 → 2048,
                    depth 5+ → 4096.
        generations: for evolution, number of generations. Defaults: 20.
        device: torch device string (e.g. "cuda", "cuda:0", or "cpu"). If
                None, auto-resolves to "cuda" when available, else "cpu".
        r2_target: early-exit threshold.
        n_islands: number of independent sub-populations (islands) the
                   population is split into. Islands evolve in isolation
                   with periodic ring migration of their best individuals,
                   which preserves diversity and explores multiple basins —
                   the targeted fix for the basin-trap failure mode where a
                   single panmictic population (and every seed of it)
                   collapses to one local optimum. Default 1 = original
                   panmictic evolution (byte-identical behaviour). When > 1,
                   ``population`` is rounded up to the nearest multiple of
                   ``n_islands``. Only affects the "evolution" strategy.
        migration_interval: migrate every this-many generations (ignored
                   when n_islands == 1).

    Returns:
        FitResult with the discovered expression, R², and metadata.
    """
    import time

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if n_islands < 1:
        raise ValueError(f"n_islands must be ≥ 1; got {n_islands}")

    # --- Input coercion (accepts numpy / list / torch; (N,), (N,V), or (V,N)) ---
    x, y = _coerce_inputs(x, y, device)
    V, N = x.shape

    # --- Strategy routing ---
    if strategy == "auto":
        strategy = "random" if depth <= 2 else "evolution"

    # --- Defaults ---
    if population is None:
        population = {1: 256, 2: 256, 3: 1024, 4: 2048}.get(depth, 4096)
    if generations is None:
        generations = 20

    # Island model needs an evenly divisible population; round up so each
    # island gets the same number of slots (the block-structured selection
    # in evolve() reshapes population into (n_islands, island_size)).
    if n_islands > 1 and population % n_islands != 0:
        population += n_islands - (population % n_islands)

    t0 = time.time()

    if strategy == "random":
        cfg = EvolutionConfig(
            depth=depth,
            num_vars=V,
            population=population,
            generations=1,  # one gen == random eval
            elite_fraction=0.1,
            mutations_per_child=0,
            device=device,
            r2_target=r2_target,
            normalize_inputs=normalize_inputs,
        )
        res = evolve(x, y, cfg)
        return FitResult(
            expression=res.best_expression,
            r2=res.best_r2,
            mse=res.best_mse,
            depth_used=depth,
            strategy="random",
            a=res.best_a,
            b=res.best_b,
            time_s=time.time() - t0,
            generations=res.generation_r2s,
            _tree=res.best_tree,
            _idx=res.best_idx,
            _device=device,
        )

    if strategy == "evolution":
        cfg = EvolutionConfig(
            depth=depth,
            num_vars=V,
            population=population,
            generations=generations,
            elite_fraction=0.1,
            mutations_per_child=1,
            crossover_fraction=0.3,
            device=device,
            r2_target=r2_target,
            normalize_inputs=normalize_inputs,
            n_islands=n_islands,
            migration_interval=migration_interval,
        )
        res = evolve(x, y, cfg)

        # Opt-in polish step: fine-tune '1' leaves as learnable constants
        if polish:
            var_names = [f"x{i+1}" for i in range(V)] if V > 1 else ["x"]
            pol = polish_tree(
                res.best_tree,
                res.best_idx,
                x,
                y,
                var_names=var_names,
                n_iters=polish_iters,
                lr=1e-2,
                device=device,
                warm_a=res.best_a,
                warm_b=res.best_b,
            )
            # Accept polished result only if it strictly improved
            if pol.r2 > res.best_r2:
                return FitResult(
                    expression=pol.formula,
                    r2=pol.r2,
                    mse=pol.mse,
                    depth_used=depth,
                    strategy="evolution+polish",
                    a=pol.a,
                    b=pol.b,
                    time_s=time.time() - t0,
                    generations=res.generation_r2s,
                    _tree=res.best_tree,
                    _idx=res.best_idx,
                    _device=device,
                )

        return FitResult(
            expression=res.best_expression,
            r2=res.best_r2,
            mse=res.best_mse,
            depth_used=depth,
            strategy="evolution",
            a=res.best_a,
            b=res.best_b,
            time_s=time.time() - t0,
            generations=res.generation_r2s,
            _tree=res.best_tree,
            _idx=res.best_idx,
            _device=device,
        )

    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Multi-seed fit: operationalize the topology-stability discipline.
# ---------------------------------------------------------------------------


@dataclass
class MultiSeedResult:
    """Result of `emltorch.fit_multi_seed(x, y, n_seeds=N, ...)`.

    Aggregates `N` independent runs of `fit()` with different RNG seeds.
    Useful for honest reporting of whether the discovered closed-form is
    a stable property of the data (same expression on most seeds) or a
    noise artifact (different expression every seed).
    """

    n_seeds: int
    all_results: list  # list[FitResult]
    best_fit: object  # FitResult with highest R² (also the .predict() target)
    best_r2: float
    median_r2: float
    mean_r2: float
    std_r2: float
    topology_counts: dict  # {expression_str: count}
    top_topology: str  # most-common expression string
    top_topology_count: int  # how many seeds produced top_topology
    topology_stability: float  # top_topology_count / n_seeds  ∈ [0, 1]
    n_unique_topologies: int

    def predict(self, x):
        """Evaluate the *best-R² seed's* formula on new data."""
        return self.best_fit.predict(x)

    @property
    def expression(self) -> str:
        """The best-R² seed's expression string. For the most-common-across-seeds
        expression, use `self.top_topology` instead."""
        return self.best_fit.expression

    def summary(self) -> str:
        """Human-readable single-line summary."""
        return (
            f"MultiSeedResult(n_seeds={self.n_seeds}, "
            f"best_r2={self.best_r2:.4f}, median_r2={self.median_r2:.4f}, "
            f"topology_stability={self.top_topology_count}/{self.n_seeds} "
            f"= {self.topology_stability:.2f}, "
            f"unique_topologies={self.n_unique_topologies})"
        )

    def __repr__(self) -> str:
        return self.summary()


def fit_multi_seed(
    x,
    y,
    *,
    n_seeds: int = 10,
    depth: int = 3,
    strategy: Literal["auto", "random", "evolution"] = "auto",
    population: int | None = None,
    generations: int | None = None,
    device: str | None = None,
    r2_target: float = 0.99,
    polish: bool = False,
    polish_iters: int = 2000,
    normalize_inputs: bool = False,
    seed_start: int = 0,
) -> MultiSeedResult:
    """Run `fit()` independently with `n_seeds` different RNG seeds and aggregate.

    Each seed sets `torch.manual_seed(s)` and `np.random.seed(s)` before
    calling `fit()`, so the search is reproducibly varied. The returned
    `MultiSeedResult` reports per-seed R² and the byte-equality topology
    distribution across seeds — useful for the honest stability check
    ("does the same closed-form keep emerging, or am I overfitting?").

    Args mirror `fit()`. Additional:
        n_seeds: number of independent seeds. Default 10.
        seed_start: first RNG seed; subsequent seeds are seed_start..seed_start+n_seeds-1.

    Returns:
        MultiSeedResult with `best_fit` (FitResult), per-seed list,
        and aggregate stability metrics.
    """
    import numpy as np

    if n_seeds < 1:
        raise ValueError(f"n_seeds must be ≥ 1; got {n_seeds}")

    results: list[FitResult] = []
    for s in range(seed_start, seed_start + n_seeds):
        torch.manual_seed(s)
        np.random.seed(s)
        r = fit(
            x,
            y,
            depth=depth,
            strategy=strategy,
            population=population,
            generations=generations,
            device=device,
            r2_target=r2_target,
            polish=polish,
            polish_iters=polish_iters,
            normalize_inputs=normalize_inputs,
        )
        results.append(r)

    r2s = [r.r2 for r in results]
    best_idx = int(np.argmax(r2s))
    exprs = [r.expression for r in results]

    # Byte-equality topology counting (advisor-recommended precision)
    from collections import Counter

    topology_counts = dict(Counter(exprs))
    top_topology, top_topology_count = max(
        topology_counts.items(), key=lambda kv: kv[1]
    )

    return MultiSeedResult(
        n_seeds=n_seeds,
        all_results=results,
        best_fit=results[best_idx],
        best_r2=float(max(r2s)),
        median_r2=float(np.median(r2s)),
        mean_r2=float(np.mean(r2s)),
        std_r2=float(np.std(r2s)),
        topology_counts=topology_counts,
        top_topology=top_topology,
        top_topology_count=top_topology_count,
        topology_stability=top_topology_count / n_seeds,
        n_unique_topologies=len(topology_counts),
    )


# ---------------------------------------------------------------------------
# Residual boosting: fit a sequence of EML trees, each on the residuals of
# the previous combined prediction. The final prediction is the SUM of stages.
# Same principle as gradient-boosted decision trees, but with EML as the
# base learner — keeps the result symbolic + SMT-translatable (cert each
# stage independently and sum the bounds).
# ---------------------------------------------------------------------------


@dataclass
class BoostedResult:
    """Result of `emltorch.fit_residual_boost(x, y, n_stages=K, ...)`.

    The combined predictor is the SUM of the per-stage `FitResult` objects:

        f(x) = stage_fits[0].predict(x) + stage_fits[1].predict(x) + ...

    `stage_fits` are returned in order; stage 0 fits the original target, each
    subsequent stage fits the residuals (target − cumulative prediction so
    far). Each stage is a standard FitResult so its expression, SMT cert,
    and `predict()` work independently.
    """

    n_stages: int
    stage_fits: list  # list[FitResult]
    cumulative_r2_train: list  # R² of the sum-of-stages on TRAIN, per stage
    final_r2_train: float
    time_s: float

    def predict(self, x):
        """Combined prediction: sum of all stage `predict()` outputs."""
        import torch

        ys = [stage.predict(x) for stage in self.stage_fits]
        out = ys[0]
        for y in ys[1:]:
            out = out + y
        return out

    @property
    def expression(self) -> str:
        """Human-readable additive form: `expr_0 + expr_1 + ...`."""
        parts = [f"({s.expression})" for s in self.stage_fits]
        return " + ".join(parts)

    def summary(self) -> str:
        first = (
            self.cumulative_r2_train[0] if self.cumulative_r2_train else float("nan")
        )
        last = self.final_r2_train
        delta = last - first
        return (
            f"BoostedResult(n_stages={self.n_stages}, "
            f"stage1_train_r2={first:.4f}, final_train_r2={last:.4f}, "
            f"Δ={delta:+.4f})"
        )

    def __repr__(self) -> str:
        return self.summary()


def fit_residual_boost(
    x,
    y,
    *,
    n_stages: int = 3,
    depth: int = 3,
    strategy: Literal["auto", "random", "evolution"] = "auto",
    population: int | None = None,
    generations: int | None = None,
    device: str | None = None,
    r2_target: float = 0.99,
    polish: bool = False,
    polish_iters: int = 2000,
    normalize_inputs: bool = True,
    seed_start: int = 0,
    seeds_per_stage: int = 1,
) -> BoostedResult:
    """Gradient-boosting-style residual fit with EML as the base learner.

    Fits `n_stages` EML trees sequentially: stage 0 targets `y`, stage k>0
    targets the residual `y − sum_{j<k} stage_j(x)`. The combined predictor
    is the additive sum of all stages.

    Why use this. A single EML tree of bounded depth has a finite
    expressive class. Some targets (e.g. transformer-behavior probes whose
    P_target involves rational/sigmoid-like structure) sit at the edge of
    that class — single-stage EML reaches a local optimum that misses
    informative features. Empirically, on Gemma-4-31B-it induction probes,
    3-stage residual boosting lifts HELDOUT R² by ≈ +0.02 to +0.04 over
    single-stage EML, with the second and third stages picking up
    features the first stage's local optimum dropped.

    The result is still purely symbolic: a sum of small EML expressions,
    each SMT-translatable independently (lower bounds on the sum follow
    from per-stage interval bounds).

    Numerical safety. ``normalize_inputs`` defaults to ``True`` here (unlike
    ``fit()``, which defaults to ``False``). The additive predictor sums
    ``n_stages`` affine-wrapped trees ``a_k + b_k * tree_k(x)``; the EML leaf
    ``exp`` is clamped per-node, but a single leaf can still reach
    ``exp(80) ≈ 5.5e34``, and neither the affine ``b_k`` nor the stage sum is
    bounded. On *unstandardized*, large-range features a leaf such as
    ``eml(x_i - x_j, exp(x_k))`` evaluated at an *in-distribution* held-out
    point can already extrapolate to ``±1e6..1e7`` — the raw feature
    differences are large — which dominates the sum and destroys held-out R²
    (observed: a random-split blow-up to R² ~= -5.6e7 on in-distribution test
    points). Normalizing inputs to train mean-0/std-1 keeps the leaf
    arguments at unit scale, so in-distribution held-out predictions stay
    bounded (the -5.6e7 case drops by ~3 orders of magnitude). This is a
    mitigation, not a hard bound: a point pushed several std beyond the
    training range still grows through ``exp(.)`` — symbolic regressors do not
    promise safe extrapolation outside the data manifold. The normalization is
    a *linear* pre-transform on the inputs, so the per-stage SMT-LIB2
    translation is preserved (it composes with the affine input map under
    QF_LRA). If you pass ``normalize_inputs=False`` on raw, wide-range
    features, a warning is emitted because the additive predictor can then
    extrapolate without bound.

    Args mirror `fit()`. Additional:
        n_stages: number of boosting stages. Default 3. Each stage adds
                  one EML tree to the additive predictor.
        seed_start: first RNG seed. Stage k, seed-candidate s uses
                    `seed_start + 100*k + s`.
        seeds_per_stage: candidate fits per stage; the candidate with the
                    best fit on that stage's residual target is kept
                    (default 1 = single fit). Higher values help on the
                    harder slices where a single seed lands in a weak
                    basin — e.g. Gemma L_large stage-3 lifts ≈ +0.01
                    going from 1 to 5 seeds-per-stage (compounds the
                    `fit_multi_seed` discipline into each boosting stage).

    Returns:
        BoostedResult with `stage_fits` (list of FitResult), additive
        `predict()`, and the cumulative train R² per stage.
    """
    import time

    import numpy as np

    if n_stages < 1:
        raise ValueError(f"n_stages must be ≥ 1; got {n_stages}")
    if seeds_per_stage < 1:
        raise ValueError(f"seeds_per_stage must be ≥ 1; got {seeds_per_stage}")

    # Coerce y once; we'll work with numpy residuals between stages.
    if not isinstance(y, torch.Tensor):
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    else:
        y_arr = y.detach().cpu().numpy().astype(np.float64).reshape(-1)

    # Safety: the additive predictor is unbounded; on raw wide-range features
    # an EML leaf exp(.) can extrapolate to ±1e6+ on held-out points and
    # dominate the sum. Warn if the caller opted out of normalization with
    # features whose spread makes this likely.
    if not normalize_inputs:
        try:
            x_np = (
                x.detach().cpu().numpy()
                if isinstance(x, torch.Tensor)
                else np.asarray(x, dtype=np.float64)
            )
            col_std = float(np.nanmax(np.std(np.atleast_2d(x_np), axis=-1)))
            if col_std > 5.0:
                warnings.warn(
                    "fit_residual_boost called with normalize_inputs=False on "
                    f"features whose max per-feature std is {col_std:.1f}. The "
                    "additive EML predictor is unbounded; on out-of-range "
                    "held-out points an exp(.) leaf can extrapolate to ±1e6+ "
                    "and destroy held-out R². Pass normalize_inputs=True "
                    "(the default) or standardize inputs first.",
                    stacklevel=2,
                )
        except Exception:
            pass

    stage_fits: list = []
    cumulative_train: list = []
    cum_pred = np.zeros_like(y_arr)
    t0 = time.time()

    def _residual_r2(fit_result, residual_arr) -> float:
        yp_np = (
            fit_result.predict(x).detach().cpu().numpy().astype(np.float64).reshape(-1)
        )
        ss_res = float(np.sum((residual_arr - yp_np) ** 2))
        ss_tot = float(np.sum((residual_arr - residual_arr.mean()) ** 2)) + 1e-12
        return 1.0 - ss_res / ss_tot

    for k in range(n_stages):
        residual = y_arr - cum_pred
        # Run `seeds_per_stage` candidate fits on this stage's residual; keep
        # the one that best explains the residual target.
        best_r = None
        best_stage_r2 = -float("inf")
        for s in range(seeds_per_stage):
            seed = seed_start + 100 * k + s
            torch.manual_seed(seed)
            np.random.seed(seed)
            candidate = fit(
                x,
                residual,
                depth=depth,
                strategy=strategy,
                population=population,
                generations=generations,
                device=device,
                r2_target=r2_target,
                polish=polish,
                polish_iters=polish_iters,
                normalize_inputs=normalize_inputs,
            )
            cand_r2 = _residual_r2(candidate, residual)
            if cand_r2 > best_stage_r2:
                best_stage_r2 = cand_r2
                best_r = candidate
        r = best_r
        stage_fits.append(r)
        # Update cumulative prediction on TRAIN inputs (x is the original).
        yp = r.predict(x)
        yp_np = yp.detach().cpu().numpy().astype(np.float64).reshape(-1)
        cum_pred = cum_pred + yp_np
        # Cumulative R² vs original y
        ss_res = float(np.sum((y_arr - cum_pred) ** 2))
        ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2)) + 1e-12
        cumulative_train.append(1.0 - ss_res / ss_tot)

    return BoostedResult(
        n_stages=n_stages,
        stage_fits=stage_fits,
        cumulative_r2_train=cumulative_train,
        final_r2_train=cumulative_train[-1] if cumulative_train else float("nan"),
        time_s=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# Pareto-front fit: accuracy vs. complexity trade-off.
# Returns the non-dominated subset of (complexity, R², FitResult) across depths.
# ---------------------------------------------------------------------------


def _expression_complexity(expr: str) -> int:
    """Return the number of EML operators in a formula string.

    Complexity is defined as ``expr.count("eml(")``.  This is deterministic,
    depth-agnostic, and matches the "expression size (nodes)" notion used in
    the project README.  A pure-constant or single-variable result has
    complexity 0; each nested ``eml(...)`` node adds 1.

    Examples::

        _expression_complexity("eml(1, x)")           # → 1
        _expression_complexity("eml(eml(1, x), 1)")   # → 2
        _expression_complexity("x")                   # → 0
    """
    return expr.count("eml(")


@dataclass
class ParetoResult:
    """Result of `emltorch.fit_pareto(x, y, ...)`.

    Holds the non-dominated Pareto front of (complexity, R²) pairs discovered
    by sweeping over EML tree depths.  Lets callers choose their own accuracy
    vs. complexity trade-off instead of receiving only the single highest-R²
    result.

    Attributes:
        front: list of ``(complexity: int, r2: float, fit: FitResult)`` tuples,
            NON-DOMINATED and sorted by complexity ASCENDING.  Along the front,
            both complexity and r2 are STRICTLY INCREASING — every step to higher
            complexity buys strictly better R².
        all_evaluated: list of ``(complexity, r2, fit)`` for every depth tried,
            including dominated points.
    """

    front: list  # list[(complexity:int, r2:float, fit:FitResult)], asc complexity
    all_evaluated: list  # list[(complexity, r2, fit)] — every depth tried

    def best(self) -> "FitResult":
        """Return the FitResult with the highest R² on the Pareto front."""
        return max(self.front, key=lambda t: t[1])[2]

    def select(self, max_complexity: int):
        """Return the best-R² FitResult whose complexity <= *max_complexity*.

        Returns ``None`` if no front point satisfies the budget.
        """
        candidates = [(r2, fit) for c, r2, fit in self.front if c <= max_complexity]
        if not candidates:
            return None
        return max(candidates, key=lambda t: t[0])[1]

    def predict(self, x) -> "torch.Tensor":
        """Evaluate the best (highest-R²) front formula on new data."""
        return self.best().predict(x)

    def summary(self) -> str:
        """One-line string listing the front as ``complexity->r2`` pairs."""
        pairs = " | ".join(f"{c}->{r2:.4f}" for c, r2, _ in self.front)
        return f"ParetoResult(front=[{pairs}], n_evaluated={len(self.all_evaluated)})"

    def __repr__(self) -> str:
        return self.summary()


def fit_pareto(
    x,
    y,
    *,
    depths=(1, 2, 3, 4, 5),
    seeds_per_depth: int = 1,
    population: int | None = None,
    generations: int | None = None,
    device: str | None = None,
    r2_target: float = 0.99,
    normalize_inputs: bool = False,
) -> "ParetoResult":
    """Discover a Pareto-optimal front of EML expressions (accuracy vs. complexity).

    Sweeps over each depth in *depths*, running ``fit()`` (or
    ``fit_multi_seed()`` when ``seeds_per_depth > 1``), and returns the
    non-dominated subset — every step to higher complexity on the front buys
    strictly better R².

    This mirrors PySR's Pareto-front output, letting callers choose their own
    accuracy vs. interpretability trade-off instead of receiving only the
    single highest-R² result.

    Args:
        x: features (same conventions as ``fit``).
        y: target, length N.
        depths: iterable of integer depths to try. Default ``(1, 2, 3, 4, 5)``.
        seeds_per_depth: number of random seeds per depth.  ``1`` → single
            ``fit()`` call; ``> 1`` → ``fit_multi_seed()`` and best seed kept.
        population: passed to ``fit()`` / ``fit_multi_seed()``. Defaults to
            the per-depth heuristic in ``fit()``.
        generations: passed to ``fit()`` / ``fit_multi_seed()``.
        device: torch device. Auto-resolves when ``None``.
        r2_target: early-exit threshold per depth.
        normalize_inputs: passed to ``fit()``.

    Returns:
        ParetoResult with ``.front`` (non-dominated, complexity-sorted) and
        ``.all_evaluated`` (every depth tried).
    """
    import numpy as np

    all_evaluated: list = []

    for depth_idx, d in enumerate(depths):
        # Set reproducible seeds for each depth so repeated calls are stable.
        seed = 20260528 + d
        torch.manual_seed(seed)
        np.random.seed(seed)

        if seeds_per_depth == 1:
            fit_result = fit(
                x,
                y,
                depth=d,
                population=population,
                generations=generations,
                device=device,
                r2_target=r2_target,
                normalize_inputs=normalize_inputs,
            )
        else:
            ms = fit_multi_seed(
                x,
                y,
                n_seeds=seeds_per_depth,
                depth=d,
                population=population,
                generations=generations,
                device=device,
                r2_target=r2_target,
                normalize_inputs=normalize_inputs,
                seed_start=seed,
            )
            fit_result = ms.best_fit

        complexity = _expression_complexity(fit_result.expression)
        all_evaluated.append((complexity, float(fit_result.r2), fit_result))

    # --- Build Pareto front ---
    # Step 1: collapse exact (complexity, r2) ties, keeping the entry that
    # appeared first (lowest depth index — deterministic when depths is ordered).
    seen: dict = {}  # (c, r2) -> index in all_evaluated
    for idx, (c, r2, fit_r) in enumerate(all_evaluated):
        key = (c, round(r2, 10))
        if key not in seen:
            seen[key] = idx
    deduped = [all_evaluated[i] for i in sorted(seen.values())]

    # Step 2: domination filter.
    # Point P_i is dominated if there exists P_j with
    #   c_j <= c_i  AND  r2_j >= r2_i  AND  (c_j < c_i OR r2_j > r2_i)
    front = []
    for i, (ci, ri, fi) in enumerate(deduped):
        dominated = False
        for j, (cj, rj, _) in enumerate(deduped):
            if i == j:
                continue
            if cj <= ci and rj >= ri and (cj < ci or rj > ri):
                dominated = True
                break
        if not dominated:
            front.append((ci, ri, fi))

    # Step 3: sort by complexity ascending.
    front.sort(key=lambda t: t[0])

    # Sanity check: along the sorted front, r2 is strictly increasing.
    # (By the domination definition this must hold; assert defensively.)
    for k in range(len(front) - 1):
        ck, rk, _ = front[k]
        ck1, rk1, _ = front[k + 1]
        assert ck < ck1, f"Front complexity not strictly increasing: {ck} -> {ck1}"
        assert rk1 > rk, (
            f"Front R² not strictly increasing at complexities {ck}->{ck1}: "
            f"{rk:.6f} -> {rk1:.6f}"
        )

    return ParetoResult(front=front, all_evaluated=all_evaluated)
