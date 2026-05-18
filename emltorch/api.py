"""
Stable public API for emltorch — a single `fit()` function that dispatches
to the best search strategy for the given depth.

The routing logic (as of v0.1.0):
    depth 1-2       → random search (peaked init, 256 restarts)
    depth 3-4       → evolution + affine wrapper
    depth 5+        → evolution + affine + more generations
    (any)           → fall back to gradient trainer if user asks
"""

from dataclasses import dataclass
from typing import Literal

import torch

from .evolution import EvolutionConfig, evolve
from .trainer import EMLConfig, EMLTrainer
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

    if x.ndim == 1:
        x = x.unsqueeze(0)  # → (1, N)
    elif x.ndim == 2:
        rows, cols = x.shape
        if cols == N and rows != N:
            pass  # already (V, N)
        elif rows == N and cols != N:
            x = x.t().contiguous()  # (N, V) → (V, N)
        elif rows == N and cols == N:
            # ambiguous square — assume sklearn (N, V) convention
            x = x.t().contiguous()
        else:
            raise ValueError(
                f"x shape {tuple(x.shape)} incompatible with len(y)={N}; "
                "expected (N,), (N, V), or (V, N)."
            )
    else:
        raise ValueError(f"x must be 1D or 2D; got {x.ndim}D")

    return x.to(device), y.to(device)


@dataclass
class FitResult:
    """Result of `emltorch.fit(x, y, ...)`."""

    expression: str  # "a + b * (eml-tree-formula)"
    r2: float  # coefficient of determination on x,y
    mse: float  # MSE after affine rescaling (if used)
    depth_used: int  # tree depth chosen by router
    strategy: str  # "random" | "evolution" | "gradient"
    a: float  # affine intercept (0 if strategy=gradient)
    b: float  # affine scale (1 if strategy=gradient)
    time_s: float
    generations: list[float] | None = None  # R² per generation (evolution only)


def fit(
    x,
    y,
    depth: int = 3,
    *,
    strategy: Literal["auto", "random", "evolution", "gradient"] = "auto",
    population: int | None = None,
    generations: int | None = None,
    device: str | None = None,
    r2_target: float = 0.99,
    polish: bool = False,
    polish_iters: int = 2000,
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
                  "gradient" = original Adam + hardening (usually worse).
        population: for evolution/random, number of candidate trees in
                    parallel. Defaults: depth≤3 → 1024, depth 4 → 2048,
                    depth 5+ → 4096.
        generations: for evolution, number of generations. Defaults: 20.
        device: torch device string (e.g. "cuda", "cuda:0", or "cpu"). If
                None, auto-resolves to "cuda" when available, else "cpu".
        r2_target: early-exit threshold.

    Returns:
        FitResult with the discovered expression, R², and metadata.
    """
    import time

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
        )

    if strategy == "gradient":
        x_b = x.unsqueeze(0)  # (1, V, N)
        y_b = y.unsqueeze(0)  # (1, N)
        cfg = EMLConfig(
            depth=depth,
            num_restarts=population or 32,
            num_vars=V,
            device=device,
            snap_mse_threshold=1e-3,
        )
        res = EMLTrainer(cfg).fit(x_b, y_b)
        mse = res.mse_values[0].item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        r2 = 1 - mse * N / max(ss_tot, 1e-12)
        return FitResult(
            expression=annotate(res.expressions[0]),
            r2=r2,
            mse=mse,
            depth_used=depth,
            strategy="gradient",
            a=0.0,
            b=1.0,
            time_s=time.time() - t0,
        )

    raise ValueError(f"Unknown strategy: {strategy}")
