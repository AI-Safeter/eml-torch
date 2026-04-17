"""
Push-to-the-limit sin(x) recovery.

Tries depths 5 → 6 → 7 → 8 in escalating fashion. At each depth, runs
evolution with maximum feasible population, long generations, crossover,
and final polish with many Adam iterations.

Records best R² per depth, first depth to hit R² > 0.99, and total
formula + constants at that breakthrough depth.

Memory strategy: depth 8 with population=8192 @ float32 @ N=1024 takes
~20 GB peak. Tuned so that it fits in 45 GB while leaving headroom.
"""

import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import emltorch as eml
from emltorch.evolution import EvolutionConfig, evolve
from emltorch.polish import polish

DEVICE = "cuda:5"


def make_data(n=1024, low=-math.pi, high=math.pi):
    x = torch.linspace(low, high, n)
    y = torch.sin(x)
    return x.unsqueeze(0), y  # (1, N), (N,)


def pop_for_depth(depth: int, n: int) -> int:
    """Max feasible population that fits on GPU:5 (~45 GB free)."""
    # Empirical: memory scales as ~B * 2^depth * N * bytes_per_element * 4 (live intermediates)
    # Budget 25 GB for the core pop tensors, leave rest for tree params.
    budget_bytes = 25 * 1024**3
    bytes_per_element = 4               # float32
    per_tree = (2 ** depth) * n * bytes_per_element * 4
    max_pop = budget_bytes // per_tree
    # Clamp to sensible range
    return min(max(int(max_pop), 256), 16384)


def run_depth(depth: int, generations: int, log_every: int = 10):
    x, y = make_data()
    population = pop_for_depth(depth, y.shape[0])
    print(f"\n{'=' * 70}")
    print(f"Depth {depth}  |  population={population}  |  generations={generations}")
    print(f"{'=' * 70}")

    t0 = time.time()
    cfg = EvolutionConfig(
        depth=depth, num_vars=1,
        population=population, generations=generations,
        elite_fraction=0.1, mutations_per_child=1,
        crossover_fraction=0.4,
        device=DEVICE, r2_target=0.9999,
        log_every=log_every,
    )
    evo = evolve(x, y, cfg, var_names=["x"])

    print(f"  Evolution R² = {evo.best_r2:+.4f}  time = {evo.total_time_s:.1f}s")
    print(f"  Formula: {evo.best_expression}")

    # Polish — many iterations because depth 8 has many constants
    polish_iters = 3000 + 500 * depth
    t1 = time.time()
    pol = polish(
        evo.best_tree, evo.best_idx, x, y,
        var_names=["x"], n_iters=polish_iters,
        lr=1e-2, device=DEVICE,
        warm_a=evo.best_a, warm_b=evo.best_b,
    )
    polish_time = time.time() - t1

    if pol.r2 > evo.best_r2:
        final_r2 = pol.r2
        final_formula = pol.formula
        constants_learned = pol.constants
    else:
        final_r2 = evo.best_r2
        final_formula = evo.best_expression
        constants_learned = None

    total_time = time.time() - t0
    print(f"  Polish R² = {pol.r2:+.4f}  (polish time = {polish_time:.1f}s, "
          f"{polish_iters} iters)")
    print(f"  -> Final R² = {final_r2:+.4f}  total = {total_time:.1f}s")
    print(f"  Final formula: {final_formula[:200]}")
    if constants_learned is not None:
        non_trivial = [c for c in constants_learned if abs(c - 1.0) > 0.01]
        print(f"  Non-trivial learned constants: {[round(c, 4) for c in non_trivial[:15]]}")

    # Free GPU memory between depth attempts
    del evo, pol
    torch.cuda.empty_cache()

    return {
        "depth": depth,
        "population": population,
        "generations_target": generations,
        "evolution_r2": evo.best_r2 if False else None,   # don't leak trees in dict
        "final_r2": final_r2,
        "total_time_s": total_time,
        "formula": final_formula,
        "constants_learned": constants_learned,
    }


def main():
    print("=" * 70)
    print("sin(x) push — GPU:5, full memory, depth escalation 5 → 8")
    print("=" * 70)
    print(f"Available GPU memory: ~{torch.cuda.get_device_properties(5).total_memory // 1024**3} GB")

    results = []
    schedule = [
        (5, 80),    # depth 5, 80 generations
        (6, 60),    # depth 6, 60 generations
        (7, 40),    # depth 7, 40 generations
        (8, 30),    # depth 8, 30 generations (expensive)
    ]

    breakthrough_depth = None
    for depth, gens in schedule:
        r = run_depth(depth, gens, log_every=max(5, gens // 10))
        results.append(r)
        if r["final_r2"] > 0.99 and breakthrough_depth is None:
            breakthrough_depth = depth
            print(f"\n*** BREAKTHROUGH at depth {depth}: R² = {r['final_r2']:.4f} ***\n")
            # Don't break — continue to higher depths to see if it keeps improving

    print("\n" + "=" * 70)
    print("FINAL SCORECARD")
    print("=" * 70)
    print(f"{'Depth':>5s}  {'Pop':>6s}  {'Final R²':>10s}  {'Time':>8s}")
    for r in results:
        flag = "✓" if r["final_r2"] > 0.99 else " "
        print(f"  {r['depth']:>3d}   {r['population']:>6d}   "
              f"{r['final_r2']:>+10.4f} {r['total_time_s']:>8.1f}s  {flag}")

    # Dump full results
    out = Path("/data2/workspace/sae-eml/results/sin_push_results.json")
    out.write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "constants_learned"}
         for r in results],
        indent=2,
    ))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
