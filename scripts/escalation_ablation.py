"""
Depth escalation with warm-start seeding.

Hypothesis: depth N+1 evolution fails from random init because the
~(V+2)^(4·2^N) search space is too sparse. Seeding with the best
depth-N tree places a known-good subtree in the left half; evolution
only needs to discover the wrapper layer.

Run:
  d=5 from scratch           → depth5
  d=6 from scratch           → depth6_fresh
  d=6 seeded from depth5      → depth6_seeded
  d=7 seeded from depth6      → depth7_seeded

All with range_penalty=0.1.
"""

import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emltorch.evolution import EvolutionConfig, evolve
from emltorch.polish import polish

DEVICE = "cuda:5"


def run_evo(depth, population, generations, x, y, seed_tree=None, seed_idx=None):
    cfg = EvolutionConfig(
        depth=depth, num_vars=1,
        population=population, generations=generations,
        elite_fraction=0.1, mutations_per_child=1,
        crossover_fraction=0.4,
        device=DEVICE, r2_target=0.9999,
        log_every=10,
        range_penalty=0.1,
    )
    return evolve(
        x, y, cfg, var_names=["x"],
        seed_tree=seed_tree, seed_idx=seed_idx,
        seed_fraction=0.25,
    )


def report(label, evo, x, y):
    t0 = time.time()
    pol = polish(
        evo.best_tree, evo.best_idx, x, y,
        var_names=["x"], n_iters=3000,
        lr=1e-2, device=DEVICE,
        warm_a=evo.best_a, warm_b=evo.best_b,
    )
    print(
        f"\n[{label}]  evo R²={evo.best_r2:+.4f}  polish R²={pol.r2:+.4f}  "
        f"|b|={abs(pol.b):.3f}  (polish {time.time()-t0:.0f}s)",
        flush=True,
    )
    print(f"  formula: {pol.formula[:220]}", flush=True)


def main():
    x = torch.linspace(-math.pi, math.pi, 1024).unsqueeze(0)
    y = torch.sin(x.squeeze(0))
    print("=" * 70)
    print("Depth escalation with warm-start seeding (rp=0.1)")
    print("=" * 70)

    # Step 1: depth 5 from scratch
    print("\n--- Step 1: depth 5 from scratch ---", flush=True)
    evo5 = run_evo(5, 16384, 60, x, y)
    report("depth5", evo5, x, y)

    # Step 2: depth 6 from scratch
    print("\n--- Step 2: depth 6 from scratch ---", flush=True)
    evo6_fresh = run_evo(6, 16384, 60, x, y)
    report("depth6_fresh", evo6_fresh, x, y)

    # Step 3: depth 6 seeded from depth 5
    print("\n--- Step 3: depth 6 seeded from depth 5 ---", flush=True)
    evo6_seeded = run_evo(6, 16384, 60, x, y,
                          seed_tree=evo5.best_tree, seed_idx=evo5.best_idx)
    report("depth6_seeded", evo6_seeded, x, y)

    # Step 4: depth 7 seeded from best depth 6
    best6 = evo6_seeded if evo6_seeded.best_r2 > evo6_fresh.best_r2 else evo6_fresh
    print("\n--- Step 4: depth 7 seeded from depth 6 ---", flush=True)
    evo7_seeded = run_evo(7, 8192, 50, x, y,
                          seed_tree=best6.best_tree, seed_idx=best6.best_idx)
    report("depth7_seeded", evo7_seeded, x, y)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  depth5          evo R² = {evo5.best_r2:+.4f}")
    print(f"  depth6 (fresh)  evo R² = {evo6_fresh.best_r2:+.4f}")
    print(f"  depth6 (seeded) evo R² = {evo6_seeded.best_r2:+.4f}")
    print(f"  depth7 (seeded) evo R² = {evo7_seeded.best_r2:+.4f}")


if __name__ == "__main__":
    main()
