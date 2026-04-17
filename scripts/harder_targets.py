"""
Test EML escalation recipe on harder targets.

Targets:
  (a) gaussian:   y = exp(-x^2)                    x in [-3, 3]
  (b) xsin:       y = x * sin(x)                   x in [-pi, pi]
  (c) damped:     y = exp(-x^2/4) * sin(2*x)       x in [-pi, pi]

For each, run depth-escalation recipe (d4 → d5 seeded → d6 seeded → d7 seeded)
and report polish R² + |b| at each step. A healthy recipe shows monotone
improvement (or early plateau at "perfect" R²>0.999) with |b| staying near 1.
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


TARGETS = {
    "gaussian": (
        lambda x: torch.exp(-x.pow(2)),
        (-3.0, 3.0),
    ),
    "xsin": (
        lambda x: x * torch.sin(x),
        (-math.pi, math.pi),
    ),
    "damped": (
        lambda x: torch.exp(-x.pow(2) / 4) * torch.sin(2 * x),
        (-math.pi, math.pi),
    ),
}


def run_evo(depth, population, generations, x, y, seed_tree=None, seed_idx=None,
            range_penalty=0.1):
    cfg = EvolutionConfig(
        depth=depth, num_vars=1,
        population=population, generations=generations,
        elite_fraction=0.1, mutations_per_child=1,
        crossover_fraction=0.4,
        device=DEVICE, r2_target=0.9999,
        log_every=20,
        range_penalty=range_penalty,
    )
    return evolve(
        x, y, cfg, var_names=["x"],
        seed_tree=seed_tree, seed_idx=seed_idx,
        seed_fraction=0.25,
    )


def polish_report(label, evo, x, y, n_iters=3000):
    t0 = time.time()
    pol = polish(
        evo.best_tree, evo.best_idx, x, y,
        var_names=["x"], n_iters=n_iters,
        lr=1e-2, device=DEVICE,
        warm_a=evo.best_a, warm_b=evo.best_b,
    )
    dt = time.time() - t0
    print(
        f"  [{label}]  evo R²={evo.best_r2:+.4f}  "
        f"polish R²={pol.r2:+.4f}  |b|={abs(pol.b):.3f}  "
        f"(polish {dt:.0f}s)",
        flush=True,
    )
    return pol


def run_target(name, fn, x_range):
    low, high = x_range
    x = torch.linspace(low, high, 1024).unsqueeze(0)
    y = fn(x.squeeze(0))
    print("\n" + "=" * 70)
    print(f"TARGET: {name}    domain [{low}, {high}]")
    print("=" * 70, flush=True)

    # Depth 4 fresh
    print(f"\n--- {name}: depth 4 fresh ---", flush=True)
    evo4 = run_evo(4, 16384, 40, x, y)
    polish_report(f"{name}/d4", evo4, x, y)

    # Depth 5 seeded
    print(f"\n--- {name}: depth 5 seeded from d4 ---", flush=True)
    evo5 = run_evo(5, 16384, 50, x, y,
                   seed_tree=evo4.best_tree, seed_idx=evo4.best_idx)
    polish_report(f"{name}/d5", evo5, x, y)

    # Depth 6 seeded
    print(f"\n--- {name}: depth 6 seeded from d5 ---", flush=True)
    evo6 = run_evo(6, 16384, 60, x, y,
                   seed_tree=evo5.best_tree, seed_idx=evo5.best_idx)
    pol6 = polish_report(f"{name}/d6", evo6, x, y)

    # Depth 7 seeded
    print(f"\n--- {name}: depth 7 seeded from d6 ---", flush=True)
    evo7 = run_evo(7, 8192, 50, x, y,
                   seed_tree=evo6.best_tree, seed_idx=evo6.best_idx)
    pol7 = polish_report(f"{name}/d7", evo7, x, y, n_iters=3500)

    print(f"  final formula ({name}): {pol7.formula[:220]}", flush=True)

    return {
        "target": name,
        "d4": {"evo_r2": evo4.best_r2},
        "d5": {"evo_r2": evo5.best_r2},
        "d6": {"evo_r2": evo6.best_r2, "polish_r2": pol6.r2, "b": pol6.b},
        "d7": {"evo_r2": evo7.best_r2, "polish_r2": pol7.r2, "b": pol7.b},
    }


def main():
    results = []
    for name, (fn, rng) in TARGETS.items():
        results.append(run_target(name, fn, rng))

    print("\n" + "=" * 70)
    print("FINAL SUMMARY — EML escalation recipe on harder targets")
    print("=" * 70)
    print(f"{'target':<10} {'d4 evo':>8} {'d5 evo':>8} {'d6 evo':>8} "
          f"{'d6 pol':>8} {'d6 |b|':>7} {'d7 pol':>8} {'d7 |b|':>7}")
    for r in results:
        print(f"{r['target']:<10} "
              f"{r['d4']['evo_r2']:>+8.4f} "
              f"{r['d5']['evo_r2']:>+8.4f} "
              f"{r['d6']['evo_r2']:>+8.4f} "
              f"{r['d6']['polish_r2']:>+8.4f} "
              f"{abs(r['d6']['b']):>7.3f} "
              f"{r['d7']['polish_r2']:>+8.4f} "
              f"{abs(r['d7']['b']):>7.3f}")


if __name__ == "__main__":
    main()
