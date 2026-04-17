"""
Range-aware fitness ablation — depth 7 sin(x).

Depth 7 evolution previously returned trees with |b|≈0.05 (tree flat,
affine did all the work). Test that range_penalty fixes this.

Variants:
  (a) baseline  — range_penalty=0.0
  (b) rp=0.1    — mild
  (c) rp=1.0    — strong

Reports: evolution best_r2, warm_b, polish R², polish |b|.
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


def run(label, range_penalty, x, y):
    t0 = time.time()
    cfg = EvolutionConfig(
        depth=7, num_vars=1,
        population=8192, generations=40,
        elite_fraction=0.1, mutations_per_child=1,
        crossover_fraction=0.4,
        device=DEVICE, r2_target=0.9999,
        log_every=10,
        range_penalty=range_penalty,
    )
    evo = evolve(x, y, cfg, var_names=["x"])
    evo_t = time.time() - t0

    t1 = time.time()
    pol = polish(
        evo.best_tree, evo.best_idx, x, y,
        var_names=["x"], n_iters=3500,
        lr=1e-2, device=DEVICE,
        warm_a=evo.best_a, warm_b=evo.best_b,
    )
    pol_t = time.time() - t1

    print(
        f"\n[{label}]  evo R²={evo.best_r2:+.4f}  warm_b={evo.best_b:+.4f}  "
        f"→ polish R²={pol.r2:+.4f}  |b|={abs(pol.b):.3f}  "
        f"(evo {evo_t:.0f}s + polish {pol_t:.0f}s)",
        flush=True,
    )
    print(f"  formula: {pol.formula[:220]}", flush=True)


def main():
    x = torch.linspace(-math.pi, math.pi, 1024).unsqueeze(0)
    y = torch.sin(x.squeeze(0))
    print("=" * 70)
    print("Depth 7 sin(x) — range_penalty ablation")
    print("=" * 70)

    run("rp=0.0",  0.0, x, y)
    run("rp=0.1",  0.1, x, y)
    run("rp=1.0",  1.0, x, y)


if __name__ == "__main__":
    main()
