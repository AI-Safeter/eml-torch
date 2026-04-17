"""
Polish ablation: constant regularizer + b-scale constraint.

Runs depth-6 sin(x) recovery with strong evolution once, then polishes
four ways:
  (a) baseline              — no regs
  (b) const_reg=1e-2        — snap weak constants to 1.0
  (c) min_b_abs=0.5         — force outer affine scale |b|≥0.5 so the
                              tree must carry the signal's range
  (d) both                  — interpretability (const snap) + structure
                              (b constraint) together

For each: R², |b|, # constants snapped, formula. The goal is to
demonstrate that the EML tree itself is shaping sin(x), not the
outer affine wrapper.
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


def run(label, evo_res, x, y, const_reg=0.0, min_b_abs=0.0, range_reg=0.0):
    t0 = time.time()
    pol = polish(
        evo_res.best_tree, evo_res.best_idx, x, y,
        var_names=["x"], n_iters=3000,
        lr=1e-2, device=DEVICE,
        warm_a=evo_res.best_a, warm_b=evo_res.best_b,
        const_reg=const_reg,
        min_b_abs=min_b_abs,
        range_reg=range_reg,
    )
    dt = time.time() - t0
    near_one = sum(1 for c in pol.constants if abs(c - 1.0) < 0.01)
    print(
        f"\n[{label}]  R²={pol.r2:+.4f}  |b|={abs(pol.b):.3f}  a={pol.a:+.3f}  "
        f"snapped={near_one}/{len(pol.constants)}  time={dt:.0f}s",
        flush=True,
    )
    print(f"  formula: {pol.formula[:250]}", flush=True)
    return pol


def main():
    x = torch.linspace(-math.pi, math.pi, 1024).unsqueeze(0)
    y = torch.sin(x.squeeze(0))

    print("Strong evolution: depth=6, pop=16384, gen=60", flush=True)
    cfg = EvolutionConfig(
        depth=6, num_vars=1,
        population=16384, generations=60,
        elite_fraction=0.1, mutations_per_child=1,
        crossover_fraction=0.4,
        device=DEVICE, r2_target=0.9999,
        log_every=15,
    )
    evo = evolve(x, y, cfg, var_names=["x"])
    print(f"\nEvolution R² = {evo.best_r2:+.4f}  ({evo.total_time_s:.1f}s)", flush=True)
    print(f"  warm_a={evo.best_a:+.3f}  warm_b={evo.best_b:+.4f}", flush=True)

    run("baseline",       evo, x, y)
    run("const=1e-2",     evo, x, y, const_reg=1e-2)
    run("|b|>=0.5",       evo, x, y, min_b_abs=0.5, range_reg=1.0)
    run("both",           evo, x, y, const_reg=1e-2, min_b_abs=0.5, range_reg=1.0)


if __name__ == "__main__":
    main()
