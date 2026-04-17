"""Verify polish safeguard: sin(x) d6 seeded → d7 seeded. rp=0.1.

Before fix: d7 polish R² could regress below d6.
After fix: d7 polish R² must be >= (polish initial state at warm_a,warm_b).
"""
import math, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emltorch.evolution import EvolutionConfig, evolve
from emltorch.polish import polish

DEVICE = "cuda:5"

def run_evo(depth, pop, gens, x, y, seed_tree=None, seed_idx=None):
    cfg = EvolutionConfig(
        depth=depth, num_vars=1,
        population=pop, generations=gens,
        elite_fraction=0.1, mutations_per_child=1,
        crossover_fraction=0.4,
        device=DEVICE, r2_target=0.9999,
        log_every=20,
        range_penalty=0.1,
    )
    return evolve(x, y, cfg, var_names=["x"],
                  seed_tree=seed_tree, seed_idx=seed_idx, seed_fraction=0.25)

def main():
    x = torch.linspace(-math.pi, math.pi, 1024).unsqueeze(0)
    y = torch.sin(x.squeeze(0))

    evo5 = run_evo(5, 16384, 40, x, y)
    print(f"[d5] evo R²={evo5.best_r2:+.4f}", flush=True)
    evo6 = run_evo(6, 16384, 50, x, y, seed_tree=evo5.best_tree, seed_idx=evo5.best_idx)
    print(f"[d6] evo R²={evo6.best_r2:+.4f}", flush=True)
    evo7 = run_evo(7, 8192, 40, x, y, seed_tree=evo6.best_tree, seed_idx=evo6.best_idx)
    print(f"[d7] evo R²={evo7.best_r2:+.4f}  warm_a={evo7.best_a:+.3f} warm_b={evo7.best_b:+.4f}", flush=True)

    # Compute raw initial MSE for comparison
    t0 = time.time()
    pol = polish(
        evo7.best_tree, evo7.best_idx, x, y,
        var_names=["x"], n_iters=3500, lr=1e-2, device=DEVICE,
        warm_a=evo7.best_a, warm_b=evo7.best_b,
    )
    print(f"[d7 polish] R²={pol.r2:+.4f}  |b|={abs(pol.b):.3f}  a={pol.a:+.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"  evo→polish delta: {pol.r2 - evo7.best_r2:+.4f}", flush=True)
    if pol.r2 < evo7.best_r2 - 0.01:
        print("FAIL: polish regressed more than 1% beyond evo R²", flush=True)
    else:
        print("PASS: polish did not regress", flush=True)

if __name__ == "__main__":
    main()
