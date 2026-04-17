"""Discover the symbolic refusal circuit at a mid-layer residual stream.

Simplified port of `sae-eml/scripts/hybrid_refusal.py` using the new
emltorch public API.

Workflow:
    1. Load cached harmful / benign pooled activations at a given layer.
    2. Define the "refusal direction" = mean(harmful) - mean(benign).
    3. PCA-reduce the residual stream to K components.
    4. Fit an EML formula predicting (activation projected on refusal dir)
       from the PCA components.

Expected result: R2 > 0.95 and a short symbolic expression that the
user can inspect to understand how the residual stream encodes refusal.

Run:
    python3 examples/refusal_circuit.py --layer 22 --n-components 4 --depth 3
"""

import argparse
import sys
from pathlib import Path

import torch

# Allow running as `python3 examples/refusal_circuit.py` without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import emltorch as eml  # noqa: E402


CACHE_DIR = Path("/data2/workspace/sae-eml/cache/contrastive")
DEVICE = "cuda:7" if torch.cuda.is_available() else "cpu"


def load_contrastive(layer: int):
    from safetensors.torch import load_file
    h = load_file(str(CACHE_DIR / "harmful.safetensors"))
    b = load_file(str(CACHE_DIR / "benign.safetensors"))
    return h[f"layer_{layer}"].float(), b[f"layer_{layer}"].float()


def pca_project(activations: torch.Tensor, n_components: int):
    """Center, SVD, project. Returns (features, variance_fraction)."""
    centered = activations - activations.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    components = Vh[:n_components]                    # (K, D)
    projections = centered @ components.T             # (N, K)
    pc_std = projections.std(0).clamp(min=1e-8)
    pc_norm = projections / pc_std
    var_fraction = (S[:n_components].pow(2) / S.pow(2).sum()).sum().item()
    return pc_norm, components, var_fraction


def refusal_target(harmful: torch.Tensor, benign: torch.Tensor) -> torch.Tensor:
    """Refusal score = projection of each activation onto (harmful - benign) direction."""
    direction = harmful.mean(0) - benign.mean(0)
    direction = direction / direction.norm()
    all_acts = torch.cat([harmful, benign], dim=0)
    centered = all_acts - all_acts.mean(0, keepdim=True)
    refusal = centered @ direction
    tgt_std = refusal.std().clamp(min=1e-8)
    return (refusal - refusal.mean()) / tgt_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading layer {args.layer} activations from {CACHE_DIR}")
    harmful, benign = load_contrastive(args.layer)
    print(f"  harmful: {tuple(harmful.shape)}  benign: {tuple(benign.shape)}")

    target = refusal_target(harmful, benign)
    all_acts = torch.cat([harmful, benign], dim=0)
    feats, _, var_frac = pca_project(all_acts, args.n_components)

    print(f"PCA: {args.n_components} components capture "
          f"{var_frac:.1%} of the variance")
    print(f"Fitting EML (depth={args.depth}) on (N={feats.shape[0]}, "
          f"V={feats.shape[1]}) -> scalar refusal score")

    # emltorch.fit wants x in (V, N) for multi-variable, y in (N,).
    x = feats.T.contiguous()
    y = target

    result = eml.fit(x, y, depth=args.depth, device=DEVICE)

    print()
    print("=" * 64)
    print(f"Layer {args.layer}  |  depth {args.depth}  |  "
          f"{args.n_components} PCA components")
    print("=" * 64)
    print(f"R2         = {result.r2:+.4f}")
    print(f"MSE        = {result.mse:.3e}")
    print(f"Strategy   = {result.strategy}")
    print(f"Time       = {result.time_s:.2f}s")
    print(f"a (bias)   = {result.a:+.4f}")
    print(f"b (scale)  = {result.b:+.4f}")
    print()
    print(f"Discovered formula:")
    print(f"  refusal_score ~= {result.expression}")


if __name__ == "__main__":
    main()
