"""Train-only gradient subspaces and validation audit of omitted local sensitivity."""

import os
import time

import torch

from data import ROOT, save
from model_io import load_mlp, setup


def gradients(layer, h, d):
    result = []
    for chunk in h.split(128):
        x = chunk.detach().clone().requires_grad_(True)
        coefficient = layer(x) @ d
        gradient = torch.autograd.grad(coefficient.sum(), x)[0]
        result.append(gradient.detach())
    return torch.cat(result)


def main():
    setup()
    save(
        ROOT / "active-job.json",
        {
            "pid": os.getpid(),
            "start_epoch": time.time(),
            "gpu": torch.cuda.get_device_name(),
        },
    )
    for op in ["add", "multiply", "divide"]:
        out = ROOT / op
        c = torch.load(out / "component.pt", weights_only=True)
        layer = load_mlp(c["layer"])
        d = c["direction"].cuda()
        inputs = {
            k: torch.load(out / f"inputs-{k}.pt", weights_only=True)["input"].cuda()
            for k in ["train", "validation"]
        }
        samples = {}
        torch.manual_seed(20260911)
        for split, h in inputs.items():
            ids = torch.randperm(len(h), device="cuda")[: 512 if split == "train" else 128]
            samples[split] = torch.cat(
                [h[ids, 1] * (1 - a) + h[ids, 0] * a for a in [0.0, 0.25, 0.5, 0.75, 1.0]]
            )
        gs = {k: gradients(layer, h, d) for k, h in samples.items()}
        # Uncentered gradient covariance: the mean gradient is part of the active subspace.
        g = gs["train"].double()
        covariance = g.T @ g / len(g)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        basis = eigenvectors.flip(1)[:, :64].float()
        assert torch.allclose(basis.T @ basis, torch.eye(64, device="cuda"), atol=1e-4)
        pls = torch.linalg.qr(c["encoder"].cuda().double()).Q.float()
        captures = {}
        for name, q in [
            ("pls32", pls),
            ("active16", basis[:, :16]),
            ("active32", basis[:, :32]),
            ("active64", basis),
        ]:
            captures[name] = {
                k: float((v @ q).square().sum() / v.square().sum()) for k, v in gs.items()
            }
        # A local perturbation invisible to the fixed PLS coordinates.
        h = samples["validation"][:32]
        grad = gs["validation"][:32]
        residual = grad - (grad @ pls) @ pls.T
        direction = residual / residual.norm(dim=1, keepdim=True).clamp_min(1e-12)
        eps = 0.01
        with torch.no_grad():
            plus, minus = h + eps * direction, h - eps * direction
            shift = ((plus - minus) @ c["encoder"].cuda()) / c["zstd"].cuda()
            delta = layer(plus) @ d - layer(minus) @ d
        audit = {
            "feature_gradient_energy_fraction": captures,
            "orthogonal_perturbation": {
                "epsilon": eps,
                "cases": len(h),
                "max_standardized_feature_change": float(shift.abs().max()),
                "true_coefficient_response_rms": float(delta.square().mean().sqrt()),
                "linearized_response_rms": float(
                    (2 * eps * residual.norm(dim=1)).square().mean().sqrt()
                ),
            },
            "interpretation": "Any predictor depending only on linear coordinates has its input gradient in their span. Omitted gradient energy is an irreducible local derivative error for that representation, not a bound on natural arithmetic accuracy.",
            "subspace_fit": "Training gradients only; validation gradients only audit the fixed subspaces.",
            "source": "https://arxiv.org/abs/1408.0545",
            "dtype": "float32 gradients, float64 eigendecomposition",
        }
        save(out / "gradient-audit.json", audit)
        original = torch.load(out / "features.pt", weights_only=True)
        features = {}
        zs = {}
        with torch.no_grad():
            for split, h in inputs.items():
                mixed = torch.cat(
                    [h[:, 1] * (1 - a) + h[:, 0] * a for a in [0.0, 0.25, 0.5, 0.75, 1.0]]
                )
                zs[split] = (mixed - c["input_mean"].cuda()) @ basis[:, :32]
            zm, zstd = zs["train"].mean(0), zs["train"].std(0).clamp_min(1e-6)
            for split, z in zs.items():
                features[split] = {**original[split], "x": ((z - zm) / zstd).cpu()}
        torch.save(features, out / "features-active.pt")
        torch.save(
            {
                **c,
                "encoder": basis[:, :32].cpu(),
                "zmean": zm.cpu(),
                "zstd": zstd.cpu(),
                "feature_method": "train-gradient active subspace",
            },
            out / "component-active.pt",
        )
        torch.save(
            {"basis64": basis.cpu(), "eigenvalues": eigenvalues.flip(0).cpu()},
            out / "active-subspace.pt",
        )
        print("ACTIVE FEATURES", op, captures, flush=True)
        del layer, inputs, gs, g, covariance, eigenvectors, features, zs
        torch.cuda.empty_cache()
    save(
        ROOT / "active-completed.json",
        {"completed_epoch": time.time(), "operations": 3},
    )


if __name__ == "__main__":
    main()
