"""GPU error decomposition and local state diagnostics on development activations only."""

import argparse

import torch

from .common import digest, previous, root, save, setup, sources
from .model import load_replacement


@torch.no_grad()
def decomposition(model, domains, init):
    d = model.dimension
    if model.architecture != "structured":
        decoder = model.decoder.weight.double()
        q = torch.linalg.qr(decoder, mode="reduced").Q
        singular = torch.linalg.svdvals(decoder)
    else:
        q = None
        singular = torch.linalg.svdvals(model.decoder.weight.double())
    totals = torch.zeros(3, device="cuda", dtype=torch.float64)
    mean = torch.zeros(d, device="cuda", dtype=torch.float64)
    second = torch.zeros(d, d, device="cuda", dtype=torch.float64)
    for data in domains.values():
        for a, b in zip(data["x"].split(512), data["y"].split(512)):
            x = (a.cuda().float() - model.xmean) / model.xstd
            y = ((b.cuda().float() - model.ymean) / model.yscale).double()
            pred = model.normalized(x).double()
            error = y - pred
            w = 1 / len(data["x"]) / len(domains)
            totals[0] += error.square().sum() * w / d
            if q is not None:
                within = error @ q
                totals[1] += within.square().sum() * w / d
                totals[2] += (error - within @ q.T).square().sum() * w / d
            mean += pred.sum(0) * w
            second.addmm_(pred.T, pred, alpha=w)
    eig = torch.linalg.eigvalsh(second - mean[:, None] * mean[None, :]).clamp_min(0)
    result = {
        "raw_mse": float(totals[0]),
        "within_decoder_mse": float(totals[1]) if q is not None else None,
        "outside_decoder_mse": float(totals[2]) if q is not None else None,
        "decomposition_closure_error": float((totals[0] - totals[1] - totals[2]).abs())
        if q is not None
        else None,
        "decoder_min_singular": float(singular.min()),
        "decoder_max_singular": float(singular.max()),
        "output_covariance_rank_relative_1e_minus_6": int((eig > eig.max() * 1e-6).sum()),
        "output_covariance_top_512_energy_fraction": float(eig[-512:].sum() / eig.sum()),
        "output_covariance_eigenvalues": eig.flip(0).tolist(),
    }
    if model.architecture == "shortcut":
        residual = init["residual_covariance"].cuda()
        result["training_best_affine_outside_learned_decoder_mse"] = float(
            (torch.trace(residual) - torch.trace(q.T @ residual @ q)) / d
        )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    a = p.parse_args()
    setup(91)
    out = root()
    path = out / "training" / f"{a.method}.pt"
    m = load_replacement(path, dtype=torch.float32)
    init = torch.load(out / "training/initialization.pt", weights_only=True)
    collection = previous(out) / "collection"
    results = {}
    for split in ["train", "selection"]:
        domains = {
            k: torch.load(collection / f"{k}-{split}.pt", weights_only=True)
            for k in ["arithmetic", "language"]
        }
        results[split] = decomposition(m, domains, init)
    # Random directional derivatives at observed states are sensitivity diagnostics,
    # not a proof of injectivity or semantic information preservation.
    x = domains["language"]["x"][:8].cuda().float()
    x = (x - m.xmean) / m.xstd
    ratios = []
    for _ in range(32):
        direction = torch.randn_like(x)
        direction = direction / direction.norm(dim=-1, keepdim=True)
        _, response = torch.func.jvp(m.normalized, (x,), (direction,))
        ratios += response.norm(dim=-1).tolist()
    results["local_random_direction_stretch"] = {
        "minimum": min(ratios),
        "maximum": max(ratios),
        "mean": sum(ratios) / len(ratios),
        "directions": len(ratios),
    }
    if m.architecture == "structured":
        ranks = []
        for stage in m.stages:
            for label, projection in [("arguments", stage.arguments), ("readout", stage.readout)]:
                for matrix in projection.matrices().double():
                    s = torch.linalg.svdvals(matrix)
                    ranks.append(
                        {
                            "map": label,
                            "rank_relative_1e_minus_6": int((s > s.max() * 1e-6).sum()),
                            "min_singular": float(s.min()),
                            "max_singular": float(s.max()),
                        }
                    )
        results["structured_projection_spectra"] = ranks
    results.update(
        method=a.method,
        checkpoint_sha256=digest(path),
        sources=sources("diagnose.py", "model.py"),
        note="Training and previously inspected selection data only. Covariance rank and random directional derivatives do not certify preserved information or an arithmetic mechanism.",
    )
    save(out / "diagnostics" / f"{a.method}.json", results)
    print("DIAGNOSIS COMPLETE", a.method, results["train"]["raw_mse"], flush=True)


if __name__ == "__main__":
    main()
