"""Measure trained deployment roundoff on selection activations without refitting."""

import copy

import torch

from gemma_mechanisms.student import contribution

from .common import digest, previous, root, save, setup, sources
from .model import Deployed, load_replacement


def main():
    setup(95)
    out = root()
    collection = previous(out) / "collection"
    norm = torch.load(collection / "native-postnorm.pt", weights_only=True)
    norm["weight"] = norm["weight"].cuda()
    init = torch.load(out / "training/initialization.pt", weights_only=True)
    scale = init["cscale"].cuda()
    domains = {
        k: torch.load(collection / f"{k}-selection.pt", weights_only=True)
        for k in ["arithmetic", "language"]
    }
    results = {}
    for path in sorted((out / "training").glob("*-n12000.pt")):
        m = load_replacement(path, dtype=torch.float32)
        folded = Deployed(m)
        bf = copy.deepcopy(folded).bfloat16()
        unfolded = copy.deepcopy(m).bfloat16()
        sums = torch.zeros(4, 3, device="cuda", dtype=torch.float64)
        with torch.inference_mode():
            for data in domains.values():
                for a, b, c in zip(
                    data["x"].split(512), data["y"].split(512), data["contribution"].split(512)
                ):
                    x = a.cuda().float()
                    y = b.cuda().float()
                    c = c.cuda().float()
                    reference = m(x)
                    predictions = [
                        reference,
                        folded(x),
                        bf(x.bfloat16()).float(),
                        unfolded(x.bfloat16()).float(),
                    ]
                    for i, pred in enumerate(predictions):
                        values = torch.stack(
                            [
                                ((pred - y) / m.yscale).square().mean(),
                                ((contribution(pred, norm["weight"], norm["eps"]) - c) / scale)
                                .square()
                                .mean(),
                                ((pred - reference) / m.yscale).square().mean(),
                            ]
                        )
                        sums[i] += values.double() * len(x) / len(data["x"]) / len(domains)
        values = {
            key: dict(zip(["raw_mse", "contribution_mse", "prediction_mse_vs_fp32"], row))
            for key, row in zip(
                ["training_fp32", "folded_fp32", "folded_bf16", "unfolded_bf16"], sums.tolist()
            )
        }
        assert values["folded_fp32"]["prediction_mse_vs_fp32"] < 1e-10
        results[path.stem] = {"metrics": values, "checkpoint_sha256": digest(path)}
        print("DEPLOYMENT PRECISION", path.stem, values, flush=True)
        del m, folded, bf, unfolded
    save(
        out / "precision-audit.json",
        {
            "methods": results,
            "sources": sources("precision.py", "model.py", "../gemma_mechanisms/student.py"),
            "scope": "Previously inspected selection activations, all 32768 positions. No tuning or fresh confirmation.",
        },
    )


if __name__ == "__main__":
    main()
