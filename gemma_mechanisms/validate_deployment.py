"""Validate affine folding on actual captured CUDA activations for every completed fit."""

import json

import torch

from .deploy import load_deployed
from .runtime import HERE, accounting, digest, root, save, setup
from .student import load_student


def main():
    setup(73)
    out = root()
    data = {
        d: torch.load(out / "collection" / f"{d}-selection.pt", weights_only=True)["x"]
        for d in ["arithmetic", "language"]
    }
    records = []
    for path in sorted((out / "training").glob("*-b*-d*-s*.json")):
        fit = json.loads(path.read_text())
        if fit["status"] != "complete":
            continue
        checkpoint = path.with_suffix(".pt")
        trained = load_student(checkpoint, dtype=torch.float32)
        folded = load_deployed(checkpoint, dtype=torch.float32)
        bf16 = load_deployed(checkpoint)
        metrics = {}
        with torch.inference_mode():
            for domain, inputs in data.items():
                sums = torch.zeros(3, device="cuda", dtype=torch.float64)
                for block in inputs.split(512):
                    x = block.cuda().float()
                    reference, candidate, rounded = (
                        trained(x),
                        folded(x),
                        bf16(x.bfloat16()).float(),
                    )
                    assert torch.isfinite(rounded).all()
                    sums += torch.stack(
                        [
                            (candidate - reference).square().sum(),
                            (rounded - reference).square().sum(),
                            reference.square().sum(),
                        ]
                    ).double()
                error = float((sums[0] / sums[2]).sqrt())
                assert error < 1e-4, (fit["name"], domain, error)
                metrics[domain] = {
                    "fp32_folding_nrmse": error,
                    "bf16_deployment_nrmse": float((sums[1] / sums[2]).sqrt()),
                }
        records.append(
            {
                "name": fit["name"],
                "checkpoint_sha256": digest(checkpoint),
                "metrics": metrics,
                "trained": accounting(trained),
                "deployed": accounting(bf16),
                "linear_collapsed": bf16.collapsed_linear,
            }
        )
        print("DEPLOYMENT CHECK", fit["name"], metrics, flush=True)
        del trained, folded, bf16
        torch.cuda.empty_cache()
    save(
        out / "deployment-validation.json",
        {
            "records": records,
            "sources": {
                p: digest(HERE / p) for p in ["deploy.py", "validate_deployment.py", "student.py"]
            },
            "scope": "Selection activations; algebraic and precision checks, not model-quality acceptance",
        },
    )


if __name__ == "__main__":
    main()
