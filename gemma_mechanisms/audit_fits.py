"""Replay saved objectives and measure the output-rank limitation on CUDA."""

import argparse
import json

import torch

from .runtime import HERE, digest, root, save, setup
from .student import load_student
from .train import evaluate, moments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    setup(19)
    out = root(args.output)
    collection = out / "collection"
    raw = {
        split: {
            domain: torch.load(collection / f"{domain}-{split}.pt", weights_only=True)
            for domain in ["arithmetic", "language"]
        }
        for split in ["train", "selection"]
    }
    _, cscale = moments(raw["train"])
    norm = torch.load(collection / "native-postnorm.pt", weights_only=True)
    norm["weight"] = norm["weight"].cuda()
    initialization = torch.load(out / "training/initialization.pt", weights_only=True)
    eigenvalues = initialization["eigenvalues"]
    directory = out / "fit-audits"
    directory.mkdir(exist_ok=True)
    for path in sorted((out / "training").glob("*-b*-d*-s*.json")):
        record = json.loads(path.read_text())
        if record["status"] != "complete":
            continue
        output_path = directory / path.name
        if output_path.exists():
            old = json.loads(output_path.read_text())
            assert old["fit_record_sha256"] == digest(path)
            continue
        checkpoint = path.with_suffix(".pt")
        assert digest(checkpoint) == record["checkpoint_sha256"]
        model = load_student(checkpoint, dtype=torch.float32)
        validation = evaluate(model, raw["selection"], norm, cscale)
        objective = sum(r["objective"] for r in validation.values()) / len(validation)
        assert objective == record["selection_objective"], (
            path.name,
            objective,
            record["selection_objective"],
        )
        train = evaluate(model, raw["train"], norm, cscale)
        actual = sum(r["raw_mse"] for r in train.values()) / len(train)
        bound = float(eigenvalues[model.width :].sum() / eigenvalues.sum())
        assert actual + 1e-6 >= bound
        # Deploying BF16 weights/statistics is a separate numerical change from fitting in FP32.
        bf16 = load_student(checkpoint, dtype=torch.bfloat16)
        conversion = {}
        with torch.no_grad():
            for name, data in raw["selection"].items():
                sums = torch.zeros(2, device="cuda", dtype=torch.float64)
                for block in data["x"].split(512):
                    x = block.cuda().float()
                    native = model(x)
                    rounded = bf16(x.bfloat16()).float()
                    sums += torch.stack(
                        [(rounded - native).square().sum(), native.square().sum()]
                    ).double()
                conversion[name] = float((sums[0] / sums[1]).sqrt())
        save(
            output_path,
            {
                "name": record["name"],
                "fit_record_sha256": digest(path),
                "checkpoint_sha256": digest(checkpoint),
                "objective_replay_exact": True,
                "selection": validation,
                "train": train,
                "output_rank": model.width,
                "training_raw_mse_lower_bound": bound,
                "training_raw_mse": actual,
                "excess_over_rank_bound": actual - bound,
                "bf16_vs_fp32_output_nrmse": conversion,
                "sources": {
                    p: digest(HERE / p) for p in ["audit_fits.py", "train.py", "student.py"]
                },
                "note": "The covariance tail bounds training MSE for any affine output decoder of this rank, regardless of nonlinear depth. It is not a general limit on EML.",
            },
        )
        print("AUDIT FIT", record["name"], "raw MSE", actual, "rank bound", bound, flush=True)
        del model, bf16
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
