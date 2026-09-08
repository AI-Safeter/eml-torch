"""Inspect the fitted EML argument ranges without changing heads or selection."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from runtime import HERE as study
from runtime import RUNS
from scalar_checkpoint_audit import operator_paths


def main():
    from heads import Head, parameter_count, safe_eml
    from model_io import setup

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", choices=["qwen17b", "qwen4b", "gemma"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Use a new diagnostic output file"
    setup()
    models = args.models
    report = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "cells": {},
        "scope": "Post-fit training/validation argument ranges; no training trajectory or held-out test diagnostic.",
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__).resolve(),
                study / "src/heads.py",
                study / "src/train_heads.py",
                *operator_paths(safe_eml),
            ]
        },
    }
    for model in models:
        expected = (
            json.loads((study / "gemma/model.json").read_text())["torch"]
            if model == "gemma"
            else "2.5.0a0+e000cf0ad9.nv24.10"
        )
        assert torch.__version__ == expected
        for op in ["add", "multiply", "divide"]:
            out = RUNS / model / op
            records = []
            hashes = {}
            for directory in ["heads", "heads-active-r32-g0", "heads-active-r32-g0.1"]:
                path = out / directory / "candidates.json"
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                candidates = json.loads(path.read_text())
                data_path = out / ("features.pt" if directory == "heads" else "features-active.pt")
                hashes[str(data_path)] = hashlib.sha256(data_path.read_bytes()).hexdigest()
                data = torch.load(data_path, weights_only=True)
                for row in candidates:
                    record = {
                        "method": directory,
                        "kind": row["kind"],
                        "seed": row["seed"],
                        "bad_steps": row["bad_steps"],
                        "status": row["status"],
                        "parameters": row["parameters"],
                    }
                    assert row["status"] == "complete"
                    if row["kind"] == "eml_square":
                        head = Head(row["kind"], 32, 32).cuda().eval()
                        checkpoint = out / directory / (row["name"] + ".pt")
                        hashes[str(checkpoint)] = hashlib.sha256(
                            checkpoint.read_bytes()
                        ).hexdigest()
                        head.load_state_dict(torch.load(checkpoint, weights_only=True))
                        assert parameter_count(head) == 2177
                        record["argument_ranges"] = {}
                        with torch.inference_mode():
                            for split in ["train", "validation"]:
                                x = data[split]["x"].cuda()
                                left = head.left(x)
                                right = 1 + head.right(x).square()
                                assert torch.isfinite(left).all() and torch.isfinite(right).all()
                                record["argument_ranges"][split] = {
                                    "arguments_per_branch": left.numel(),
                                    "left_min": float(left.min()),
                                    "left_max": float(left.max()),
                                    "right_min": float(right.min()),
                                    "right_max": float(right.max()),
                                    "exponential_outside_clamp": int(
                                        ((left < -80) | (left > 80)).sum()
                                    ),
                                    "logarithm_outside_clamp": int(
                                        ((right < 1e-6) | (right > 1e30)).sum()
                                    ),
                                }
                    records.append(record)
            report["cells"][model + "/" + op] = {"fits": records, "input_sha256": hashes}
            print("NUMERICS CHECKED", model, op, flush=True)
    destination = args.output.resolve()
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print("HEAD NUMERICS COMPLETE", destination, flush=True)


if __name__ == "__main__":
    main()
