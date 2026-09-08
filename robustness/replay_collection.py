"""Reproduce a model's full feature pipeline and compare every tensor archive on GPU."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from runtime import HERE, RUNS


def compare(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype
        assert torch.equal(left.cuda(), right.cuda())
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare(left[key], right[key])
    else:
        assert left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen17b", "qwen4b", "smollm"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.output.resolve() != (RUNS / args.model).resolve()
    subprocess.run(
        [
            sys.executable,
            str(HERE / "run_collection.py"),
            args.model,
            "--output",
            str(args.output.resolve()),
        ],
        check=True,
    )
    archives = []
    for op in ["add", "multiply", "divide"]:
        for name in [
            "component.pt",
            "features.pt",
            "inputs-train.pt",
            "inputs-validation.pt",
            "features-active.pt",
            "component-active.pt",
            "active-subspace.pt",
            "gradients-active.pt",
        ]:
            original = RUNS / args.model / op / name
            replay = args.output / op / name
            compare(torch.load(original, weights_only=True), torch.load(replay, weights_only=True))
            archives.append(
                {
                    "path": f"{op}/{name}",
                    "tensor_values_bitwise_equal_on_gpu": True,
                    "archive_bytes_identical": original.read_bytes() == replay.read_bytes(),
                    "sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                }
            )
    report = {
        "model": args.model,
        "archives": archives,
        "gpu": torch.cuda.get_device_name(),
        "pipeline": "Fresh pinned model collection, PLS, active subspaces, and derivative targets; no copied activations or test outputs.",
    }
    (HERE / f"replay-{args.model}.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FRESH PIPELINE GPU REPLAY PASSED", args.model, len(archives), flush=True)


if __name__ == "__main__":
    main()
