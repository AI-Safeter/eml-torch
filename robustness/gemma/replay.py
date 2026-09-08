"""Recollect Gemma features in an isolated layout and compare all archives on GPU."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from replay_collection import compare
from runtime import HERE, RUNS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    assert not destination.exists(), "Use a new isolated replay directory"
    assert not destination.is_relative_to(HERE.parent) and not destination.is_relative_to(RUNS)
    study = destination / "eml-torch/robustness"
    shutil.copytree(HERE, study, ignore=shutil.ignore_patterns("__pycache__", "results"))
    subprocess.run([sys.executable, str(study / "gemma/collect.py")], check=True)
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
            original = RUNS / "gemma" / op / name
            replay = destination / "emltorch-robustness-runs/gemma" / op / name
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
        "model": "gemma",
        "archives": archives,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "pipeline": "Fresh pinned Gemma collection, PLS, active subspaces, and derivatives in an isolated copied source layout; no copied activations or test outputs.",
    }
    (HERE / "replay-gemma.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FRESH GEMMA PIPELINE GPU REPLAY PASSED", len(archives), flush=True)


if __name__ == "__main__":
    main()
