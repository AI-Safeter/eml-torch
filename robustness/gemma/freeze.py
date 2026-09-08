"""Freeze and verify the Gemma adapter before its feature collection."""

import argparse
import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    path = HERE / "source-freeze.json"
    names = ["GEMMA_AMENDMENT.md"] + [
        "gemma/" + name
        for name in [
            "adapter.py",
            "execute.py",
            "collect.py",
            "sparse_neurons.py",
            "validate.py",
            "freeze.py",
            "model.json",
        ]
    ]
    names += [
        "freeze.json",
        "extension-freeze.json",
        "evaluation-freeze.json",
        "deployment-freeze.json",
    ]
    hashes = {name: hashlib.sha256((STUDY / name).read_bytes()).hexdigest() for name in names}
    if args.verify:
        assert json.loads(path.read_text())["files"] == hashes
        print("GEMMA ADAPTER SOURCE FREEZE VERIFIED", len(hashes), flush=True)
    else:
        assert not path.exists(), "Do not overwrite a frozen Gemma implementation"
        path.write_text(
            json.dumps(
                {
                    "epoch": time.time(),
                    "stage": "Gemma adapter and collection before Gemma feature collection",
                    "files": hashes,
                    "test_outputs_collected": False,
                },
                indent=2,
            )
            + "\n"
        )
        print("GEMMA ADAPTER SOURCE FROZEN", len(hashes), flush=True)


if __name__ == "__main__":
    main()
