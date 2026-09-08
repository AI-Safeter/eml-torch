"""Freeze source and inputs before collection; verify them before every stage."""

import argparse
import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    target = HERE / "freeze.json"
    if args.verify or target.exists():
        record = json.loads(target.read_text())
        for name, sha in record["files"].items():
            assert digest(HERE / name) == sha, f"Frozen source/input changed: {name}"
        print("FREEZE VERIFIED", len(record["files"]), flush=True)
        return
    paths = [
        p
        for p in HERE.rglob("*")
        if p.is_file()
        and not any(x.startswith(".") or x == "__pycache__" for x in p.relative_to(HERE).parts)
        and p.suffix in [".py", ".md", ".json"]
    ]
    record = {
        "frozen_epoch": time.time(),
        "test_model_outputs_unseen": True,
        "files": {str(p.relative_to(HERE)): digest(p) for p in sorted(paths)},
    }
    target.write_text(json.dumps(record, indent=2) + "\n")
    print("FROZEN", len(paths), flush=True)


if __name__ == "__main__":
    main()
