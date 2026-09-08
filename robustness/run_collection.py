"""Collect training/validation features for all operations with pinned inputs."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen17b", "qwen4b", "smollm"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, str(HERE / "freeze.py"), "--verify"], check=True)
    spec = json.loads((HERE / "models.json").read_text())[args.model]
    path = (
        Path.home()
        / ".cache/huggingface/hub"
        / ("models--" + spec["id"].replace("/", "--"))
        / "snapshots"
        / spec["revision"]
    )
    env = {
        **os.environ,
        "EMLTORCH_RESEARCH_ROOT": str(args.output.resolve()),
        "EMLTORCH_MODEL_ID": spec["id"],
        "EMLTORCH_MODEL_REVISION": spec["revision"],
        "EMLTORCH_MODEL_PATH": str(path),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for op in ["add", "multiply", "divide"]:
        folder = args.output / op
        folder.mkdir(exist_ok=True)
        for name in ["problems.json", "extra-problems.json", "protocol.json"]:
            source = HERE / "data" / op / name
            destination = folder / name
            if destination.exists():
                assert destination.read_bytes() == source.read_bytes()
            else:
                shutil.copyfile(source, destination)
        if not (folder / "collection.json").exists():
            subprocess.run([sys.executable, str(HERE / "src/collect.py"), op], env=env, check=True)
    for name, marker in [
        ("active_features.py", "active-completed.json"),
        ("derivative_data.py", "derivative-completed.json"),
    ]:
        if not (args.output / marker).exists():
            subprocess.run([sys.executable, str(HERE / "src" / name)], env=env, check=True)
    print("FEATURE PIPELINE COMPLETE", args.model, flush=True)


if __name__ == "__main__":
    main()
