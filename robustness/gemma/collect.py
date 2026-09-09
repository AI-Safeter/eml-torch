"""Collect the complete Gemma arm from the existing frozen arithmetic splits."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parent / ".artifacts/robustness/gemma"


def stage(name, *args):
    subprocess.run([sys.executable, str(HERE / "execute.py"), name, *args], check=True)


def main():
    subprocess.run([sys.executable, str(STUDY / "verify_sources.py")], check=True)
    subprocess.run([sys.executable, str(HERE / "freeze.py"), "--verify"], check=True)
    for op in ["add", "multiply", "divide"]:
        out = ROOT / op
        out.mkdir(parents=True, exist_ok=True)
        for name in ["problems.json", "extra-problems.json", "protocol.json"]:
            source, target = STUDY / "data" / op / name, out / name
            if target.exists():
                assert target.read_bytes() == source.read_bytes()
            else:
                shutil.copyfile(source, target)
        if not (out / "collection.json").exists():
            stage("collect", op)
    for name, marker in [
        ("active_features", "active-completed.json"),
        ("derivative_data", "derivative-completed.json"),
    ]:
        if not (ROOT / marker).exists():
            stage(name)
    (ROOT / "gemma-collection-completed.json").write_text(json.dumps({"status": "complete"}) + "\n")
    print("GEMMA FEATURE COLLECTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
