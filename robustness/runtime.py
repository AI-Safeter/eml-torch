"""Shared paths and pinned model environment for the extension stages."""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / ".artifacts/robustness"
sys.path.insert(0, str(HERE / "src"))


def configure(name):
    if name == "gemma":
        from gemma.adapter import SPEC, install

        install()
        return SPEC
    spec = json.loads((HERE / "models.json").read_text())[name]
    path = (
        Path.home()
        / ".cache/huggingface/hub"
        / ("models--" + spec["id"].replace("/", "--"))
        / "snapshots"
        / spec["revision"]
    )
    os.environ.update(
        EMLTORCH_MODEL_ID=spec["id"],
        EMLTORCH_MODEL_REVISION=spec["revision"],
        EMLTORCH_MODEL_PATH=str(path),
        EMLTORCH_RESEARCH_ROOT=str(RUNS / name),
    )
    return spec


def run_stage(script, *args):
    if os.environ.get("EMLTORCH_MODEL_ID") == "google/gemma-4-E2B-it":
        command = [sys.executable, str(HERE / "gemma/execute.py"), Path(script).stem, *args]
    else:
        command = [sys.executable, str(HERE / script), *args]
    subprocess.run(command, check=True)
