"""Shared paths and pinned model environment for the extension stages."""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent.parent / "emltorch-robustness-runs"
sys.path.insert(0, str(HERE / "src"))


def configure(name):
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
