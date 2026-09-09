"""Pinned shared runtime and isolated, provenance-bound stage paths."""

import json
import os
from pathlib import Path

from gemma_mechanisms.runtime import (  # noqa: F401
    SNAPSHOT,
    accounting,
    digest,
    load,
    prompt,
    save,
    setup,
    text_layers,
    tokenizer,
)

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "protocol.json").read_text())


def root(path=None):
    out = Path(
        path
        or os.environ.get(
            "EML_ARCHITECTURE_RUNS", "/home/ubuntu/samuel/emltorch-gemma-architecture-runs"
        )
    ).resolve()
    assert "replacement-runs" not in str(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def previous(out):
    return Path(json.loads((out / "data/freeze.json").read_text())["previous_root"])


def name(architecture, activation, depth, seed, steps=12000):
    return f"{architecture}-{activation}-d{depth}-s{seed}-n{steps}"


def sources(*names):
    paths = [HERE / n for n in names]
    paths += [
        HERE / "protocol.json",
        HERE.parent / "emltorch/operator.py",
        HERE.parent / "gemma_mechanisms/runtime.py",
    ]
    return {str(p.relative_to(HERE.parent)): digest(p) for p in paths}


def freeze(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, ("Changed frozen inputs", path)
    else:
        save(path, value)
    return digest(path)
