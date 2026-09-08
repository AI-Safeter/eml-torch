"""Install the validated Qwen-only cache in one scalar evaluator process."""

import hashlib
import json
import os
from pathlib import Path

from prefix_cache import PrefixCache

HERE = Path(__file__).resolve().parent


def install(model, stop, namespace):
    if model.config.model_type != "qwen3":
        return None
    specs = json.loads((HERE / "models.json").read_text())
    identity = (os.environ.get("EMLTORCH_MODEL_ID"), os.environ.get("EMLTORCH_MODEL_REVISION"))
    assert identity in {
        (specs[name]["id"], specs[name]["revision"]) for name in ["qwen17b", "qwen4b"]
    }
    frozen = json.loads((HERE / "prefix-freeze.json").read_text())["files"]
    for name in ["prefix_cache.py", "prefix_execution.py"]:
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == frozen[name], name
    import evaluate_heads
    import model_io

    assert not hasattr(model, "_emltorch_prefix_cache"), "Cache already installed"
    cache = PrefixCache(model, stop, model_io.logits)
    model._emltorch_prefix_cache = cache
    model_io.logits = evaluate_heads.logits = namespace["logits"] = cache.logits
    print("QWEN PREFIX CACHE ENABLED", identity[0], "unchanged layers", stop, flush=True)
    return cache
