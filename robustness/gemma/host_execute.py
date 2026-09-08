"""Run frozen Gemma scalar stages with separately validated execution changes."""

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma import adapter


def verify():
    from verify_sources import main as verify_sources

    verify_sources()
    frozen = json.loads((adapter.HERE / "host-execution-freeze.json").read_text())
    for name, expected in frozen["files"].items():
        actual = hashlib.sha256((adapter.STUDY / name).read_bytes()).hexdigest()
        assert actual == expected, f"Changed execution source or evidence: {name}"
    validation = json.loads((adapter.HERE / "host-validation.json").read_text())
    replay = json.loads((adapter.HERE / "host-prefix-replay.json").read_text())
    assert (
        validation["passed"]
        and replay["raw_records_identical"]
        and replay["raw_file_bytes_identical"]
    )
    assert replay["records"] == 221184


def install_prefix(model, stop, namespace):
    import evaluate_heads
    import model_io
    from gemma.prefix_cache import GemmaPrefixCache

    assert model.config.model_type == "gemma4_text"
    assert not hasattr(model, "_emltorch_prefix_cache")
    cache = GemmaPrefixCache(model, stop, adapter.logits)
    model._emltorch_prefix_cache = cache
    model_io.logits = evaluate_heads.logits = namespace["logits"] = cache.logits
    print("VALIDATED GEMMA PREFIX ENABLED", stop, "unchanged layers", flush=True)
    return cache


def run_stage(script, *args):
    subprocess.run(
        [sys.executable, str(adapter.HERE / "host_execute.py"), Path(script).stem, *args],
        check=True,
    )


def main():
    verify()
    script, *args = sys.argv[1:]
    from gemma.host_embeddings import load

    # The frozen adapter installs these functions before importing evaluator
    # globals. Child processes enter through this same explicit driver.
    adapter.load = load
    adapter.install()
    import prefix_execution
    import runtime

    prefix_execution.install = install_prefix
    runtime.run_stage = run_stage
    sys.argv = [script, *args]
    module = importlib.import_module(
        "gemma.sparse_neurons" if script == "sparse_neurons" else script
    )
    module.main()


if __name__ == "__main__":
    main()
