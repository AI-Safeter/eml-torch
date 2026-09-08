"""Run fixed Gemma stages with the separately audited generation prefill cache."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma import adapter, host_execute
from gemma.generation_prefix import GenerationPrefix


def verify():
    host_execute.verify()
    frozen = json.loads((adapter.HERE / "generation-execution-freeze.json").read_text())
    for name, expected in frozen["files"].items():
        assert hashlib.sha256((adapter.STUDY / name).read_bytes()).hexdigest() == expected, name
    evidence = json.loads((adapter.HERE / "generation-validation.json").read_text())
    assert evidence["passed"] and evidence["replay_raw_bytes_identical"]
    assert evidence["native_comparisons"] == 72 and evidence["replay_records"] == 36864


def run_stage(script, *args):
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), Path(script).stem, *args], check=True
    )


def main():
    verify()
    original_install = host_execute.install_prefix

    def install(model, stop, namespace):
        import evaluate_heads
        import model_io

        cache = original_install(model, stop, namespace)
        assert not hasattr(model, "_emltorch_generation_prefix")
        generation = model._emltorch_generation_prefix = GenerationPrefix(model, stop)

        def logits(model, inputs):
            # The next intervention forward needs no saved generation prefill.
            generation.clear()
            return cache.logits(model, inputs)

        model_io.logits = evaluate_heads.logits = namespace["logits"] = logits
        print("VALIDATED GEMMA GENERATION PREFIX ENABLED", stop, flush=True)
        return cache

    host_execute.install_prefix = install
    host_execute.run_stage = run_stage
    host_execute.main()


if __name__ == "__main__":
    main()
