"""Replay the complete native Gemma cohort before enabling execution changes."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma import adapter
from gemma.host_embeddings import load
from gemma.prefix_cache import GemmaPrefixCache


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    assert not destination.exists(), "Use a new replay directory"
    adapter.install()
    import evaluate_heads as evaluate
    from data import ROOT, STYLES, save
    from model_io import expand, logits, setup
    from trace_audit import check

    setup()
    validation = json.loads((adapter.HERE / "host-validation.json").read_text())
    assert validation["passed"]
    assert validation["source_hashes"]["host_embeddings.py"] == digest(
        adapter.HERE / "host_embeddings.py"
    )
    out = ROOT / "add"
    source = out / "interventions-test-known-formats.json"
    reference_hash = digest(source)
    destination.mkdir(parents=True)
    freeze = {
        "native_reference_sha256": reference_hash,
        "validation_sha256": digest(adapter.HERE / "host-validation.json"),
        "source_hashes": {
            name: digest(adapter.STUDY / name)
            for name in [
                "gemma/host_embeddings.py",
                "gemma/replay_host_prefix.py",
                "gemma/prefix_cache.py",
                "prefix_cache.py",
            ]
        },
    }
    save(destination / "freeze.json", freeze)
    selected = json.loads((out / "selection.json").read_text())
    rep = evaluate.Replacements(out, selected)
    rows = expand(json.loads((out / "problems.json").read_text())["test"], STYLES)
    model, tok = load()
    started = time.monotonic()
    with torch.inference_mode(), GemmaPrefixCache(model, rep.layer, logits) as cache:
        evaluate.logits = cache.logits
        try:
            actual = evaluate.interventions(model, tok, rep, rows)
        finally:
            evaluate.logits = logits
        trace = check(actual, rows, rep.kinds, alphas=evaluate.ALPHAS)
        assert digest(source) == reference_hash, "Native reference changed during replay"
        save(destination / "interventions.json", actual)
        assert actual == json.loads(source.read_text()), (
            "Execution changes differ from native records"
        )
        assert digest(destination / "interventions.json") == reference_hash
        report = {
            "model": adapter.SPEC,
            "gpu": torch.cuda.get_device_name(),
            "cohort": "Gemma addition primary interventions; all selected methods and controls",
            "raw_records_identical": True,
            "raw_file_bytes_identical": True,
            "records": sum(len(records) for records in actual.values()),
            "trace_audit": trace,
            "unchanged_layers_cached": rep.layer,
            "prefix_executions": cache.executions,
            "prefix_cache_hits": cache.hits,
            "seconds": time.monotonic() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            **freeze,
        }
    save(destination / "report.json", report)
    print("GEMMA COMPLETE HOST/PREFIX REPLAY PASSED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
