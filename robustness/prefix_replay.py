"""Replay a completed intervention cohort with a cached, unchanged prefix."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from prefix_cache import PrefixCache
from runtime import HERE, RUNS, configure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen17b", "qwen4b"])
    parser.add_argument("operation", choices=["add", "multiply", "divide"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure(args.model)
    import evaluate_heads
    from data import STYLES, save
    from evaluate_heads import Replacements
    from model_io import expand, load, logits, setup
    from trace_audit import check

    setup()
    out = RUNS / args.model / args.operation
    source = out / "interventions-test-known-formats.json"
    assert source.exists(), "A complete uncached reference cohort is required"
    target = args.output.resolve()
    assert not target.exists() and not target.is_relative_to(RUNS)
    assert not target.is_relative_to(HERE.parent)
    selected = json.loads((out / "selection.json").read_text())
    rep = Replacements(out, selected)
    rows = expand(json.loads((out / "problems.json").read_text())["test"], STYLES)
    model, tok = load()
    started = time.perf_counter()
    with torch.inference_mode(), PrefixCache(model, rep.layer, logits) as cache:
        evaluate_heads.logits = cache.logits
        try:
            actual = evaluate_heads.interventions(model, tok, rep, rows)
        finally:
            evaluate_heads.logits = logits
        audit = check(actual, rows, rep.kinds, alphas=[0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
        assert actual == json.loads(source.read_text()), "Cached and uncached records differ"
        save(target, actual)
        report = {
            "model": args.model,
            "operation": args.operation,
            "scope": "Entire previously completed primary intervention cohort; all raw records exactly equal to the frozen uncached evaluator",
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cached_layers": rep.layer,
            "prefix_executions": cache.executions,
            "prefix_cache_hits": cache.hits,
            "trace_audit": audit,
            "raw_records_identical": True,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "replay_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "cache_source_sha256": hashlib.sha256(
                (HERE / "prefix_cache.py").read_bytes()
            ).hexdigest(),
            "seconds": time.perf_counter() - started,
        }
    save(HERE / f"prefix-replay-{args.model}-{args.operation}.json", report)
    print("FULL CACHED INTERVENTION REPLAY PASSED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
