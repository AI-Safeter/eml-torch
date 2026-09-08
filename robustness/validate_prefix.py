"""Reproduce the Qwen addition cache validation and preliminary forward timing."""

import json
import sys
import time

import torch
from runtime import HERE, RUNS, configure


def main():
    assert sys.argv[1] in ["qwen17b", "qwen4b"]
    configure(sys.argv[1])
    from data import STYLES
    from evaluate_heads import Replacements, generation
    from model_io import expand, inputs, load, logits, setup
    from prefix_cache import PrefixCache

    setup()
    model, tok = load()
    out = RUNS / sys.argv[1] / "add"
    selected = json.loads((out / "selection.json").read_text())
    rep = Replacements(out, selected)
    rows = expand(json.loads((out / "problems.json").read_text())["validation"][:4], STYLES)
    checks = []
    with torch.inference_mode():
        clean, corrupt = inputs(tok, rows), inputs(tok, rows, True)
        hs = []
        for inp in [clean, corrupt]:
            with rep.hook(model, "original") as captured:
                logits(model, inp)
            hs.append(captured["input"].clone())
        with PrefixCache(model, rep.layer, logits) as cache:
            for alpha in [0.0, 0.125, 0.5, 1.0, 1.25]:
                mixed = (1 - alpha) * hs[1] + alpha * hs[0]
                for kind in rep.kinds:
                    with rep.hook(model, kind, mixed):
                        expected = logits(model, corrupt)
                    with rep.hook(model, kind, mixed):
                        actual = cache.logits(model, corrupt)
                    assert torch.equal(expected, actual), (
                        alpha,
                        kind,
                        float((expected - actual).abs().max()),
                    )
                    checks.append([alpha, kind])
            for inp in [clean, corrupt, inputs(tok, rows[:2]), clean]:
                assert torch.equal(logits(model, inp), cache.logits(model, inp))
            changed = {k: v.clone() for k, v in clean.items()}
            cache.logits(model, changed)
            changed["input_ids"][0, -1] = corrupt["input_ids"][0, 0]
            assert torch.equal(logits(model, changed), cache.logits(model, changed))
            score, text = generation(model, tok, inputs(tok, rows[:2]))
            before_hits = cache.hits
            score2, text2 = generation(model, tok, inputs(tok, rows[:2]))
            assert torch.equal(score, score2) and text == text2 and cache.hits == before_hits
            for _ in range(2):
                logits(model, corrupt)
                cache.logits(model, corrupt)
            timing = []
            for _ in range(8):
                pair = {}
                for label, fn in [("original", logits), ("cached", cache.logits)]:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    fn(model, corrupt)
                    torch.cuda.synchronize()
                    pair[label] = time.perf_counter() - start
                timing.append(pair)
            report = {
                "model": sys.argv[1],
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cached_layers": rep.layer,
                "validation_interventions_bitwise_identical": len(checks),
                "all_kinds": rep.kinds,
                "input_content_and_batch_invalidation_verified": True,
                "generation_does_not_use_cache": True,
                "prefix_executions": cache.executions,
                "prefix_cache_hits": cache.hits,
                "paired_seconds": timing,
                "speed_ratio": sum(p["original"] for p in timing)
                / sum(p["cached"] for p in timing),
                "scope": "Validation only; optional optimization not installed in live study jobs",
            }
        assert all(
            layer.forward == original for layer, original in zip(cache.layers, cache.originals)
        )
    (HERE / f"prefix-validation-reproduction-{sys.argv[1]}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
