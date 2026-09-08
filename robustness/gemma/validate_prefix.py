"""Reproduce native Gemma shared-state cache checks on validation prompts."""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma import adapter


def main():
    adapter.install()
    from data import STYLES
    from evaluate_heads import generation
    from gemma.prefix_cache import GemmaPrefixCache
    from model_io import expand, inputs, load, logits, setup
    from runtime import HERE, RUNS

    setup()
    model, tok = load()
    checks = []
    timings = []
    with torch.inference_mode():
        for op in ["add", "multiply", "divide"]:
            component = torch.load(RUNS / "gemma" / op / "component.pt", weights_only=True)
            layer = component["layer"]
            direction = component["direction"].cuda()
            rows = expand(
                json.loads((HERE / "data" / op / "problems.json").read_text())["validation"][:4],
                STYLES,
            )
            clean, corrupt = inputs(tok, rows), inputs(tok, rows, True)
            with GemmaPrefixCache(model, layer, logits) as cache:
                for inp in [clean, corrupt, inputs(tok, rows[:2]), clean]:
                    for strength in [0.0, -0.25, 0.25, 1.0]:

                        def edit(module, args, out):
                            result = out.clone()
                            result[:, -1] += strength * direction
                            return result

                        handle = model.model.layers[layer].mlp.register_forward_hook(edit)
                        try:
                            reference = logits(model, inp)
                            actual = cache.logits(model, inp)
                            assert torch.equal(reference, actual), (
                                op,
                                strength,
                                float((reference - actual).abs().max()),
                            )
                        finally:
                            handle.remove()
                        checks.append([op, layer, len(inp["input_ids"]), strength])
                before_hits = cache.hits
                reference_score, reference_text = generation(model, tok, inputs(tok, rows[:2]))
                cached_score, cached_text = generation(model, tok, inputs(tok, rows[:2]))
                assert (
                    reference_text == cached_text
                    and torch.equal(reference_score, cached_score)
                    and cache.hits == before_hits
                )
                for _ in range(2):
                    logits(model, corrupt)
                    cache.logits(model, corrupt)
                pairs = []
                for _ in range(6):
                    pair = {}
                    for name, fn in [("native", logits), ("cached", cache.logits)]:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        fn(model, corrupt)
                        torch.cuda.synchronize()
                        pair[name] = time.perf_counter() - start
                    pairs.append(pair)
                timings.append(
                    {
                        "operation": op,
                        "layer": layer,
                        "paired_seconds": pairs,
                        "speed_ratio": sum(p["native"] for p in pairs)
                        / sum(p["cached"] for p in pairs),
                        "shared_state_entries": sum(len(v) for v in cache.shared_updates.values()),
                        "cache_hits": cache.hits,
                    }
                )
    report = {
        "scope": "Gemma validation-only output interventions; not yet installed in live evaluation. No learned heads or held-out outputs used.",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "bitwise_logit_checks": checks,
        "generation_unchanged": True,
        "timings": timings,
    }
    (HERE / "gemma/prefix-validation-reproduction.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
