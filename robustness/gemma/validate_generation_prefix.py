"""Check native decoding/cache equivalence and replay a complete saved cohort."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma import adapter
from gemma.generation_prefix import GenerationPrefix
from gemma.host_embeddings import load
from gemma.prefix_cache import GemmaPrefixCache


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(value):
    if isinstance(value, torch.Tensor):
        return {
            "tensor": hashlib.sha256(
                value.detach().contiguous().cpu().numpy().tobytes()
            ).hexdigest(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {k: signature(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [signature(v) for v in value]
    if hasattr(value, "__dict__"):
        return {"class": type(value).__name__, "state": signature(vars(value))}
    if value is None or type(value) in [bool, int, float, str]:
        return value
    return {"type": type(value).__name__, "repr": str(value)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    assert not destination.exists(), "Use a new validation directory"
    from gemma.host_execute import verify

    verify()
    adapter.install()
    import evaluate_heads as evaluate
    from data import ROOT, STYLES, save
    from model_io import expand, inputs, setup
    from trace_audit import check

    setup()
    sources = {
        str(path.relative_to(adapter.STUDY)): sha(path)
        for path in [Path(__file__), adapter.HERE / "generation_prefix.py"]
    }
    destination.mkdir(parents=True)
    save(destination / "freeze.json", {"sources": sources})
    model, tok = load()
    checks, timing = [], []
    started = time.monotonic()
    for operation in ["add", "multiply", "divide"]:
        out = ROOT / operation
        selected = evaluate.freeze(out, ["heads", "heads-active-r32-g0", "heads-active-r32-g0.1"])
        rep = evaluate.Replacements(out, selected)
        problems = json.loads((out / "problems.json").read_text())
        rows = expand(problems["validation"], STYLES)
        with (
            torch.inference_mode(),
            GemmaPrefixCache(model, rep.layer, adapter.logits) as logits_cache,
            GenerationPrefix(model, rep.layer) as cache,
        ):
            inp = inputs(tok, rows[:32])
            records = {}
            for enabled in [False, True]:
                cache.enabled = enabled
                for all_tokens in [False, True]:
                    for kind in rep.kinds:
                        with rep.hook(model, kind, all_tokens=all_tokens) as patched:
                            result = model.generate(
                                **inp,
                                max_new_tokens=16,
                                do_sample=False,
                                use_cache=True,
                                pad_token_id=tok.eos_token_id,
                                return_dict_in_generate=True,
                                output_scores=True,
                                logits_to_keep=1,
                            )
                        actual = signature(
                            {
                                "sequences": result.sequences,
                                "scores": result.scores,
                                "cache": result.past_key_values,
                                "patch": patched,
                            }
                        )
                        key = (all_tokens, kind)
                        if enabled:
                            assert actual == records[key], (operation, key)
                            checks.append(
                                {
                                    "operation": operation,
                                    "all_tokens": all_tokens,
                                    "kind": kind,
                                    "rows": len(inp.input_ids),
                                    "signature": actual,
                                }
                            )
                        else:
                            records[key] = actual
                        del result
                print("GENERATION MATCH" if enabled else "NATIVE RECORDED", operation, flush=True)
            assert cache.misses == 1 and cache.hits == 23
            # Non-generative calls use the independently validated logits cache.
            for input_batch in [inp, inputs(tok, rows[32:64])]:
                native = adapter.logits(model, input_batch)
                assert torch.equal(native, logits_cache.logits(model, input_batch))
                assert torch.equal(native, logits_cache.logits(model, input_batch))
            # A different input with the same batch size must invalidate the prefill.
            changed = inputs(tok, rows[32:64])
            with rep.hook(model, "original"):
                cache.enabled = False
                reference = evaluate.generation(model, tok, changed)
            with rep.hook(model, "original"):
                cache.enabled = True
                actual = evaluate.generation(model, tok, changed)
            assert torch.equal(reference[0], actual[0]) and reference[1] == actual[1]
            assert cache.misses == 2
            # Failed/unsupported generation must not leave an active cache context.
            try:
                model.generate(**changed, max_new_tokens=1)
            except AssertionError:
                assert not cache.active and not cache.generating
            else:
                raise AssertionError("Unsupported generation accepted")
            if operation == "add":
                for repeat in range(6):
                    batch = inputs(tok, rows[repeat * 32 : (repeat + 1) * 32])
                    for enabled in [False, True] if repeat % 2 == 0 else [True, False]:
                        cache.enabled = enabled
                        cache.clear()
                        torch.cuda.synchronize()
                        before = time.perf_counter()
                        for kind in rep.kinds:
                            with rep.hook(model, kind):
                                evaluate.generation(model, tok, batch)
                        torch.cuda.synchronize()
                        timing.append(
                            {
                                "repeat": repeat,
                                "cache": enabled,
                                "seconds": time.perf_counter() - before,
                            }
                        )
            checks.append(
                {
                    "operation": operation,
                    "logits_cache_coexistence": True,
                    "input_change_invalidates": True,
                    "unsupported_settings_rejected": True,
                }
            )
        del rep
    save(destination / "native-validation.json", {"checks": checks, "timing": timing})
    print("ALL OPERATIONS NATIVE VALIDATION PASSED", flush=True)

    out = ROOT / "add"
    rep = evaluate.Replacements(out, json.loads((out / "selection.json").read_text()))
    rows = expand(json.loads((out / "problems.json").read_text())["test"], STYLES)
    reference = out / "ordinary-test-known-formats.json"
    reference_hash = sha(reference)
    with torch.inference_mode(), GenerationPrefix(model, rep.layer) as cache:
        actual = evaluate.ordinary(model, tok, rep, rows)
        traces = check(actual, rows, rep.kinds)
        save(destination / "ordinary.json", actual)
        assert sha(reference) == reference_hash
        assert actual == json.loads(reference.read_text())
        assert sha(destination / "ordinary.json") == reference_hash
        report = {
            "passed": True,
            "model": adapter.SPEC,
            "gpu": torch.cuda.get_device_name(),
            "sources": sources,
            "native_validation_sha256": sha(destination / "native-validation.json"),
            "native_comparisons": 72,
            "rows_per_comparison": 32,
            "compared": [
                "generated tokens",
                "every generation score",
                "final KV cache",
                "patch records",
            ],
            "replay_records": sum(len(v) for v in actual.values()),
            "replay_raw_bytes_identical": True,
            "replay_sha256": reference_hash,
            "trace_audit": traces,
            "replay_cache_hits": cache.hits,
            "replay_cache_misses": cache.misses,
            "timing": timing,
            "seconds": time.monotonic() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
    assert all(sha(adapter.STUDY / name) == digest for name, digest in sources.items())
    save(destination / "report.json", report)
    print("COMPLETE GENERATION REPLAY PASSED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
