"""Check all-token restoration on real models using validation prompts only."""

import argparse
import json

import torch
from runtime import HERE, RUNS, configure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen17b", "qwen4b", "smollm"])
    args = parser.parse_args()
    configure(args.model)
    from data import STYLES, save
    from evaluate_heads import Replacements, ordinary
    from model_io import expand, load, setup

    setup()
    model, tok = load()
    checks = {}
    with torch.inference_mode():
        for op in ["add", "multiply", "divide"]:
            out = RUNS / args.model / op
            problems = json.loads((out / "problems.json").read_text())
            rows = expand(problems["validation"][:4], STYLES)
            rep = Replacements(out, {"heads": {}, "sparse": {}, "linear": {}})
            values = ordinary(model, tok, rep, rows, batch_size=12, all_tokens=True)
            assert all(r["exact_text_agreement"] for r in values["restore"])
            assert any(r["patched_calls"] > 1 for r in values["mean"])
            checks[op] = {
                "validation_prompts": len(rows),
                "restoration_logits_and_text_bitwise_equal": True,
                "constant_replacement_called_during_decode": True,
                "maximum_generation_hook_calls": max(r["patched_calls"] for r in values["mean"]),
            }
    save(
        HERE / f"generation-audit-{args.model}.json",
        {
            "model": args.model,
            "gpu": torch.cuda.get_device_name(),
            "checks": checks,
            "test_outputs_collected": False,
            "scope": "Generation implementation check with original, restoration, and constant controls; no learned-head selection or performance claim.",
        },
    )
    print("REAL-MODEL ALL-TOKEN GPU AUDIT PASSED", args.model, flush=True)


if __name__ == "__main__":
    main()
