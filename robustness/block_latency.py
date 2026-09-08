"""Standalone eager block timings; these are not end-to-end LLM speedups."""

import time

import torch
from runtime import RUNS, configure
from whole_evaluate import select, students


def elapsed(module, x):
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(100):
        module(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1e6 / 100


def main():
    configure("qwen17b")
    from data import save
    from model_io import load_mlp, setup

    setup()
    out = RUNS / "whole-block"
    selected = select(out)
    original = load_mlp(selected["layer"])
    replacements = students(out, selected)
    sample = torch.load(out / "activations-arithmetic-validation.pt", weights_only=True)["x"][
        :1024
    ].cuda()
    records = []
    with torch.inference_mode():
        for kind, student in replacements.items():
            for batch in [1, 8]:
                for length in [1, 128]:
                    x = sample[: batch * length].reshape(batch, length, -1)
                    for module in [original, student]:
                        for _ in range(10):
                            module(x)
                    pairs = []
                    for index in range(50):
                        order = [("original", original), ("replacement", student)]
                        if index % 2:
                            order.reverse()
                        pairs.append({name: elapsed(module, x) for name, module in order})
                    records.append(
                        {
                            "student": kind,
                            "batch": batch,
                            "sequence_length": length,
                            "paired_microseconds_per_call": pairs,
                            "ratio_of_means": sum(p["original"] for p in pairs)
                            / sum(p["replacement"] for p in pairs),
                        }
                    )
    save(
        out / "block-latency.json",
        {
            "mode": "eager float32",
            "timing": "100 calls per synchronized wall-clock sample",
            "gpu": torch.cuda.get_device_name(),
            "records": records,
        },
    )
    print("BLOCK LATENCY COMPLETE", flush=True)


if __name__ == "__main__":
    main()
