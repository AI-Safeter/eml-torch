"""Paired end-to-end eager latency with actual whole-module replacement."""

import subprocess
import time

import torch
from runtime import RUNS, configure
from whole_evaluate import replace, select, students


def prefill(model, ids):
    return model(input_ids=ids, use_cache=True, logits_to_keep=1)


def decode(model, ids, steps=32):
    current = prefill(model, ids)
    past = current.past_key_values
    fixed = ids[:, :1]
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(steps):
        current = model(input_ids=fixed, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = current.past_key_values
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000


def measure(model, ids, phase):
    if phase == "decode":
        return decode(model, ids)
    torch.cuda.synchronize()
    started = time.perf_counter()
    prefill(model, ids)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000


def main():
    configure("qwen17b")
    from data import save
    from model_io import load, setup

    setup()
    out = RUNS / "whole-block"
    selected = select(out)
    replacements = students(out, selected)
    model, _ = load()
    ids = torch.load(out / "language-validation.pt", weights_only=True)[:8, :128].cuda()
    records = []
    environment = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    with torch.inference_mode():
        for kind, student in replacements.items():
            for batch in [1, 8]:
                for phase in ["prefill", "decode"]:
                    for current in [None, student]:
                        with replace(model, selected["layer"], current, check_finite=False):
                            for _ in range(10):
                                measure(model, ids[:batch], phase)
                    pairs = []
                    for repetition in range(50):
                        order = [("original", None), ("replacement", student)]
                        if repetition % 2:
                            order.reverse()
                        pair = {}
                        for label, current in order:
                            with replace(model, selected["layer"], current, check_finite=False):
                                pair[label] = measure(model, ids[:batch], phase)
                        pairs.append(pair)
                    baseline = torch.tensor(
                        [p["original"] for p in pairs], dtype=torch.float64, device="cuda"
                    )
                    replacement = torch.tensor(
                        [p["replacement"] for p in pairs], dtype=torch.float64, device="cuda"
                    )
                    generator = torch.Generator(device="cuda").manual_seed(20260928)
                    indices = torch.randint(50, (10000, 50), device="cuda", generator=generator)
                    ratios = baseline[indices].mean(1) / replacement[indices].mean(1)
                    ci = torch.quantile(
                        ratios, torch.tensor([0.025, 0.975], device="cuda", dtype=torch.float64)
                    ).tolist()
                    record = {
                        "student": kind,
                        "batch": batch,
                        "phase": phase,
                        "input_tokens": 128,
                        "decode_steps": 32 if phase == "decode" else 0,
                        "baseline_ms_mean": float(baseline.mean()),
                        "replacement_ms_mean": float(replacement.mean()),
                        "speedup_mean_ratio": float(baseline.mean() / replacement.mean()),
                        "paired_ratio_ci95": ci,
                        "speedup_exceeds_sampling_uncertainty": ci[0] > 1,
                        "paired_measurements_ms": pairs,
                    }
                    records.append(record)
                    save(
                        out / "latency.json",
                        {
                            "mode": "eager",
                            "dtype": "float32",
                            "gpu": torch.cuda.get_device_name(),
                            "shared_gpu_snapshot": environment,
                            "records": records,
                            "limitations": "Shared hardware; paired intervals quantify these measurements, not all deployment conditions.",
                        },
                    )
                    print(
                        "WHOLE LATENCY",
                        {k: v for k, v in record.items() if k != "paired_measurements_ms"},
                        flush=True,
                    )
    print("WHOLE BENCHMARK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
