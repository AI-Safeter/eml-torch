"""Synchronized, paired timing of the actual replacement model, with native caches."""

import argparse
import gc
import json
import os
import subprocess
import time

import torch

from .common import HERE, SPEC, accounting, digest, load, root, save, setup, text_layers
from .model import load_replacement


def telemetry():
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader",
    ]
    return subprocess.check_output(command, text=True).strip().splitlines()


def cache_accounting(cache):
    """Count unique live tensor storages, including aliases and cache metadata."""
    objects, storages = set(), {}

    def visit(value):
        if id(value) in objects:
            return
        objects.add(id(value))
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr())
            storages[key] = storage.nbytes()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif hasattr(value, "__dict__"):
            visit(vars(value))

    visit(cache)
    return {
        "cache_cuda_storage_bytes": sum(
            size for (device, _), size in storages.items() if device.startswith("cuda")
        ),
        "cache_total_storage_bytes": sum(storages.values()),
    }


@torch.inference_mode()
def workload(model, ids, forced):
    mask = torch.ones_like(ids)
    pre_start, pre_end, dec_start, dec_end = [
        torch.cuda.Event(enable_timing=True) for _ in range(4)
    ]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    pre_start.record()
    result = model(input_ids=ids, attention_mask=mask, use_cache=True, logits_to_keep=1)
    pre_end.record()
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - start
    decode_start = time.perf_counter()
    dec_start.record()
    for step in range(forced.shape[1]):
        token = forced[:, step : step + 1]
        mask = torch.cat([mask, torch.ones_like(token)], -1)
        result = model(
            input_ids=token,
            attention_mask=mask,
            past_key_values=result.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
    dec_end.record()
    torch.cuda.synchronize()
    end = time.perf_counter()
    assert torch.isfinite(result.logits).all()
    elapsed = end - start
    return {
        "prefill_seconds": prefill_seconds,
        "decode_seconds": end - decode_start,
        "end_to_end_seconds": elapsed,
        "prefill_gpu_ms": pre_start.elapsed_time(pre_end),
        "decode_gpu_ms": dec_start.elapsed_time(dec_end),
        "prefill_tokens_per_second": ids.numel() / prefill_seconds,
        "decode_tokens_per_second": forced.numel() / (end - decode_start),
        "output_tokens_per_second": forced.numel() / elapsed,
        "total_tokens_per_second": (ids.numel() + forced.numel()) / elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        **cache_accounting(result.past_key_values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    setup(73)
    out = root(args.output)
    methods = [args.method]
    directory = out / "benchmark"
    directory.mkdir(exist_ok=True)
    freeze = {
        "sources": {
            p: digest(HERE / p) for p in ["benchmark.py", "model.py", "common.py", "protocol.json"]
        },
        "checkpoints": {m: digest(out / "training" / f"{m}.pt") for m in methods},
        "protocol": SPEC["benchmark"],
        "placement": "Entire native model and PLE table on GPU; staged comparison modules on CPU outside timing",
        "optimization": "Native eager SDPA; affine statistics folded in every student; linear factors collapsed when cheaper. No compilation.",
        "input_sha256": digest(out / "data/language-development.pt"),
    }
    frozen = directory / f"{args.method}-freeze.json"
    if frozen.exists():
        assert json.loads(frozen.read_text()) == freeze
    else:
        save(frozen, freeze)
    model = load(host_ple=False)
    base_accounting = accounting(model)
    original = text_layers(model)[26].mlp
    original_ids = {id(p) for p in original.parameters()}
    native_forward = original.forward
    blocks = torch.load(out / "data/language-development.pt", weights_only=True)
    # One native BOS per synthetic request; no BOS injected into forced decoding.
    assert (blocks[:, 0] == model.config.text_config.bos_token_id).all()
    tokens = torch.cat([blocks[:8], blocks[8:16, 1:]], -1)
    forced = blocks[16:24, 1:33].cuda()
    shapes = [(b, length) for b in [1, 8] for length in [128, 512]]

    def forbidden(*a, **kw):
        raise AssertionError("Original MLP executed in the measured student path")

    for method in methods:
        student = load_replacement(out / "training" / f"{method}.pt", deployed=True, device="cpu")

        def activate(name, student=student):
            if name == "original":
                student.cpu()
                original.forward = native_forward
                text_layers(model)[26].mlp = original.cuda()
            else:
                original.cpu()
                original.forward = forbidden
                text_layers(model)[26].mlp = student.cuda()
                assert not original_ids.intersection(id(p) for p in model.parameters())
            gc.collect()

        for batch, length in shapes:
            path = directory / f"{method}-b{batch}-p{length}.json"
            if path.exists():
                assert json.loads(path.read_text())["freeze_sha256"] == digest(frozen)
                continue
            ids, decode = tokens[:batch, :length].cuda(), forced[:batch]
            before = telemetry()
            isolated_memory = {}
            for name in ["original", method]:
                activate(name)
                if name == method:
                    student_accounting = accounting(student)
                    assert set(student_accounting["by_device"]) == {"cuda"}
                torch.cuda.empty_cache()
                first = workload(model, ids, decode)
                isolated_memory[name] = {
                    k: v
                    for k, v in first.items()
                    if k.startswith("peak_") or k.startswith("cache_")
                }
            # Ten warm-ups per path in total. Preserve the warmed allocator
            # between later switches; clearing it would time artificial cold allocations.
            for _ in range(SPEC["benchmark"]["warmup"] - 1):
                for name in ["original", method]:
                    activate(name)
                    workload(model, ids, decode)
            samples = []
            for repeat in range(SPEC["benchmark"]["paired_repeats"]):
                order = ["original", method] if repeat % 2 == 0 else [method, "original"]
                pair = {"repeat": repeat, "order": order}
                for name in order:
                    activate(name)
                    pair[name] = workload(model, ids, decode)
                samples.append(pair)
                if repeat % 10 == 0:
                    print("BENCHMARK", method, batch, length, repeat, flush=True)
            save(
                path,
                {
                    "method": method,
                    "batch": batch,
                    "prefill_tokens": length,
                    "decode_steps": 32,
                    "samples": samples,
                    "isolated_memory": isolated_memory,
                    "allocator_policy": "First warm-up per path starts with an empty unused allocator cache and records isolated working memory. Nine interleaved warm-ups follow. No allocator clearing between timed switches; reserved memory in timed samples can retain comparison allocations.",
                    "telemetry_before": before,
                    "telemetry_after": telemetry(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "device_name": torch.cuda.get_device_name(),
                    "device_uuid": str(
                        getattr(
                            torch.cuda.get_device_properties(0),
                            "uuid",
                            "See device telemetry and CUDA_VISIBLE_DEVICES",
                        )
                    ),
                    "original_accounting": base_accounting,
                    "student_module_accounting": student_accounting,
                    "freeze_sha256": digest(frozen),
                    "sharing": "Other processes are present. These paired measurements describe this shared-device workload, not exclusive serving.",
                    "timing_excludes": "Model loading, comparison module transfers, tokenization and warm-up; includes native prefill, cached decoding, fixed-token feeding, and phase synchronization.",
                },
            )
            print("BENCHMARK COMPLETE", path.name, flush=True)
        text_layers(model)[26].mlp = original.cuda()
        original.forward = native_forward
        del student
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
