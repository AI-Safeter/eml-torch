"""Diagnostic module timing; never substitute it for end-to-end model timing."""

import json
import os
import time

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP

from .deploy import load_deployed
from .evaluate import roster
from .runtime import HERE, SNAPSHOT, accounting, digest, root, save, setup


@torch.inference_mode()
def timed(module, x, repeats):
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    begin.record()
    for _ in range(repeats):
        y = module(x)
    end.record()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    assert torch.isfinite(y).all()
    return {
        "gpu_ms_per_call": begin.elapsed_time(end) / repeats,
        "wall_ms_per_call": seconds * 1000 / repeats,
    }


def main():
    setup(73)
    out = root()
    destination = out / "microbenchmark.json"
    if destination.exists():
        old = json.loads(destination.read_text())
        assert old["source_sha256"] == digest(HERE / "microbenchmark.py")
        assert old["selection_sha256"] == digest(out / "training/selection.json")
        assert old["input_sha256"] == digest(out / "collection/language-selection.pt")
        print("PRESERVE MODULE TIMINGS", flush=True)
        return
    prefix = "model.language_model.layers.26.mlp."
    state = {}
    for path in SNAPSHOT.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state[key.removeprefix(prefix)] = handle.get_tensor(key)
    assert len(state) == 3
    original = Gemma4TextMLP(
        AutoConfig.from_pretrained(SNAPSHOT, local_files_only=True).text_config, 26
    )
    original.load_state_dict(state, strict=True)
    original = original.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    assert accounting(original)["parameters"] == 56623104
    data = torch.load(out / "collection/language-selection.pt", weights_only=True)["x"]
    records = []
    for name in roster(out):
        if not name.endswith("-s1103"):
            continue
        student = load_deployed(out / "training" / f"{name}.pt")
        for tokens in [1, 8, 128, 512, 4096]:
            x = data[:tokens].cuda()
            repeats = 20 if tokens <= 128 else 5
            for module in [original, student]:
                for _ in range(10):
                    timed(module, x, repeats)
            pairs = []
            for i in range(50):
                modules = [("original", original), ("student", student)]
                if i % 2:
                    modules.reverse()
                pairs.append({kind: timed(module, x, repeats) for kind, module in modules})
            records.append(
                {
                    "method": name,
                    "tokens": tokens,
                    "inner_repeats": repeats,
                    "samples": pairs,
                    "student_accounting": accounting(student),
                }
            )
            print("MODULE TIMING", name, tokens, flush=True)
        del student
        torch.cuda.empty_cache()
    save(
        out / "microbenchmark.json",
        {
            "records": records,
            "source_sha256": digest(HERE / "microbenchmark.py"),
            "selection_sha256": digest(out / "training/selection.json"),
            "input_sha256": digest(out / "collection/language-selection.pt"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")),
            "scope": "Diagnostic module timings on captured inputs, eager BF16, shared H100. Does not measure attention, native norms, KV cache, or end-to-end generation.",
        },
    )
    print("MODULE TIMING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
