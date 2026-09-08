"""Equal-compiler module diagnostic, separate from frozen eager quality/timing."""

import json
import os
import time

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP

from .deploy import load_deployed
from .evaluate import roster
from .microbenchmark import timed
from .runtime import HERE, SNAPSHOT, accounting, digest, root, save, setup


@torch.inference_mode()
def main():
    setup(73)
    out = root()
    directory = out / "compiler-diagnostic"
    directory.mkdir(exist_ok=True)
    plan = {
        "sources": {
            p: digest(HERE / p)
            for p in [
                "compile_probe.py",
                "microbenchmark.py",
                "deploy.py",
                "student.py",
                "runtime.py",
            ]
        },
        "selection_sha256": digest(out / "training/selection.json"),
        "input_sha256": digest(out / "collection/language-selection.pt"),
        "tokens": [1, 4096],
        "warmup": 10,
        "paired_repeats": 50,
        "compiler": "torch.compile, inductor, default mode, fullgraph=True, dynamic=False; same policy for native and all replacements",
        "scope": "Post hoc module diagnostic only. No final quality or end-to-end claim for compiled execution. Failures are recorded without eager fallback. First-call setup can reuse the compiler disk cache.",
    }
    frozen = directory / "freeze.json"
    if frozen.exists():
        assert json.loads(frozen.read_text()) == plan
    else:
        save(frozen, plan)
    prefix = "model.language_model.layers.26.mlp."
    with safe_open(SNAPSHOT / "model.safetensors", framework="pt", device="cpu") as handle:
        state = {
            key.removeprefix(prefix): handle.get_tensor(key)
            for key in handle.keys()
            if key.startswith(prefix)
        }
    original = Gemma4TextMLP(
        AutoConfig.from_pretrained(SNAPSHOT, local_files_only=True).text_config, 26
    )
    original.load_state_dict(state, strict=True)
    original = original.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    data = torch.load(out / "collection/language-selection.pt", weights_only=True)["x"]
    for name in roster(out):
        if not name.endswith("-s1103"):
            continue
        path = directory / f"{name}.json"
        if path.exists():
            assert json.loads(path.read_text())["freeze_sha256"] == digest(frozen)
            continue
        student = load_deployed(out / "training" / f"{name}.pt")
        records = []
        for tokens in plan["tokens"]:
            torch.compiler.reset()
            x = data[:tokens].cuda()
            modules = {"original_eager": original, "student_eager": student}
            record = {"tokens": tokens, "first_call_setup_seconds": {}, "precision": {}}
            try:
                for kind, native in [("original", original), ("student", student)]:
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    compiled = torch.compile(
                        native, backend="inductor", fullgraph=True, dynamic=False
                    )
                    actual = compiled(x)
                    torch.cuda.synchronize()
                    record["first_call_setup_seconds"][kind] = time.perf_counter() - start
                    expected = native(x)
                    assert torch.isfinite(actual).all()
                    difference = actual.float() - expected.float()
                    record["precision"][kind] = {
                        "output_nrmse": float(
                            (
                                difference.square().mean()
                                / expected.float().square().mean().clamp_min(1e-12)
                            ).sqrt()
                        ),
                        "max_absolute_error": float(difference.abs().max()),
                        "bitwise_equal": bool(
                            torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                        ),
                    }
                    modules[f"{kind}_compiled"] = compiled
                repeats = 20 if tokens == 1 else 5
                for _ in range(plan["warmup"]):
                    for module in modules.values():
                        timed(module, x, repeats)
                pairs = []
                for repeat in range(plan["paired_repeats"]):
                    order = list(modules)
                    order = order[repeat % 4 :] + order[: repeat % 4]
                    pairs.append({kind: timed(modules[kind], x, repeats) for kind in order})
                record.update(status="complete", inner_repeats=repeats, samples=pairs)
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__, error=str(error))
            records.append(record)
            print("COMPILER DIAGNOSTIC", name, tokens, record["status"], flush=True)
            del modules, x
            torch.cuda.empty_cache()
        save(
            path,
            {
                "method": name,
                "records": records,
                "freeze_sha256": digest(frozen),
                "checkpoint_sha256": digest(out / "training" / f"{name}.pt"),
                "original_accounting": accounting(original),
                "student_accounting": accounting(student),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "device_uuid": str(
                    getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")
                ),
            },
        )
        del student
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
