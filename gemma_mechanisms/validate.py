"""CUDA integration checks for accounting, native normalization, and MLP removal."""

import argparse
import gc
import json
import time

import torch

from .runtime import HERE, SPEC, accounting, load, prompt, save, setup, sources, tokenizer
from .student import Student, contribution, install


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-model", action="store_true")
    args = parser.parse_args()
    setup(73)
    started = time.perf_counter()
    checks, counts = [], []
    for budget in SPEC["replacement"]["budgets"]:
        for depth in SPEC["replacement"]["depths"]:
            models = [
                Student(1536, budget["coefficients"], budget["bottleneck"], depth, kind)
                for kind in ["eml", "silu"]
            ]
            a, b = [m.coefficients() for m in models]
            assert abs(a - b) / max(a, b) <= 0.001
            for model in models:
                model.cuda()
                x = torch.randn(8, 1536, device="cuda")
                y = model(x)
                y.square().mean().backward()
                assert torch.isfinite(y).all()
                for stage in model.stages:
                    grad = stage.arguments.weight.grad
                    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
                counts.append({**model.specification(), **accounting(model)})
            del models, model, x, y
            gc.collect()
    checks.append(
        "All matched budgets/depths within 0.1%; every nonlinear stage receives finite nonzero CUDA gradients"
    )
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    norm = Gemma4RMSNorm(1536).cuda()
    with torch.no_grad():
        norm.weight.uniform_(0.1, 2)
    for dtype in [torch.float32, torch.bfloat16]:
        y = torch.randn(32, 1536, device="cuda", dtype=dtype)
        assert torch.equal(norm(y), contribution(y, norm.weight, norm.eps))
    checks.append(
        "Contribution objective is bitwise equal to native Gemma4 RMSNorm in FP32 and BF16"
    )
    del norm, y
    gc.collect()
    torch.cuda.empty_cache()
    full = None
    if args.full_model:
        model, tok = load(), tokenizer()
        initial = accounting(model)
        inp = tok(
            [prompt(tok, x) for x in ["What is 582 + 736?", "Explain evaporation briefly."]],
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            reference = model(**inp, use_cache=True, logits_to_keep=1)
            assert torch.isfinite(reference.logits).all()
            del reference
        student = Student(1536, 3000000, 256, 4, "eml").cuda().bfloat16().eval()
        old = install(model, student)
        old_count = accounting(old)

        def forbidden(*args, **kwargs):
            raise AssertionError("Removed original MLP executed")

        old.forward = forbidden
        old.cpu()
        del old
        gc.collect()
        torch.cuda.empty_cache()
        deployed = accounting(model)
        assert (
            deployed["parameters"]
            == initial["parameters"] - old_count["parameters"] + accounting(student)["parameters"]
        )
        student_calls = []
        hook = student.register_forward_hook(lambda m, a, o: student_calls.append(tuple(o.shape)))
        try:
            with torch.inference_mode():
                result = model(**inp, use_cache=True, logits_to_keep=1)
                assert torch.isfinite(result.logits).all()
                mask = inp.attention_mask
                for _ in range(3):
                    next_id = result.logits[:, -1].argmax(-1, keepdim=True)
                    mask = torch.cat([mask, torch.ones_like(next_id)], dim=1)
                    result = model(
                        input_ids=next_id,
                        attention_mask=mask,
                        past_key_values=result.past_key_values,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    assert torch.isfinite(result.logits).all()
        finally:
            hook.remove()
        assert len(student_calls) == 4 and all(shape[1] == 1 for shape in student_calls[1:])
        full = {
            "original": initial,
            "deployed": deployed,
            "removed_mlp": old_count,
            "student_calls": student_calls,
            "original_forward_poisoned": True,
        }
        checks.append(
            "Actual native Gemma prefill and cached decoding execute the student with the original MLP removed"
        )
    torch.cuda.synchronize()
    record = {
        "checks": checks,
        "grid": counts,
        "full_model": full,
        "seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "sources": sources(),
        "quality_evaluation": False,
    }
    save(
        HERE / ("integration-validation.json" if args.full_model else "student-validation.json"),
        record,
    )
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
