"""Validate native inference against CUDA Torch, and time standalone scalar predictors."""

import json
import statistics
import time

import torch

from data import ROOT, save
from heads import Head
from model_io import load_mlp, setup
from native_head import NativeHead


def timing(function, x):
    for _ in range(10):
        function(x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(32):
            answer = function(x)
    elapsed = []
    for _ in range(7):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end) * 1000 / 32)
    return {
        "median_us": statistics.median(elapsed),
        "min_us": min(elapsed),
        "max_us": max(elapsed),
        "repeats_us": elapsed,
    }, answer


def main():
    setup()
    out = ROOT / "add"
    raw = torch.load(out / "inputs-validation.pt", weights_only=True)["input"]
    h = raw.reshape(-1, raw.shape[-1]).cuda().contiguous()
    component = torch.load(out / "component.pt", weights_only=True)
    teacher = load_mlp(component["layer"])
    direction = component["direction"].cuda()
    # Compute the exact scalar without materializing the full output projection.
    readout = (teacher.down_proj.weight.double().T @ direction.double()).float()

    def exact_scalar(x):
        return (torch.nn.functional.silu(teacher.gate_proj(x)) * teacher.up_proj(x)) @ readout

    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "scope": "Standalone scalar coefficient only. The full LLM still evaluates the original MLP to preserve orthogonal contributions.",
        "timing": "CUDA graph contains 32 predictor calls, 7 separately timed replays, CUDA events, shared H100; float32, TF32 disabled. Calls inside each graph eliminate Python replay gaps.",
        "teacher_baseline": "Folded d^T down_proj, so no full MLP output is computed by the scalar baseline",
        "teacher_scalar_coefficients": teacher.gate_proj.weight.numel()
        + teacher.up_proj.weight.numel()
        + readout.numel(),
        "models": {},
    }
    with torch.inference_mode():
        true = teacher(h[:256]) @ direction
        folded = exact_scalar(h[:256])
        report["folded_teacher_max_error"] = float((true - folded).abs().max())
        torch.testing.assert_close(folded, true, atol=0.003, rtol=2e-4)
        report["teacher_timings"] = {
            str(batch): timing(exact_scalar, h[:batch])[0] for batch in [1, 16, 128, 1024]
        }
        for directory, feature in [("heads", "pls"), ("heads-active-r32-g0", "active")]:
            c = torch.load(
                out / ("component.pt" if feature == "pls" else "component-active.pt"),
                weights_only=True,
            )
            c = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in c.items()}
            rows = json.loads((out / directory / "candidates.json").read_text())
            for kind in sorted({r["kind"] for r in rows if r["status"] == "complete"}):
                row = min(
                    [r for r in rows if r["status"] == "complete" and r["kind"] == kind],
                    key=lambda r: r["validation_mse"] + 4 * r["validation_response_mse"],
                )
                head = Head(kind, 32, row["width"]).cuda().eval()
                head.load_state_dict(
                    torch.load(out / directory / (row["name"] + ".pt"), weights_only=True)
                )
                native = NativeHead(head, c)

                def reference(x, c=c, head=head):
                    z = (((x - c["input_mean"]) @ c["encoder"]) - c["zmean"]) / c["zstd"]
                    return head(z) * c["ystd"] + c["ymean"]

                expected, predicted = reference(h), native(h)
                error = (predicted - expected).abs()
                # Report actual error, with a tolerance relative to the training scalar scale.
                tolerance = float(c["ystd"]) * 2e-4
                torch.testing.assert_close(predicted, expected, atol=tolerance, rtol=2e-4)
                result = {
                    "checkpoint": f"{directory}/{row['name']}.pt",
                    "width": head.width,
                    "samples_validated": len(h),
                    "max_absolute_error": float(error.max()),
                    "rmse": float(error.square().mean().sqrt()),
                    "absolute_tolerance": tolerance,
                    "max_error_in_training_standard_deviations": float(error.max() / c["ystd"]),
                    "native_stored_coefficients_including_padding": sum(
                        t.numel() for t in native.tensors
                    ),
                    "timings": {},
                }
                for batch in [1, 16, 128, 1024]:
                    x = h[:batch].contiguous()
                    result["timings"][str(batch)] = {
                        "torch_head": timing(reference, x)[0],
                        "native_head": timing(native, x)[0],
                    }
                report["models"][directory + "/" + kind] = result
                save(ROOT / "native-benchmark.json", report)
                print(
                    "NATIVE VALIDATED",
                    directory,
                    kind,
                    float(error.max()),
                    result["timings"]["1"],
                    flush=True,
                )
    report["completed_epoch"] = time.time()
    save(ROOT / "native-benchmark.json", report)
    print("NATIVE BENCHMARK DONE", flush=True)


if __name__ == "__main__":
    main()
