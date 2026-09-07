"""Compare identical frozen heads in eager Torch, compiled Torch, and native CUDA."""

import json
import time

import torch

from benchmark_native import timing
from data import ROOT, save
from heads import Head
from model_io import setup
from native_head import NativeHead
from packed_head import PackedHead


def main():
    setup()
    out = ROOT / "add"
    c = torch.load(out / "component-active.pt", weights_only=True)
    c = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in c.items()}
    raw = torch.load(out / "inputs-validation.pt", weights_only=True)["input"]
    h = raw.reshape(-1, raw.shape[-1]).cuda().contiguous()
    rows = json.loads((out / "heads-active-r32-g0/candidates.json").read_text())
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "scope": "Standalone coefficient, including input projection; same checkpoints; TF32 disabled; shared GPU",
        "models": {},
    }
    if (ROOT / "compiled-comparison.json").exists():
        report = json.loads((ROOT / "compiled-comparison.json").read_text())
    with torch.inference_mode():
        for kind in ["eml_square", "eml_log", "silu", "silu_two"]:
            if len(report["models"].get(kind, {}).get("batches", {})) == 4:
                continue
            torch._dynamo.reset()
            row = min(
                [r for r in rows if r["status"] == "complete" and r["kind"] == kind],
                key=lambda r: r["validation_objective"],
            )
            head = Head(kind, 32, row["width"]).cuda().eval()
            head.load_state_dict(
                torch.load(out / "heads-active-r32-g0" / f"{row['name']}.pt", weights_only=True)
            )
            packed = PackedHead(head, c).eval()
            compiled = torch.compile(packed, fullgraph=True, dynamic=False)
            native = NativeHead(head, c)

            def reference(x, head=head):
                z = (((x - c["input_mean"]) @ c["encoder"]) - c["zmean"]) / c["zstd"]
                return head(z) * c["ystd"] + c["ymean"]

            record = report["models"].get(kind, {"checkpoint": row["name"], "batches": {}})
            for batch in [1, 16, 128, 1024]:
                if str(batch) in record["batches"]:
                    continue
                x = h[:batch]
                expected = reference(x)
                start = time.perf_counter()
                prediction = compiled(x)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    prediction, expected, atol=float(c["ystd"]) * 2e-4, rtol=2e-4
                )
                entry = {
                    "compile_and_first_call_seconds": time.perf_counter() - start,
                    "max_error": float((prediction - expected).abs().max()),
                }
                for label, function in [
                    ("reference", reference),
                    ("packed_eager", packed),
                    ("packed_compiled", compiled),
                    ("native", native),
                ]:
                    entry[label] = timing(function, x)[0]
                record["batches"][str(batch)] = entry
                report["models"][kind] = record
                save(ROOT / "compiled-comparison.json", report)
                print(
                    "COMPILED",
                    kind,
                    batch,
                    {k: v["median_us"] for k, v in entry.items() if isinstance(v, dict)},
                    flush=True,
                )
    report["completed_epoch"] = time.time()
    save(ROOT / "compiled-comparison.json", report)
    print("COMPILED COMPARISON DONE", flush=True)


if __name__ == "__main__":
    main()
