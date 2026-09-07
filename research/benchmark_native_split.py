"""Compare one-block and split-projection CUDA kernels on the same frozen heads."""

import json
import time

import torch

from benchmark_native import timing
from data import ROOT, save
from heads import Head
from model_io import setup
from native_head import NativeHead


def main():
    setup()
    folder = ROOT / "add"
    c = torch.load(folder / "component-active.pt", weights_only=True)
    c = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in c.items()}
    raw = torch.load(folder / "inputs-validation.pt", weights_only=True)["input"]
    x = raw.reshape(-1, raw.shape[-1]).cuda().contiguous()
    rows = json.loads((folder / "heads-active-r32-g0/candidates.json").read_text())
    results = {}
    with torch.inference_mode():
        for kind in ["eml_square", "eml_log", "silu", "silu_two"]:
            row = min(
                [r for r in rows if r["status"] == "complete" and r["kind"] == kind],
                key=lambda r: r["validation_objective"],
            )
            head = Head(kind, 32, row["width"]).cuda().eval()
            head.load_state_dict(
                torch.load(
                    folder / "heads-active-r32-g0" / f"{row['name']}.pt",
                    weights_only=True,
                )
            )
            one, split = NativeHead(head, c), NativeHead(head, c, split_projection=True)

            def reference(h, head=head):
                z = (((h - c["input_mean"]) @ c["encoder"]) - c["zmean"]) / c["zstd"]
                return head(z) * c["ystd"] + c["ymean"]

            expected, prediction = reference(x), split(x)
            tolerance = float(c["ystd"]) * 2e-4
            torch.testing.assert_close(prediction, expected, atol=tolerance, rtol=2e-4)
            record = {
                "checkpoint": row["name"],
                "max_error": float((prediction - expected).abs().max()),
                "batch": {},
            }
            for batch in [1, 16, 128, 1024]:
                record["batch"][str(batch)] = {
                    name: timing(function, x[:batch])[0]
                    for name, function in [
                        ("torch", reference),
                        ("one_block", one),
                        ("split_projection", split),
                    ]
                }
            results[kind] = record
            save(
                ROOT / "native-split-benchmark.json",
                {
                    "gpu": torch.cuda.get_device_name(),
                    "records": results,
                    "timing": "32 calls inside a single CUDA graph, 7 timed replays; shared GPU; TF32 off; exact same checkpoints for each implementation",
                },
            )
            print(kind, record["batch"]["1"], flush=True)
    save(ROOT / "native-split-completed.json", {"completed_epoch": time.time()})
    print("SPLIT BENCHMARK DONE", flush=True)


if __name__ == "__main__":
    main()
