"""Check native buffer/device contracts and default-dtype independence on CUDA."""

import json
import time

import torch

from data import ROOT, save
from heads import Head
from model_io import setup
from native_head import NativeHead


def rejects(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("Invalid native call was accepted")


def main():
    setup()
    out = ROOT / "add"
    c = torch.load(out / "component-active.pt", weights_only=True)
    chosen = json.loads((out / "selection.json").read_text())["heads"]["heads-active-r32-g0/eml"]
    head = Head(chosen["kind"], chosen["rank"], chosen["width"]).cuda().eval()
    head.load_state_dict(torch.load(out / chosen["checkpoint"], weights_only=True))
    inputs = (
        torch.load(out / "inputs-validation.pt", weights_only=True)["input"][:32, 0]
        .cuda()
        .contiguous()
    )
    cc = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in c.items()}
    with torch.no_grad():
        z = (((inputs - cc["input_mean"]) @ cc["encoder"]) - cc["zmean"]) / cc["zstd"]
        expected = head(z) * cc["ystd"] + cc["ymean"]
    original_dtype = torch.get_default_dtype()
    errors = {}
    try:
        torch.set_default_dtype(torch.float64)
        for split in [False, True]:
            native = NativeHead(head, c, split_projection=split)
            assert all(t.dtype == torch.float32 for t in native.tensors)
            actual = native(inputs)
            torch.testing.assert_close(actual, expected, atol=float(c["ystd"]) * 2e-4, rtol=2e-4)
            errors[str(split)] = float((actual - expected).abs().max())
            assert native(inputs[:0]).shape == (0,)
            buffer = torch.empty(len(inputs), dtype=torch.float32, device="cuda")
            assert native(inputs, out=buffer) is buffer
            rejects(lambda native=native: native(inputs.double()))
            rejects(lambda native=native: native(inputs[:, ::2]))
            rejects(lambda native=native: native(inputs.detach().requires_grad_(True)))
            rejects(lambda native=native: native(inputs, out=inputs.flatten()[: len(inputs)]))
            rejects(
                lambda native=native, buffer=buffer: native(
                    inputs, out=buffer.detach().requires_grad_(True)
                )
            )
            if torch.cuda.device_count() > 1:
                with torch.cuda.device(1):
                    wrong_device = inputs.to("cuda:1")
                    rejects(lambda native=native, wrong_device=wrong_device: native(wrong_device))
    finally:
        torch.set_default_dtype(original_dtype)
    save(
        ROOT / "native-contract-validation.json",
        {
            "completed_epoch": time.time(),
            "gpu": torch.cuda.get_device_name(),
            "max_absolute_errors": errors,
            "checks": "Default float64; explicit output; empty input; invalid dtype/layout/autograd/output alias/device rejected",
            "status": "passed",
        },
    )
    print("NATIVE CONTRACT VALIDATION PASSED")


if __name__ == "__main__":
    main()
