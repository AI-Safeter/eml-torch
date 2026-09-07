"""CUDA integration validation for the reusable core EML head."""

import copy
import io
import json
import time

import torch

from data import ROOT, save
from emltorch import EMLHead
from heads import Head
from model_io import setup


def main():
    setup()
    records = {}
    folder = ROOT / "replication-0.6b/add"
    selection = json.loads((folder / "selection.json").read_text())["heads"][
        "heads-active-r32-g0/eml"
    ]
    state = torch.load(folder / selection["checkpoint"], weights_only=True)
    reference = Head("eml_square", 32, 32).cuda()
    reusable = EMLHead(32, 32, device="cuda")
    reference.load_state_dict(state)
    reusable.load_state_dict(state)
    x = torch.load(folder / "features-active.pt", weights_only=True)["validation"]["x"].cuda()
    with torch.no_grad():
        expected, actual = reference(x), reusable(x).squeeze(-1)
        assert torch.equal(expected, actual)
    xr, xc = x[:16].clone().requires_grad_(True), x[:16].clone().requires_grad_(True)
    reference(xr).square().mean().backward()
    reusable(xc).square().mean().backward()
    torch.testing.assert_close(xr.grad, xc.grad, atol=1e-7, rtol=1e-5)
    for (name, a), (other, b) in zip(reference.named_parameters(), reusable.named_parameters()):
        assert name == other
        torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=1e-5)
    records["frozen_llm_head"] = {
        "checkpoint": selection["checkpoint"],
        "validation_samples": len(x),
        "outputs_bitwise_equal": True,
        "input_and_parameter_gradients_agree": True,
    }
    for linear_skip in [False, True]:
        module = EMLHead(3, 4, 2, linear_skip=linear_skip, device="cuda", dtype=torch.float64)
        value = torch.randn(2, 3, device="cuda", dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(module, (value,), fast_mode=True)
        assert torch.autograd.gradgradcheck(module, (value,), fast_mode=True)
    records["numerical_derivatives"] = (
        "First and second input derivatives checked against finite differences on CUDA float64, with and without linear skip"
    )
    for shape in [(5,), (3, 5), (2, 3, 5), (0, 5)]:
        module = EMLHead(5, 8, 2, device="cuda")
        assert module(torch.randn(*shape, device="cuda")).shape == (*shape[:-1], 2)
    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        module = EMLHead(5, 8, 2, device="cuda", dtype=dtype)
        value = torch.randn(16, 5, device="cuda", dtype=dtype, requires_grad=True)
        result = module(value)
        assert result.dtype == dtype and torch.isfinite(result).all()
        result.float().square().mean().backward()
        assert torch.isfinite(value.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    records["layout_and_dtype"] = (
        "Unbatched, batched, sequence, and empty inputs; float16/bfloat16/float32/float64 forward and backward on CUDA"
    )
    buffer = io.BytesIO()
    torch.save(reusable.state_dict(), buffer)
    buffer.seek(0)
    restored = EMLHead(32, 32, device="cuda")
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    with torch.no_grad():
        assert torch.equal(restored(x), reusable(x))
    records["serialization"] = (
        "Weights-only state-dict roundtrip reproduces every validation output exactly"
    )
    eager = EMLHead(5, 8, 2, device="cuda")
    compiled_source = copy.deepcopy(eager)
    compiled = torch.compile(compiled_source, fullgraph=True)
    a = torch.randn(16, 5, device="cuda", requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    first, second = eager(a), compiled(b)
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-4)
    first.square().mean().backward()
    second.square().mean().backward()
    torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)
    for p, q in zip(eager.parameters(), compiled_source.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)
    records["compile"] = (
        "Full-graph compiled CUDA forward, input gradients, and parameter gradients agree with eager mode"
    )
    # Verify a real optimizer update through a downstream nonlinear module.
    network = torch.nn.Sequential(
        EMLHead(5, 8, 4, device="cuda"),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 2, device="cuda"),
    )
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
    inputs = torch.randn(64, 5, device="cuda")
    targets = torch.stack(
        [inputs[:, 0].sin() + inputs[:, 1], inputs[:, 2].square() - inputs[:, 3]], 1
    )
    initial = float((network(inputs) - targets).square().mean())
    for _ in range(500):
        optimizer.zero_grad(set_to_none=True)
        loss = (network(inputs) - targets).square().mean()
        loss.backward()
        optimizer.step()
    final = float((network(inputs) - targets).square().mean())
    assert final < initial * 0.05, (initial, final)
    records["optimizer_integration"] = {
        "initial_mse": initial,
        "final_mse": final,
        "steps": 500,
    }
    for invalid in [0, -1, True, 1.5]:
        try:
            EMLHead(invalid, 8, device="cuda")
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid feature size accepted: {invalid}")
    save(
        ROOT / "core-head-validation.json",
        {
            "completed_epoch": time.time(),
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "checks": records,
        },
    )
    print("CORE HEAD CUDA VALIDATION PASSED", flush=True)


if __name__ == "__main__":
    main()
