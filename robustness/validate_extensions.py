"""CUDA checks for all-token scalar hooks and whole-vector student gradients."""

from types import SimpleNamespace

import torch
from runtime import configure
from torch import nn
from whole_student import Student


def main():
    configure("qwen17b")
    from evaluate_heads import Replacements
    from model_io import setup

    setup()
    mlp = nn.Linear(4, 4, bias=False).cuda()
    model = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp)]))
    rep = Replacements.__new__(Replacements)
    rep.layer = 0
    rep.d = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")
    rep.normsq = rep.d.square().sum()
    rep.coefficient = lambda h, kind: h.sum(-1)
    prompt, decode = torch.randn(2, 5, 4, device="cuda"), torch.randn(2, 1, 4, device="cuda")
    expected_prompt, expected_decode = mlp(prompt), mlp(decode)
    with rep.hook(model, "student", all_tokens=True) as cache:
        patched_prompt, patched_decode = mlp(prompt), mlp(decode)
    torch.testing.assert_close(patched_prompt[:, -1, 0], prompt[:, -1].sum(-1))
    torch.testing.assert_close(patched_decode[:, -1, 0], decode[:, -1].sum(-1))
    torch.testing.assert_close(patched_prompt[:, :-1], expected_prompt[:, :-1], rtol=0, atol=0)
    torch.testing.assert_close(patched_decode[:, :, 1:], expected_decode[:, :, 1:], rtol=0, atol=0)
    torch.testing.assert_close(cache["input"], prompt[:, -1], rtol=0, atol=0)
    assert cache["patched_calls"] == 2
    with rep.hook(model, "student"):
        mlp(prompt)
        prefill_only_decode = mlp(decode)
    torch.testing.assert_close(prefill_only_decode, expected_decode, rtol=0, atol=0)
    with rep.hook(model, "restore", all_tokens=True):
        restored_prompt, restored_decode = mlp(prompt), mlp(decode)
    torch.testing.assert_close(restored_prompt, expected_prompt, rtol=0, atol=0)
    torch.testing.assert_close(restored_decode, expected_decode, rtol=0, atol=0)
    assert not mlp._forward_hooks
    stats = {
        "xmean": torch.zeros(4),
        "xstd": torch.ones(4),
        "ymean": torch.zeros(4),
        "yscale": torch.tensor(1.0),
    }
    counts = {}
    for kind in ["eml", "swiglu", "linear"]:
        student = Student(4, 4, kind, stats).cuda().double()
        x = torch.randn(2, 3, 4, device="cuda", dtype=torch.float64, requires_grad=True) * 0.2
        assert student(x).shape == x.shape
        assert torch.autograd.gradcheck(student, (x,), fast_mode=True)
        assert torch.autograd.gradgradcheck(student, (x,), fast_mode=True)
        counts[kind] = student.stored_coefficients()
    assert counts["eml"] == counts["swiglu"]
    print("GPU EXTENSION CHECKS PASSED", counts, flush=True)


if __name__ == "__main__":
    main()
