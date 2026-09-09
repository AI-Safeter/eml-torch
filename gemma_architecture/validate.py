"""CUDA numerical and actual-model removal checks before fitting."""

import argparse
import copy

import torch

from .common import accounting, load, root, save, setup, sources, text_layers
from .model import Deployed, Replacement, StructuredProjection


def units():
    records = []
    for groups in [1, 2]:
        layer = StructuredProjection(16, 3, groups).cuda().double()
        x = torch.randn(2, 5, 16, device="cuda", dtype=torch.float64, requires_grad=True)
        y = layer(x)
        dense = torch.einsum("...d,god->...go", x, layer.matrices()) + layer.bias
        if groups == 1:
            dense = dense.squeeze(-2)
        torch.testing.assert_close(y, dense, atol=1e-12, rtol=1e-12)
        v = torch.randn_like(y)
        torch.testing.assert_close(
            torch.autograd.grad(y, x, v)[0],
            torch.autograd.grad(dense, x, v)[0],
            atol=1e-12,
            rtol=1e-12,
        )
    for arch in ["bottleneck", "shortcut", "structured"]:
        for act in ["eml", "silu"]:
            for depth in [1, 2]:
                net = Replacement(arch, act, depth).cuda()
                with torch.no_grad():
                    net.xmean.normal_(0, 0.1)
                    net.xstd.uniform_(0.7, 1.3)
                    net.ymean.normal_(0, 0.1)
                    net.yscale.fill_(1.7)
                    for stage in net.stages:
                        if arch == "structured":
                            stage.readout.up.normal_(0, 0.001)
                            stage.readout.diagonal.normal_(0, 0.01)
                        else:
                            stage.readout.weight.normal_(0, 0.001)
                        stage.readout.bias.normal_(0, 0.01)
                x = torch.randn(7, 1536, device="cuda")
                y = net(x)
                y.square().mean().backward()
                assert all(
                    p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()
                )
                deployed = Deployed(net).cuda()
                with torch.no_grad():
                    reference = y.detach()
                    error = float(
                        (deployed(x) - reference).square().mean() / reference.square().mean()
                    )
                    assert error < 1e-10, (arch, act, depth, error)
                    bf = copy.deepcopy(deployed).bfloat16()(x.bfloat16()).float()
                    bferror = float((bf - reference).square().mean() / reference.square().mean())
                records.append(
                    {
                        "architecture": arch,
                        "activation": act,
                        "depth": depth,
                        "training": accounting(net),
                        "deployed": accounting(deployed),
                        "folding_relative_mse_fp32": error,
                        "bf16_vs_fp32_relative_mse": bferror,
                    }
                )
                del net, deployed, y, x
                torch.cuda.empty_cache()
    # Covariance rank is not an information-preservation certificate.
    x = torch.randn(512, 8, device="cuda", dtype=torch.float64)
    powers = torch.stack([x[:, 0] ** k for k in range(1, 9)], -1)
    assert int(torch.linalg.matrix_rank(powers - powers.mean(0))) == 8
    assert torch.equal(powers, torch.stack([x[:, 0] ** k for k in range(1, 9)], -1))
    return records


def integration():
    model = load()
    assert accounting(model)["parameters"] == 5104297504
    layer = text_layers(model)[26]
    original = layer.mlp
    ids_original = {id(p) for p in original.parameters()}
    original.cpu()

    def forbidden(*a, **kw):
        raise AssertionError("Original MLP executed")

    original.forward = forbidden
    records = []
    ids = torch.tensor([[model.config.text_config.bos_token_id, 100, 101, 102]], device="cuda")
    for arch in ["bottleneck", "shortcut", "structured"]:
        for act in ["eml", "silu"]:
            layer.mlp = Deployed(Replacement(arch, act)).cuda().bfloat16().eval()
            assert not ids_original.intersection(id(p) for p in model.parameters())
            with torch.inference_mode():
                result = model(input_ids=ids, use_cache=True, logits_to_keep=1)
                assert torch.isfinite(result.logits).all()
                result = model(
                    input_ids=ids[:, -1:],
                    past_key_values=result.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
                assert torch.isfinite(result.logits).all()
            records.append(
                {
                    "architecture": arch,
                    "activation": act,
                    "model": accounting(model),
                    "original_calls": 0,
                    "cached_decode_finite": True,
                }
            )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", action="store_true")
    args = parser.parse_args()
    setup(712)
    data = integration() if args.integration else units()
    save(
        root() / ("integration.json" if args.integration else "validation.json"),
        {
            "checks": data,
            "sources": sources("validate.py", "model.py"),
            "gpu": torch.cuda.get_device_name(),
        },
    )
    print("CUDA VALIDATION COMPLETE", len(data), flush=True)


if __name__ == "__main__":
    main()
