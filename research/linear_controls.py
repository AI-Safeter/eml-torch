"""Exact training least squares for linear controls, folded to a single input vector."""

import time

import torch

from data import ROOT, save
from heads import objective
from model_io import setup


def main():
    setup()
    for op in ["add", "multiply", "divide"]:
        out = ROOT / op
        validation_inputs = torch.load(out / "inputs-validation.pt", weights_only=True)["input"][
            :32, 1
        ].cuda()
        rows = []
        for feature in ["pls", "active"]:
            data = torch.load(
                out / ("features.pt" if feature == "pls" else "features-active.pt"),
                weights_only=True,
            )
            c = torch.load(
                out / ("component.pt" if feature == "pls" else "component-active.pt"),
                weights_only=True,
            )
            c = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in c.items()}
            for rank in [2, 4, 8, 16, 32]:
                x, y = data["train"]["x"][:, :rank].cuda(), data["train"]["y"].cuda()
                vx, vy = (
                    data["validation"]["x"][:, :rank].cuda(),
                    data["validation"]["y"].cuda(),
                )
                cases, vc = data["train"]["cases"], data["validation"]["cases"]
                a = torch.cat([torch.ones(len(x), 1, device="cuda"), x], 1)
                ar, yr = a.reshape(5, cases, -1), y.reshape(5, cases)
                design = torch.cat([a, 5**0.5 * (ar[1:] - ar[:1]).flatten(0, 1)]).double()
                target = torch.cat([y, 5**0.5 * (yr[1:] - yr[:1]).flatten()]).double()
                gram = design.T @ design
                gram.diagonal().add_(1e-4)
                weights = torch.linalg.solve(gram, design.T @ target)
                pred = weights[0].float() + vx @ weights[1:].float()
                value, absolute, response = objective(pred, vy, vc, 4)
                encoder = c["encoder"][:, :rank].double()
                scale = weights[1:] / c["zstd"][:rank].double()
                weight = c["ystd"].double() * (encoder @ scale)
                bias = c["ymean"].double() + c["ystd"].double() * (
                    weights[0]
                    - (c["input_mean"].double() @ encoder + c["zmean"][:rank].double()) @ scale
                )
                name = f"linear-{feature}-r{rank}"
                exported = validation_inputs @ weight.float() + bias.float()
                expected = pred[:32] * c["ystd"] + c["ymean"]
                torch.testing.assert_close(
                    exported, expected, atol=float(c["ystd"]) * 2e-4, rtol=2e-4
                )
                torch.save(
                    {"weight": weight.float().cpu(), "bias": bias.float().cpu()},
                    out / f"{name}.pt",
                )
                rows.append(
                    {
                        "name": name,
                        "feature": feature,
                        "rank": rank,
                        "validation_objective": float(value),
                        "validation_mse": float(absolute),
                        "validation_response_mse": float(response),
                        "stored_coefficients": weight.numel() + 1,
                    }
                )
        save(
            out / "linear-controls.json",
            {
                "completed_epoch": time.time(),
                "candidates": rows,
                "selected": min(rows, key=lambda r: r["validation_objective"]),
                "method": "Training least squares, value MSE plus 4 times response MSE; validation selects rank and feature basis. Folded to one input-weight vector and bias.",
                "late_control": (out / "selection.json").exists(),
                "test_outputs_used_to_fit": False,
            },
        )
        print(
            "LINEAR CONTROL",
            op,
            min(rows, key=lambda r: r["validation_objective"]),
            flush=True,
        )


if __name__ == "__main__":
    main()
