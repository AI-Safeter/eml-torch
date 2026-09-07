"""Training-only sparse original-neuron controls with matched encoder storage."""

import argparse
import os
import time

import torch

from data import ROOT, save
from heads import objective
from model_io import load_mlp, setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["add", "multiply", "divide"])
    args = parser.parse_args()
    setup()
    out = ROOT / args.operation
    save(
        out / "sparse-job.json",
        {
            "pid": os.getpid(),
            "start_epoch": time.time(),
            "gpu": torch.cuda.get_device_name(),
        },
    )
    c = torch.load(out / "component.pt", weights_only=True)
    mlp = load_mlp(c["layer"])
    data = torch.load(out / "features.pt", weights_only=True)
    activities = {}
    with torch.inference_mode():
        for split in ["train", "validation"]:
            h = torch.load(out / f"inputs-{split}.pt", weights_only=True)["input"].cuda()
            mixed = torch.cat(
                [(1 - alpha) * h[:, 1] + alpha * h[:, 0] for alpha in [0, 0.25, 0.5, 0.75, 1]]
            )
            activities[split] = torch.cat(
                [
                    torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
                    for x in mixed.split(128)
                ]
            )
        mean, std = (
            activities["train"].mean(0),
            activities["train"].std(0).clamp_min(1e-6),
        )
        a = (activities["train"] - mean) / std
        v = (activities["validation"] - mean) / std
        y, vy = data["train"]["y"].cuda(), data["validation"]["y"].cuda()
        cases, vc = data["train"]["cases"], data["validation"]["cases"]
        ar, yr = a.reshape(5, cases, -1), y.reshape(5, cases)
        # The augmented least-squares mean differs by a constant factor only.
        aug = torch.cat([a, 5**0.5 * (ar[1:] - ar[:1]).flatten(0, 1)])
        target = torch.cat([y, 5**0.5 * (yr[1:] - yr[:1]).flatten()])
        intercept = torch.cat(
            [torch.ones(len(a), device="cuda"), torch.zeros(4 * cases, device="cuda")]
        )
        norm = aug.square().sum(0).sqrt().clamp_min(1e-8)
        residual = target.clone()
        chosen = []
        candidates = []
        for step in range(64):
            score = (aug.T @ residual).abs() / norm
            score[chosen] = -1
            chosen.append(int(score.argmax()))
            design = torch.cat([intercept[:, None], aug[:, chosen]], 1).double()
            gram = design.T @ design
            gram.diagonal().add_(1e-4)
            weights = torch.linalg.solve(gram, design.T @ target.double()).float()
            residual = target - design.float() @ weights
            if step + 1 not in [4, 8, 16, 32, 64]:
                continue
            prediction = weights[0] + v[:, chosen] @ weights[1:]
            value, absolute, response = objective(prediction, vy, vc, 4)
            readout = weights[1:] * c["ystd"].cuda() / std[chosen]
            bias = c["ymean"].cuda() + c["ystd"].cuda() * weights[0] - readout @ mean[chosen]
            state = {
                "gate": mlp.gate_proj.weight[chosen].cpu(),
                "up": mlp.up_proj.weight[chosen].cpu(),
                "readout": readout.cpu(),
                "bias": bias.cpu(),
                "neuron_indices": torch.tensor(chosen),
            }
            exported = (
                torch.nn.functional.silu(mixed[:32] @ state["gate"].cuda().T)
                * (mixed[:32] @ state["up"].cuda().T)
            ) @ readout + bias
            expected = prediction[:32] * c["ystd"].cuda() + c["ymean"].cuda()
            torch.testing.assert_close(exported, expected, atol=0.003, rtol=2e-4)
            torch.save(state, out / f"neurons-{step + 1}.pt")
            row = {
                "neurons": step + 1,
                "indices": chosen.copy(),
                "validation_objective": float(value),
                "validation_mse": float(absolute),
                "validation_response_mse": float(response),
                "stored_coefficients": sum(
                    state[k].numel() for k in ["gate", "up", "readout", "bias"]
                ),
            }
            candidates.append(row)
            print("SPARSE", args.operation, row, flush=True)
    best = min(candidates, key=lambda r: r["validation_objective"])
    save(
        out / "sparse-neurons.json",
        {
            "completed_epoch": time.time(),
            "candidates": candidates,
            "selected_neurons": best["neurons"],
            "matched_budget_neurons": 16,
            "method": "Training OMP over original SwiGLU neuron activations; training least squares includes coefficient responses. Validation selects among 4/8/16/32/64 neurons.",
            "parameters": "Two original input weight vectors per neuron plus a refitted scalar readout and bias. Sixteen neurons have approximately the same storage as a rank-32 learned head encoder.",
            "test_data_used": False,
        },
    )
    print("SPARSE DONE", args.operation, flush=True)


if __name__ == "__main__":
    main()
