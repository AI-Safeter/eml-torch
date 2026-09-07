"""Replay the earlier real-data distillation task with trainable EML heads."""

import copy
import json
import os
import time

import numpy as np
import torch

from data import ROOT, save
from emltorch import EMLHead
from heads import Head
from model_io import setup

SOURCE = ROOT / "airfoil-replay/source"
OUT = ROOT / "airfoil-replay"


def prediction(model, x):
    return model(x).reshape(-1)


def make_head(kind, width):
    return EMLHead(5, width, device="cuda") if kind == "eml_square" else Head(kind, 5, width).cuda()


def queries(x, n, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    a, b = [torch.randint(len(x), (n,), generator=generator, device="cuda") for _ in range(2)]
    alpha = torch.rand(n, 1, generator=generator, device="cuda")
    value = alpha * x[a] + (1 - alpha) * x[b]
    value += 0.05 * torch.randn(value.shape, generator=generator, device="cuda")
    return value.clamp(x.min(0).values, x.max(0).values)


def teacher_data(teacher, x):
    ys, gradients = [], []
    for chunk in x.split(512):
        h = chunk.detach().clone().requires_grad_(True)
        y = prediction(teacher, h)
        g = torch.autograd.grad(y.sum(), h)[0]
        ys.append(y.detach())
        gradients.append(g.detach())
    return torch.cat(ys), torch.cat(gradients)


def fit(kind, width, seed, gradient_weight, data):
    torch.manual_seed(seed)
    model = make_head(kind, width)
    x, y, gradient = data["x"], data["y"], data["gradient"]
    vx, vy = data["validation_x"], data["validation_y"]
    groups = data["validation_real_cases"]
    with torch.no_grad():
        a = torch.cat([torch.ones(len(x), 1, device="cuda"), x], 1).double()
        gram = a.T @ a
        gram.diagonal().add_(0.001)
        w = torch.linalg.solve(gram, a.T @ y.double()).float()
        model.skip.weight.copy_(w[1:][None])
        model.skip.bias.copy_(w[:1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    generator = torch.Generator(device="cuda").manual_seed(seed + 100000)
    scale = gradient.square().mean().clamp_min(1e-8)
    best, state, stale = float("inf"), None, 0
    started = time.perf_counter()
    for step in range(8000):
        ids = torch.randint(len(x), (1024,), generator=generator, device="cuda")
        batch = x[ids].detach().requires_grad_(bool(gradient_weight))
        output = prediction(model, batch)
        loss = (output - y[ids]).square().mean()
        if gradient_weight:
            derivative = torch.autograd.grad(output.sum(), batch, create_graph=True)[0]
            loss = loss + gradient_weight * (derivative - gradient[ids]).square().mean() / scale
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            break
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        if not torch.isfinite(norm):
            break
        optimizer.step()
        if step % 100 == 0:
            with torch.no_grad():
                error = (prediction(model, vx) - vy).square()
                real_mse, query_mse = (
                    float(error[:groups].mean()),
                    float(error[groups:].mean()),
                )
                value = 0.5 * (real_mse + query_mse)
            if value < best:
                best, stale = value, 0
                state = copy.deepcopy(model.state_dict())
                selected_step = step
                selected_real, selected_query = real_mse, query_mse
            else:
                stale += 100
            if step >= 3000 and stale >= 2000:
                break
    assert state is not None
    model.load_state_dict(state)
    return model, {
        "kind": kind,
        "width": width,
        "seed": seed,
        "gradient_weight": gradient_weight,
        "validation_mse": best,
        "validation_real_mse": selected_real,
        "validation_query_mse": selected_query,
        "selected_step": selected_step,
        "steps": step + 1,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": time.perf_counter() - started,
    }


def metrics(pred, teacher, target, group_ids):
    pred, teacher, target = pred.double(), teacher.double(), target.double()
    groups = int(group_ids.max()) + 1
    count = torch.bincount(group_ids, minlength=groups).double()
    error, base = (pred - target).square(), (teacher - target).square()
    sums = torch.zeros(groups, device="cuda", dtype=torch.float64).scatter_add_(0, group_ids, error)
    base_sums = torch.zeros_like(sums).scatter_add_(0, group_ids, base)
    indices = torch.randint(groups, (5000, groups), device="cuda")
    ratios = sums[indices].sum(1) / base_sums[indices].sum(1).clamp_min(1e-12)
    return {
        "rmse": float(error.mean().sqrt()),
        "teacher_rmse": float(base.mean().sqrt()),
        "teacher_approximation_rmse": float((pred - teacher).square().mean().sqrt()),
        "mse_ratio_to_teacher": float(error.mean() / base.mean()),
        "mse_ratio_ci95": torch.quantile(
            ratios, torch.tensor([0.025, 0.975], device="cuda", dtype=torch.float64)
        ).tolist(),
        "operating_groups": groups,
        "cases": int(count.sum()),
        "finite": bool(torch.isfinite(pred).all()),
    }


def main():
    setup(seed=20260918)
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "candidates.json").exists(), "Preserve the previous search"
    array = np.loadtxt(SOURCE / "airfoil_self_noise.dat")
    splits = json.loads((SOURCE / "split-indices.json").read_text())
    teacher_state = torch.load(SOURCE / "network.pt", weights_only=True, map_location="cpu")
    teacher = (
        torch.nn.Sequential(
            torch.nn.Linear(5, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 1),
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    teacher.load_state_dict(teacher_state["state"])
    mean, std = teacher_state["xmean"].cuda(), teacher_state["xstd"].cuda()
    ym, ys = teacher_state["ymean"].cuda(), teacher_state["ystd"].cuda()
    raw_x = torch.tensor(array[:, :5], device="cuda", dtype=torch.float32)
    normalized = (raw_x - mean) / std
    training, validation = normalized[splits["train"]], normalized[splits["validation"]]
    x = torch.cat([training, queries(training, 8192, 20260918)])
    vx = torch.cat([validation, queries(training, 2048, 20260919)])
    y, gradient = teacher_data(teacher, x)
    vy, _ = teacher_data(teacher, vx)
    data = {
        "x": x,
        "y": y,
        "gradient": gradient,
        "validation_x": vx,
        "validation_y": vy,
        "validation_real_cases": len(validation),
    }
    grid = [
        (kind, width, seed, weight)
        for width in [4, 8, 16, 32]
        for seed in [17, 29]
        for weight in [0.0, 0.1]
        for kind in ["eml_square", "eml_log", "silu"]
    ]
    save(
        OUT / "protocol.json",
        {
            "created_epoch": time.time(),
            "pid": os.getpid(),
            "gpu": torch.cuda.get_device_name(),
            "grid": grid,
            "steps_max": 8000,
            "teacher_frozen": True,
            "training_queries": len(x),
            "validation_queries": len(vx),
            "selection": "Smallest EML head with validation MSE <= .0025, otherwise minimum validation MSE. Objective averages teacher-approximation MSE on original validation operating groups and independent synthetic teacher queries. No ground-truth test labels used for selection.",
            "test_reuse": "The earlier airfoil test and velocity-shift sets were already evaluated. This is an explicitly labeled replay and improvement study, not a fresh data replication.",
            "synthetic_queries": "Convex combinations of training inputs plus small noise, clipped to training bounds; targets and derivatives supplied by the frozen teacher.",
            "source": str(SOURCE),
            "success": "Test MSE <= 1.05 times teacher MSE and fewer stored coefficients",
        },
    )
    rows = []
    for kind, width, seed, weight in grid:
        model, row = fit(kind, width, seed, weight, data)
        name = f"{kind}-w{width}-s{seed}-g{weight:g}"
        row["name"] = name
        torch.save(model.state_dict(), OUT / f"{name}.pt")
        rows.append(row)
        save(OUT / "candidates.json", rows)
        print("AIRFOIL", name, row, flush=True)
    eml = [r for r in rows if r["kind"].startswith("eml")]
    eligible = [r for r in eml if r["validation_mse"] <= 0.0025]
    best = (
        min(eligible, key=lambda r: (r["parameters"], r["validation_mse"]))
        if eligible
        else min(eml, key=lambda r: r["validation_mse"])
    )
    neural = min([r for r in rows if r["kind"] == "silu"], key=lambda r: r["validation_mse"])
    save(
        OUT / "selection.json",
        {
            "frozen_epoch": time.time(),
            "eml": best,
            "neural": neural,
            "validation_fidelity_threshold_met": bool(eligible),
        },
    )
    results = {}
    predictions = {}
    with torch.inference_mode():
        baseline = prediction(teacher, normalized) * ys + ym
        for label, spec in [("eml", best), ("neural", neural)]:
            model = make_head(spec["kind"], spec["width"]).eval()
            model.load_state_dict(torch.load(OUT / (spec["name"] + ".pt"), weights_only=True))
            pred = prediction(model, normalized) * ys + ym
            results[label] = {}
            for split, ids in splits.items():
                _, inverse = torch.unique(raw_x[ids, 1:5], dim=0, return_inverse=True)
                results[label][split] = metrics(
                    pred[ids],
                    baseline[ids],
                    torch.tensor(array[ids, 5], device="cuda"),
                    inverse,
                )
            predictions[label] = pred.cpu()
    torch.save(
        {
            "predictions": predictions,
            "teacher": baseline.cpu(),
            "teacher_state": teacher_state,
            "splits": splits,
        },
        OUT / "predictions.pt",
    )
    save(
        OUT / "results.json",
        {
            "metrics": results,
            "teacher_parameters": sum(p.numel() for p in teacher.parameters()),
            "eml_parameters": best["parameters"],
            "completed_epoch": time.time(),
            "test_reused": True,
        },
    )
    print("AIRFOIL REPLAY DONE", results, flush=True)


if __name__ == "__main__":
    main()
