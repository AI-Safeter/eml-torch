"""Bounded multi-seed head search; checkpoints selected exclusively by validation."""

import argparse
import copy
import json
import os
import time

import torch

from data import ROOT, save
from heads import Head, objective, parameter_count
from model_io import setup


def train(kind, width, seed, response_weight, data, max_steps=20000, gradient_weight=0.0):
    torch.manual_seed(seed)
    x, y = data["train"]["x"].cuda(), data["train"]["y"].cuda()
    vx, vy = data["validation"]["x"].cuda(), data["validation"]["y"].cuda()
    cases = data["train"]["cases"]
    vc = data["validation"]["cases"]
    grad_target = data["train"]["gradient"].cuda() if gradient_weight else None
    grad_scale = grad_target.square().mean().clamp_min(1e-8) if gradient_weight else None
    model = Head(kind, x.shape[1], width).cuda()
    model.initialize_linear(x, y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    batches = torch.Generator(device="cuda").manual_seed(seed + 100000)
    best = float("inf")
    state = None
    record = None
    stale = 0
    bad_steps = 0
    started = time.perf_counter()
    for step in range(max_steps):
        ids = torch.randint(cases, (min(512, cases),), device="cuda", generator=batches)
        indices = torch.cat([ids + a * cases for a in range(5)])
        batch = x[indices].detach().requires_grad_(bool(gradient_weight))
        pred = model(batch)
        loss, _, _ = objective(pred, y[indices], len(ids), response_weight)
        if gradient_weight:
            derivative = torch.autograd.grad(pred.sum(), batch, create_graph=True)[0]
            loss = (
                loss
                + gradient_weight * (derivative - grad_target[indices]).square().mean() / grad_scale
            )
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            bad_steps += 1
            if state is None or bad_steps > 5:
                break
            model.load_state_dict(state)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
            continue
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            bad_steps += 1
            if bad_steps > 5:
                break
            continue
        optimizer.step()
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                val, absolute, response = objective(model(vx), vy, vc, response_weight)
                value = float(val)
            if value < best:
                best = value
                state = copy.deepcopy(model.state_dict())
                stale = 0
                record = {
                    "validation_objective": value,
                    "validation_mse": float(absolute),
                    "validation_response_mse": float(response),
                    "selected_step": step,
                }
            else:
                stale += 100
            model.train()
            if step >= 3000 and stale >= 2500:
                break
    if state is None:
        return None, {
            "status": "failed",
            "kind": kind,
            "width": width,
            "seed": seed,
            "response_weight": response_weight,
            "bad_steps": bad_steps,
        }
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    return model, {
        "status": "complete",
        "kind": kind,
        "width": width,
        "seed": seed,
        "response_weight": response_weight,
        "gradient_weight": gradient_weight,
        "rank": x.shape[1],
        "parameters": parameter_count(model),
        "eml_nodes": width if kind.startswith("eml") else 0,
        "steps": step + 1,
        "bad_steps": bad_steps,
        "seconds": time.perf_counter() - started,
        **record,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["add", "multiply", "divide"])
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--deadline-epoch", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--features", choices=["pls", "active"], default="pls")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--gradient-weight", type=float, default=0.0)
    parser.add_argument("--widths", type=int, nargs="+", default=[32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 211, 307, 401, 503])
    parser.add_argument(
        "--kinds", nargs="+", choices=["eml_square", "silu_two"], default=["eml_square", "silu_two"]
    )
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_seconds < 0 or not 1 <= args.rank <= 32:
        parser.error("max-steps must be positive, max-seconds nonnegative, and rank in 1..32")
    if any(width < 1 for width in args.widths) or args.gradient_weight < 0:
        parser.error("Widths must be positive and gradient-weight nonnegative")
    setup()
    tag = (
        "heads"
        if args.features == "pls"
        else f"heads-active-r{args.rank}-g{args.gradient_weight:g}"
    )
    out = ROOT / args.operation / tag
    out.mkdir(parents=True, exist_ok=True)
    name = "features.pt" if args.features == "pls" else "features-active.pt"
    data = torch.load(ROOT / args.operation / name, weights_only=True)
    for split in data:
        data[split]["x"] = data[split]["x"][:, : args.rank]
    if args.gradient_weight:
        assert args.features == "active"
        grads = torch.load(ROOT / args.operation / "gradients-active.pt", weights_only=True)
        for split in data:
            data[split]["gradient"] = grads[split]["gradient"][:, : args.rank]
    end = time.time() + args.max_seconds
    if args.deadline_epoch is not None:
        end = min(end, args.deadline_epoch)
    assert args.resume or not (out / "candidates.json").exists(), (
        "Use --resume to preserve a started search"
    )
    save(
        out / "job.json",
        {
            "pid": os.getpid(),
            "start_epoch": time.time(),
            "stop_new_candidates_epoch": end,
            "gpu": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    grid = [
        (kind, width, seed, 4.0)
        for seed in args.seeds
        for width in args.widths
        for kind in args.kinds
    ]
    protocol = {
        "grid": grid,
        "steps_max": args.max_steps,
        "batch_cases": 512,
        "checkpoint_selection": "Validation absolute coefficient MSE + response_weight times validation coefficient-response MSE",
        "primary_objective_weight": 4.0,
        "seed_order": args.seeds,
        "families": "EML squared log arguments and a parameter-matched two-layer SiLU control.",
        "time_limit_seconds": args.max_seconds,
        "checkpoint_cadence_steps": 100,
        "features": args.features,
        "rank": args.rank,
        "gradient_weight": args.gradient_weight,
    }
    protocol_path = out / "protocol.json"
    if args.resume and protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        current = json.loads(json.dumps(protocol))
        for field in ["grid", "steps_max", "features", "rank", "gradient_weight"]:
            assert field not in previous or previous[field] == current[field], (
                f"Resume would change {field}"
            )
    else:
        save(protocol_path, protocol)
    rows = (
        json.loads((out / "candidates.json").read_text())
        if args.resume and (out / "candidates.json").exists()
        else []
    )
    existing = {row["name"] for row in rows}
    for kind, width, seed, weight in grid:
        if time.time() >= end:
            break
        name = f"{kind}-w{width}-s{seed}-r{int(weight)}"
        if name in existing:
            continue
        model, row = train(kind, width, seed, weight, data, args.max_steps, args.gradient_weight)
        row["name"] = name
        if model is not None:
            torch.save(model.state_dict(), out / f"{name}.pt")
        rows.append(row)
        save(out / "candidates.json", rows)
        print(
            args.operation,
            name,
            {
                k: row.get(k)
                for k in [
                    "status",
                    "parameters",
                    "validation_mse",
                    "validation_response_mse",
                    "seconds",
                ]
            },
            flush=True,
        )
    save(
        out / "completed.json",
        {
            "finished_epoch": time.time(),
            "candidates": len(rows),
            "grid_total": len(grid),
            "status": "complete_grid" if len(rows) == len(grid) else "time_limited",
        },
    )
    print("HEAD SEARCH DONE", args.operation, flush=True)


if __name__ == "__main__":
    main()
