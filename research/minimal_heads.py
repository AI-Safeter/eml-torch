"""Bounded low-rank, few-node head search for a fresh-operand confirmation."""

import argparse
import json
import os
import time

import torch

from data import ROOT, save
from model_io import setup
from train_heads import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["add", "multiply", "divide"])
    parser.add_argument("--max-seconds", type=int, default=2400)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--deadline-epoch", type=float, default=None)
    args = parser.parse_args()
    setup()
    out = ROOT / args.operation / "minimal-heads"
    out.mkdir(exist_ok=True)
    assert args.resume or not (out / "candidates.json").exists(), (
        "Do not overwrite a started search"
    )
    data = torch.load(out.parent / "features-active.pt", weights_only=True)
    dimension = torch.load(out.parent / "component-active.pt", weights_only=True)["encoder"].shape[
        0
    ]
    protocol = json.loads((ROOT / "minimal-protocol.json").read_text())
    grid = [
        (rank, width, seed, kind)
        for rank in protocol["ranks"]
        for width in protocol["widths"]
        for seed in protocol["seeds"]
        for kind in protocol["kinds"]
    ]
    stop = time.time() + args.max_seconds
    if args.deadline_epoch is not None:
        stop = min(stop, args.deadline_epoch)
    save(
        out / "job.json",
        {
            "pid": os.getpid(),
            "start_epoch": time.time(),
            "stop_new_candidates_epoch": stop,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "grid_candidates": len(grid),
        },
    )
    rows = json.loads((out / "candidates.json").read_text()) if args.resume else []
    existing = {row["name"] for row in rows}
    for rank, width, seed, kind in grid:
        name = f"r{rank}-{kind}-w{width}-s{seed}"
        if name in existing:
            continue
        if time.time() >= stop:
            break
        current = {split: {**values, "x": values["x"][:, :rank]} for split, values in data.items()}
        model, row = train(kind, width, seed, 4.0, current, max_steps=protocol["max_steps"])
        row.update(
            name=name,
            encoder_coefficients=dimension * rank,
            stored_coefficients=dimension * rank + row.get("parameters", 0),
        )
        if model is not None:
            torch.save(model.state_dict(), out / f"{name}.pt")
        rows.append(row)
        save(out / "candidates.json", rows)
        print(
            args.operation,
            name,
            row.get("validation_objective"),
            row.get("seconds"),
            flush=True,
        )
    save(
        out / "completed.json",
        {
            "completed_epoch": time.time(),
            "candidates": len(rows),
            "grid_total": len(grid),
            "complete_grid": len(rows) == len(grid),
        },
    )
    print("MINIMAL SEARCH DONE", args.operation, flush=True)


if __name__ == "__main__":
    main()
