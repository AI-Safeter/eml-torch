"""Fit the frozen vector-student grid on CUDA; select using validation only."""

import copy
import json
import time

import torch
from runtime import RUNS, configure
from whole_student import Student


def main():
    configure("qwen17b")
    from data import save
    from model_io import setup

    setup(20260924)
    out = RUNS / "whole-block"
    assert (out / "collection.json").exists()
    raw = {
        split: {
            domain: torch.load(out / f"activations-{domain}-{split}.pt", weights_only=True)
            for domain in ["arithmetic", "language"]
        }
        for split in ["train", "validation"]
    }
    train = raw["train"]
    xm = sum(r["x"].double().mean(0) for r in train.values()) / 2
    xs = (
        (sum(r["x"].double().square().mean(0) for r in train.values()) / 2 - xm.square())
        .clamp_min(1e-12)
        .sqrt()
    )
    ym = sum(r["y"].double().mean(0) for r in train.values()) / 2
    ys = (
        (sum((r["y"].double() - ym).square().mean() for r in train.values()) / 2)
        .sqrt()
        .clamp_min(1e-6)
    )
    statistics = {
        "xmean": xm.float(),
        "xstd": xs.float(),
        "ymean": ym.float(),
        "yscale": ys.float(),
    }
    del xm, xs, ym, ys
    data = {}
    for split, domains in raw.items():
        data[split] = {}
        for domain, r in domains.items():
            data[split][domain] = {
                "x": ((r["x"] - statistics["xmean"]) / statistics["xstd"]).cuda(),
                "y": ((r["y"] - statistics["ymean"]) / statistics["yscale"]).cuda(),
            }
    del raw, train
    path = out / "candidates.json"
    records = json.loads(path.read_text()) if path.exists() else []
    completed = {r["name"] for r in records}
    for seed in [101, 211, 307]:
        for width in [128, 256, 512]:
            for kind in ["eml", "swiglu", "linear"]:
                name = f"{kind}-w{width}-s{seed}"
                if name in completed:
                    continue
                torch.manual_seed(seed)
                model = Student(len(statistics["xmean"]), width, kind, statistics).cuda()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
                generator = torch.Generator(device="cuda").manual_seed(seed + 100000)
                best, state, selected_step, stale = float("inf"), None, None, 0
                started = time.perf_counter()
                failed = False
                for step in range(20000):
                    losses = []
                    for r in data["train"].values():
                        ids = torch.randint(len(r["x"]), (512,), device="cuda", generator=generator)
                        losses.append((model.normalized(r["x"][ids]) - r["y"][ids]).square().mean())
                    loss = sum(losses) / 2
                    if not torch.isfinite(loss):
                        failed = True
                        break
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    if not torch.isfinite(norm):
                        failed = True
                        break
                    optimizer.step()
                    if step % 100 == 0:
                        with torch.no_grad():
                            val = {}
                            for domain, r in data["validation"].items():
                                total = sum(
                                    (model.normalized(x) - y).square().sum()
                                    for x, y in zip(r["x"].split(1024), r["y"].split(1024))
                                )
                                val[domain] = float(total / r["y"].numel())
                        objective = sum(val.values()) / 2
                        if objective < best:
                            best, state, selected_step, stale = (
                                objective,
                                copy.deepcopy(model.state_dict()),
                                step,
                                0,
                            )
                            validation = val
                        else:
                            stale += 100
                        if step >= 3000 and stale >= 2500:
                            break
                if state is not None:
                    torch.save({k: v.cpu() for k, v in state.items()}, out / f"{name}.pt")
                row = {
                    "name": name,
                    "kind": kind,
                    "width": width,
                    "seed": seed,
                    "status": "failed" if failed or state is None else "complete",
                    "steps": step + 1,
                    "selected_step": selected_step,
                    "validation_objective": best if state is not None else None,
                    "validation_domain_mse": validation if state is not None else None,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "stored_coefficients": model.stored_coefficients(),
                    "seconds": time.perf_counter() - started,
                }
                records.append(row)
                save(path, records)
                print("WHOLE FIT", row, flush=True)
                del model, optimizer, state
                torch.cuda.empty_cache()
    save(
        out / "fits-completed.json",
        {"candidates": len(records), "expected": 27, "epoch": time.time()},
    )


if __name__ == "__main__":
    main()
