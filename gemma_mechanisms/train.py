"""Train the preregistered complete-MLP grid without loading a teacher model."""

import argparse
import gc
import json
import time

import torch

from .runtime import HERE, SPEC, accounting, digest, root, save, setup
from .student import Student, contribution


def moments(domains):
    means = {}
    for name in ["x", "y", "contribution"]:
        avg, square = 0, 0
        for data in domains.values():
            total, ss = 0, 0
            for block in data[name].split(2048):
                block = block.cuda().double()
                total = total + block.sum(0)
                ss = ss + block.square().sum(0)
            avg = avg + total / len(data[name]) / len(domains)
            square = square + ss / len(data[name]) / len(domains)
        means[name] = (avg, square)
    xm, xx = means["x"]
    ym, yy = means["y"]
    statistics = {
        "xmean": xm.float(),
        "xstd": (xx - xm.square()).clamp_min(1e-10).sqrt().float(),
        "ymean": ym.float(),
        "yscale": (yy - ym.square()).clamp_min(1e-10).mean().sqrt().float(),
    }
    cscale = means["contribution"][1].mean().sqrt().float()
    return statistics, cscale


def initialization(domains, stats):
    dim = len(stats["xmean"])
    xx, xy, yy = [torch.zeros(dim, dim, device="cuda", dtype=torch.float64) for _ in range(3)]
    with torch.no_grad():
        for data in domains.values():
            for x, y in zip(data["x"].split(2048), data["y"].split(2048)):
                x = ((x.cuda().float() - stats["xmean"]) / stats["xstd"]).double()
                y = ((y.cuda().float() - stats["ymean"]) / stats["yscale"]).double()
                scale = 1 / len(data["x"]) / len(domains)
                xx.addmm_(x.T, x, alpha=scale)
                xy.addmm_(x.T, y, alpha=scale)
                yy.addmm_(y.T, y, alpha=scale)
        ridge = torch.linalg.solve(xx + torch.eye(dim, device="cuda") * 0.001, xy)
        eigenvalues, basis = torch.linalg.eigh(yy)
    return {
        "ridge": ridge.float().cpu(),
        "basis": basis.flip(1).float().cpu(),
        "eigenvalues": eigenvalues.flip(0).cpu(),
    }


def initialize(model, state):
    basis = state["basis"][:, : model.width].cuda()
    encoder = state["ridge"].cuda() @ basis
    with torch.no_grad():
        model.encoder.weight.copy_(encoder.T)
        model.encoder.bias.zero_()
        model.decoder.weight.copy_(basis)
        model.decoder.bias.zero_()


def objective(model, x, y, c, norm, cscale):
    yn = model.normalized((x - model.xmean) / model.xstd)
    raw = (yn - (y - model.ymean) / model.yscale).square().mean()
    predicted_c = contribution(yn * model.yscale + model.ymean, norm["weight"], norm["eps"])
    normalized = ((predicted_c - c) / cscale).square().mean()
    return raw + normalized, raw, normalized


@torch.no_grad()
def evaluate(model, domains, norm, cscale):
    record = {}
    model.eval()
    for name, data in domains.items():
        sums = torch.zeros(3, device="cuda", dtype=torch.float64)
        for x, y, c in zip(
            data["x"].split(512), data["y"].split(512), data["contribution"].split(512)
        ):
            terms = objective(
                model, x.cuda().float(), y.cuda().float(), c.cuda().float(), norm, cscale
            )
            sums += torch.stack(terms).double() * len(x)
        values = (sums / len(data["x"])).tolist()
        record[name] = dict(zip(["objective", "raw_mse", "contribution_mse"], values))
    model.train()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument(
        "--candidate", help="Run one named grid member; scientific policy is unchanged"
    )
    args = parser.parse_args()
    setup(SPEC["data_seed"])
    out = root(args.output)
    collection = out / "collection"
    assert (collection / "complete.json").exists()
    directory = out / "training"
    directory.mkdir(exist_ok=True)
    paths = [
        collection / f"{domain}-{split}.pt"
        for split in ["train", "selection"]
        for domain in ["arithmetic", "language"]
    ]
    paths += [
        collection / "native-postnorm.pt",
        collection / "freeze.json",
        out / "data/freeze.json",
    ]
    freeze = {
        "sources": {
            name: digest(HERE / name)
            for name in ["student.py", "train.py", "runtime.py", "protocol.json", "PROTOCOL.md"]
        },
        "inputs": {str(path.relative_to(out)): digest(path) for path in paths},
    }
    if (directory / "freeze.json").exists():
        assert json.loads((directory / "freeze.json").read_text()) == freeze
    else:
        save(directory / "freeze.json", freeze)
    raw = {
        split: {
            domain: torch.load(collection / f"{domain}-{split}.pt", weights_only=True)
            for domain in ["arithmetic", "language"]
        }
        for split in ["train", "selection"]
    }
    norm = torch.load(collection / "native-postnorm.pt", weights_only=True)
    norm["weight"] = norm["weight"].cuda()
    preparation_started = time.perf_counter()
    statistics, cscale = moments(raw["train"])
    init_path = directory / "initialization.pt"
    if init_path.exists():
        init = torch.load(init_path, weights_only=True)
    else:
        init = initialization(raw["train"], statistics)
        torch.save(init, init_path)
    torch.cuda.synchronize()
    save(
        directory / "initialization.json",
        {
            "seconds": time.perf_counter() - preparation_started,
            "fixed_training_output_subspace_residual": {
                str(width): float(init["eigenvalues"][width:].sum() / init["eigenvalues"].sum())
                for width in [256, 512, 976, 1536]
            },
            "note": "Fixed training eigenspaces only; learned student decoder may move. No teacher at inference.",
        },
    )
    spec = SPEC["replacement"]
    grid = [
        (seed, budget, depth, kind)
        for seed in spec["seeds"]
        for budget in spec["budgets"]
        for depth in spec["depths"]
        for kind in spec["families"]
        if kind != "linear" or depth == 1
    ]
    for seed, budget, depth, kind in grid:
        name = f"{kind}-b{budget['coefficients']}-d{depth if kind != 'linear' else 0}-s{seed}"
        if args.candidate and name != args.candidate:
            continue
        result_path = directory / f"{name}.json"
        if result_path.exists():
            print("PRESERVE FIT", name, flush=True)
            continue
        setup(seed)
        model = Student(
            1536, budget["coefficients"], budget["bottleneck"], depth, kind, statistics
        ).cuda()
        initialize(model, init)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"]
        )
        generator = torch.Generator().manual_seed(seed + 100000)
        best, best_state, stale, selected_step = float("inf"), None, 0, None
        curves, grad_clips, grad_steps, failed = [], 0, 0, None
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        event_start, event_end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        event_start.record()
        for step in range(spec["max_steps"]):
            for stage in model.stages:
                stage.diagnostics = step % spec["validation_interval"] == 0
            losses = []
            for data in raw["train"].values():
                ids = torch.randint(
                    len(data["x"]), (spec["batch_per_domain"],), generator=generator
                )
                loss, _, _ = objective(
                    model,
                    *(data[k][ids].cuda().float() for k in ["x", "y", "contribution"]),
                    norm,
                    cscale,
                )
                losses.append(loss)
            loss = sum(losses) / len(losses)
            if not torch.isfinite(loss):
                failed = "nonfinite loss"
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), spec["clip_gradient_norm"])
            grad_steps += 1
            grad_clips += int(grad > spec["clip_gradient_norm"])
            if not torch.isfinite(grad):
                failed = "nonfinite gradient"
                break
            optimizer.step()
            if (step + 1) % spec["validation_interval"] == 0:
                for stage in model.stages:
                    stage.diagnostics = False
                validation = evaluate(model, raw["selection"], norm, cscale)
                value = sum(r["objective"] for r in validation.values()) / len(validation)
                if value < best:
                    best, selected_step, stale = value, step + 1, 0
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                else:
                    stale += spec["validation_interval"]
                curves.append(
                    {
                        "step": step + 1,
                        "train_batch_objective": float(loss),
                        "selection": validation,
                        "gradient_norm": float(grad),
                    }
                )
                if (step + 1) % 500 == 0:
                    print("FIT", name, step + 1, value, flush=True)
                if step + 1 >= spec["minimum_steps"] and stale >= spec["patience_steps"]:
                    break
        event_end.record()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        result = {
            "name": name,
            "specification": model.specification(),
            "seed": seed,
            "status": "failed" if failed or best_state is None else "complete",
            "failure": failed,
            "steps": step + 1,
            "selected_step": selected_step,
            "selection_objective": best if best_state else None,
            "training_seconds": elapsed,
            "cuda_elapsed_ms": event_start.elapsed_time(event_end),
            "training_tokens": grad_steps * spec["batch_per_domain"] * 2,
            "gradient_clipped_steps": grad_clips,
            "gradient_steps": grad_steps,
            "exponent_diagnostics": [
                {
                    "clamps": s.clamps,
                    "sampled_arguments": s.arguments_seen,
                    "max_abs_argument": s.max_argument,
                }
                for s in model.stages
            ],
            "accounting": accounting(model),
            "curves": curves,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "gpu": torch.cuda.get_device_name(),
            "training_freeze_sha256": digest(directory / "freeze.json"),
        }
        if best_state is not None:
            torch.save(
                {"specification": model.specification(), "state": best_state},
                directory / f"{name}.pt",
            )
            result["checkpoint_sha256"] = digest(directory / f"{name}.pt")
        save(result_path, result)
        print("FIT COMPLETE", name, result["status"], best, elapsed, flush=True)
        del model, optimizer, best_state, loss, losses
        gc.collect()
        torch.cuda.empty_cache()
    if not args.candidate:
        records = [
            json.loads(
                (
                    directory
                    / f"{kind}-b{budget['coefficients']}-d{depth if kind != 'linear' else 0}-s{seed}.json"
                ).read_text()
            )
            for seed, budget, depth, kind in grid
        ]
        selection = {}
        for kind in spec["families"]:
            groups = {}
            for record in records:
                if record["specification"]["kind"] != kind:
                    continue
                key = (record["specification"]["budget"], record["specification"]["depth"])
                groups.setdefault(key, []).append(record)
            complete = {
                key: rs
                for key, rs in groups.items()
                if len(rs) == len(spec["seeds"]) and all(r["status"] == "complete" for r in rs)
            }
            if not complete:
                selection[kind] = {"status": "all configurations failed"}
                continue
            chosen = min(complete, key=lambda k: sum(r["selection_objective"] for r in complete[k]))
            selection[kind] = {
                "budget": chosen[0],
                "depth": chosen[1],
                "checkpoints": {r["name"]: r["checkpoint_sha256"] for r in complete[chosen]},
            }
        save(
            directory / "selection.json",
            {
                "families": selection,
                "training_freeze_sha256": digest(directory / "freeze.json"),
                "fit_records": {
                    r["name"]: digest(directory / f"{r['name']}.json") for r in records
                },
                "gate_evaluated": False,
                "final_evaluated": False,
            },
        )
        print("GRID COMPLETE", len(records), flush=True)


if __name__ == "__main__":
    main()
