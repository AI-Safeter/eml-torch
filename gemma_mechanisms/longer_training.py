"""A preregistered selection-only diagnostic of the 12k-step optimization limit."""

import copy
import gc
import json
import time

import torch

from .runtime import HERE, SPEC, digest, root, save, setup
from .student import Student
from .train import evaluate, initialize, moments, objective


def main():
    setup(73)
    out = root()
    directory = out / "optimization-extension"
    directory.mkdir(exist_ok=True)
    selection = json.loads((out / "training/selection.json").read_text())
    chosen = selection["families"]["eml"]
    budget, depth = chosen["budget"], chosen["depth"]
    width = next(
        b["bottleneck"] for b in SPEC["replacement"]["budgets"] if b["coefficients"] == budget
    )
    plan = {
        "budget": budget,
        "width": width,
        "depth": depth,
        "families": ["eml", "silu"],
        "seeds": SPEC["replacement"]["seeds"],
        "steps": 24000,
        "baseline_steps": 12000,
        "recipe": "Restart from the same initialization; identical optimizer, minibatch stream, losses and 100-step validation. Run twice as many updates. No gate/final outputs or task scores used.",
        "selection_sha256": digest(out / "training/selection.json"),
        "training_freeze_sha256": digest(out / "training/freeze.json"),
        "sources": {
            p: digest(HERE / p)
            for p in ["longer_training.py", "train.py", "student.py", "runtime.py"]
        },
        "scope": "Exploratory optimization diagnostic. These weights cannot replace the preregistered held-out candidates or justify expansion.",
    }
    frozen = directory / "freeze.json"
    if frozen.exists():
        assert json.loads(frozen.read_text()) == plan
    else:
        save(frozen, plan)
    data = {
        split: {
            domain: torch.load(out / "collection" / f"{domain}-{split}.pt", weights_only=True)
            for domain in ["arithmetic", "language"]
        }
        for split in ["train", "selection"]
    }
    stats, cscale = moments(data["train"])
    initial = torch.load(out / "training/initialization.pt", weights_only=True)
    norm = torch.load(out / "collection/native-postnorm.pt", weights_only=True)
    norm["weight"] = norm["weight"].cuda()
    for seed in plan["seeds"]:
        for kind in plan["families"]:
            name = f"{kind}-b{budget}-d{depth}-s{seed}"
            path = directory / f"{name}.json"
            if path.exists():
                assert json.loads(path.read_text())["freeze_sha256"] == digest(frozen)
                continue
            setup(seed)
            model = Student(1536, budget, width, depth, kind, stats).cuda()
            initialize(model, initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
            generator = torch.Generator().manual_seed(seed + 100000)
            best, state, selected_step = float("inf"), None, None
            curves, prefix, clips = [], None, 0
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for step in range(plan["steps"]):
                losses = []
                for domain in data["train"].values():
                    ids = torch.randint(len(domain["x"]), (256,), generator=generator)
                    terms = objective(
                        model,
                        *(domain[k][ids].cuda().float() for k in ["x", "y", "contribution"]),
                        norm,
                        cscale,
                    )
                    losses.append(terms[0])
                loss = sum(losses) / 2
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                assert torch.isfinite(grad)
                clips += int(grad > 1)
                optimizer.step()
                if (step + 1) % 100 == 0:
                    validation = evaluate(model, data["selection"], norm, cscale)
                    value = sum(x["objective"] for x in validation.values()) / 2
                    if value < best:
                        best, state, selected_step = (
                            value,
                            copy.deepcopy(model.state_dict()),
                            step + 1,
                        )
                    curves.append(
                        {
                            "step": step + 1,
                            "selection": validation,
                            "train_batch_loss": float(loss.detach()),
                            "gradient_norm": float(grad),
                        }
                    )
                    if step + 1 == 12000:
                        baseline = json.loads((out / "training" / f"{name}.json").read_text())
                        prefix = {
                            "replayed_best_objective": best,
                            "primary_best_objective": baseline["selection_objective"],
                            "absolute_difference": abs(best - baseline["selection_objective"]),
                        }
                        assert prefix["absolute_difference"] < 1e-7, prefix
                    if (step + 1) % 1000 == 0:
                        print("EXTENDED FIT", name, step + 1, value, flush=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            model.load_state_dict(state)
            validation = evaluate(model, data["selection"], norm, cscale)
            train = evaluate(model, data["train"], norm, cscale)
            torch.save(
                {
                    "specification": model.specification(),
                    "state": {k: v.cpu() for k, v in state.items()},
                },
                path.with_suffix(".pt"),
            )
            save(
                path,
                {
                    "name": name,
                    "steps": 24000,
                    "selected_step": selected_step,
                    "selection_objective": best,
                    "train": train,
                    "selection": validation,
                    "prefix_replay": prefix,
                    "training_seconds": elapsed,
                    "gradient_clipped_steps": clips,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "curves": curves,
                    "checkpoint_sha256": digest(path.with_suffix(".pt")),
                    "freeze_sha256": digest(frozen),
                    "scope": plan["scope"],
                },
            )
            del model, optimizer, state, loss, losses
            gc.collect()
            torch.cuda.empty_cache()
            print("EXTENDED FIT COMPLETE", name, best, flush=True)
    print("OPTIMIZATION DIAGNOSTIC COMPLETE", flush=True)


if __name__ == "__main__":
    main()
