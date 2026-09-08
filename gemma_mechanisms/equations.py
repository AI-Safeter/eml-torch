"""Compact equations between eight decoded quantities; observational fit is diagnostic."""

import copy
import json
import time

import torch

from .runtime import HERE, SPEC, digest, root, save, setup
from .student import Student


def quantities(states, probe):
    weights, biases = [], []
    for name in SPEC["mechanism"]["candidate_variables"]:
        lo, hi = probe["blocks"][name]
        c = torch.ones(1) if hi - lo == 1 else torch.arange(10).float()
        weights.append(probe["weight"][:, lo:hi] @ c)
        biases.append(probe["bias"][lo:hi] @ c)
    weight = torch.stack(weights, -1).cuda()
    bias = torch.stack(biases).cuda()
    return states.cuda().float() @ weight + bias


def symbolic(x):
    # These are inferred internal quantities, not ground-truth operand labels.
    a, b, at, bt = [x[:, k].round().clamp(0, 9) for k in range(4)]
    carry = (a + b >= 10).float()
    return torch.stack(
        [
            a,
            b,
            at,
            bt,
            carry,
            (at + bt + carry >= 10).float(),
            (a + b) % 10,
            (at + bt + carry) % 10,
        ],
        -1,
    )


def metrics(predicted, y, scale):
    error = ((predicted - y) / scale).square().mean(0)
    return {"normalized_mse": error.tolist(), "mean": float(error.mean())}


def main():
    setup(SPEC["data_seed"])
    base = root()
    out = base / "residual"
    directory = out / "equations"
    directory.mkdir(exist_ok=True)
    source, target, prefix = 26, 32, 2
    probes = {
        layer: torch.load(out / "mechanism" / f"layer{layer}-prefix{prefix}.pt", weights_only=True)
        for layer in [source, target]
    }
    data = {}
    input_files = {}
    for split in ["train", "selection"]:
        path = out / "collection" / f"mechanism-{split}.pt"
        input_files[str(path.relative_to(out))] = digest(path)
        raw = torch.load(path, weights_only=True)
        ids = torch.tensor(
            [i for i, row in enumerate(raw["rows"]) if row["answer_prefix_tokens"] == prefix]
        )
        data[split] = {
            "x": quantities(raw["states"][source]["input"][ids], probes[source]),
            "y": quantities(raw["states"][target]["input"][ids], probes[target]),
        }
    plan = {
        "source_layer": source,
        "target_layer": target,
        "prefix": prefix,
        "variables": SPEC["mechanism"]["candidate_variables"],
        "seeds": SPEC["replacement"]["seeds"],
        "depths": [1, 2, 4],
        "max_coefficients": 2048,
        "width": 8,
        "max_steps": 5000,
        "validation_interval": 100,
        "minimum_steps": 1000,
        "patience": 700,
        "learning_rate": 0.001,
        "batch_size": 256,
        "gradient_clip": 1.0,
        "selection_uses": "Observational quantity error only; no intervention objective",
        "scope": "Diagnostic propagation fit after failed probe-direction mediation. No algorithm claim.",
        "source_sha256": digest(HERE / "equations.py"),
        "inputs": input_files,
    }
    frozen = directory / "freeze.json"
    if frozen.exists():
        assert json.loads(frozen.read_text()) == plan
    else:
        save(frozen, plan)
    x, y = data["train"]["x"], data["train"]["y"]
    vx, vy = data["selection"]["x"], data["selection"]["y"]
    statistics = {
        "xmean": x.mean(0),
        "xstd": x.std(0).clamp_min(1e-5),
        "ymean": y.mean(0),
        "yscale": y.std(0).clamp_min(1e-5),
    }
    scale = statistics["yscale"]
    xx = torch.cat([x, torch.ones(len(x), 1, device="cuda")], -1).double()
    ridge = torch.eye(xx.shape[-1], device="cuda", dtype=torch.float64) * 0.001
    ridge[-1, -1] = 0
    linear = torch.linalg.solve(xx.T @ xx / len(xx) + ridge, xx.T @ y.double() / len(xx)).float()
    torch.save({"weight": linear.cpu(), "scale": scale.cpu()}, directory / "linear.pt")
    results = {
        "linear": metrics(
            torch.cat([vx, torch.ones(len(vx), 1, device="cuda")], -1) @ linear, vy, scale
        ),
        "symbolic": metrics(symbolic(vx), vy, scale),
        "identity": metrics(vx, vy, scale),
    }
    for seed in plan["seeds"]:
        for depth in plan["depths"]:
            # lcm(3*8+2, 2*8+1)=442 gives exactly matched stage budgets.
            budget = 176 + 24 * depth + 1768
            for kind in ["eml", "silu"]:
                name = f"{kind}-d{depth}-s{seed}"
                path = directory / f"{name}.json"
                if path.exists():
                    results[name] = json.loads(path.read_text())
                    continue
                setup(seed)
                model = Student(8, budget, 8, depth, kind, statistics).cuda()
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
                best, best_state, stale, curves, clipped = float("inf"), None, 0, [], 0
                started = time.perf_counter()
                for step in range(plan["max_steps"]):
                    ids = torch.randint(len(x), (256,), device="cuda")
                    loss = ((model(x[ids]) - y[ids]) / scale).square().mean()
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    assert torch.isfinite(grad)
                    clipped += int(grad > 1)
                    optimizer.step()
                    if (step + 1) % 100 == 0:
                        with torch.no_grad():
                            value = float(((model(vx) - vy) / scale).square().mean())
                        curves.append(
                            {
                                "step": step + 1,
                                "selection_mse": value,
                                "train_batch_mse": float(loss),
                            }
                        )
                        if value < best:
                            best, best_state, stale = value, copy.deepcopy(model.state_dict()), 0
                        else:
                            stale += 100
                        if step + 1 >= 1000 and stale >= 700:
                            break
                model.load_state_dict(best_state)
                with torch.no_grad():
                    result = metrics(model(vx), vy, scale)
                result.update(
                    name=name,
                    seed=seed,
                    depth=depth,
                    kind=kind,
                    coefficients=model.coefficients(),
                    seconds=time.perf_counter() - started,
                    steps=step + 1,
                    gradient_clipped_steps=clipped,
                    curves=curves,
                    upstream_projection_coefficients=1537 * 8,
                    downstream_measurement_coefficients=1537 * 8,
                )
                torch.save(
                    {
                        "state": {k: v.cpu() for k, v in best_state.items()},
                        "specification": model.specification(),
                    },
                    directory / f"{name}.pt",
                )
                result["checkpoint_sha256"] = digest(directory / f"{name}.pt")
                save(path, result)
                results[name] = result
                print("EQUATION", name, best, flush=True)
    save(
        directory / "observational-results.json",
        {
            "results": results,
            "freeze_sha256": digest(frozen),
            "note": "Eight upstream quantities include estimated carry/result digits. This tests state propagation, not derivation of arithmetic from operands. Causal mediation already failed for these readout directions.",
        },
    )
    print("EQUATIONS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
