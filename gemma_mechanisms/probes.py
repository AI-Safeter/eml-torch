"""Decode arithmetic hypotheses; decoding is not evidence of causal mediation."""

import argparse
import json
import time

import torch

from .runtime import HERE, SPEC, digest, root, save, setup


def labels(rows):
    columns, blocks, offset = [], {}, 0
    for name in SPEC["mechanism"]["candidate_variables"]:
        values = torch.tensor([r["variables"][name] for r in rows], device="cuda")
        values = (
            values[:, None].float()
            if name.startswith("carry")
            else torch.nn.functional.one_hot(values, 10).float()
        )
        columns.append(values)
        blocks[name] = (offset, offset + values.shape[1])
        offset += values.shape[1]
    return torch.cat(columns, 1), blocks


def score(predicted, target, blocks):
    record = {}
    for name, (start, stop) in blocks.items():
        p, t = predicted[:, start:stop], target[:, start:stop]
        correct = (
            (p[:, 0] > 0.5) == (t[:, 0] > 0.5)
            if stop - start == 1
            else p.argmax(-1) == t.argmax(-1)
        )
        mse = (p - t).square().mean()
        record[name] = {
            "accuracy": float(correct.float().mean()),
            "mse": float(mse),
            "r2": float(1 - mse / (t - t.mean(0)).square().mean().clamp_min(1e-12)),
        }
    return record


def fit(x, y, vx, vy, blocks):
    mean, std = x.mean(0), x.std(0).clamp_min(1e-6)
    z, vz = (x - mean) / std, (vx - mean) / std
    ym = y.mean(0)
    zz = z.double().T @ z.double() / len(x)
    zy = z.double().T @ (y - ym).double() / len(x)
    identity = torch.eye(x.shape[1], device="cuda", dtype=torch.float64)
    winners, curves = {}, {}
    result_weight = torch.empty(x.shape[1], y.shape[1], device="cuda")
    for penalty in SPEC["mechanism"]["ridge_penalties"]:
        w = torch.linalg.solve(zz + penalty * identity, zy).float()
        values = score(vz @ w + ym, vy, blocks)
        curves[str(penalty)] = values
        for name, (start, stop) in blocks.items():
            if name not in winners or values[name]["mse"] < winners[name]["mse"]:
                winners[name] = {"penalty": penalty, **values[name]}
                result_weight[:, start:stop] = w[:, start:stop]
    weight = result_weight / std[:, None]
    bias = ym - mean @ weight
    assert torch.isfinite(weight).all()
    return {"weight": weight.cpu(), "bias": bias.cpu(), "blocks": blocks}, winners, curves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    setup(SPEC["data_seed"])
    out = root(args.output)
    directory = out / "mechanism"
    directory.mkdir(exist_ok=True)
    files = [out / "collection" / f"mechanism-{s}.pt" for s in ["train", "selection"]]
    freeze = {
        "sources": {p: digest(HERE / p) for p in ["probes.py", "protocol.json", "PROTOCOL.md"]},
        "inputs": {str(p.relative_to(out)): digest(p) for p in files},
    }
    frozen_path = directory / "probe-freeze.json"
    if frozen_path.exists():
        assert json.loads(frozen_path.read_text()) == freeze
    else:
        save(frozen_path, freeze)
    raw = {s: torch.load(p, weights_only=True) for s, p in zip(["train", "selection"], files)}
    record = {}
    started = time.perf_counter()
    for prefix in range(4):
        ids = {
            s: torch.tensor(
                [i for i, row in enumerate(r["rows"]) if row["answer_prefix_tokens"] == prefix]
            )
            for s, r in raw.items()
        }
        yy = {s: labels([raw[s]["rows"][i] for i in ids[s]]) for s in raw}
        target, blocks = yy["train"]
        validation_target = yy["selection"][0]
        for layer in SPEC["mechanism"]["candidate_layers"]:
            name = f"layer{layer}-prefix{prefix}"
            path = directory / f"{name}.pt"
            if path.exists() and path.with_suffix(".json").exists():
                record[name] = json.loads(path.with_suffix(".json").read_text())
                continue
            x = raw["train"]["states"][layer]["input"][ids["train"]].cuda().float()
            vx = raw["selection"]["states"][layer]["input"][ids["selection"]].cuda().float()
            probe, selected, curves = fit(x, target, vx, validation_target, blocks)
            generator = torch.Generator(device="cuda").manual_seed(
                SPEC["data_seed"] + layer + prefix
            )
            perm = torch.randperm(len(target), generator=generator, device="cuda")
            _, shuffled, _ = fit(x, target[perm], vx, validation_target, blocks)
            torch.save(probe, path)
            row = {
                "layer": layer,
                "answer_prefix_tokens": prefix,
                "selected": selected,
                "shuffled_labels": shuffled,
                "penalty_curves": curves,
                "train_observations": len(x),
                "selection_observations": len(vx),
                "checkpoint_sha256": digest(path),
                "probe_coefficients": sum(probe[k].numel() for k in ["weight", "bias"]),
                "causal_evidence": False,
            }
            save(path.with_suffix(".json"), row)
            record[name] = row
            print("PROBE", name, "carry_units", selected["carry_units"], flush=True)
    save(
        directory / "probes.json",
        {
            "results": record,
            "probe_freeze_sha256": digest(frozen_path),
            "seconds": time.perf_counter() - started,
            "gpu": torch.cuda.get_device_name(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "interpretation": "Selection-set decoding only. Layer/variable labels remain hypotheses; no mediation test yet.",
        },
    )
    print("PROBES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
