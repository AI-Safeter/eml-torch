"""Separate learned-decoder subspace error from error inside that subspace."""

import json
import time

import torch

from .runtime import HERE, SPEC, digest, root, save, setup
from .student import load_student


@torch.inference_mode()
def decompose(model, data, basis):
    sums = torch.zeros(4, device="cuda", dtype=torch.float64)
    bias = model.decoder.bias.double()
    for x, target in zip(data["x"].split(512), data["y"].split(512)):
        x, target = x.cuda().float(), target.cuda().float()
        target = (target - model.ymean) / model.yscale
        prediction = model.normalized((x - model.xmean) / model.xstd)
        error = target.double() - prediction.double()
        centered = target.double() - bias
        perpendicular = centered - (centered @ basis) @ basis.T
        within = error @ basis
        decoded = prediction.double() - bias
        rounding = decoded - (decoded @ basis) @ basis.T
        sums += torch.stack(
            [
                error.square().sum(),
                perpendicular.square().sum(),
                within.square().sum(),
                rounding.square().sum(),
            ]
        )
    total, outside, inside, rounding = (sums / data["y"].numel()).tolist()
    closure = abs(total - outside - inside)
    assert closure < 1e-6 * max(total, 1.0), (total, outside, inside, closure)
    return {
        "raw_normalized_mse": total,
        "outside_learned_affine_decoder_mse": outside,
        "within_decoder_prediction_mse": inside,
        "decoder_rounding_outside_subspace_mse": rounding,
        "pythagorean_closure_error": closure,
        "outside_fraction": outside / total,
        "tokens": len(data["x"]),
    }


def main():
    setup(73)
    out = root()
    directory = out / "decoder-error"
    directory.mkdir(exist_ok=True)
    selection = json.loads((out / "training/selection.json").read_text())["families"]["eml"]
    candidates = [
        (phase, f"{kind}-b{selection['budget']}-d{selection['depth']}-s{seed}")
        for phase in ["training", "optimization-extension"]
        for kind in ["eml", "silu"]
        for seed in SPEC["replacement"]["seeds"]
    ]
    inputs = [
        out / "collection" / f"{d}-{s}.pt"
        for s in ["train", "selection"]
        for d in ["arithmetic", "language"]
    ]
    inputs += [out / "training/initialization.pt", out / "training/selection.json"]
    plan = {
        "sources": {
            name: digest(HERE / name) for name in ["decoder_error.py", "student.py", "runtime.py"]
        },
        "inputs": {str(p.relative_to(out)): digest(p) for p in inputs},
        "checkpoints": {
            f"{phase}/{name}.pt": digest(out / phase / f"{name}.pt") for phase, name in candidates
        },
        "scope": "Post hoc decomposition on training and selection only. No fitting, candidate changes, or held-out outcome optimization. Within-subspace error combines encoder information loss, function capacity, optimization and objective tradeoffs; it is not an optimization-only measure. Raw-output error does not determine answer accuracy.",
    }
    frozen = directory / "freeze.json"
    if frozen.exists():
        assert json.loads(frozen.read_text()) == plan
    else:
        save(frozen, plan)
    raw = {
        s: {
            d: torch.load(out / "collection" / f"{d}-{s}.pt", weights_only=True)
            for d in ["arithmetic", "language"]
        }
        for s in ["train", "selection"]
    }
    eigenvalues = torch.load(out / "training/initialization.pt", weights_only=True)["eigenvalues"]
    for phase, name in candidates:
        path = directory / f"{phase}-{name}.json"
        if path.exists():
            assert json.loads(path.read_text())["freeze_sha256"] == digest(frozen)
            continue
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        model = load_student(out / phase / f"{name}.pt", dtype=torch.float32)
        basis = torch.linalg.qr(model.decoder.weight.double(), mode="reduced").Q
        assert (
            torch.max(
                torch.abs(
                    basis.T @ basis - torch.eye(model.width, device="cuda", dtype=torch.float64)
                )
            )
            < 1e-10
        )
        results = {
            split: {domain: decompose(model, data, basis) for domain, data in values.items()}
            for split, values in raw.items()
        }
        averaged = {
            split: {
                key: sum(r[key] for r in values.values()) / len(values)
                for key in [
                    "raw_normalized_mse",
                    "outside_learned_affine_decoder_mse",
                    "within_decoder_prediction_mse",
                ]
            }
            for split, values in results.items()
        }
        for r in averaged.values():
            r["outside_fraction"] = (
                r["outside_learned_affine_decoder_mse"] / r["raw_normalized_mse"]
            )
        floor = float(eigenvalues[model.width :].sum() / eigenvalues.sum())
        assert averaged["train"]["outside_learned_affine_decoder_mse"] + 1e-6 >= floor
        if phase == "training":
            audited = json.loads((out / "fit-audits" / f"{name}.json").read_text())
            assert abs(averaged["train"]["raw_normalized_mse"] - audited["training_raw_mse"]) < 1e-6
        save(
            path,
            {
                "phase": phase,
                "name": name,
                "results": results,
                "balanced": averaged,
                "optimal_training_rank_floor": floor,
                "rank": model.width,
                "seconds": time.perf_counter() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "freeze_sha256": digest(frozen),
            },
        )
        print("DECODER ERROR", phase, name, averaged, flush=True)
        del model, basis
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
