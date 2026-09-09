"""Fixed-effort teacher-free fitting, exact continuation, and empirical capacity bounds."""

import argparse
import json
import time

import torch

from gemma_mechanisms.train import evaluate, moments, objective

from .common import SPEC, accounting, digest, freeze, name, previous, root, save, setup, sources
from .model import Deployed, Replacement


def data(out):
    collection = previous(out) / "collection"
    raw = {
        split: {
            domain: torch.load(collection / f"{domain}-{split}.pt", weights_only=True)
            for domain in ["arithmetic", "language"]
        }
        for split in ["train", "selection"]
    }
    norm = torch.load(collection / "native-postnorm.pt", weights_only=True)
    norm["weight"] = norm["weight"].cuda()
    return raw, norm


@torch.no_grad()
def initialize_data(raw):
    stats, cscale = moments(raw)
    d = len(stats["xmean"])
    xx = torch.zeros(d + 1, d + 1, device="cuda", dtype=torch.float64)
    xy = torch.zeros(d + 1, d, device="cuda", dtype=torch.float64)
    yy = torch.zeros(d, d, device="cuda", dtype=torch.float64)
    for domain in raw.values():
        for a, b in zip(domain["x"].split(2048), domain["y"].split(2048)):
            x = ((a.cuda().float() - stats["xmean"]) / stats["xstd"]).double()
            y = ((b.cuda().float() - stats["ymean"]) / stats["yscale"]).double()
            x = torch.cat([x, torch.ones(len(x), 1, device="cuda", dtype=torch.float64)], -1)
            w = 1 / len(domain["x"]) / len(raw)
            xx.addmm_(x.T, x, alpha=w)
            xy.addmm_(x.T, y, alpha=w)
            yy.addmm_(y.T, y, alpha=w)
    ridge_penalty = torch.eye(d + 1, device="cuda", dtype=torch.float64) * 0.001
    ridge_penalty[-1, -1] = 0
    ridge = torch.linalg.solve(xx + ridge_penalty, xy)
    eigx, basisx = torch.linalg.eigh(xx[:d, :d])
    eigy, basisy = torch.linalg.eigh(yy)
    # Unregularized solve: ridge residuals cannot certify this bound.
    ols = torch.linalg.solve(xx, xy)
    residual_cov = torch.zeros_like(yy)
    normal_residual = torch.zeros_like(xy)
    for domain in raw.values():
        for a, b in zip(domain["x"].split(2048), domain["y"].split(2048)):
            x = ((a.cuda().float() - stats["xmean"]) / stats["xstd"]).double()
            y = ((b.cuda().float() - stats["ymean"]) / stats["yscale"]).double()
            x = torch.cat([x, torch.ones(len(x), 1, device="cuda", dtype=torch.float64)], -1)
            residual = y - x @ ols
            w = 1 / len(domain["x"]) / len(raw)
            residual_cov.addmm_(residual.T, residual, alpha=w)
            normal_residual.addmm_(x.T, residual, alpha=w)
    eigr, basisr = torch.linalg.eigh(residual_cov)
    condition = float(torch.linalg.cond(xx))
    ne = float(normal_residual.norm() / xy.norm())
    identity_error = float((residual_cov - (yy - xy.T @ ols)).norm() / residual_cov.norm())
    certified = (
        condition < 1e10 and ne < 1e-9 and identity_error < 1e-8 and float(eigr.min()) > -1e-10
    )
    result = {
        "statistics": {k: v.cpu() for k, v in stats.items()},
        "cscale": cscale.cpu(),
        "ridge": ridge.cpu(),
        "ols": ols.cpu(),
        "y_basis": basisy.flip(1).float().cpu(),
        "x_basis": basisx.flip(1).float().cpu(),
        "residual_basis": basisr.flip(1).float().cpu(),
        "y_eigenvalues": eigy.flip(0).cpu(),
        "residual_eigenvalues": eigr.flip(0).cpu(),
        "residual_covariance": residual_cov.cpu(),
        "output_covariance": yy.cpu(),
    }
    diagnostics = {
        "normal_equation_condition": condition,
        "normal_equation_relative_residual": ne,
        "direct_residual_covariance_relative_disagreement": identity_error,
        "shortcut_bound_numerically_certified": certified,
        "raw_mse_bounds": {
            "bottleneck": float(eigy[:-512].sum() / d),
            "shortcut": float(eigr[:-512].sum() / d) if certified else None,
            "structured": 0.0,
        },
        "full_affine_raw_mse": float(eigr.sum() / d),
        "output_total_normalized_second_moment": float(eigy.sum() / d),
        "note": "Finite training-distribution MSE lower bounds. They do not bound downstream quality or establish information preservation.",
    }
    return result, diagnostics


def initialize(model, init):
    ridge = init["ridge"].to(device="cuda", dtype=torch.float32)
    with torch.no_grad():
        if model.architecture == "bottleneck":
            basis = init["y_basis"][:, : model.width].cuda()
            model.encoder.weight.copy_((ridge[:-1] @ basis).T)
            model.encoder.bias.copy_(ridge[-1] @ basis)
            model.decoder.weight.copy_(basis)
            model.decoder.bias.zero_()
        elif model.architecture == "shortcut":
            model.shortcut.weight.copy_(ridge[:-1].T)
            model.shortcut.bias.copy_(ridge[-1])
            model.encoder.weight.copy_(init["x_basis"][:, : model.width].T)
            model.encoder.bias.zero_()
            model.decoder.weight.copy_(init["residual_basis"][:, : model.width])
        else:
            model.decoder.weight.copy_(ridge[:-1].T)
            model.decoder.bias.copy_(ridge[-1])


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--architecture", choices=SPEC["training"]["architectures"])
    p.add_argument("--activation", choices=["eml", "silu"])
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--seed", type=int, default=1103)
    p.add_argument("--steps", type=int, choices=[12000, 24000], default=12000)
    a = p.parse_args()
    setup(a.seed)
    out = root()
    directory = out / "training"
    directory.mkdir(exist_ok=True)
    collection = previous(out) / "collection"
    inputs = [
        collection / f"{d}-{s}.pt"
        for d in ["arithmetic", "language"]
        for s in ["train", "selection"]
    ] + [collection / "native-postnorm.pt", out / "data/freeze.json"]
    frozen = freeze(
        directory / "freeze.json",
        {
            "sources": sources(
                "train.py",
                "model.py",
                "../gemma_mechanisms/train.py",
                "../gemma_mechanisms/student.py",
            ),
            "inputs": {str(f): digest(f) for f in inputs},
        },
    )
    raw, norm = data(out)
    if a.prepare_only:
        assert not (directory / "initialization.pt").exists()
        start = time.perf_counter()
        init, bounds = initialize_data(raw["train"])
        torch.save(init, directory / "initialization.pt")
        bounds["seconds"] = time.perf_counter() - start
        bounds["initialization_sha256"] = digest(directory / "initialization.pt")
        save(directory / "initialization.json", bounds)
        print("CAPACITY BOUNDS", json.dumps(bounds), flush=True)
        # Both activations must start from the same affine map in each architecture.
        x = raw["selection"]["language"]["x"][:32].cuda().float()
        for arch in SPEC["training"]["architectures"]:
            predictions = []
            for act in ["eml", "silu"]:
                m = Replacement(arch, act, statistics=init["statistics"]).cuda()
                initialize(m, init)
                with torch.no_grad():
                    predictions.append(m(x))
                del m
            torch.testing.assert_close(*predictions, atol=2e-6, rtol=2e-6)
        print("INITIAL AFFINE MATCH CHECKED", flush=True)
        return
    assert a.architecture and a.activation
    assert (out / "validation.json").exists() and (out / "integration.json").exists()
    key = name(a.architecture, a.activation, a.depth, a.seed, a.steps)
    assert not (directory / f"{key}.json").exists()
    init = torch.load(directory / "initialization.pt", weights_only=True)
    m = Replacement(a.architecture, a.activation, a.depth, statistics=init["statistics"]).cuda()
    initialize(m, init)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.0001)
    rng = torch.Generator().manual_seed(a.seed + 100000)
    cscale = init["cscale"].cuda()
    start_step = 0
    best = float("inf")
    best_state = None
    chosen = None
    curves = []
    clips = 0
    if a.steps == 24000:
        prior = name(a.architecture, a.activation, a.depth, a.seed, 12000)
        continuation = torch.load(directory / f"{prior}-continuation.pt", weights_only=True)
        m.load_state_dict(continuation["state"])
        opt.load_state_dict(continuation["optimizer"])
        rng.set_state(continuation["rng"])
        start_step = 12000
        old = json.loads((directory / f"{prior}.json").read_text())
        assert old["status"] == "complete" and old["steps"] == 12000
        best = old["selection_objective"]
        chosen = old["selected_step"]
        curves = old["curves"]
        clips = old["gradient_clipped_steps"]
        best_state = torch.load(directory / f"{prior}.pt", weights_only=True)["state"]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    events[0].record()
    failed = None
    for step in range(start_step, a.steps):
        for stage in m.stages:
            stage.diagnostics = step % 200 == 0
        losses = []
        for domain in raw["train"].values():
            ids = torch.randint(len(domain["x"]), (256,), generator=rng)
            losses.append(
                objective(
                    m,
                    *(domain[k][ids].cuda().float() for k in ["x", "y", "contribution"]),
                    norm,
                    cscale,
                )[0]
            )
        loss = sum(losses) / 2
        if not torch.isfinite(loss):
            failed = "nonfinite loss"
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        if not torch.isfinite(grad):
            failed = "nonfinite gradient"
            break
        clips += int(grad > 1)
        opt.step()
        if (step + 1) % 200 == 0:
            for stage in m.stages:
                stage.diagnostics = False
            val = evaluate(m, raw["selection"], norm, cscale)
            value = sum(v["objective"] for v in val.values()) / 2
            if value < best:
                best = value
                chosen = step + 1
                best_state = cpu_state(m)
            curves.append(
                {
                    "step": step + 1,
                    "selection": val,
                    "train_batch_objective": float(loss),
                    "gradient_norm": float(grad),
                }
            )
            if (step + 1) % 1000 == 0:
                print("FIT", key, step + 1, value, time.perf_counter() - start, flush=True)
    events[1].record()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    torch.save(
        {"state": cpu_state(m), "optimizer": opt.state_dict(), "rng": rng.get_state()},
        directory / f"{key}-continuation.pt",
    )
    result = {
        "name": key,
        "specification": m.specification(),
        "seed": a.seed,
        "status": "failed" if failed or best_state is None else "complete",
        "failure": failed,
        "steps": step + 1,
        "selected_step": chosen,
        "selection_objective": best if best_state else None,
        "training_seconds": elapsed,
        "cuda_elapsed_ms": events[0].elapsed_time(events[1]),
        "additional_training_tokens": (step + 1 - start_step) * 512,
        "gradient_clipped_steps": clips,
        "curves": curves,
        "accounting": accounting(m),
        "training_freeze_sha256": frozen,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "exponent_diagnostics": [
            {"clamps": s.clamps, "arguments": s.arguments_seen, "max_argument": s.max_argument}
            for s in m.stages
        ],
    }
    if best_state is not None:
        torch.save(
            {"specification": m.specification(), "state": best_state}, directory / f"{key}.pt"
        )
        result["checkpoint_sha256"] = digest(directory / f"{key}.pt")
        m.load_state_dict(best_state)
        result["selection"] = evaluate(m, raw["selection"], norm, cscale)
        result["train"] = evaluate(m, raw["train"], norm, cscale)
        result["deployed_accounting"] = accounting(Deployed(m).bfloat16())
    save(directory / f"{key}.json", result)
    print("FIT COMPLETE", key, result["status"], best, elapsed, flush=True)


if __name__ == "__main__":
    main()
