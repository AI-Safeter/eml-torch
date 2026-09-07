"""Real-data distillation and numerical-simulation surrogates, fitted on CUDA."""

import argparse
import hashlib
import io
import json
import time
import urllib.request

import numpy as np
import torch

from .shared import (
    ROOT,
    deploy,
    design,
    fit_baselines,
    fit_equation,
    metrics,
    predict_equation,
    save,
    setup,
    timed,
    train_network,
)


def airfoil(out):
    url = "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip"
    data_path = out / "airfoil_self_noise.dat"
    if not data_path.exists():
        import zipfile

        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            data_path.write_bytes(z.read("airfoil_self_noise.dat"))
    a = np.loadtxt(data_path)
    assert a.shape == (1503, 6) and np.isfinite(a).all()
    # Entire operating configurations stay together; frequency sweeps cannot leak across splits.
    groups = np.unique(a[:, 1:5], axis=0)
    rng = np.random.default_rng(20260907)
    groups = groups[groups[:, 2] < a[:, 3].max()]
    rng.shuffle(groups)
    n = len(groups)
    group_sets = {
        "train": groups[: int(0.6 * n)],
        "validation": groups[int(0.6 * n) : int(0.8 * n)],
        "test": groups[int(0.8 * n) :],
    }
    split_indices = {
        k: np.flatnonzero(np.any(np.all(a[:, None, 1:5] == g[None], axis=2), axis=1))
        for k, g in group_sets.items()
    }
    split_indices["shift"] = np.flatnonzero(a[:, 3] == a[:, 3].max())
    sets = [set(v.tolist()) for v in split_indices.values()]
    assert sum(map(len, sets)) == len(a)
    assert all(not x.intersection(y) for i, x in enumerate(sets) for y in sets[i + 1 :])
    save(out / "split-indices.json", {k: v.tolist() for k, v in split_indices.items()})
    splits = {
        k: (
            torch.tensor(a[v, :5], device="cuda", dtype=torch.float32),
            torch.tensor(a[v, 5], device="cuda", dtype=torch.float32),
        )
        for k, v in split_indices.items()
    }
    return splits, {
        "dataset": "UCI Airfoil Self-Noise",
        "doi": "10.24432/C5VW2C",
        "url": url,
        "license": "CC BY 4.0",
        "attribution": "Brooks, T., Pope, D., & Marcolini, M. (1989)",
        "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "features": [
            "frequency_Hz",
            "attack_angle_deg",
            "chord_m",
            "velocity_m_per_s",
            "thickness_m",
        ],
        "target": "sound_pressure_dB",
        "groups": n,
        "split": "60/20/20% of operating groups; maximum velocity reserved for shift",
    }


def pendulum(x, steps=600):
    """RK4 for theta'' + damping theta' + sin(theta) = 0, omega(0)=0, T=6."""
    theta, damping = x[:, 0].clone(), x[:, 1]
    omega = torch.zeros_like(theta)
    dt = 6.0 / steps
    for _ in range(steps):
        k1t, k1w = omega, -damping * omega - torch.sin(theta)
        k2t = omega + dt * 0.5 * k1w
        k2w = -damping * k2t - torch.sin(theta + dt * 0.5 * k1t)
        k3t = omega + dt * 0.5 * k2w
        k3w = -damping * k3t - torch.sin(theta + dt * 0.5 * k2t)
        k4t = omega + dt * k3w
        k4w = -damping * k4t - torch.sin(theta + dt * k3t)
        theta = theta + dt / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
        omega = omega + dt / 6 * (k1w + 2 * k2w + 2 * k3w + k4w)
    return 0.5 * omega.square() + 1 - torch.cos(theta)


def simulation(out, energy_prior=False):
    splits = {}
    torch.manual_seed(20260908 if energy_prior else 20260907)
    for split, n in [("train", 2048), ("validation", 512), ("test", 512), ("shift", 512)]:
        x = torch.rand(n, 2, device="cuda", dtype=torch.float64)
        x[:, 0] = 0.1 + x[:, 0] * 1.9 if split != "shift" else 2.0 + x[:, 0] * 0.7
        x[:, 1] = 0.02 + x[:, 1] * 0.28
        y = pendulum(x)
        assert (y >= 0).all() and (y <= 1 - torch.cos(x[:, 0]) + 1e-8).all()
        splits[split] = x.float(), y.float()
    # Independent discretization audit before fitting; use float64 to expose integration error.
    audit_x = torch.cat([splits["test"][0][:32], splits["shift"][0][:32]]).double()
    difference = float((pendulum(audit_x, 600) - pendulum(audit_x, 1200)).abs().max())
    assert difference < 1e-7
    torch.save({k: (x.cpu(), y.cpu()) for k, (x, y) in splits.items()}, out / "simulation-data.pt")
    return splits, {
        "equation": "theta'' + damping*theta' + sin(theta) = 0",
        "initial_velocity": 0,
        "final_time": 6,
        "RK4_steps": 600,
        "features": ["initial_angle_rad", "damping"],
        "target": "final_energy",
        "solver_convergence_max_abs": difference,
        "in_domain": [[0.1, 2.0], [0.02, 0.3]],
        "shift": [[2.0, 2.7], [0.02, 0.3]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("application", choices=["distillation", "surrogate"])
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing fitted models; rerun reporting and export checks",
    )
    parser.add_argument(
        "--energy-prior",
        action="store_true",
        help="Learn log remaining-energy fraction; use new simulation draws",
    )
    args = parser.parse_args()
    environment = setup()
    out = ROOT / "results" / args.application
    out.mkdir(parents=True, exist_ok=True)
    save(
        out / "protocol.json",
        {
            "environment": environment,
            "selection": "training fits, validation selects; test and shift evaluated only afterward",
            "eml": {
                "depths": [2, 3, 4],
                "seeds": [17, 29],
                "population": 1024,
                "generations": 80,
                "max_residual_stages": 4,
            },
            "success_rule": "distillation: test MSE <= 1.05 * teacher test MSE and export < teacher parameter bytes; surrogate: test NRMSE <= .05 and export faster than CPU RK4",
            "note": "A surrogate win against RK4 does not imply a win against other fitted surrogates.",
        },
    )
    splits, provenance = (
        airfoil(out) if args.application == "distillation" else simulation(out, args.energy_prior)
    )
    provenance["energy_prior"] = args.energy_prior
    provenance["training_target"] = (
        "log(final_energy / initial_energy)" if args.energy_prior else provenance["target"]
    )
    xt, yt = splits["train"]
    mean, std = xt.mean(0), xt.std(0).clamp_min(1e-6)

    def learning_target(xx, yy):
        return torch.log(yy / (1 - torch.cos(xx[:, 0]))) if args.energy_prior else yy

    learned_y = learning_target(xt, yt)
    ym, ys = learned_y.mean(), learned_y.std().clamp_min(1e-6)
    normalized = {
        k: ((x - mean) / std, (learning_target(x, y) - ym) / ys) for k, (x, y) in splits.items()
    }
    x, y = normalized["train"]
    xv, yv = normalized["validation"]
    start = time.perf_counter()
    if args.resume:
        teacher = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 1),
        ).cuda()
        teacher.load_state_dict(torch.load(out / "network.pt", weights_only=True)["state"])
        teacher.eval().requires_grad_(False)
        teacher_search = (
            json.loads((out / "teacher-search.json").read_text())
            if (out / "teacher-search.json").exists()
            else []
        )
    else:
        teacher, teacher_search = train_network(x, y, xv, yv)
        save(out / "teacher-search.json", teacher_search)
    teacher_time = time.perf_counter() - start
    torch.save(
        {"state": teacher.state_dict(), "xmean": mean, "xstd": std, "ymean": ym, "ystd": ys},
        out / "network.pt",
    )
    with torch.no_grad():
        target = teacher(x).squeeze(1) if args.application == "distillation" else y
        val_target = teacher(xv).squeeze(1) if args.application == "distillation" else yv
        equation = (
            json.loads((out / "equation.json").read_text())
            if args.resume
            else fit_equation(x, target, xv, val_target, out)
        )
        baselines = fit_baselines(x, target, xv, val_target)
    save(
        out / "equation.json",
        {
            **equation,
            "xmean": mean.tolist(),
            "xstd": std.tolist(),
            "ymean": float(ym),
            "ystd": float(ys),
            "feature_names": provenance["features"],
        },
    )
    torch.save(
        {k: {**v, "weights": v["weights"].cpu()} for k, v in baselines.items()},
        out / "baselines.pt",
    )
    result = {
        "environment": environment,
        "provenance": provenance,
        "counts": {k: len(v[0]) for k, v in splits.items()},
        "teacher_search": teacher_search,
        "teacher_seconds": None if args.resume else teacher_time,
        "resumed_frozen_models": args.resume,
        "eml_search_seconds": equation["search_seconds"],
        "stages": len(equation["expressions"]),
        "eml_nodes": sum(e.count("eml(") for e in equation["expressions"]),
        "suites": {},
    }
    predictions = {}
    with torch.no_grad():
        for split, (xx, yy) in normalized.items():
            target_raw = splits[split][1]
            preds = {
                "eml": predict_equation(equation, xx) * ys + ym,
                "network": teacher(xx).squeeze(1) * ys + ym,
                "mean": torch.full_like(yy, ym),
            }
            preds.update(
                {k: (design(xx, k) @ v["weights"]) * ys + ym for k, v in baselines.items()}
            )
            if args.energy_prior:
                initial = 1 - torch.cos(splits[split][0][:, 0])
                preds = {k: initial * torch.exp(v) for k, v in preds.items()}
                preds["small_angle_envelope"] = initial * torch.exp(-6 * splits[split][0][:, 1])
            result["suites"][split] = {k: metrics(p, target_raw) for k, p in preds.items()}
            if args.application == "distillation":
                result["suites"][split]["teacher_fidelity"] = metrics(
                    preds["eml"], preds["network"]
                )
            predictions[split] = {
                "x": splits[split][0].cpu(),
                "target": target_raw.cpu(),
                **{k: v.cpu() for k, v in preds.items()},
            }
    torch.save(predictions, out / "predictions.pt")
    all_inputs = torch.cat([v[0] for v in splits.values()])
    bounds = (xt.min(0).values, xt.max(0).values)
    result["deployment"] = deploy(
        equation, mean, std, ym, ys, all_inputs, out / "equation.py", bounds, args.energy_prior
    )
    cpu_network = teacher.cpu()
    row = splits["test"][0][:1].cpu()
    m, s, cy, cs = mean.cpu(), std.cpu(), ym.cpu(), ys.cpu()
    with torch.inference_mode():

        def network_predict():
            value = cpu_network((row - m) / s) * cs + cy
            return (1 - torch.cos(row[:, 0])) * torch.exp(value) if args.energy_prior else value

        result["deployment"]["network_cpu_single"] = timed(network_predict)
    result["deployment"]["teacher_parameter_bytes"] = sum(
        p.numel() * p.element_size() for p in teacher.parameters()
    )
    if args.application == "surrogate":
        result["deployment"]["solver_cpu_single"] = timed(
            lambda: pendulum(row.double()), repeats=20
        )
        error = result["suites"]["test"]["eml"]["rmse"] / float(
            splits["test"][1].std(unbiased=False)
        )
        result["test_nrmse"] = error
        result["acceptance_passed"] = (
            error <= 0.05
            and result["deployment"]["cpu_single"]["median_us"]
            < result["deployment"]["solver_cpu_single"]["median_us"]
        )
    else:
        result["acceptance_passed"] = (
            result["suites"]["test"]["eml"]["rmse"] ** 2
            <= 1.05 * result["suites"]["test"]["network"]["rmse"] ** 2
            and result["deployment"]["bytes"] < result["deployment"]["teacher_parameter_bytes"]
        )
    save(out / "metrics.json", result)
    print(args.application, "DONE", json_summary(result), flush=True)


def json_summary(result):
    return {
        "test": result["suites"]["test"],
        "shift": result["suites"]["shift"],
        "accepted": result["acceptance_passed"],
    }


if __name__ == "__main__":
    main()
