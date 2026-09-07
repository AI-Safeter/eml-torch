"""Audit deployable equations and paired errors on CUDA; time CPU deployment separately."""

import importlib.util
import json
import math

import numpy as np
import torch
from scipy.integrate import solve_ivp

from .shared import ROOT, save, setup, timed


def main():
    environment = setup()
    report = {"environment": environment, "applications": {}}
    for name in ["distillation", "surrogate"]:
        out = ROOT / "results" / name
        rows = torch.load(out / "predictions.pt", weights_only=True)
        spec = importlib.util.spec_from_file_location("equation", out / "equation.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        checks = {}
        for split, data in rows.items():
            ref = data["eml"].cuda()
            actual, idx = [], []
            rejected = 0
            for i, x in enumerate(data["x"].tolist()):
                in_box = all(lo <= v <= hi for v, lo, hi in zip(x, module.LOW, module.HIGH))
                try:
                    value = module.predict_one(x)
                except ValueError:
                    assert not in_box
                    rejected += 1
                else:
                    assert in_box and math.isfinite(value)
                    actual.append(value)
                    idx.append(i)
            p = torch.tensor(actual, device="cuda")
            expected = ref[idx]
            assert torch.allclose(p, expected, atol=0.002, rtol=2e-4)
            checks[split] = {
                "accepted": len(idx),
                "rejected": rejected,
                "max_export_error": float((p - expected).abs().max()) if idx else None,
            }
        for bad in [[math.nan] * len(module.MEAN), [math.inf] * len(module.MEAN), []]:
            try:
                module.predict_one(bad)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid input accepted")
        # Resample whole operating groups for airfoil; independent draws for simulation.
        data = rows["test"]
        if name == "distillation":
            _, group_ids = torch.unique(data["x"][:, 1:].cuda(), dim=0, return_inverse=True)
        else:
            group_ids = torch.arange(len(data["x"]), device="cuda")
        n = int(group_ids.max()) + 1
        counts = torch.bincount(group_ids, minlength=n).double()
        indices = torch.randint(n, (5000, n), device="cuda")
        sums = {}
        for method in ["eml", "network", "polynomial3"]:
            err = (data[method].cuda().double() - data["target"].cuda().double()).square()
            sums[method] = torch.zeros(n, device="cuda", dtype=torch.float64).scatter_add_(
                0, group_ids, err
            )
        boot = {k: (v[indices].sum(1) / counts[indices].sum(1)).sqrt() for k, v in sums.items()}
        uncertainty = {
            k: torch.quantile(
                v, torch.tensor([0.025, 0.975], device="cuda", dtype=torch.float64)
            ).tolist()
            for k, v in boot.items()
        }
        uncertainty["eml_minus_network_rmse_ci95"] = torch.quantile(
            boot["eml"] - boot["network"],
            torch.tensor([0.025, 0.975], device="cuda", dtype=torch.float64),
        ).tolist()
        checkpoint = torch.load(out / "network.pt", weights_only=True, map_location="cpu")
        weights = checkpoint["state"]
        xrow = data["x"][0].numpy()
        mean, std = checkpoint["xmean"].numpy(), checkpoint["xstd"].numpy()
        layers = [(weights[f"{i}.weight"].numpy(), weights[f"{i}.bias"].numpy()) for i in [0, 2, 4]]

        def numpy_network():
            z = (xrow - mean) / std
            for j, (w, b) in enumerate(layers):
                z = w @ z + b
                if j < 2:
                    z = z / (1 + np.exp(-z))
            value = float(z[0]) * float(checkpoint["ystd"]) + float(checkpoint["ymean"])
            return (1 - math.cos(xrow[0])) * math.exp(value) if name == "surrogate" else value

        network_error = abs(numpy_network() - float(data["network"][0]))
        assert network_error < 0.002
        entry = {
            "export_checks": checks,
            "invalid_inputs_rejected": True,
            "bootstrap_test_rmse_ci95": uncertainty,
            "bootstrap_unit": "operating group" if name == "distillation" else "problem",
            "bootstrap_groups": n,
            "numpy_network_cpu_single": timed(numpy_network),
            "numpy_network_check_error": network_error,
        }
        if name == "surrogate":
            amplitude, damping = xrow.tolist()

            def scipy_solver():
                s = solve_ivp(
                    lambda t, z: [z[1], -damping * z[1] - math.sin(z[0])],
                    (0, 6),
                    [amplitude, 0.0],
                    rtol=1e-8,
                    atol=1e-10,
                )
                assert s.success
                angle, velocity = s.y[:, -1]
                return 0.5 * velocity**2 + 1 - math.cos(angle)

            reference = scipy_solver()
            gpu_error = float(
                (torch.tensor(reference, device="cuda") - data["target"][0].cuda()).abs()
            )
            assert gpu_error < 1e-6
            entry["scipy_cpu_single"] = timed(scipy_solver, repeats=30)
            entry["scipy_vs_gpu_RK4_error"] = gpu_error
        report["applications"][name] = entry
    llm = ROOT / "llm/replay"
    assert (llm / "metadata.json").exists()
    for p in llm.glob("ordinary-*.json"):
        d = json.loads(p.read_text())
        assert len({len(v) for v in d.values()}) == 1
        assert all(r["correct"] == s["correct"] for r, s in zip(d["original"], d["restore"]))
    report["llm"] = json.loads((llm / "metadata.json").read_text())
    save(ROOT / "results/validation.json", report)
    print("Application GPU validation passed", flush=True)


if __name__ == "__main__":
    main()
