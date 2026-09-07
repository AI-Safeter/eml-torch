"""Fit observed and intervention-trained replacements before test evaluation."""

import json
import time

import numpy as np
import torch
import torch.nn as nn
from common import OUT, dump, load, setup

import emltorch as eml


def design(z, kind):
    parts = [torch.ones_like(z[:, 0])]
    parts.extend(z.unbind(1))
    if kind in ["quadratic", "cubic"]:
        parts.extend(z[:, i] * z[:, j] for i in range(3) for j in range(i, 3))
    if kind == "cubic":
        parts.extend(
            z[:, i] * z[:, j] * z[:, k] for i in range(3) for j in range(i, 3) for k in range(j, 3)
        )
    if kind == "fourier":
        for freq in [1.0, 2.0, 4.0]:
            parts.extend(torch.sin(z * freq).unbind(1))
            parts.extend(torch.cos(z * freq).unbind(1))
    return torch.stack(parts, 1)


def main():
    setup()
    model, _ = load()
    component = torch.load(OUT / "component.pt", weights_only=True)
    data = torch.load(OUT / "component-data.pt", weights_only=True)
    layer = model.model.layers[component["layer"]].mlp
    direction = component["direction"].cuda()
    encoder = component["encoder"].cuda()
    mean = component["input_mean"].cuda()
    splits = {}
    with torch.inference_mode():
        for split in data:
            h = data[split]["input"].cuda()
            ys = []
            zs = []
            for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                mixed = h[:, 1] * (1 - alpha) + h[:, 0] * alpha
                ys.append(torch.cat([layer(c) @ direction for c in mixed.split(128)]))
                zs.append((mixed - mean) @ encoder)
            # Alpha-major ordering; endpoint observations occupy first and last groups.
            splits[split] = {"z": torch.cat(zs), "y": torch.cat(ys), "n": len(h)}
            observed = torch.cat([ys[0], ys[-1]])
            original = data[split]["output"].cuda()
            assert torch.allclose(
                observed,
                torch.cat([original[:, 1] @ direction, original[:, 0] @ direction]),
                atol=1e-3,
                rtol=1e-4,
            )
    zmean = splits["train"]["z"].mean(0)
    zstd = splits["train"]["z"].std(0).clamp(min=1e-6)
    ymean = splits["train"]["y"].mean()
    ystd = splits["train"]["y"].std().clamp(min=1e-6)
    for s in splits.values():
        s["z"] = (s["z"] - zmean) / zstd
        s["y"] = (s["y"] - ymean) / ystd
    checkpoint = {
        "zmean": zmean.cpu(),
        "zstd": zstd.cpu(),
        "ymean": ymean.cpu(),
        "ystd": ystd.cpu(),
        "baselines": {},
    }
    choices = []
    best = {}
    t0 = time.perf_counter()
    dump(
        "fit-protocol.json",
        {
            "inputs": component.get("feature_method", "three training PCA coordinates")
            + " of actual MLP input, standardized with training mixtures",
            "output": "selected direction dot actual MLP output, standardized with training mixtures",
            "train_alphas": [0, 0.25, 0.5, 0.75, 1],
            "test_alphas": [0, 0.125, 0.375, 0.625, 0.875, 1],
            "variants": ["eml_observed", "eml_intervention"],
            "main_variant": "eml_intervention",
            "selection": "validation mixture coefficient MSE; test prompts and test interventions untouched",
            "features_and_coefficients_use_training_only": True,
        },
    )
    for mode in ["observed", "intervention"]:
        train = splits["train"]
        n = train["n"]
        if mode == "observed":
            x = torch.cat([train["z"][:n], train["z"][-n:]])
            y = torch.cat([train["y"][:n], train["y"][-n:]])
        else:
            x = train["z"]
            y = train["y"]
        for depth in [2, 3, 4]:
            for seed in [0, 1, 2]:
                torch.manual_seed(seed)
                np.random.seed(seed)
                fit = eml.fit(
                    x,
                    y,
                    depth=depth,
                    strategy="evolution",
                    population=1024,
                    generations=50,
                    r2_target=0.9999,
                    normalize_inputs=True,
                    use_mul=True,
                    device="cuda",
                    polish=True,
                    polish_optimizer="lbfgs",
                    polish_iters=200,
                    var_names=["z0", "z1", "z2"],
                )
                pred = fit.predict(splits["validation"]["z"]).cuda()
                mse = float((pred - splits["validation"]["y"]).square().mean())
                entry = {
                    "mode": mode,
                    "depth": depth,
                    "seed": seed,
                    "validation_normalized_mse": mse,
                    "train_r2": fit.r2,
                    "expression": fit.expression,
                    "nodes": fit.expression.count("eml("),
                }
                choices.append(entry)
                if mode not in best or mse < best[mode]["validation_normalized_mse"]:
                    best[mode] = entry
                    torch.save(fit, OUT / f"eml-{mode}.pt")
                print(
                    "EML",
                    json.dumps({k: v for k, v in entry.items() if k != "expression"}),
                    flush=True,
                )
                dump("candidates.json", choices)
    x = splits["train"]["z"]
    y = splits["train"]["y"]
    v = splits["validation"]
    scores = {}
    for kind in ["linear", "quadratic", "cubic", "fourier"]:
        matrix = design(x.double(), kind)
        w = torch.linalg.lstsq(matrix, y.double()).solution.float()
        checkpoint["baselines"][kind] = w.cpu()
        scores[kind] = float((design(v["z"], kind) @ w - v["y"]).square().mean())
    # A small neural control with the same three inputs and scalar output.
    best_mse = float("inf")
    best_state = None
    for seed in [0, 1]:
        torch.manual_seed(seed)
        network = nn.Sequential(
            nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1)
        ).cuda()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.005)
        for step in range(1500):
            loss = (network(x).squeeze(1) - y).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 25 == 0:
                with torch.no_grad():
                    mse = float((network(v["z"]).squeeze(1) - v["y"]).square().mean())
                if mse < best_mse:
                    best_mse = mse
                    best_state = {
                        k: t.detach().cpu().clone() for k, t in network.state_dict().items()
                    }
    checkpoint["network"] = best_state
    scores["network"] = best_mse
    torch.save(checkpoint, OUT / "replacements.pt")
    dump(
        "fits.json",
        {
            "selected": best,
            "baseline_validation_normalized_mse": scores,
            "fitting_seconds": time.perf_counter() - t0,
            "all_candidates": choices,
        },
    )
    torch.save(
        {
            k: {a: b.cpu() if isinstance(b, torch.Tensor) else b for a, b in v.items()}
            for k, v in splits.items()
        },
        OUT / "fit-data.pt",
    )
    print("FROZEN", json.dumps({"eml": best, "baselines": scores}), flush=True)


if __name__ == "__main__":
    main()
