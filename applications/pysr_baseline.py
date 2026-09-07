"""Optional established symbolic-regression control; CPU search, CUDA scoring."""

import argparse
import json
import time

import numpy as np
import pysr
import torch

from .shared import ROOT, metrics, save, setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("application", choices=["distillation", "surrogate"])
    args = parser.parse_args()
    setup()
    out = ROOT / "results" / args.application
    saved = torch.load(out / "predictions.pt", weights_only=True)
    state = json.loads((out / "equation.json").read_text())
    prior = args.application == "surrogate"
    mean, std = np.array(state["xmean"]), np.array(state["xstd"])
    ym, ys = state["ymean"], state["ystd"]

    def xy(split):
        r = saved[split]
        x = r["x"].numpy()
        y = (
            r["network"].numpy()
            if not prior
            else np.log(r["target"].numpy() / (1 - np.cos(x[:, 0])))
        )
        return ((x - mean) / std).astype(np.float32), ((y - ym) / ys).astype(np.float32)

    xt, yt = xy("train")
    xv, yv = xy("validation")
    model = pysr.PySRRegressor(
        niterations=40,
        populations=8,
        population_size=32,
        maxsize=25,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sin"],
        parallelism="serial",
        deterministic=True,
        random_state=17,
        timeout_in_seconds=120,
        progress=False,
        verbosity=0,
        output_directory=str(out / "pysr-search"),
    )
    start = time.perf_counter()
    model.fit(xt, yt)
    candidates = []
    for index, row in model.equations_.iterrows():
        with np.errstate(all="ignore"):
            p = model.predict(xv, index=index)
        if np.isfinite(p).all():
            error = float(
                (torch.tensor(p, device="cuda") - torch.tensor(yv, device="cuda")).square().mean()
            )
            candidates.append(
                {
                    "index": int(index),
                    "validation_mse": error,
                    "complexity": int(row["complexity"]),
                    "equation": str(row["equation"]),
                }
            )
    choice = min(candidates, key=lambda r: r["validation_mse"])
    result = {
        "selected": choice,
        "candidates": candidates,
        "seconds": time.perf_counter() - start,
        "search_device": "CPU",
        "scoring_device": "CUDA",
        "budget": "40 iterations, 8 populations of 32; 120 second search timeout",
        "comparison": "same splits and learning targets; budgets are not compute-matched",
        "suites": {},
    }
    for split, r in saved.items():
        with np.errstate(all="ignore"):
            pred = model.predict(xy(split)[0], index=choice["index"]) * ys + ym
        p = torch.as_tensor(pred, device="cuda")
        if prior:
            p = (1 - torch.cos(r["x"][:, 0].cuda())) * torch.exp(p)
        result["suites"][split] = metrics(p, r["target"].cuda())
    save(out / "pysr.json", result)
    print(args.application, "PySR DONE", result["suites"]["test"], flush=True)


if __name__ == "__main__":
    main()
