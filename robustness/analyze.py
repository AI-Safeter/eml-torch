"""Analyze every scalar suite, with nondegenerate accuracy and simultaneous CIs."""

import argparse
import json

import torch
from robust_statistics import loss_bound, response_metrics
from runtime import RUNS, configure


def group(row):
    return row["a"], row["b"], row["c"], row.get("corrupt_b", row["b"])


def ordinary(data):
    original = data["original"]
    groups = sorted({group(r) for r in original})
    styles = sorted({r["style"] for r in original})
    tables = {k: {(group(r), r["style"]): r for r in rows} for k, rows in data.items()}
    truth = torch.tensor(
        [[tables["original"][(g, s)]["correct"] for s in styles] for g in groups], device="cuda"
    )
    result = {}
    for name, table in tables.items():
        actual = torch.tensor(
            [[table[(g, s)]["correct"] for s in styles] for g in groups], device="cuda"
        )
        result[name] = {
            "original_accuracy": float(truth.double().mean()),
            "accuracy": float(actual.double().mean()),
            **loss_bound(truth, actual),
            "answer_agreement": sum(r["answer_agreement"] for r in data[name]) / len(data[name]),
            "mean_first_token_kl": sum(r["kl"] for r in data[name]) / len(data[name]),
        }
    return result


def interventions(data):
    rows = data["original"]
    groups = sorted({group(r) for r in rows})
    conditions = sorted({(r["style"], r.get("feature", -1)) for r in rows})
    available = sorted({r["alpha"] for r in rows})
    alphas = [a for a in available if a not in [0, 1]]
    effects = {}
    for kind, records in data.items():
        table = {(group(r), (r["style"], r.get("feature", -1)), r["alpha"]): r for r in records}
        effects[kind] = torch.tensor(
            [
                [
                    table[(g, s, a)]["margin"] - table[(g, s, 0)]["margin"]
                    for s in conditions
                    for a in alphas
                ]
                for g in groups
            ],
            device="cuda",
            dtype=torch.float64,
        )
    metrics = {k: response_metrics(effects["original"], p) for k, p in effects.items()}
    comparisons = {}
    generator = torch.Generator(device="cuda").manual_seed(20260929)
    indices = torch.randint(len(groups), (10000, len(groups)), device="cuda", generator=generator)
    for name, value in effects.items():
        if not name.endswith("/eml"):
            continue
        other = name.removesuffix("/eml") + "/neural"
        if other not in effects:
            continue
        difference = (
            (value - effects["original"]).square() - (effects[other] - effects["original"]).square()
        ).mean(1)
        interval = torch.quantile(
            difference[indices].mean(1),
            torch.tensor([0.025, 0.975], dtype=torch.float64, device="cuda"),
        )
        comparisons[name] = {
            "control": other,
            "eml_minus_control_response_mse": float(difference.mean()),
            "paired_difference_ci95": interval.tolist(),
        }
    return {"models": metrics, "comparisons": comparisons, "strengths": alphas}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen17b", "qwen4b", "smollm"])
    parser.add_argument("operation", choices=["add", "multiply", "divide"])
    args = parser.parse_args()
    configure(args.model)
    from data import save
    from model_io import setup

    setup(20260926)
    out = RUNS / args.model / args.operation
    result = {"ordinary": {}, "interventions": {}}
    for path in sorted(out.glob("ordinary-*.json")):
        if "validation" not in path.name:
            result["ordinary"][path.stem] = ordinary(json.loads(path.read_text()))
    paths = (
        sorted(out.glob("interventions-*.json"))
        + sorted(out.glob("*coordinate-interventions.json"))
        + sorted(out.glob("geometry-*.json"))
    )
    for path in paths:
        if "validation" not in path.name:
            result["interventions"][path.stem] = interventions(json.loads(path.read_text()))
    primary = "heads-active-r32-g0.1/eml"
    if (
        "ordinary-test-known-formats" in result["ordinary"]
        and "interventions-test-known-formats" in result["interventions"]
    ):
        normal = result["ordinary"]["ordinary-test-known-formats"][primary]
        causal = result["interventions"]["interventions-test-known-formats"]["models"][primary]
        checks = {
            "accuracy_bound": normal["passes_one_percentage_point_bound"],
            "nrmse_bound": causal["nrmse_simultaneous_ci95"] is not None
            and causal["nrmse_simultaneous_ci95"][1] <= 0.2,
            "correlation_bound": causal["correlation_simultaneous_ci95"] is not None
            and causal["correlation_simultaneous_ci95"][0] >= 0.9,
        }
        result["primary_checks"] = {**checks, "all_pass": all(checks.values())}
    save(out / "robust-results.json", result)
    print(
        "ROBUST ANALYSIS COMPLETE",
        args.model,
        args.operation,
        result.get("primary_checks"),
        flush=True,
    )


if __name__ == "__main__":
    main()
