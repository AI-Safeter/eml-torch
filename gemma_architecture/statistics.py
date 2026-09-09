"""Paired CUDA uncertainty and fixed development selection; no confirmation tuning."""

import argparse
import json

import torch
from scipy.stats import beta

from gemma_mechanisms.quality_summary import bootstrap

from .common import SPEC, digest, name, root, save, setup, sources


def accuracy(teacher, student, alpha=0.05 / 15):
    def groups(rows):
        result = {}
        for r in rows:
            values = result.setdefault(r["id"], {})
            style = r.get("style", "qa")
            assert style not in values
            values[style] = int(r["correct"])
        return result

    t, s = groups(teacher), groups(student)
    assert t.keys() == s.keys()
    keys = sorted(t)
    for k in keys:
        assert t[k].keys() == s[k].keys()
    tv = torch.tensor(
        [[t[k][style] for style in sorted(t[k])] for k in keys], device="cuda", dtype=torch.float64
    )
    sv = torch.tensor(
        [[s[k][style] for style in sorted(t[k])] for k in keys], device="cuda", dtype=torch.float64
    )
    losses = (tv - sv).mean(-1)
    regressions = int(((tv == 1) & (sv == 0)).any(-1).sum())
    gains = int(((tv == 0) & (sv == 1)).any(-1).sum())
    n = len(keys)
    upper = (
        1.0 if regressions == n else float(beta.ppf(1 - alpha, regressions + 1, n - regressions))
    )
    return {
        "teacher_accuracy": float(tv.mean()),
        "student_accuracy": float(sv.mean()),
        "net_loss": float(losses.mean()),
        "net_loss_bootstrap": bootstrap(losses.tolist()),
        "regressing_groups": regressions,
        "improving_groups": gains,
        "groups": n,
        "loss_upper_bound": upper,
        "alpha": alpha,
        "passes": upper <= 0.01,
    }


def point(result):
    rows = result["results"]
    values = {}
    for op in ["add", "multiply", "divide"]:
        records = [r for r in rows["arithmetic"] if r["op"] == op]
        values[op] = sum(r["correct"] for r in records) / len(records)
        values[op + "_parse"] = sum(r["prediction"] is not None for r in records) / len(records)
    values["arc"] = sum(r["correct"] for r in rows["arc"]) / len(rows["arc"])
    values["ce"] = sum(r["ce_nats"] for r in rows["language"]) / len(rows["language"])
    return values


def read(out, method):
    return json.loads((out / "training" / f"{method}.json").read_text())


def screen(out, depth=1, steps=12000):
    directory = out / "evaluation/development"
    base = point(json.loads((directory / "original.json").read_text()))
    methods = {}
    architectures = {}
    for arch in SPEC["training"]["architectures"]:
        pair = []
        for act in ["eml", "silu"]:
            key = name(arch, act, depth, 1103, steps)
            path = directory / f"{key}.json"
            if not path.exists():
                continue
            record = read(out, key)
            control = name("bottleneck", act, depth, 1103, steps)
            c = read(out, control)
            values = point(json.loads(path.read_text()))
            cv = point(json.loads((directory / f"{control}.json").read_text()))

            def avg(r, metric):
                return sum(x[metric] for x in r["selection"].values()) / 2

            raw = 1 - avg(record, "raw_mse") / avg(c, "raw_mse")
            contrib = 1 - avg(record, "contribution_mse") / avg(c, "contribution_mse")
            checks = {
                "raw": raw >= 0.1,
                "contribution": contrib >= 0.1,
                "multiplication": values["multiply"] - cv["multiply"] >= 0.03,
                "other_accuracy": all(values[k] - cv[k] >= -0.02 for k in ["add", "divide", "arc"]),
                "language": values["ce"] - base["ce"] <= 0.02,
                "complete_effort": record["status"] == "complete" and record["steps"] == steps,
            }
            entry = {
                "point": values,
                "raw_relative_improvement": raw,
                "contribution_relative_improvement": contrib,
                "objective_relative_improvement": 1
                - record["selection_objective"] / c["selection_objective"],
                "checks": checks,
                "passes": all(checks.values()),
                "selected_step": record["selected_step"],
                "method": key,
            }
            methods[key] = entry
            pair.append(entry)
        if len(pair) == 2:
            architectures[arch] = {
                "passes": any(v["passes"] for v in pair)
                and all(v["checks"]["complete_effort"] for v in pair),
                "rank_score": sum(v["objective_relative_improvement"] for v in pair) / 2,
                "activation_error_improvement_both": all(
                    min(v["raw_relative_improvement"], v["contribution_relative_improvement"])
                    >= 0.1
                    for v in pair
                ),
                "late_best": any(v["selected_step"] >= 0.9 * steps for v in pair),
            }
    ordered = sorted(
        [k for k in architectures if k != "bottleneck"],
        key=lambda k: (-architectures[k]["rank_score"], k != "shortcut"),
    )
    promising = [k for k in ordered if architectures[k]["passes"]]
    diagnostic = [k for k in ordered if architectures[k]["activation_error_improvement_both"]]
    return {
        "depth": depth,
        "steps": steps,
        "original": base,
        "methods": methods,
        "architectures": architectures,
        "promising": promising[0] if promising else None,
        "diagnostic_leader": diagnostic[0] if diagnostic else None,
        "sources": sources("statistics.py"),
        "input_sha256": {str(p.relative_to(out)): digest(p) for p in directory.glob("*.json")},
    }


def confirmation(out):
    selected = json.loads((out / "selection.json").read_text())
    directory = out / "evaluation/confirmation"
    records = {
        k: json.loads((directory / f"{k}.json").read_text())
        for k in ["original", *selected["checkpoints"]]
    }
    original = records["original"]["results"]
    documents = json.loads((out / "data/language-documents.json").read_text())["confirmation"]
    summaries = {}
    for method, record in records.items():
        if method == "original":
            continue
        results = record["results"]
        metrics = {}
        for op in ["add", "multiply", "divide"]:
            t = [r for r in original["arithmetic"] if r["op"] == op]
            s = [r for r in results["arithmetic"] if r["op"] == op]
            assert len(t) == len(s) == 1024
            metrics[op] = accuracy(t, s)
        assert len(original["arc"]) == len(results["arc"]) == 1024
        metrics["arc"] = accuracy(original["arc"], results["arc"])
        assert (
            [r["document"] for r in original["language"]]
            == [r["document"] for r in results["language"]]
            == list(range(256))
        )
        delta = [
            s["ce_nats"] - t["ce_nats"] for s, t in zip(results["language"], original["language"])
        ]
        ce = bootstrap(delta, 0.05 / 15)
        domains = {}
        for doc, v in zip(documents, delta):
            domains.setdefault(doc["domain"], []).append(v)
        totals = torch.tensor(
            [[sum(v), len(v)] for v in domains.values()], device="cuda", dtype=torch.float64
        )
        rng = torch.Generator(device="cuda").manual_seed(917)
        samples = []
        for _ in range(100):
            ids = torch.randint(len(totals), (100, len(totals)), device="cuda", generator=rng)
            parts = totals[ids].sum(1)
            samples.append(parts[:, 0] / parts[:, 1])
        ce["hostname_clustered_upper"] = float(torch.quantile(torch.cat(samples), 1 - 0.05 / 15))
        ce["hostname_clusters"] = len(domains)
        ce["passes"] = max(ce["upper"], ce["hostname_clustered_upper"]) <= 0.02
        coefficients = record["student_accounting"]["coefficients"]
        storage = {
            "coefficients": coefficients,
            "block_compression": 56623104 / coefficients,
            "total_parameter_reduction": 1
            - record["accounting"]["parameters"] / records["original"]["accounting"]["parameters"],
        }
        summaries[method] = {
            "accuracy": metrics,
            "language_ce_increase": ce,
            "storage": storage,
            "quality_gate_passes": all(v["passes"] for v in metrics.values())
            and ce["passes"]
            and storage["block_compression"] >= 4,
            "point": point(record),
        }
    comparisons = {}
    for arch in ["bottleneck", selected["architecture"]]:
        for seed in [1103, 2207, 3301]:
            eml = name(arch, "eml", selected["depth"], seed, selected["steps"])
            silu = name(arch, "silu", selected["depth"], seed, selected["steps"])
            e, s = records[eml]["results"], records[silu]["results"]
            comparisons[f"{arch}-s{seed}"] = {
                op: accuracy(
                    [r for r in s["arithmetic"] if r["op"] == op],
                    [r for r in e["arithmetic"] if r["op"] == op],
                )
                for op in ["add", "multiply", "divide"]
            }
            comparisons[f"{arch}-s{seed}"]["arc"] = accuracy(s["arc"], e["arc"])
            comparisons[f"{arch}-s{seed}"]["ce_eml_minus_silu"] = bootstrap(
                [a["ce_nats"] - b["ce_nats"] for a, b in zip(e["language"], s["language"])]
            )
    return {
        "methods": summaries,
        "eml_vs_silu": comparisons,
        "original": point(records["original"]),
        "all_eml_seeds_pass": all(
            summaries[
                name(selected["architecture"], "eml", selected["depth"], seed, selected["steps"])
            ]["quality_gate_passes"]
            for seed in [1103, 2207, 3301]
        ),
        "inputs": {k: digest(directory / f"{k}.json") for k in records},
        "selection_sha256": digest(out / "selection.json"),
        "sources": sources("statistics.py"),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--confirmation", action="store_true")
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--steps", type=int, default=12000)
    a = p.parse_args()
    setup(73)
    out = root()
    if a.confirmation:
        result = confirmation(out)
        path = out / "confirmation-summary.json"
    else:
        result = screen(out, a.depth, a.steps)
        path = out / f"screen-d{a.depth}-n{a.steps}.json"
    save(path, result)
    print("SUMMARY", path, result.get("promising", result.get("all_eml_seeds_pass")), flush=True)


if __name__ == "__main__":
    main()
