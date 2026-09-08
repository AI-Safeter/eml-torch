"""Paired retention bounds and a conservative, explicit one-block expansion gate."""

import argparse
import json

import torch
from scipy.stats import beta

from .evaluate import roster
from .runtime import HERE, SPEC, digest, root, save, setup


def bootstrap(values, alpha=0.05, seed=73):
    values = torch.tensor(values, device="cuda", dtype=torch.float64)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    means = []
    for _ in range(100):
        ids = torch.randint(len(values), (100, len(values)), device="cuda", generator=generator)
        means.append(values[ids].mean(-1))
    samples = torch.cat(means)
    return {
        "mean": float(values.mean()),
        "two_sided_95": torch.quantile(
            samples, torch.tensor([0.025, 0.975], device="cuda", dtype=torch.float64)
        ).tolist(),
        "upper": float(torch.quantile(samples, 1 - alpha)),
    }


def accuracy(teacher, student, alpha, arithmetic=False):
    def group(rows):
        result = {}
        for row in rows:
            group = result.setdefault(row["id"], {})
            style = row.get("style", "qa")
            assert style not in group, "Duplicate evaluation identity"
            group[style] = row["correct"]
        return result

    t, s = group(teacher), group(student)
    assert t.keys() == s.keys()
    losses, regressions, gains, ta, sa = [], [], [], [], []
    for key in sorted(t):
        assert t[key].keys() == s[key].keys()
        tv = torch.tensor(list(t[key].values()), device="cuda", dtype=torch.float64)
        sv = torch.tensor([s[key][k] for k in t[key]], device="cuda", dtype=torch.float64)
        losses.append(float((tv - sv).mean()))
        regressions.append(bool(((tv == 1) & (sv == 0)).any()))
        gains.append(bool(((tv == 0) & (sv == 1)).any()))
        ta.append(float(tv.mean()))
        sa.append(float(sv.mean()))
    n, k = len(regressions), sum(regressions)
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))
    return {
        "teacher_accuracy": sum(ta) / n,
        "student_accuracy": sum(sa) / n,
        "net_loss": sum(losses) / n,
        "net_loss_bootstrap": bootstrap(losses),
        "regressing_groups": k,
        "improving_groups": sum(gains),
        "groups": n,
        "loss_upper_bound": upper,
        "alpha": alpha,
        "passes": upper <= SPEC["acceptance"]["accuracy_loss_upper_bound"],
        "teacher_zero_accuracy": not any(ta),
        "bound": "Clopper-Pearson bound on any within-group regression; improvements are reported separately",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--split", choices=["gate", "final"], default="gate")
    args = parser.parse_args()
    setup(73)
    out = root(args.output)
    directory = out / "evaluation" / args.split
    methods = roster(out)
    records = {name: json.loads((directory / f"{name}.json").read_text()) for name in methods}
    frozen_sha = digest(directory / "freeze.json")
    assert all(r["freeze_sha256"] == frozen_sha for r in records.values())
    # Five retention endpoints times three training seeds, within each family.
    # This is at least as conservative as the original twelve-accuracy-test adjustment.
    alpha = SPEC["acceptance"]["alpha"] / (5 * len(SPEC["replacement"]["seeds"]))
    original = records["original"]["results"]
    documents = json.loads((out / "data/language-documents.json").read_text())[args.split]
    expected_arithmetic = {
        (r["id"], r["op"], style)
        for r in json.loads((out / "data" / f"arithmetic-{args.split}.json").read_text())
        for style in SPEC["arithmetic"]["formats"]
    }
    expected_arc = {
        r["id"] for r in json.loads((out / "data" / f"arc-{args.split}.json").read_text())
    }
    for record in records.values():
        result = record["results"]
        identities = [(r["id"], r["op"], r["style"]) for r in result["arithmetic"]]
        assert (
            len(identities) == len(expected_arithmetic) and set(identities) == expected_arithmetic
        )
        assert (
            len(result["arc"]) == len(expected_arc)
            and {r["id"] for r in result["arc"]} == expected_arc
        )
        assert [r["document"] for r in result["language"]] == list(range(len(documents)))
        assert record["nonfinite_outputs"] == 0
    summaries = {}
    for name, record in records.items():
        if name == "original":
            continue
        results = record["results"]
        a = {
            op: accuracy(
                [r for r in original["arithmetic"] if r["op"] == op],
                [r for r in results["arithmetic"] if r["op"] == op],
                alpha,
                True,
            )
            for op in SPEC["arithmetic"]["operations"]
        }
        a["arc"] = accuracy(original["arc"], results["arc"], alpha)
        assert [r["document"] for r in original["language"]] == [
            r["document"] for r in results["language"]
        ]
        deltas = [
            s["ce_nats"] - t["ce_nats"] for s, t in zip(results["language"], original["language"])
        ]
        ce = bootstrap(deltas, alpha)
        ce["passes"] = ce["upper"] <= SPEC["acceptance"]["language_ce_increase_upper_bound_nats"]
        # Sensitivity check: resample whole hostnames, retaining document weights.
        by_domain = {}
        for document, delta in zip(documents, deltas):
            by_domain.setdefault(document["domain"], []).append(delta)
        totals = torch.tensor(
            [[sum(v), len(v)] for v in by_domain.values()], device="cuda", dtype=torch.float64
        )
        generator = torch.Generator(device="cuda").manual_seed(917)
        samples = []
        for _ in range(100):
            ids = torch.randint(len(totals), (100, len(totals)), device="cuda", generator=generator)
            selected = totals[ids].sum(1)
            samples.append(selected[:, 0] / selected[:, 1])
        domain_upper = float(torch.quantile(torch.cat(samples), 1 - alpha))
        ce["hostname_clustered_upper"] = domain_upper
        ce["hostname_clusters"] = len(totals)
        ce["hostname_sensitivity_passes"] = (
            domain_upper <= SPEC["acceptance"]["language_ce_increase_upper_bound_nats"]
        )
        base_count = records["original"]["accounting"]
        model_count = record["accounting"]
        student_coefficients = 56623104 - (base_count["coefficients"] - model_count["coefficients"])
        storage = {
            "student_coefficients": student_coefficients,
            "block_compression": 56623104 / student_coefficients,
            "model_parameter_reduction": 1 - model_count["parameters"] / base_count["parameters"],
        }
        passes = (
            all(r["passes"] for r in a.values())
            and ce["passes"]
            and ce["hostname_sensitivity_passes"]
            and storage["block_compression"] >= 4
        )
        summaries[name] = {
            "accuracy": a,
            "language_ce_increase": ce,
            "storage": storage,
            "quality_gate_passes": passes,
        }
    selection = json.loads((out / "training/selection.json").read_text())
    eml = selection["families"]["eml"].get("checkpoints", {})
    expand = len(eml) == len(SPEC["replacement"]["seeds"]) and all(
        summaries[name]["quality_gate_passes"] for name in eml
    )
    save(
        out / "evaluation" / f"{args.split}-summary.json",
        {
            "split": args.split,
            "methods": summaries,
            "eml_expansion_allowed": expand,
            "original_accuracy": {
                op: sum(r["correct"] for r in original["arithmetic"] if r["op"] == op)
                / sum(r["op"] == op for r in original["arithmetic"])
                for op in SPEC["arithmetic"]["operations"]
            },
            "evaluation_freeze_sha256": frozen_sha,
            "selection_sha256": digest(out / "training/selection.json"),
            "source_sha256": digest(HERE / "quality_summary.py"),
            "raw_sha256": {name: digest(directory / f"{name}.json") for name in methods},
            "deployment_targets_satisfied": False,
            "note": "Quality alone cannot establish deployment targets; parameters and latency are separate. A zero-accuracy teacher task is uninformative about arithmetic retention.",
        },
    )
    print("QUALITY SUMMARY", args.split, "expansion allowed", expand, flush=True)


if __name__ == "__main__":
    main()
