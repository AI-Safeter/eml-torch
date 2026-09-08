"""Grouped paired uncertainty for arithmetic tests; numerical analysis runs on CUDA."""

import argparse
import json

import torch

from data import ROOT, save
from model_io import setup


def tensor(values):
    return torch.tensor(values, dtype=torch.float64, device="cuda")


def interval(samples):
    return torch.quantile(samples, tensor([0.025, 0.975])).cpu().tolist()


def pair_key(row):
    return row["a"], row["b"], row["c"]


def ordinary(data):
    groups = sorted({pair_key(r) for r in data["original"]})
    styles = sorted({r["style"] for r in data["original"]})
    tables = {k: {(pair_key(r), r["style"]): r for r in v} for k, v in data.items()}
    indices = torch.randint(len(groups), (5000, len(groups)), device="cuda")
    truth = tensor([[tables["original"][(g, s)]["correct"] for s in styles] for g in groups])
    result = {}
    for name, table in tables.items():

        def field(key, table=table):
            return tensor([[table[(g, s)][key] for s in styles] for g in groups])

        correct = field("correct")
        paired_delta = (correct - truth).mean(1) * 100
        y, pred = field("true_coefficient"), field("predicted_coefficient")
        result[name] = {
            "independent_operand_pairs": len(groups),
            "prompt_cases": correct.numel(),
            "exact_answer_accuracy": float(correct.mean()),
            "accuracy_ci95": interval(correct.mean(1)[indices].mean(1)),
            "accuracy_change_pp": float(paired_delta.mean()),
            "accuracy_change_pp_ci95": interval(paired_delta[indices].mean(1)),
            "answer_agreement": float(field("answer_agreement").mean()),
            "exact_text_agreement": float(field("exact_text_agreement").mean()),
            "first_digit_agreement": float(field("first_digit_agreement").mean()),
            "first_digit_accuracy": float(field("first_digit_correct").mean()),
            "mean_first_digit_kl": float(field("kl").mean()),
            "coefficient_rmse": float((pred - y).square().mean().sqrt()),
            "coefficient_r2": float(1 - (pred - y).square().sum() / (y - y.mean()).square().sum()),
        }
    return result


def correlation(pred, truth):
    p, t = pred.flatten() - pred.mean(), truth.flatten() - truth.mean()
    den = p.norm() * t.norm()
    return float(p @ t / den) if den > 1e-12 else None


def intervention(data):
    groups = sorted({pair_key(r) for r in data["original"]})

    def condition(row):
        return row["style"], row.get("feature", -1)

    styles = sorted({condition(r) for r in data["original"]})
    alphas = [-0.25, 0.25] if "feature" in data["original"][0] else [0.125, 0.375, 0.625, 0.875]
    tables = {
        k: {(pair_key(r), condition(r), r["alpha"]): r for r in rows} for k, rows in data.items()
    }
    effects = {}
    for name, table in tables.items():
        effects[name] = tensor(
            [
                [
                    table[(g, s, a)]["margin"] - table[(g, s, 0.0)]["margin"]
                    for s in styles
                    for a in alphas
                ]
                for g in groups
            ]
        )
    truth = effects["original"]
    energy = truth.square().mean(1)
    indices = torch.randint(len(groups), (5000, len(groups)), device="cuda")
    boot_energy = energy[indices].mean(1).clamp_min(1e-12)
    result = {}
    errors = {}
    for name, pred in effects.items():
        error = (pred - truth).square().mean(1)
        errors[name] = error
        active = truth.abs() > 0.1
        rows = data[name]
        y, fit = (
            tensor([r["true_coefficient"] for r in rows]),
            tensor([r["predicted_coefficient"] for r in rows]),
        )
        # Bootstrap moments by operand group, preserving all formats and alphas together.
        moments = torch.stack(
            [
                pred.mean(1),
                truth.mean(1),
                pred.square().mean(1),
                truth.square().mean(1),
                (pred * truth).mean(1),
            ],
            1,
        )
        bm = moments[indices].mean(1)
        variance_product = (bm[:, 2] - bm[:, 0].square()).clamp_min(0) * (
            bm[:, 3] - bm[:, 1].square()
        ).clamp_min(0)
        valid = variance_product > 1e-20
        boot_corr = (bm[valid, 4] - bm[valid, 0] * bm[valid, 1]) / variance_product[valid].sqrt()
        result[name] = {
            "independent_operand_pairs": len(groups),
            "offgrid_interventions": truth.numel(),
            "response_correlation": correlation(pred, truth),
            "response_correlation_ci95": interval(boot_corr) if valid.all() else None,
            "response_nrmse": float((error.mean() / energy.mean()).sqrt()),
            "response_nrmse_ci95": interval((error[indices].mean(1) / boot_energy).sqrt()),
            "reference_response_rms_logits": float(energy.mean().sqrt()),
            "response_sign_agreement_above_0_1_logit": float(
                (pred[active].sign() == truth[active].sign()).double().mean()
            )
            if active.any()
            else None,
            "mean_kl_vs_original_same_intervention": float(
                tensor([r["kl_vs_same_intervention_original"] for r in rows]).mean()
            ),
            "coefficient_r2": float(1 - (y - fit).square().sum() / (y - y.mean()).square().sum()),
        }
    comparisons = {}
    for name in effects:
        if not name.endswith("/eml"):
            continue
        other = name.removesuffix("/eml") + "/neural"
        delta = errors[name] - errors[other]
        comparisons[name] = {
            "neural_control": other,
            "eml_minus_neural_response_mse_logits_squared": float(delta.mean()),
            "paired_difference_ci95": interval(delta[indices].mean(1)),
            "eml_to_neural_response_rmse_ratio": float(
                (errors[name].mean() / errors[other].mean()).sqrt()
            ),
            "ratio_ci95": interval(
                (
                    errors[name][indices].mean(1) / errors[other][indices].mean(1).clamp_min(1e-12)
                ).sqrt()
            ),
        }
    return result, comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["add", "multiply", "divide"])
    args = parser.parse_args()
    setup(seed=20260915)
    out = ROOT / args.operation
    results = {}
    for path in sorted(out.glob("ordinary-*.json")):
        suite = path.stem.removeprefix("ordinary-")
        other = out / f"interventions-{suite}.json"
        if not other.exists():
            continue
        normal = json.loads(path.read_text())
        causal = json.loads(other.read_text())
        linear_normal = out / f"linear-ordinary-{suite}.json"
        linear_causal = out / f"linear-interventions-{suite}.json"
        if linear_normal.exists() and linear_causal.exists():
            normal["linear"] = json.loads(linear_normal.read_text())["linear"]
            causal["linear"] = json.loads(linear_causal.read_text())["linear"]
        all_styles = [None] + sorted({r["style"] for r in normal["original"]})
        for style in all_styles:
            n = (
                normal
                if style is None
                else {k: [r for r in rows if r["style"] == style] for k, rows in normal.items()}
            )
            c = (
                causal
                if style is None
                else {k: [r for r in rows if r["style"] == style] for k, rows in causal.items()}
            )
            effects, comparisons = intervention(c)
            results[suite if style is None else f"{suite}/{style}"] = {
                "ordinary": ordinary(n),
                "interventions": effects,
                "eml_vs_neural": comparisons,
            }
        save(out / "results.json", results)
    coordinates = out / "coordinate-interventions.json"
    if coordinates.exists():
        coordinate_data = json.loads(coordinates.read_text())
        linear_coordinates = out / "linear-coordinate-interventions.json"
        if linear_coordinates.exists():
            coordinate_data["linear"] = json.loads(linear_coordinates.read_text())["linear"]
        metrics, comparisons = intervention(coordinate_data)
        results["coordinate-interventions"] = {
            "interventions": metrics,
            "eml_vs_neural": comparisons,
            "interpretation": "Local off-manifold interventions along active input coordinates; operand-group bootstrap preserves all features and formats.",
        }
        for feature in [0, 1, 7, 31]:
            filtered = {
                name: [r for r in rows if r["feature"] == feature]
                for name, rows in coordinate_data.items()
            }
            metrics, comparisons = intervention(filtered)
            results[f"coordinate-interventions/feature-{feature}"] = {
                "interventions": metrics,
                "eml_vs_neural": comparisons,
            }
        save(out / "results.json", results)
    if "test-known-formats" in results:
        main_result = results["test-known-formats"]
        checks = {}
        for name in main_result["ordinary"]:
            if not name.endswith("/eml"):
                continue
            n, c = main_result["ordinary"][name], main_result["interventions"][name]
            point = {
                "accuracy": n["accuracy_change_pp"] >= -1,
                "correlation": c["response_correlation"] is not None
                and c["response_correlation"] >= 0.9,
                "normalized_error": c["response_nrmse"] <= 0.2,
            }
            conservative = {
                "accuracy": n["accuracy_change_pp_ci95"][0] >= -1,
                "correlation": c["response_correlation_ci95"] is not None
                and c["response_correlation_ci95"][0] >= 0.9,
                "normalized_error": c["response_nrmse_ci95"][1] <= 0.2,
            }
            checks[name] = {
                "point_estimate_checks": point,
                "all_point_checks_pass": all(point.values()),
                "interval_checks": conservative,
                "all_interval_checks_pass": all(conservative.values()),
            }
        save(
            out / "outcome.json",
            {
                "checks": checks,
                "bootstrap": "5000 paired operand-group resamples; formats and alpha strengths remain grouped",
                "nrmse_denominator": "RMS original margin response, not centered standard deviation",
                "status": "Exploratory family comparisons; extensive validation tuning disclosed",
            },
        )
    print("ANALYSIS DONE", args.operation, flush=True)


if __name__ == "__main__":
    main()
