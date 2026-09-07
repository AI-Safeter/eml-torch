"""Aggregate held-out results and paired uncertainty using CUDA."""

import json

import torch
from common import OUT, dump, setup


def tensor(x):
    return torch.tensor(x, device="cuda", dtype=torch.float64)


def correlation(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 1e-12 else None


def ordinary(data):
    result = {}
    base = data["original"]
    n = len(base)
    idx = torch.randint(n, (5000, n), device="cuda")
    for name, rows in data.items():
        correct = tensor([r["correct"] for r in rows])
        basecorrect = tensor([r["correct"] for r in base])
        delta = correct - basecorrect
        y = tensor([r["true_coefficient"] for r in rows])
        pred = tensor([r["predicted_coefficient"] for r in rows])
        ss = (y - y.mean()).square().sum()
        result[name] = {
            "n": n,
            "exact_answer_accuracy": float(correct.mean()),
            "accuracy_change_pp": float(delta.mean() * 100),
            "accuracy_change_pp_ci95": torch.quantile(
                delta[idx].mean(1) * 100, tensor([0.025, 0.975])
            )
            .cpu()
            .tolist(),
            "answer_agreement": float(tensor([r["answer_agreement"] for r in rows]).mean()),
            "first_digit_accuracy": float(tensor([r["first_digit_correct"] for r in rows]).mean()),
            "first_digit_agreement": float(
                tensor([r["first_digit_agreement"] for r in rows]).mean()
            ),
            "mean_first_digit_kl": float(tensor([r["kl"] for r in rows]).mean()),
            "coefficient_r2": float(1 - (y - pred).square().sum() / ss),
            "coefficient_rmse": float((y - pred).square().mean().sqrt()),
        }
    return result


def intervention(data):
    result = {}
    keys = [(r["a"], r["b"], r["c"]) for r in data["original"] if r["alpha"] == 0]
    alphas = [0.125, 0.375, 0.625, 0.875]
    tables = {
        name: {(r["a"], r["b"], r["c"], r["alpha"]): r for r in rows} for name, rows in data.items()
    }
    effects = {}
    for name, table in tables.items():
        effects[name] = tensor(
            [[table[(*k, a)]["margin"] - table[(*k, 0.0)]["margin"] for a in alphas] for k in keys]
        )
    truth = effects["original"]
    norm = truth.square().mean().sqrt()
    idx = torch.randint(len(keys), (5000, len(keys)), device="cuda")
    for name, rows in data.items():
        pred = effects[name]
        err = pred - truth
        boots = (
            err[idx].square().mean((1, 2)) / truth[idx].square().mean((1, 2)).clamp(min=1e-12)
        ).sqrt()
        table = tables[name]
        base = tables["original"]
        coefficient = tensor([r["true_coefficient"] for r in rows])
        fit = tensor([r["predicted_coefficient"] for r in rows])
        endpoint = tensor([table[(*k, 1.0)]["margin"] - table[(*k, 0.0)]["margin"] for k in keys])
        trueend = tensor([base[(*k, 1.0)]["margin"] - base[(*k, 0.0)]["margin"] for k in keys])
        active = truth.abs() > 0.1
        result[name] = {
            "pairs": len(keys),
            "offgrid_interventions": truth.numel(),
            "response_correlation": correlation(pred.flatten(), truth.flatten()),
            "response_nrmse": float(err.square().mean().sqrt() / norm),
            "response_nrmse_ci95": torch.quantile(boots, tensor([0.025, 0.975])).cpu().tolist(),
            "response_sign_agreement_above_0_1_logit": float(
                (pred[active].sign() == truth[active].sign()).double().mean()
            ),
            "reference_response_rms_logits": float(norm),
            "endpoint_effect_correlation": correlation(endpoint, trueend),
            "mean_endpoint_margin_effect": float(endpoint.mean()),
            "mean_kl_vs_original_same_intervention": float(
                tensor([r["kl_vs_same_intervention_original"] for r in rows]).mean()
            ),
            "coefficient_r2": float(
                1
                - (coefficient - fit).square().sum()
                / (coefficient - coefficient.mean()).square().sum()
            ),
        }
    return result


def main():
    setup()
    result = {}
    for path in sorted(OUT.glob("ordinary-*.json")):
        name = path.stem.removeprefix("ordinary-")
        causal = OUT / f"interventions-{name}.json"
        if not causal.exists():
            continue
        result[name] = {
            "ordinary": ordinary(json.loads(path.read_text())),
            "interventions": intervention(json.loads(causal.read_text())),
        }
    dump("summary.json", result)
    for suite, res in result.items():
        print(
            suite,
            json.dumps(
                {
                    k: {
                        "accuracy": v["exact_answer_accuracy"],
                        "coefficient_r2": v["coefficient_r2"],
                        "response_correlation": res["interventions"]
                        .get(k, {})
                        .get("response_correlation"),
                        "response_nrmse": res["interventions"].get(k, {}).get("response_nrmse"),
                    }
                    for k, v in res["ordinary"].items()
                    if k
                    in ["original", "eml_observed", "eml_intervention", "cubic", "network", "mean"]
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
