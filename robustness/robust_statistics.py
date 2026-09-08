"""Operand-group GPU statistics and conservative rare-regression bounds."""

import math

import torch
from scipy.stats import beta


def exact_upper(k, n, alpha):
    if n < 1 or not 0 <= k <= n or not 0 < alpha < 1:
        raise ValueError("Require n>=1, 0<=k<=n, and 0<alpha<1")
    return 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))


def loss_bound(original, replacement, cells=9, alpha=0.05):
    """Rows are independent operands; columns are correlated prompt formats.

    E[net loss]/S <= E[max(net loss,0)]/S = sum P(net loss>=j)/S.
    Simultaneous upper binomial bounds follow by a union bound over j and cells.
    Source for exact intervals: scipy.stats.binomtest(...).proportion_ci.
    """
    assert original.shape == replacement.shape and original.ndim == 2
    assert original.is_cuda and replacement.is_cuda
    assert torch.all((original == 0) | (original == 1))
    assert torch.all((replacement == 0) | (replacement == 1))
    n, styles = original.shape
    loss = (original.to(torch.int64) - replacement.to(torch.int64)).sum(1)
    counts = [int((loss >= j).sum()) for j in range(1, styles + 1)]
    bounds = [exact_upper(k, n, alpha / (styles * cells)) for k in counts]
    return {
        "operand_groups": n,
        "formats": styles,
        "simultaneous_cells": cells,
        "alpha": alpha,
        "loss_threshold_counts": counts,
        "threshold_upper_bounds": bounds,
        "accuracy_loss_pp": float(loss.double().mean() / styles * 100),
        "accuracy_loss_upper_pp": sum(bounds) / styles * 100,
        "passes_one_percentage_point_bound": sum(bounds) / styles <= 0.01,
        "assumption": "Independent sampled operand groups; formats within each group may depend.",
    }


def response_metrics(truth, pred, cells=9, draws=20000, seed=20260926):
    assert truth.shape == pred.shape and truth.ndim == 2 and truth.is_cuda
    truth, pred = truth.double(), pred.double()
    assert torch.isfinite(truth).all() and torch.isfinite(pred).all()
    error = (pred - truth).square().mean(1)
    energy = truth.square().mean(1)
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
    generator = torch.Generator(device="cuda").manual_seed(seed)
    boot = []
    for start in range(0, draws, 500):
        indices = torch.randint(
            len(truth), (min(500, draws - start), len(truth)), device="cuda", generator=generator
        )
        m = moments[indices].mean(1)
        variance = ((m[:, 2] - m[:, 0] ** 2) * (m[:, 3] - m[:, 1] ** 2)).clamp_min(0)
        corr = (m[:, 4] - m[:, 0] * m[:, 1]) / variance.sqrt().clamp_min(1e-20)
        boot.append(
            torch.stack(
                [(error[indices].mean(1) / energy[indices].mean(1).clamp_min(1e-20)).sqrt(), corr],
                1,
            )
        )
    boot = torch.cat(boot)
    p, t = pred - pred.mean(), truth - truth.mean()
    den = p.norm() * t.norm()
    result = {
        "operand_groups": len(truth),
        "response_rms": float(energy.mean().sqrt()),
        "nrmse": float((error.mean() / energy.mean()).sqrt()) if energy.mean() > 0 else None,
        "correlation": float((p * t).sum() / den) if den > 0 else None,
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "simultaneous_cells": cells,
    }
    for label, tail in [("ci95", 0.025), ("simultaneous_ci95", 0.025 / cells)]:
        bounds = torch.quantile(
            boot, torch.tensor([tail, 1 - tail], device="cuda", dtype=torch.float64), dim=0
        )
        result[f"nrmse_{label}"] = bounds[:, 0].tolist() if energy.mean() > 0 else None
        result[f"correlation_{label}"] = bounds[:, 1].tolist() if den > 0 else None
    return result


def validate():
    from scipy.stats import binomtest

    for n in [1, 32, 1024]:
        for k in sorted({0, n // 2, n}):
            a = 0.05 / 27
            reference = (
                binomtest(k, n, alternative="less").proportion_ci(confidence_level=1 - a).high
            )
            assert math.isclose(exact_upper(k, n, a), reference, abs_tol=1e-10)
    base = torch.ones(1024, 3, device="cuda", dtype=torch.bool)
    zero = loss_bound(base, base)
    assert 0 < zero["accuracy_loss_upper_pp"] < 1
    regression = loss_bound(base, torch.zeros_like(base))
    assert regression["accuracy_loss_upper_pp"] == 100
    truth = torch.arange(1, 1025, device="cuda", dtype=torch.float64).reshape(256, 4)
    identity = response_metrics(truth, truth, draws=1000)
    assert identity["nrmse"] == 0 and math.isclose(identity["correlation"], 1)
    scaled = response_metrics(truth, truth * 1.1, draws=1000)
    assert math.isclose(scaled["nrmse"], 0.1, abs_tol=1e-12)
    print(
        "GPU STATISTICS PASSED; zero-regression simultaneous upper loss",
        zero["accuracy_loss_upper_pp"],
        "pp",
    )


if __name__ == "__main__":
    validate()
