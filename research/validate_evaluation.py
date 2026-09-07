"""GPU validation of patching invariants and grouped metric calculations."""

import json
import time

import torch

from analyze_heads import intervention
from data import ROOT, STYLES, save
from evaluate_heads import Replacements, interventions, ordinary, parse_answer
from model_io import expand, load, setup


def main():
    setup()
    out = ROOT / "add"
    selected = {"heads": {}}
    for directory, feature in [("heads", "pls"), ("heads-active-r32-g0", "active")]:
        rows = json.loads((out / directory / "candidates.json").read_text())
        for prefix in ["eml", "silu"]:
            row = min(
                [
                    r
                    for r in rows
                    if r["status"] == "complete"
                    and r["kind"].startswith(prefix)
                    and r["response_weight"] == 4
                ],
                key=lambda r: r["validation_objective"],
            )
            selected["heads"][directory + "/" + ("eml" if prefix == "eml" else "neural")] = {
                **row,
                "features": feature,
                "rank": 32,
                "checkpoint": f"{directory}/{row['name']}.pt",
            }
    rep = Replacements(out, selected)
    model, tok = load()
    data = json.loads((out / "problems.json").read_text())
    rows = expand(data["validation"][:4], STYLES)
    with torch.inference_mode():
        normal = ordinary(model, tok, rep, rows)
        causal = interventions(model, tok, rep, rows)
    assert all(r["true_coefficient"] == r["predicted_coefficient"] for r in normal["restore"])
    assert parse_answer(" 123. ") == 123
    assert parse_answer("1234") == 1234
    assert parse_answer("123 is the result") is None
    assert parse_answer("12.5") is None
    metrics, comparisons = intervention(causal)
    assert metrics["restore"]["response_nrmse"] == 0
    assert abs(metrics["restore"]["response_correlation"] - 1) < 1e-12
    assert abs(metrics["mean"]["response_nrmse"] - 1) < 1e-12
    assert metrics["mean"]["response_correlation"] is None
    assert metrics["restore"]["independent_operand_pairs"] == 4
    assert metrics["restore"]["offgrid_interventions"] == 48
    # Artificial prediction at half the true response has exactly one-half NRMSE.
    synthetic = {"original": causal["original"], "half": []}
    baseline = {(r["row_index"]): r["margin"] for r in causal["original"] if r["alpha"] == 0}
    for row in causal["original"]:
        base = baseline[row["row_index"]]
        synthetic["half"].append({**row, "margin": base + 0.5 * (row["margin"] - base)})
    m, _ = intervention(synthetic)
    assert abs(m["half"]["response_nrmse"] - 0.5) < 1e-12
    save(
        ROOT / "evaluation-validation.json",
        {
            "completed_epoch": time.time(),
            "gpu": torch.cuda.get_device_name(),
            "validation_pairs": 4,
            "prompt_cases": 12,
            "checks": [
                "Exact restore logits and generation",
                "Zero-intervention reference equivalence",
                "Mean replacement has zero response",
                "Pair-grouping across formats",
                "Analytic half-response NRMSE",
                "Strict complete-answer parser",
            ],
            "test_data_used": False,
            "validation_models_not_final_selection": selected,
            "metrics": metrics,
            "comparisons": comparisons,
        },
    )
    print("EVALUATION VALIDATION PASSED", flush=True)


if __name__ == "__main__":
    main()
