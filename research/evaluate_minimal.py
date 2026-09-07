"""Freeze compact heads and evaluate additional, previously untouched operand pairs."""

import argparse
import json
import time

import torch

from data import ROOT, STYLES, save
from evaluate_heads import Replacements, digest, interventions, ordinary
from model_io import expand, load, setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["add", "multiply", "divide"])
    args = parser.parse_args()
    setup()
    out = ROOT / args.operation
    target = out / "minimal-selection.json"
    if target.exists():
        frozen = json.loads(target.read_text())
        selection = frozen["selection"]
    else:
        assert (out / "minimal-heads/completed.json").exists()
        rows = json.loads((out / "minimal-heads/candidates.json").read_text())
        complete = [r for r in rows if r["status"] == "complete"]
        eml = [
            r
            for r in complete
            if r["kind"].startswith("eml")
            and any(
                n["kind"] == "silu" and n["rank"] == r["rank"] and n["width"] == r["width"]
                for n in complete
            )
        ]
        eligible = [
            r for r in eml if r["validation_mse"] <= 0.005 and r["validation_response_mse"] <= 0.005
        ]
        best = min(eml, key=lambda r: r["validation_objective"])
        compact = (
            min(
                eligible,
                key=lambda r: (r["stored_coefficients"], r["validation_objective"]),
            )
            if eligible
            else best
        )
        selection = json.loads((out / "selection.json").read_text())
        for name, row in [("minimal", compact), ("best_small", best)]:
            neural = min(
                [
                    r
                    for r in complete
                    if r["kind"] == "silu"
                    and r["rank"] == row["rank"]
                    and r["width"] == row["width"]
                ],
                key=lambda r: r["validation_objective"],
            )
            for family, chosen in [("eml", row), ("neural", neural)]:
                checkpoint = f"minimal-heads/{chosen['name']}.pt"
                selection["heads"][f"{name}/{family}"] = {
                    **chosen,
                    "features": "active",
                    "checkpoint": checkpoint,
                    "sha256": digest(out / checkpoint),
                }
        linear = json.loads((out / "linear-controls.json").read_text())["selected"]
        checkpoint = linear["name"] + ".pt"
        selection["linear"] = {
            "linear": {
                **linear,
                "checkpoint": checkpoint,
                "sha256": digest(out / checkpoint),
            }
        }
        frozen = {
            "frozen_epoch": time.time(),
            "small_head_validation_threshold_met": bool(eligible),
            "selection": selection,
            "test_sha256": digest(out / "minimal-problems.json"),
            "test_inputs_and_outputs_not_used_in_fitting_or_selection": True,
        }
        save(target, frozen)
    assert digest(out / "minimal-problems.json") == frozen["test_sha256"]
    for spec in selection["heads"].values():
        assert digest(out / spec["checkpoint"]) == spec["sha256"]
    rep = Replacements(out, selection)
    rows = expand(json.loads((out / "minimal-problems.json").read_text())["test"], STYLES)
    model, tok = load()
    with torch.inference_mode():
        for label, function in [
            ("ordinary", ordinary),
            ("interventions", interventions),
        ]:
            path = out / f"{label}-minimal-fresh.json"
            if not path.exists():
                save(path, function(model, tok, rep, rows))
    save(
        out / "minimal-evaluation-completed.json",
        {"completed_epoch": time.time(), "independent_pairs": len(rows) // len(STYLES)},
    )
    print("MINIMAL EVALUATION DONE", args.operation, flush=True)


if __name__ == "__main__":
    main()
