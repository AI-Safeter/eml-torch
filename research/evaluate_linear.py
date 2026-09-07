"""Disclosed supplementary linear baseline; preserve the original frozen selections."""

import json
import time

import torch

from data import ROOT, STYLES, save
from evaluate_heads import (
    Replacements,
    coordinate_interventions,
    digest,
    interventions,
    ordinary,
)
from model_io import expand, load, setup


def main():
    setup()
    model, tok = load()
    for op in ["add", "multiply", "divide"]:
        out = ROOT / op
        selection = json.loads((out / "selection.json").read_text())
        row = json.loads((out / "linear-controls.json").read_text())["selected"]
        name = row["name"] + ".pt"
        selection["linear"] = {"linear": {**row, "checkpoint": name, "sha256": digest(out / name)}}
        save(
            out / "linear-selection.json",
            {
                "frozen_epoch": time.time(),
                "selection": selection["linear"],
                "supplementary_control_added_after_initial_test_results": True,
                "fit_and_selection_data": "Training and validation only",
            },
        )
        rep = Replacements(out, selection)
        rep.kinds = ["original", "restore", "linear"]
        problems = json.loads((out / "problems.json").read_text())
        suites = [
            ("test-known-formats", expand(problems["test"], STYLES)),
            ("test-new-format", expand(problems["test"], ["instruction"])),
            ("shift", expand(problems["shift"], STYLES)),
        ]
        if "carry" in problems:
            suites.append(("carry", expand(problems["carry"], STYLES)))
        with torch.inference_mode():
            for suite, rows in suites:
                for label, function in [
                    ("ordinary", ordinary),
                    ("interventions", interventions),
                ]:
                    target = out / f"linear-{label}-{suite}.json"
                    if not target.exists():
                        save(target, function(model, tok, rep, rows))
            target = out / "linear-coordinate-interventions.json"
            if not target.exists():
                save(
                    target,
                    coordinate_interventions(
                        model, tok, rep, expand(problems["test"][:32], STYLES)
                    ),
                )
        save(out / "linear-evaluation-completed.json", {"completed_epoch": time.time()})
        print("LINEAR EVALUATION DONE", op, flush=True)


if __name__ == "__main__":
    main()
