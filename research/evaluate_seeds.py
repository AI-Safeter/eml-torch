"""Evaluate every new initialization on the same additional operand cohort."""

import json
import time

import torch

from data import ROOT, STYLES, save
from evaluate_heads import Replacements, digest, interventions, ordinary
from model_io import expand, load, setup


def main():
    setup()
    out = ROOT / "add"
    folder = out / "seed-replication"
    assert all((folder / f"completed-{seed}.json").exists() for seed in [59, 71, 83, 97])
    selection = json.loads((out / "selection.json").read_text())
    selection["heads"], selection["sparse"], selection["linear"] = {}, {}, {}
    for seed in [59, 71, 83, 97]:
        for row in json.loads((folder / f"seed-{seed}.json").read_text()):
            family = "eml" if row["kind"] == "eml_square" else "neural"
            checkpoint = f"seed-replication/{row['name']}.pt"
            selection["heads"][f"seed-{seed}/{family}"] = {
                **row,
                "features": "active",
                "rank": 32,
                "checkpoint": checkpoint,
                "sha256": digest(out / checkpoint),
            }
    path = folder / "selection.json"
    if path.exists():
        frozen = json.loads(path.read_text())
        assert frozen["selection"] == selection, "Preserve the frozen seed selection"
        assert frozen["test_sha256"] == digest(out / "minimal-problems.json")
    else:
        save(
            path,
            {
                "frozen_epoch": time.time(),
                "selection": selection,
                "test_sha256": digest(out / "minimal-problems.json"),
                "no_selection_among_seeds": True,
            },
        )
    rep = Replacements(out, selection)
    rows = expand(json.loads((out / "minimal-problems.json").read_text())["test"], STYLES)
    model, tok = load()
    with torch.inference_mode():
        for name, function in [
            ("ordinary", ordinary),
            ("interventions", interventions),
        ]:
            path = out / f"{name}-seed-replication.json"
            if not path.exists():
                save(path, function(model, tok, rep, rows))
    save(folder / "evaluation-completed.json", {"completed_epoch": time.time()})
    print("SEED EVALUATION DONE", flush=True)


if __name__ == "__main__":
    main()
