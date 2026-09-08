"""Evaluate every frozen scalar seed on primary ordinary and causal tests."""

import argparse
import json

import torch
from runtime import RUNS, configure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen17b", "qwen4b", "smollm"])
    parser.add_argument("operation", choices=["add", "multiply", "divide"])
    args = parser.parse_args()
    configure(args.model)
    from data import STYLES, save
    from evaluate_heads import Replacements, digest, freeze, interventions, ordinary
    from model_io import expand, load, setup

    setup()
    out = RUNS / args.model / args.operation
    directories = ["heads", "heads-active-r32-g0", "heads-active-r32-g0.1"]
    selected = freeze(out, directories)
    heads = {}
    for directory in directories:
        protocol = json.loads((out / directory / "protocol.json").read_text())
        records = json.loads((out / directory / "candidates.json").read_text())
        assert len(records) == 10
        for row in records:
            if row["status"] != "complete":
                continue
            checkpoint = f"{directory}/{row['name']}.pt"
            heads[f"{directory}/{row['kind']}-seed-{row['seed']}"] = {
                **row,
                "features": protocol["features"],
                "rank": protocol["rank"],
                "checkpoint": checkpoint,
                "sha256": digest(out / checkpoint),
            }
    selected = {**selected, "heads": heads, "sparse": {}, "linear": {}}
    path = out / "all-seeds-selection.json"
    if path.exists():
        assert json.loads(path.read_text()) == selected
    else:
        save(path, selected)
    rep = Replacements(out, selected)
    rep.kinds = ["original", *rep.models]
    model, tok = load()
    problems = json.loads((out / "problems.json").read_text())
    rows = expand(problems["test"], STYLES)
    with torch.inference_mode():
        for label, function in [("ordinary", ordinary), ("interventions", interventions)]:
            target = out / f"{label}-all-seeds.json"
            if not target.exists():
                save(target, function(model, tok, rep, rows))
    print("ALL SEEDS EVALUATED", args.model, args.operation, flush=True)


if __name__ == "__main__":
    main()
