"""Disaggregate primary and held-out formats without changing selection criteria."""

import argparse
import json

from analyze import interventions, ordinary
from runtime import RUNS, configure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen17b", "qwen4b", "smollm", "gemma"])
    parser.add_argument("operation", choices=["add", "multiply", "divide"])
    args = parser.parse_args()
    configure(args.model)
    from data import save
    from model_io import setup

    setup(20260926)
    out = RUNS / args.model / args.operation
    results = {"ordinary": {}, "interventions": {}}
    for suite in ["test-known-formats", "test-new-formats"]:
        normal = json.loads((out / f"ordinary-{suite}.json").read_text())
        causal = json.loads((out / f"interventions-{suite}.json").read_text())
        for style in sorted({r["style"] for r in normal["original"]}):
            n = {k: [r for r in rows if r["style"] == style] for k, rows in normal.items()}
            c = {k: [r for r in rows if r["style"] == style] for k, rows in causal.items()}
            results["ordinary"][f"ordinary-{suite}/{style}"] = ordinary(n)
            results["interventions"][f"interventions-{suite}/{style}"] = interventions(c)
    save(out / "per-style-results.json", results)
    print("PER-FORMAT ANALYSIS COMPLETE", args.model, args.operation, flush=True)


if __name__ == "__main__":
    main()
