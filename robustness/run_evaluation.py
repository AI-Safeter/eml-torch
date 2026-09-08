"""Run audits first, then the complete frozen scalar evaluation and analysis."""

import argparse
import json

from runtime import RUNS, configure
from runtime import run_stage as run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen17b", "qwen4b", "smollm", "gemma"])
    args = parser.parse_args()
    configure(args.model)
    run("verify_sources.py")
    for op in ["add", "multiply", "divide"]:
        out = RUNS / args.model / op
        for directory in ["heads", "heads-active-r32-g0", "heads-active-r32-g0.1"]:
            status = json.loads((out / directory / "completed.json").read_text())
            assert (
                status["status"] == "complete_grid"
                and status["candidates"] == status["grid_total"] == 10
            )
        # A previous launcher omitted the operation argument for this control.
        # Ensure every required control exists before freezing/evaluating choices.
        if not (out / "sparse-neurons.json").exists():
            run("src/sparse_neurons.py", op)
    if any(
        not (RUNS / args.model / op / "linear-controls.json").exists()
        for op in ["add", "multiply", "divide"]
    ):
        run("src/linear_controls.py")
    for op in ["add", "multiply", "divide"]:
        run("src/evaluate_heads.py", op, "--validate-only")
        run("geometry_evaluate.py", args.model, op, "--validation-only")
        run("src/evaluate_heads.py", op)
        run("seed_evaluate.py", args.model, op)
        run("geometry_evaluate.py", args.model, op)
        run("analyze.py", args.model, op)
        run("per_style.py", args.model, op)
    print("COMPLETE SCALAR EVALUATION", args.model, flush=True)


if __name__ == "__main__":
    main()
