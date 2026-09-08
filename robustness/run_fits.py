"""Complete the fixed scalar search and controls; never read test outputs."""

import argparse
import subprocess
import sys

from runtime import HERE, RUNS, configure, run_stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen17b", "qwen4b", "smollm", "gemma"])
    args = parser.parse_args()
    configure(args.model)
    assert (RUNS / args.model / "derivative-completed.json").exists()
    subprocess.run([sys.executable, str(HERE / "verify_sources.py")], check=True)
    for op in ["add", "multiply", "divide"]:
        for features, gradient in [("active", ".1"), ("active", "0"), ("pls", "0")]:
            run_stage(
                "src/train_heads.py",
                op,
                "--features",
                features,
                "--gradient-weight",
                gradient,
                "--max-seconds",
                "864000",
                "--resume",
            )
    for op in ["add", "multiply", "divide"]:
        run_stage("src/sparse_neurons.py", op)
    run_stage("src/linear_controls.py")
    print("ALL SCALAR FITS COMPLETE", args.model, flush=True)


if __name__ == "__main__":
    main()
