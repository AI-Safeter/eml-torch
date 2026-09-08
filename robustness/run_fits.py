"""Complete the fixed scalar search and controls; never read test outputs."""

import argparse
import subprocess
import sys

from runtime import HERE, RUNS, configure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen17b", "qwen4b", "smollm"])
    args = parser.parse_args()
    configure(args.model)
    assert (RUNS / args.model / "derivative-completed.json").exists()
    subprocess.run([sys.executable, str(HERE / "freeze.py"), "--verify"], check=True)
    for op in ["add", "multiply", "divide"]:
        for features, gradient in [("active", ".1"), ("active", "0"), ("pls", "0")]:
            subprocess.run(
                [
                    sys.executable,
                    str(HERE / "src/train_heads.py"),
                    op,
                    "--features",
                    features,
                    "--gradient-weight",
                    gradient,
                    "--max-seconds",
                    "864000",
                    "--resume",
                ],
                check=True,
            )
    for op in ["add", "multiply", "divide"]:
        subprocess.run([sys.executable, str(HERE / "src/sparse_neurons.py"), op], check=True)
    subprocess.run([sys.executable, str(HERE / "src/linear_controls.py")], check=True)
    print("ALL SCALAR FITS COMPLETE", args.model, flush=True)


if __name__ == "__main__":
    main()
