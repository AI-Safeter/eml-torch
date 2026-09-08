"""Run disjoint grid shards after one serialized initialization step."""

import argparse
import subprocess
import sys

from .runtime import SPEC, root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=3)
    args = parser.parse_args()
    assert 0 <= args.shard < args.shards
    out = root(args.output)
    assert (out / "training/initialization.json").exists()
    spec = SPEC["replacement"]
    names = [
        f"{kind}-b{budget['coefficients']}-d{depth if kind != 'linear' else 0}-s{seed}"
        for seed in spec["seeds"]
        for budget in spec["budgets"]
        for depth in spec["depths"]
        for kind in spec["families"]
        if kind != "linear" or depth == 1
    ]
    for name in names[args.shard :: args.shards]:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "gemma_mechanisms.train",
                "--output",
                str(out),
                "--candidate",
                name,
            ],
            check=True,
        )
    print("SHARD COMPLETE", args.shard, flush=True)


if __name__ == "__main__":
    main()
