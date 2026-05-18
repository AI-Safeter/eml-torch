"""Recover exp(x), ln(x), and -x from synthetic data via emltorch.fit().

Run:
    python3 examples/recover_elementary.py

Expected: all three recover with R2 > 0.99 in under a few seconds on GPU.
"""

import math
import sys
import time
from pathlib import Path

import torch

# Allow running as `python3 examples/recover_elementary.py` without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import emltorch as eml  # noqa: E402


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGETS = [
    # (name, callable, depth, lo, hi)
    ("exp(x)",  torch.exp,                    1, -2.0, 2.0),
    ("e - x",   lambda x: math.e - x,         2,  0.1, 2.0),
    ("ln(x)",   torch.log,                    3,  0.5, 5.0),
    ("-x",      lambda x: -x,                 4, -2.0, 2.0),
]


def main():
    print("=" * 64)
    print(f"emltorch elementary-function recovery  (device={DEVICE})")
    print("=" * 64)

    for name, fn, depth, lo, hi in TARGETS:
        x = torch.linspace(lo, hi, 512)
        y = fn(x)

        t0 = time.time()
        result = eml.fit(x, y, depth=depth, device=DEVICE)
        dt = time.time() - t0

        status = "OK" if result.r2 > 0.99 else "fail"
        print(f"\n[{status}] {name}  (depth={depth})")
        print(f"    R2         = {result.r2:+.4f}")
        print(f"    MSE        = {result.mse:.3e}")
        print(f"    time       = {dt:.2f}s")
        print(f"    expression = {result.expression}")

    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
