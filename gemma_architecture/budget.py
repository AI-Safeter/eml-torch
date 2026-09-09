"""Meter GPU subprocesses and enforce the prospective eight-device-hour cap."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager

from .common import HERE, SPEC, digest, root, save

PHASES = ["screen", "refinement", "confirmation", "benchmark_and_audit"]


@contextmanager
def locked(out):
    with (out / "budget.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        path = out / "budget.json"
        data = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "limit_seconds": SPEC["budget"]["device_hours"] * 3600,
                "protocol_sha256": digest(HERE / "protocol.json"),
                "jobs": {},
            }
        )
        assert data["protocol_sha256"] == digest(HERE / "protocol.json")
        yield data
        save(path, data)


def remaining(data, phase):
    def charged(row):
        return row.get("elapsed_seconds", row["allowance_seconds"])

    total = data["limit_seconds"] - sum(charged(r) for r in data["jobs"].values())
    used = sum(charged(r) for r in data["jobs"].values() if r["phase"] == phase)
    limit = SPEC["budget"][phase + "_hours"] * 3600
    if phase == "confirmation":
        # Preserve the last hour for timing and audits; earlier unused budgets roll forward.
        return max(0, min(total - 3600, total))
    if phase == "benchmark_and_audit":
        return max(0, total)
    return max(0, min(limit - used, total))


def run(out, phase, label, gpu, allowance, module, options):
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    with locked(out) as data:
        if label in data["jobs"]:
            row = data["jobs"][label]
            assert row["status"] == "complete", ("Do not overwrite a failed/running job", label)
            print("PRESERVE JOB", label, flush=True)
            return 0
        available = remaining(data, phase)
        if available < allowance:
            raise RuntimeError(
                f"Budget cannot reserve {allowance}s for {label}; {available:.1f}s remains"
            )
        data["jobs"][label] = {
            "phase": phase,
            "gpu": str(gpu),
            "module": module,
            "options": options,
            "allowance_seconds": allowance,
            "status": "running",
            "started_unix": time.time(),
        }
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), EML_ARCHITECTURE_RUNS=str(out))
    started = time.perf_counter()
    status, code = "failed", -1
    print("START", label, "GPU", gpu, "reserved seconds", allowance, flush=True)
    try:
        with (logs / f"{label}.log").open("w") as handle:
            child = subprocess.Popen(
                [sys.executable, "-m", "gemma_architecture." + module, *options],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            try:
                code = child.wait(timeout=allowance)
                status = "complete" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                status, code = "timeout", -9
    finally:
        with locked(out) as data:
            data["jobs"][label].update(
                status=status,
                exit_code=code,
                elapsed_seconds=time.perf_counter() - started,
                finished_unix=time.time(),
            )
    print(status.upper(), label, "seconds", time.perf_counter() - started, flush=True)
    return code


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output")
    p.add_argument("--phase", choices=PHASES, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--gpu", required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("module")
    p.add_argument("options", nargs=argparse.REMAINDER)
    a = p.parse_args()
    raise SystemExit(run(root(a.output), a.phase, a.label, a.gpu, a.seconds, a.module, a.options))


if __name__ == "__main__":
    main()
