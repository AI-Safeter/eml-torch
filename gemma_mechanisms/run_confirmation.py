"""Run the fixed post-training checks and confirmation, preserving completed files."""

import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import time

from .evaluate import roster
from .runtime import root, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--mechanisms-output", required=True)
    args = parser.parse_args()
    out, mechanisms = root(args.output), root(args.mechanisms_output)
    logs = out / "confirmation-logs"
    logs.mkdir(exist_ok=True)

    def run(module, options, gpu, label, destination=out):
        environment = dict(
            os.environ, CUDA_VISIBLE_DEVICES=str(gpu), EML_GEMMA_MECHANISMS_RUNS=str(destination)
        )
        print("START", label, "GPU", gpu, flush=True)
        with (logs / f"{label}.log").open("a") as handle:
            subprocess.run(
                [sys.executable, "-m", f"gemma_mechanisms.{module}", *options],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print("COMPLETE", label, flush=True)

    def parallel(jobs, gpus):
        pending = queue.Queue()
        for job in jobs:
            pending.put(job)

        def worker(gpu):
            while True:
                try:
                    module, options, label, destination = pending.get_nowait()
                except queue.Empty:
                    return
                run(module, options, gpu, label, destination)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, gpu) for gpu in gpus]
            for future in futures:
                future.result()

    last = -1
    while True:
        fits = list((out / "training").glob("*-b*-d*-s*.json"))
        if len(fits) != last:
            print("WAIT FOR GRID", len(fits), "/42", flush=True)
            last = len(fits)
        if len(fits) == 42:
            break
        time.sleep(10)
    if not (out / "training/selection.json").exists():
        run("train", ["--output", str(out)], 2, "freeze-selection")
    run("audit_fits", ["--output", str(out)], 2, "audit-fits")
    run("validate_deployment", [], 2, "validate-deployment")
    assert len(json.loads((out / "deployment-validation.json").read_text())["records"]) == 42
    run("validate_evaluation", ["--deployed"], 0, "validate-native-scoring")
    names = list(roster(out))
    jobs = [
        ("evaluate", ["--split", "gate", "--method", name], f"gate-{name}", out) for name in names
    ]
    for split in ["gate", "shift"]:
        for fresh in [False, True]:
            options = ["--split", split, "--groups", "256"] + (["--new-formats"] if fresh else [])
            jobs.append(
                (
                    "state_sufficiency",
                    options,
                    f"causal-{split}-{'new' if fresh else 'known'}",
                    mechanisms,
                )
            )
    parallel(jobs, [0, 0, 1, 3])
    run("quality_summary", ["--split", "gate"], 2, "gate-summary")
    gate = json.loads((out / "evaluation/gate-summary.json").read_text())
    if gate["eml_expansion_allowed"]:
        save(
            out / "confirmation-status.json",
            {
                "status": "One-block gate passed; freeze a fresh multi-block stage before final evaluation"
            },
        )
        return
    parallel(
        [
            ("evaluate", ["--split", split, "--method", name], f"{split}-{name}", out)
            for split in ["final", "shift"]
            for name in names
        ],
        [0, 0, 1, 3],
    )
    run("quality_summary", ["--split", "final"], 2, "final-summary")
    for split in ["gate", "shift"]:
        for style in ["known", "new"]:
            cohort = f"{split}-256-{style}"
            run("equation_responses", ["--cohort", cohort], 2, f"equations-{cohort}", mechanisms)
    parallel(
        [
            ("benchmark", ["--method", name], f"benchmark-{name}", out)
            for name in names
            if name.endswith("-s1103")
        ],
        [0, 3],
    )
    save(
        out / "confirmation-status.json",
        {
            "status": "Complete",
            "one_block_gate_passed": False,
            "multi_block_expansion": "Stopped by frozen quality criteria",
            "methods": names,
        },
    )
    print("CONFIRMATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
