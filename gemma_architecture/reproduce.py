"""Resume the frozen screen and its negative-result release without repeating completed jobs."""

import argparse
import json
import subprocess

from .budget import run
from .common import SPEC, digest, freeze, name, root


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fit-gpu", default="0")
    p.add_argument("--model-gpu", default="3")
    p.add_argument("--audit-gpu", default="2")
    p.add_argument("--release", action="store_true")
    a = p.parse_args()
    out = root()

    def job(phase, label, gpu, seconds, module, *options):
        ledger = (
            json.loads((out / "budget.json").read_text())
            if (out / "budget.json").exists()
            else {"jobs": {}}
        )
        if label not in ledger["jobs"]:
            free = int(
                subprocess.check_output(
                    [
                        "nvidia-smi",
                        "-i",
                        str(gpu),
                        "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
            )
            required = (
                11500
                if module == "benchmark"
                else 7800
                if module == "evaluate" or "--integration" in options
                else 1250
            )
            if free < required:
                raise RuntimeError(
                    f"GPU {gpu}: {free} MiB free, require {required}; select another GPU without stopping other jobs"
                )
        code = run(out, phase, label, gpu, seconds, module, list(options))
        if code:
            raise RuntimeError(
                f"Job {label} did not complete; preserve failure and inspect its log"
            )

    job("screen", "prepare-data", a.audit_gpu, 900, "prepare")
    job("screen", "cuda-units", a.audit_gpu, 120, "validate")
    job("screen", "cuda-integration", a.model_gpu, 180, "validate", "--integration")
    job("screen", "initialization", a.fit_gpu, 180, "train", "--prepare-only")
    methods = []
    for arch in SPEC["training"]["architectures"]:
        for act in ["eml", "silu"]:
            method = name(arch, act, 1, 1103)
            methods.append(method)
            job(
                "screen",
                f"fit-{arch}-{act}",
                a.fit_gpu,
                900,
                "train",
                "--architecture",
                arch,
                "--activation",
                act,
            )
    job("screen", "dev-original", a.model_gpu, 450, "evaluate", "--method", "original")
    for method in methods:
        arch, act = method.split("-")[:2]
        job("screen", f"dev-{arch}-{act}", a.model_gpu, 450, "evaluate", "--method", method)
    job("screen", "screen-summary", a.audit_gpu, 120, "statistics")
    result = json.loads((out / "screen-d1-n12000.json").read_text())
    decision = {
        "status": "stop"
        if not result["promising"] and not result["diagnostic_leader"]
        else "refinement_eligible",
        "promising_architecture": result["promising"],
        "diagnostic_leader": result["diagnostic_leader"],
        "screen_sha256": digest(out / "screen-d1-n12000.json"),
        "confirmation_opened": False,
        "additional_seeds_run": False,
        "depth_screen_run": False,
        "longer_training_run": False,
        "reason": "No architecture passed the frozen screen. None improved both activation errors by 10% in both activations, so the conditional longer-training diagnostic did not activate."
        if not result["promising"] and not result["diagnostic_leader"]
        else "Use the preregistered matched refinement and confirmation rules before opening held-out inference.",
    }
    freeze(out / "decision.json", decision)
    print("DECISION", decision, flush=True)
    if not a.release:
        return
    if decision["status"] != "stop":
        raise RuntimeError(
            "This release command handles the observed screen-stop path. Refine eligible architectures using the frozen protocol before confirmation."
        )
    for method in methods:
        arch, act = method.split("-")[:2]
        job(
            "benchmark_and_audit",
            f"diagnose-{arch}-{act}",
            a.audit_gpu,
            120,
            "diagnose",
            "--method",
            method,
        )
        job(
            "benchmark_and_audit",
            f"export-{arch}-{act}",
            a.audit_gpu,
            120,
            "export",
            "--method",
            method,
        )
        job(
            "benchmark_and_audit",
            f"benchmark-{arch}-{act}",
            a.model_gpu,
            900,
            "benchmark",
            "--method",
            method,
        )
    job("benchmark_and_audit", "deployment-precision", a.audit_gpu, 120, "precision")
    job("benchmark_and_audit", "release-audit-v2", a.audit_gpu, 180, "audit")
    job("benchmark_and_audit", "release-summary", a.audit_gpu, 180, "summarize")


if __name__ == "__main__":
    main()
