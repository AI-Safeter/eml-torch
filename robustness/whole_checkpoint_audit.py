"""Recompute all vector-checkpoint validation scores; reuse only hash-matching audits."""

import hashlib
import json
from pathlib import Path

import torch
from runtime import HERE, RUNS, configure
from whole_student import Student


def validate():
    out = RUNS / "whole-block"
    candidates = json.loads((out / "candidates.json").read_text())
    assert len(candidates) == 27
    paths = [out / "candidates.json", HERE / "whole_student.py", Path(__file__)]
    paths += [out / f"activations-{domain}-validation.pt" for domain in ["arithmetic", "language"]]
    paths += [out / f"{r['name']}.pt" for r in candidates if r["status"] == "complete"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    destination = out / "checkpoint-audit.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous["input_sha256"] == hashes:
            print("WHOLE CHECKPOINT AUDIT HASHES VERIFIED", flush=True)
            return previous
    validation = {
        domain: torch.load(out / f"activations-{domain}-validation.pt", weights_only=True)
        for domain in ["arithmetic", "language"]
    }
    checked = []
    for row in candidates:
        if row["status"] != "complete":
            continue
        state = torch.load(out / f"{row['name']}.pt", weights_only=True)
        stats = {k: state[k] for k in ["xmean", "xstd", "ymean", "yscale"]}
        student = Student(len(stats["xmean"]), row["width"], row["kind"], stats).cuda()
        student.load_state_dict(state)
        errors = {}
        for domain, values in validation.items():
            x = ((values["x"] - stats["xmean"]) / stats["xstd"]).cuda()
            y = ((values["y"] - stats["ymean"]) / stats["yscale"]).cuda()
            with torch.no_grad():
                actual = (
                    sum(
                        (student.normalized(a) - b).square().sum()
                        for a, b in zip(x.split(1024), y.split(1024))
                    )
                    / y.numel()
                )
            errors[domain] = abs(float(actual) - row["validation_domain_mse"][domain])
            assert errors[domain] < 1e-6, row["name"]
        checked.append({"name": row["name"], "validation_score_absolute_errors": errors})
    record = {
        "input_sha256": hashes,
        "checkpoints": checked,
        "gpu": torch.cuda.get_device_name(),
        "validation_scores_recomputed": True,
    }
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print("WHOLE CHECKPOINT GPU AUDIT PASSED", len(checked), flush=True)
    return record


if __name__ == "__main__":
    configure("qwen17b")
    from model_io import setup

    setup()
    validate()
