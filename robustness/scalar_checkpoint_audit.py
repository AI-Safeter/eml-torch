"""Audit completed scalar searches independently of the later test evaluations."""

import hashlib
import json
from pathlib import Path

import torch
from runtime import HERE, RUNS, configure


def validate(out, directory):
    from heads import Head, objective, parameter_count

    folder = out / directory
    status = json.loads((folder / "completed.json").read_text())
    assert status["status"] == "complete_grid"
    records = json.loads((folder / "candidates.json").read_text())
    assert len(records) == 10
    assert {(r["kind"], r["width"], r["seed"], r["response_weight"]) for r in records} == {
        (kind, 32, seed, 4.0)
        for kind in ["eml_square", "silu_two"]
        for seed in [101, 211, 307, 401, 503]
    }
    protocol = json.loads((folder / "protocol.json").read_text())
    filename = "features-active.pt" if protocol["features"] == "active" else "features.pt"
    paths = [
        folder / "candidates.json",
        folder / "protocol.json",
        out / filename,
        HERE / "src/heads.py",
        Path(__file__),
    ]
    paths += [folder / f"{r['name']}.pt" for r in records if r["status"] == "complete"]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    destination = folder / "checkpoint-audit.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous["input_sha256"] == hashes:
            return previous
    data = torch.load(out / filename, weights_only=True)["validation"]
    x, y = data["x"].cuda(), data["y"].cuda()
    checked = []
    for row in records:
        if row["status"] != "complete":
            continue
        head = Head(row["kind"], 32, 32).cuda()
        head.load_state_dict(torch.load(folder / f"{row['name']}.pt", weights_only=True))
        assert parameter_count(head) == row["parameters"] == 2177
        with torch.no_grad():
            values = objective(head(x), y, data["cases"], 4.0)
        errors = {}
        for actual, name in zip(
            values, ["validation_objective", "validation_mse", "validation_response_mse"]
        ):
            errors[name] = abs(float(actual) - row[name])
            assert errors[name] <= 1e-7 + abs(row[name]) * 1e-5, (row["name"], name)
        checked.append({"name": row["name"], "validation_score_absolute_errors": errors})
    report = {
        "input_sha256": hashes,
        "checkpoints": checked,
        "gpu": torch.cuda.get_device_name(),
        "validation_scores_recomputed": True,
    }
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    configure("qwen17b")
    from model_io import setup

    setup()
    total = 0
    for model in ["qwen17b", "qwen4b", "smollm"]:
        for op in ["add", "multiply", "divide"]:
            out = RUNS / model / op
            for directory in ["heads", "heads-active-r32-g0", "heads-active-r32-g0.1"]:
                marker = out / directory / "completed.json"
                if marker.exists() and json.loads(marker.read_text())["status"] == "complete_grid":
                    total += len(validate(out, directory)["checkpoints"])
    print("SCALAR CHECKPOINT GPU AUDIT PASSED FOR COMPLETED SEARCHES", total, flush=True)
