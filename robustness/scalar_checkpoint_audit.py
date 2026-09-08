"""Audit completed scalar searches independently of the later test evaluations."""

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from runtime import HERE, RUNS, configure


def operator_paths(operator):
    """Bind cached validation to the operator actually imported by the head."""
    manifest = HERE / "operator-dependency.json"
    spec = json.loads(manifest.read_text())
    bundled = (HERE / spec["path"]).resolve()
    imported = Path(inspect.getsourcefile(operator)).resolve()
    for path in {bundled, imported}:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["sha256"], (
            f"EML operator differs from the study dependency: {path}"
        )
    return [manifest, bundled, imported, Path(__file__).resolve()]


def validate(out, directory):
    from heads import Head, objective, parameter_count, safe_eml

    folder = out / directory
    expected_torch = (
        json.loads((HERE / "gemma/model.json").read_text())["torch"]
        if out.parent.name == "gemma"
        else "2.5.0a0+e000cf0ad9.nv24.10"
    )
    if torch.__version__ != expected_torch:
        key = "GEMMA_PYTHON" if out.parent.name == "gemma" else "QWEN_PYTHON"
        python = os.environ.get(key)
        assert python, f"Set {key} to the model's pinned Python environment for GPU replay"
        assert os.path.abspath(python) != os.path.abspath(sys.executable), key
        subprocess.run(
            [
                python,
                str(Path(__file__).resolve()),
                "--model",
                out.parent.name,
                "--operation",
                out.name,
                "--directory",
                directory,
            ],
            check=True,
        )
        report = json.loads((folder / "checkpoint-audit.json").read_text())
        assert report["torch"] == expected_torch
        return report
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
    paths += operator_paths(safe_eml)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    destination = folder / "checkpoint-audit.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous["input_sha256"] == hashes and previous.get("torch") == torch.__version__:
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
        "torch": torch.__version__,
        "validation_scores_recomputed": True,
    }
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen17b", "qwen4b", "smollm", "gemma"])
    parser.add_argument("--operation", choices=["add", "multiply", "divide"])
    parser.add_argument(
        "--directory", choices=["heads", "heads-active-r32-g0", "heads-active-r32-g0.1"]
    )
    args = parser.parse_args()
    configure("qwen17b")
    from model_io import setup

    setup()
    total = 0
    roster = json.loads((HERE / "study.json").read_text())
    models = (
        [args.model] if args.model else [*roster["primary_models"], *roster["superseded_models"]]
    )
    for model in models:
        for op in [args.operation] if args.operation else ["add", "multiply", "divide"]:
            out = RUNS / model / op
            for directory in (
                [args.directory]
                if args.directory
                else ["heads", "heads-active-r32-g0", "heads-active-r32-g0.1"]
            ):
                marker = out / directory / "completed.json"
                if marker.exists() and json.loads(marker.read_text())["status"] == "complete_grid":
                    total += len(validate(out, directory)["checkpoints"])
    print("SCALAR CHECKPOINT GPU AUDIT PASSED FOR COMPLETED SEARCHES", total, flush=True)
