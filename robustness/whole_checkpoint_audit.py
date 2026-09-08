"""Recompute all vector-checkpoint validation scores; reuse only hash-matching audits."""

import hashlib
import json
from pathlib import Path

import torch
from runtime import HERE, RUNS, configure
from scalar_checkpoint_audit import operator_paths
from whole_student import Student, safe_eml


def validate():
    assert torch.__version__ == "2.5.0a0+e000cf0ad9.nv24.10", (
        "Run the whole-block checkpoint audit in the original Qwen environment"
    )
    out = RUNS / "whole-block"
    candidates = json.loads((out / "candidates.json").read_text())
    assert len(candidates) == 27
    assert {(r["kind"], r["width"], r["seed"]) for r in candidates} == {
        (kind, width, seed)
        for kind in ["eml", "swiglu", "linear"]
        for width in [128, 256, 512]
        for seed in [101, 211, 307]
    }
    selected = json.loads((out / "selection.json").read_text())
    assert selected["model"] == json.loads((HERE / "models.json").read_text())["qwen17b"]
    for filename, key in [
        ("WHOLE_BLOCK_PROTOCOL.md", "protocol_sha256"),
        ("language-freeze.json", "language_freeze_sha256"),
    ]:
        assert hashlib.sha256((HERE / filename).read_bytes()).hexdigest() == selected[key]
    language = json.loads((HERE / "language-freeze.json").read_text())
    for split, frozen in language["splits"].items():
        assert (
            hashlib.sha256((out / f"language-{split}.pt").read_bytes()).hexdigest()
            == frozen["file_sha256"]
        )
    for kind, spec in selected["students"].items():
        best = min(
            (r for r in candidates if r["kind"] == kind and r["status"] == "complete"),
            key=lambda r: r["validation_objective"],
        )
        assert spec["name"] == best["name"]
        assert hashlib.sha256((out / spec["checkpoint"]).read_bytes()).hexdigest() == spec["sha256"]
    for row in candidates:
        if row["status"] == "complete":
            assert row["steps"] <= 20000 and row["selected_step"] % 100 == 0
            assert (
                abs(row["validation_objective"] - sum(row["validation_domain_mse"].values()) / 2)
                < 1e-7
            )
    paths = [out / "candidates.json", HERE / "whole_student.py", Path(__file__)]
    paths += [out / f"activations-{domain}-validation.pt" for domain in ["arithmetic", "language"]]
    paths += [out / f"{r['name']}.pt" for r in candidates if r["status"] == "complete"]
    paths += operator_paths(safe_eml)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    destination = out / "checkpoint-audit.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous["input_sha256"] == hashes and previous.get("torch") == torch.__version__:
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
        "torch": torch.__version__,
        "validation_scores_recomputed": True,
        "complete_grid_validation_selection_and_language_hashes_verified": True,
    }
    destination.write_text(json.dumps(record, indent=2) + "\n")
    print("WHOLE CHECKPOINT GPU AUDIT PASSED", len(checked), flush=True)
    return record


if __name__ == "__main__":
    configure("qwen17b")
    from model_io import setup

    setup()
    validate()
