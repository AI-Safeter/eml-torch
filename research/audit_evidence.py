"""Audit frozen selections, operand separation, validation objectives, and raw metrics on CUDA."""

import hashlib
import json
import time
from pathlib import Path

import torch

from data import ROOT, key, old_addition_pairs, save
from heads import Head, objective
from model_io import setup


def read(path):
    return json.loads(path.read_text())


def check_hash(path, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path


def main():
    setup()
    records = []
    for folder in [ROOT, ROOT / "replication-0.6b"]:
        for op in ["add", "multiply", "divide"]:
            out = folder / op
            problems = read(out / "problems.json")
            groups = {
                split: {key(op, row[t], row["b"]) for row in rows for t in ["a", "c"]}
                for split, rows in problems.items()
            }
            for split, rows in problems.items():
                assert len(groups[split]) == 2 * len(rows), (op, split)
                for other, other_groups in groups.items():
                    assert split == other or not groups[split] & other_groups
            used = set.union(*groups.values())
            if op == "add":
                assert not used & old_addition_pairs()
            if (out / "minimal-problems.json").exists():
                extra = read(out / "minimal-problems.json")["test"]
                extra_groups = {key(op, row[t], row["b"]) for row in extra for t in ["a", "c"]}
                assert len(extra_groups) == 2 * len(extra) and not extra_groups & used
            selection = read(out / "selection.json")
            check_hash(out / "problems.json", selection["problems_sha256"])
            for name, checksum in selection["component_sha256"].items():
                check_hash(out / name, checksum)
            for family in ["sparse", "linear"]:
                for spec in selection.get(family, {}).values():
                    check_hash(out / spec["checkpoint"], spec["sha256"])
            specs = dict(selection["heads"])
            for name, spec in specs.items():
                candidates = read(out / Path(spec["checkpoint"]).parent / "candidates.json")
                prefix = "eml" if name.endswith("/eml") else "silu"
                eligible = [
                    r
                    for r in candidates
                    if r["status"] == "complete"
                    and r["kind"].startswith(prefix)
                    and r["response_weight"] == 4
                ]
                assert (
                    min(eligible, key=lambda r: r["validation_objective"])["name"] == spec["name"]
                )
            if (out / "minimal-selection.json").exists():
                minimal = read(out / "minimal-selection.json")
                check_hash(out / "minimal-problems.json", minimal["test_sha256"])
                specs.update(minimal["selection"]["heads"])
            seed_file = out / "seed-replication/selection.json"
            if seed_file.exists():
                frozen = read(seed_file)
                check_hash(out / "minimal-problems.json", frozen["test_sha256"])
                specs.update(frozen["selection"]["heads"])
            objectives = {}
            for name, spec in specs.items():
                check_hash(out / spec["checkpoint"], spec["sha256"])
                data = torch.load(
                    out / ("features-active.pt" if spec["features"] == "active" else "features.pt"),
                    weights_only=True,
                )["validation"]
                head = Head(spec["kind"], spec["rank"], spec["width"]).cuda().eval()
                head.load_state_dict(torch.load(out / spec["checkpoint"], weights_only=True))
                with torch.no_grad():
                    value, absolute, response = objective(
                        head(data["x"][:, : spec["rank"]].cuda()),
                        data["y"].cuda(),
                        data["cases"],
                        spec["response_weight"],
                    )
                for field, actual in [
                    ("validation_objective", value),
                    ("validation_mse", absolute),
                    ("validation_response_mse", response),
                ]:
                    torch.testing.assert_close(
                        actual,
                        torch.tensor(spec[field], device="cuda"),
                        rtol=2e-4,
                        atol=2e-7,
                    )
                objectives[name] = float(value)
            results = read(out / "results.json")
            for path in out.glob("ordinary-*.json"):
                suite = path.stem.removeprefix("ordinary-")
                if suite not in results:
                    continue
                raw = read(path)
                for name, rows in raw.items():
                    correctness = torch.tensor(
                        [r["correct"] for r in rows], device="cuda", dtype=torch.float64
                    )
                    expected = results[suite]["ordinary"][name]["exact_answer_accuracy"]
                    assert abs(float(correctness.mean()) - expected) < 1e-12
                    assert (
                        all(r["text"] == r["base_text"] for r in rows)
                        if name in ["original", "restore"]
                        else True
                    )
            records.append(
                {
                    "model_folder": str(folder.relative_to(ROOT)),
                    "operation": op,
                    "split_pairs": {k: len(v) for k, v in problems.items()},
                    "validated_heads": objectives,
                }
            )
    save(
        ROOT / "evidence-audit.json",
        {
            "completed_epoch": time.time(),
            "gpu": torch.cuda.get_device_name(),
            "checks": records,
            "status": "passed",
        },
    )
    print("FROZEN EVIDENCE AUDIT PASSED")


if __name__ == "__main__":
    main()
