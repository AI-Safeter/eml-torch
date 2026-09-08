"""Require complete evidence and recompute every scalar validation objective on GPU."""

import json

import torch
from runtime import HERE, RUNS, configure
from verify_sources import main as verify_sources


def main():
    configure("qwen17b")
    from data import save
    from evaluate_heads import digest
    from heads import Head, objective
    from model_io import setup

    setup()
    verify_sources()
    audit = {"status": "complete", "scalar": {}, "whole_block": {}}
    directories = ["heads", "heads-active-r32-g0", "heads-active-r32-g0.1"]
    for model in ["qwen17b", "qwen4b", "smollm"]:
        for op in ["add", "multiply", "divide"]:
            out = RUNS / model / op
            selected = json.loads((out / "selection.json").read_text())
            assert digest(out / "problems.json") == selected["problems_sha256"]
            assert digest(out / "extra-problems.json") == selected["extra_problems_sha256"]
            assert (out / "problems.json").read_bytes() == (
                HERE / "data" / op / "problems.json"
            ).read_bytes()
            fits = 0
            failed = []
            for directory in directories:
                folder = out / directory
                protocol = json.loads((folder / "protocol.json").read_text())
                records = json.loads((folder / "candidates.json").read_text())
                assert {
                    (r["kind"], r["width"], r["seed"], r["response_weight"]) for r in records
                } == {
                    (kind, 32, seed, 4.0)
                    for kind in ["eml_square", "silu_two"]
                    for seed in [101, 211, 307, 401, 503]
                }
                filename = (
                    "features-active.pt" if protocol["features"] == "active" else "features.pt"
                )
                data = torch.load(out / filename, weights_only=True)["validation"]
                x, y = data["x"].cuda(), data["y"].cuda()
                for record in records:
                    fits += 1
                    if record["status"] != "complete":
                        failed.append(f"{directory}/{record['name']}")
                        continue
                    head = Head(record["kind"], 32, 32).cuda()
                    head.load_state_dict(
                        torch.load(folder / f"{record['name']}.pt", weights_only=True)
                    )
                    with torch.no_grad():
                        value, absolute, response = objective(head(x), y, data["cases"], 4.0)
                    for actual, name in [
                        (value, "validation_objective"),
                        (absolute, "validation_mse"),
                        (response, "validation_response_mse"),
                    ]:
                        assert (
                            abs(float(actual) - record[name]) <= 1e-7 + abs(record[name]) * 1e-5
                        ), (model, op, record["name"], name)
                for family, prefix in [("eml", "eml"), ("neural", "silu")]:
                    candidates = [
                        r
                        for r in records
                        if r["status"] == "complete" and r["kind"].startswith(prefix)
                    ]
                    best = min(candidates, key=lambda r: r["validation_objective"])
                    spec = selected["heads"][f"{directory}/{family}"]
                    assert spec["name"] == best["name"]
                    assert digest(out / spec["checkpoint"]) == spec["sha256"]
            expected = {
                "test-known-formats": 3072,
                "test-new-formats": 4096,
                "shift": 1536,
                "both-operands": 768,
                "all-seeds": 3072,
            }
            if op == "add":
                expected["carry"] = 1536
            file_counts = {}
            for suite, count in expected.items():
                for category, multiplier in [("ordinary", 1), ("interventions", 6)]:
                    filename = f"{category}-{suite}.json"
                    data = json.loads((out / filename).read_text())
                    assert all(len(rows) == count * multiplier for rows in data.values()), filename
                    assert "original" in data and len(data) >= 7
                    file_counts[filename] = {
                        "models": len(data),
                        "rows_per_model": count * multiplier,
                    }
            for name in [
                "ordinary-unconditioned",
                "ordinary-all-tokens-unconditioned",
                "ordinary-all-tokens-test-known-formats",
            ]:
                data = json.loads((out / f"{name}.json").read_text())
                assert all(len(rows) == 3072 for rows in data.values())
                if "all-tokens" in name:
                    assert all(any(r["patched_calls"] > 1 for r in rows) for rows in data.values())
            for filename, count in [
                ("coordinate-interventions.json", 32 * 3 * 4 * 3),
                ("interventions-extrapolation.json", 128 * 3 * 3),
                ("geometry-ambient.json", 32 * 3 * 5),
                ("geometry-nullspace.json", 32 * 3 * 5),
            ]:
                data = json.loads((out / filename).read_text())
                assert all(len(rows) == count for rows in data.values()), filename
            assert (out / "robust-results.json").exists()
            per_style = json.loads((out / "per-style-results.json").read_text())
            assert len(per_style["ordinary"]) == len(per_style["interventions"]) == 7
            audit["scalar"][f"{model}/{op}"] = {
                "fits": fits,
                "failed": failed,
                "validation_objectives_recomputed_on_gpu": True,
                "primary_and_stress_counts": file_counts,
            }
    whole = RUNS / "whole-block"
    from whole_checkpoint_audit import validate as validate_whole_checkpoints

    validate_whole_checkpoints()
    for kind in ["original", "eml", "swiglu", "linear"]:
        data = json.loads((whole / f"evaluation-test-{kind}.json").read_text())
        assert len(data["language"]) == 256
        assert len(data["arithmetic"]) == 6 and all(
            len(rows) == 3072 for rows in data["arithmetic"].values()
        )
    assert len(json.loads((whole / "latency.json").read_text())["records"]) == 12
    assert len(json.loads((whole / "block-latency.json").read_text())["records"]) == 12
    assert len(json.loads((whole / "output-fidelity.json").read_text())) == 2
    assert (whole / "utility-results.json").exists()
    storage = json.loads((whole / "model-storage.json").read_text())["students"]
    assert set(storage) == {"eml", "swiglu", "linear"}
    for row in storage.values():
        assert (
            row["original_model_parameters"]
            - row["original_block_parameters"]
            + row["deployed_student_parameters"]
            == row["replaced_model_parameters"]
        )
    audit["whole_block"] = {
        "fits": 27,
        "required_test_and_latency_artifacts_present": True,
        "all_validation_objectives_recomputed_on_gpu": True,
    }
    save(RUNS / "audit.json", audit)
    print("COMPLETE STUDY AUDIT PASSED", flush=True)


if __name__ == "__main__":
    main()
