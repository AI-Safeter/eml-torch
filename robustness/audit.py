"""Require complete evidence and recompute every scalar validation objective on GPU."""

import json
import math
from collections import Counter

from runtime import HERE, RUNS, configure
from verify_sources import main as verify_sources


def audit_selection(out, selected):
    """Verify frozen components, control selection, and completion before selection."""
    from evaluate_heads import digest

    assert set(selected["component_sha256"]) == {"component.pt", "component-active.pt"}
    for name, expected in selected["component_sha256"].items():
        assert digest(out / name) == expected, name
    for category in ["heads", "sparse", "linear"]:
        for spec in selected[category].values():
            assert digest(out / spec["checkpoint"]) == spec["sha256"], spec["checkpoint"]
    for directory, status in selected["searches"].items():
        assert status == json.loads((out / directory / "completed.json").read_text())
        assert status["status"] == "complete_grid"
        assert status["finished_epoch"] <= selected["frozen_epoch"]
    sparse = json.loads((out / "sparse-neurons.json").read_text())
    assert sparse["completed_epoch"] <= selected["frozen_epoch"]
    assert sparse["test_data_used"] is False
    candidates = sparse["candidates"]
    assert len(candidates) == 5 and {r["neurons"] for r in candidates} == {4, 8, 16, 32, 64}
    best = min(candidates, key=lambda r: r["validation_objective"])
    assert sparse["selected_neurons"] == best["neurons"]
    assert set(selected["sparse"]) == {"sparse16", "sparse_selected"}
    for label, neurons in [("sparse16", 16), ("sparse_selected", best["neurons"])]:
        spec = selected["sparse"][label]
        record = next(r for r in candidates if r["neurons"] == neurons)
        assert all(spec[key] == value for key, value in record.items()), label
        assert spec["checkpoint"] == f"neurons-{neurons}.pt"
    linear = json.loads((out / "linear-controls.json").read_text())
    assert linear["completed_epoch"] <= selected["frozen_epoch"]
    assert linear["test_outputs_used_to_fit"] is False
    assert len(linear["candidates"]) == 10
    assert {(r["feature"], r["rank"]) for r in linear["candidates"]} == {
        (feature, rank) for feature in ["pls", "active"] for rank in [2, 4, 8, 16, 32]
    }
    best = min(linear["candidates"], key=lambda r: r["validation_objective"])
    assert linear["selected"] == best and set(selected["linear"]) == {"linear"}
    spec = selected["linear"]["linear"]
    assert all(spec[key] == value for key, value in best.items())
    assert spec["checkpoint"] == best["name"] + ".pt"
    return {
        "component_and_checkpoint_hashes_verified": True,
        "control_grids_and_validation_selection_verified": True,
        "fits_and_controls_completed_before_selection": True,
    }


def audit_whole():
    """Audit the completed whole-block arm independently of pending scalar tests."""
    configure("qwen17b")
    import torch
    from data import STYLES
    from evaluate_heads import parse_answer
    from model_io import expand
    from trace_audit import identity
    from whole_evaluate import summarize

    whole = RUNS / "whole-block"
    selected = json.loads((whole / "selection.json").read_text())
    assert set(selected["students"]) == {"eml", "swiglu", "linear"}
    component = torch.load(RUNS / "qwen17b/add/component.pt", weights_only=True)
    assert selected["layer"] == component["layer"]
    assert selected["layer"] == json.loads((whole / "collection.json").read_text())["layer"]
    from whole_checkpoint_audit import validate as validate_whole_checkpoints

    validate_whole_checkpoints()
    raw = {}
    for kind in ["original", "eml", "swiglu", "linear"]:
        data = json.loads((whole / f"evaluation-test-{kind}.json").read_text())
        raw[kind] = data
        assert len(data["language"]) == 256
        assert all(math.isfinite(value) for value in data["language"])
        assert len(data["arithmetic"]) == 6 and all(
            len(rows) == 3072 for rows in data["arithmetic"].values()
        )
        for op in ["add", "multiply", "divide"]:
            problems = json.loads((HERE / "data" / op / "problems.json").read_text())
            extra = json.loads((HERE / "data" / op / "extra-problems.json").read_text())
            for label, frozen in [("test", problems["test"]), ("unconditioned", extra["ordinary"])]:
                rows = data["arithmetic"][f"{op}/{label}"]
                assert Counter(map(identity, rows)) == Counter(
                    map(identity, expand(frozen, STYLES))
                )
                assert all(
                    row["correct"] == (parse_answer(row["text"]) == row["answer"]) for row in rows
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
    counts = {"original": storage["eml"]["original_block_parameters"]}
    counts.update({kind: row["deployed_student_parameters"] for kind, row in storage.items()})
    assert summarize(raw, counts) == json.loads((whole / "utility-results.json").read_text())
    return {
        "fits": 27,
        "required_test_and_latency_artifacts_present": True,
        "all_validation_objectives_recomputed_on_gpu": True,
        "complete_grid_selection_language_hashes_and_arithmetic_cohorts_verified": True,
        "all_utility_metrics_recomputed_exactly_on_gpu": True,
    }


def main():
    configure("qwen17b")
    from data import STYLES, save
    from evaluate_heads import digest
    from model_io import expand, setup
    from scalar_checkpoint_audit import validate as validate_scalar_checkpoints
    from trace_audit import check as check_traces

    setup()
    verify_sources()
    roster = json.loads((HERE / "study.json").read_text())
    audit = {"status": "complete", "scalar": {}, "whole_block": {}, "roster": roster}
    directories = ["heads", "heads-active-r32-g0", "heads-active-r32-g0.1"]
    for model in roster["primary_models"]:
        for op in ["add", "multiply", "divide"]:
            out = RUNS / model / op
            selected = json.loads((out / "selection.json").read_text())
            selection_audit = audit_selection(out, selected)
            assert digest(out / "problems.json") == selected["problems_sha256"]
            assert digest(out / "extra-problems.json") == selected["extra_problems_sha256"]
            assert (out / "problems.json").read_bytes() == (
                HERE / "data" / op / "problems.json"
            ).read_bytes()
            assert (out / "extra-problems.json").read_bytes() == (
                HERE / "data" / op / "extra-problems.json"
            ).read_bytes()
            problems = json.loads((out / "problems.json").read_text())
            extra = json.loads((out / "extra-problems.json").read_text())
            kinds = [
                "original",
                "restore",
                "mean",
                *selected["heads"],
                *selected["sparse"],
                *selected["linear"],
            ]
            seeds = ["original"] + [
                f"{directory}/{family}-seed-{seed}"
                for directory in directories
                for family in ["eml_square", "silu_two"]
                for seed in [101, 211, 307, 401, 503]
            ]
            cohort = {
                "test-known-formats": expand(problems["test"], STYLES),
                "test-new-formats": expand(
                    problems["test"], ["completion", "named", "reversed", "distractor"]
                ),
                "shift": expand(problems["shift"], STYLES),
                "both-operands": expand(extra["both_operands"], STYLES),
                "all-seeds": expand(problems["test"], STYLES),
            }
            if op == "add":
                cohort["carry"] = expand(problems["carry"], STYLES)
            fits = 0
            failed = []
            for directory in directories:
                folder = out / directory
                protocol = json.loads((folder / "protocol.json").read_text())
                assert protocol["features"] == ("pls" if directory == "heads" else "active")
                assert protocol["gradient_weight"] == (0.1 if directory.endswith("g0.1") else 0.0)
                assert protocol["steps_max"] == 20000 and protocol["rank"] == 32
                records = json.loads((folder / "candidates.json").read_text())
                assert all(
                    r["steps"] <= 20000 and r["selected_step"] % 100 == 0
                    for r in records
                    if r["status"] == "complete"
                )
                assert {
                    (r["kind"], r["width"], r["seed"], r["response_weight"]) for r in records
                } == {
                    (kind, 32, seed, 4.0)
                    for kind in ["eml_square", "silu_two"]
                    for seed in [101, 211, 307, 401, 503]
                }
                validate_scalar_checkpoints(out, directory)
                fits += len(records)
                failed.extend(
                    f"{directory}/{r['name']}" for r in records if r["status"] != "complete"
                )
                for family, prefix in [("eml", "eml"), ("neural", "silu")]:
                    candidates = [
                        r
                        for r in records
                        if r["status"] == "complete" and r["kind"].startswith(prefix)
                    ]
                    best = min(candidates, key=lambda r: r["validation_objective"])
                    spec = selected["heads"][f"{directory}/{family}"]
                    assert spec["name"] == best["name"]
                    assert all(spec[key] == value for key, value in best.items())
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
                    check_traces(
                        data,
                        cohort[suite],
                        seeds if suite == "all-seeds" else kinds,
                        alphas=[0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
                        if category == "interventions"
                        else None,
                    )
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
                check_traces(
                    data,
                    expand(extra["ordinary"], STYLES)
                    if "unconditioned" in name
                    else cohort["test-known-formats"],
                    kinds,
                )
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
                if filename == "coordinate-interventions.json":
                    check_traces(
                        data,
                        expand(problems["test"][:32], STYLES),
                        kinds,
                        alphas=[0.0, -0.25, 0.25],
                        features=[0, 1, 7, 31],
                    )
                elif filename == "interventions-extrapolation.json":
                    check_traces(
                        data,
                        expand(problems["test"][:128], STYLES),
                        kinds,
                        alphas=[0.0, -0.25, 1.25],
                    )
                else:
                    check_traces(
                        data,
                        expand(problems["test"][:32], STYLES),
                        kinds,
                        alphas=[0.0, -0.1, -0.01, 0.01, 0.1],
                        geometry=filename.removeprefix("geometry-").removesuffix(".json"),
                    )
            assert (out / "robust-results.json").exists()
            per_style = json.loads((out / "per-style-results.json").read_text())
            assert len(per_style["ordinary"]) == len(per_style["interventions"]) == 7
            audit["scalar"][f"{model}/{op}"] = {
                "fits": fits,
                "failed": failed,
                "validation_objectives_recomputed_on_gpu": True,
                "primary_and_stress_counts": file_counts,
                "raw_conditions_controls_and_restoration_verified": True,
                "selection": selection_audit,
            }
    assert set(audit["scalar"]) == {
        f"{model}/{op}"
        for model in roster["primary_models"]
        for op in ["add", "multiply", "divide"]
    }
    audit["superseded"] = {}
    for model, spec in roster["superseded_models"].items():
        from analyze import ordinary

        fits = 0
        ordinary_metrics = {}
        for op in ["add", "multiply", "divide"]:
            for directory in directories:
                fits += len(
                    validate_scalar_checkpoints(RUNS / model / op, directory)["checkpoints"]
                )
            out = RUNS / model / op
            path = out / "ordinary-test-known-formats.json"
            if path.exists():
                selected = json.loads((out / "selection.json").read_text())
                audit_selection(out, selected)
                problems = json.loads((out / "problems.json").read_text())
                raw = json.loads(path.read_text())
                check_traces(
                    raw,
                    expand(problems["test"], STYLES),
                    [
                        "original",
                        "restore",
                        "mean",
                        *selected["heads"],
                        *selected["sparse"],
                        *selected["linear"],
                    ],
                )
                ordinary_metrics[op] = ordinary(raw)
        assert fits == spec["completed_fits"]
        audit["superseded"][model] = {
            **spec,
            "fits": fits,
            "validation_objectives_recomputed_on_gpu": True,
            "ordinary_primary_metrics": ordinary_metrics,
            "completed_evaluation_files": sorted(
                str(path.relative_to(RUNS / model))
                for pattern in ["ordinary-*.json", "interventions-*.json"]
                for path in (RUNS / model).rglob(pattern)
            ),
        }
    audit["independent_feature_replays"] = {}
    for model in ["qwen17b", "gemma", *roster["superseded_models"]]:
        replay = json.loads((HERE / f"replay-{model}.json").read_text())
        assert len(replay["archives"]) == 24
        for item in replay["archives"]:
            assert item["tensor_values_bitwise_equal_on_gpu"] and item["archive_bytes_identical"]
            path = RUNS / model / item["path"]
            # Large input tensors are regenerated, and may be omitted from a bundle.
            if path.exists():
                assert digest(path) == item["sha256"]
        audit["independent_feature_replays"][model] = {"tensor_archives_bitwise_equal": 24}
    audit["whole_block"] = audit_whole()
    save(RUNS / "audit.json", audit)
    print("COMPLETE STUDY AUDIT PASSED", flush=True)


if __name__ == "__main__":
    main()
