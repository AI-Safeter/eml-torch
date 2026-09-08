"""Recompute saved scalar summaries from raw traces with the frozen CUDA statistics."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from runtime import HERE, RUNS


def read(path):
    return json.loads(path.read_text())


def raw_paths(out):
    ordinary = sorted(out.glob("ordinary-*.json"))
    interventions = (
        sorted(out.glob("interventions-*.json"))
        + sorted(out.glob("*coordinate-interventions.json"))
        + sorted(out.glob("geometry-*.json"))
    )
    return {
        category: [p for p in paths if "validation" not in p.name]
        for category, paths in [("ordinary", ordinary), ("interventions", interventions)]
    }


def fingerprint(out):
    paths = [p for group in raw_paths(out).values() for p in group]
    paths += [out / name for name in ["robust-results.json", "per-style-results.json"]]
    paths += [
        HERE / name
        for name in [
            "metric_audit.py",
            "analyze.py",
            "per_style.py",
            "robust_statistics.py",
            "runtime.py",
            "src/model_io.py",
            "study.json",
            "gemma/model.json",
        ]
    ]
    hashes = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        hashes[str(path)] = digest.hexdigest()
    return hashes


def expected_torch(out):
    return (
        read(HERE / "gemma/model.json")["torch"]
        if out.parent.name == "gemma"
        else "2.5.0a0+e000cf0ad9.nv24.10"
    )


def equal(actual, expected, label):
    # Compare serialized types too: e.g. true must not silently equal 1.
    def encode(value):
        return json.dumps(value, sort_keys=True, allow_nan=False)

    assert encode(actual) == encode(expected), f"Scalar summary differs from raw traces: {label}"


def require(out):
    """Read-only report guard; hashes bind the record to current sources and evidence."""
    record = read(out / "metric-audit.json")
    assert record["status"] == "complete" and record["recomputed_on_cuda"]
    assert record["exact_summary_agreement"] and record["exact_per_style_agreement"]
    assert record["model"] == out.parent.name and record["operation"] == out.name
    assert record["environment"]["torch"] == expected_torch(out)
    assert record["input_sha256"] == fingerprint(out), f"Stale scalar metric audit: {out}"
    return record


def validate(out, force=False):
    import numpy
    import scipy
    import torch

    if torch.__version__ != expected_torch(out):
        key = "GEMMA_PYTHON" if out.parent.name == "gemma" else "QWEN_PYTHON"
        python = os.environ.get(key)
        assert python and os.path.abspath(python) != os.path.abspath(sys.executable), key
        subprocess.run(
            [
                python,
                str(Path(__file__).resolve()),
                "--model",
                out.parent.name,
                "--operation",
                out.name,
                *(["--force"] if force else []),
            ],
            check=True,
        )
        return require(out)

    environment = {
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "python": sys.version,
    }
    inputs = fingerprint(out)
    destination = out / "metric-audit.json"
    if destination.exists() and not force:
        previous = read(destination)
        if previous.get("input_sha256") == inputs and previous.get("environment") == environment:
            return require(out)

    from analyze import interventions, ordinary
    from model_io import setup
    from verify_sources import main as verify_sources

    setup(20260926)
    verify_sources()
    saved, styled = read(out / "robust-results.json"), read(out / "per-style-results.json")
    assert set(saved) == {"ordinary", "interventions", "primary_checks"}
    assert set(styled) == {"ordinary", "interventions"}
    result, per_style, counts = {}, {}, {}
    for category, paths in raw_paths(out).items():
        calculate = ordinary if category == "ordinary" else interventions
        assert set(saved[category]) == {p.stem for p in paths}
        result[category], per_style[category], counts[category] = {}, {}, {}
        for path in paths:
            raw = read(path)
            fresh = calculate(raw)
            equal(fresh, saved[category][path.stem], path.name)
            result[category][path.stem] = fresh
            counts[category][path.stem] = sum(map(len, raw.values()))
            if path.stem in [f"{category}-test-known-formats", f"{category}-test-new-formats"]:
                for style in sorted({row["style"] for row in raw["original"]}):
                    subset = {
                        key: [r for r in rows if r["style"] == style] for key, rows in raw.items()
                    }
                    name = f"{path.stem}/{style}"
                    fresh_style = calculate(subset)
                    equal(fresh_style, styled[category][name], name)
                    per_style[category][name] = fresh_style
            print("SCALAR METRICS REPLAYED", out.parent.name, out.name, path.stem, flush=True)
    primary = "heads-active-r32-g0.1/eml"
    normal = result["ordinary"]["ordinary-test-known-formats"][primary]
    causal = result["interventions"]["interventions-test-known-formats"]["models"][primary]
    checks = {
        "accuracy_bound": normal["passes_one_percentage_point_bound"],
        "nrmse_bound": causal["nrmse_simultaneous_ci95"] is not None
        and causal["nrmse_simultaneous_ci95"][1] <= 0.2,
        "correlation_bound": causal["correlation_simultaneous_ci95"] is not None
        and causal["correlation_simultaneous_ci95"][0] >= 0.9,
    }
    result["primary_checks"] = {**checks, "all_pass": all(checks.values())}
    equal(result, saved, "complete summary and primary checks")
    equal(per_style, styled, "complete per-style summary")
    assert fingerprint(out) == inputs, "Inputs changed during metric replay"
    record = {
        "status": "complete",
        "model": out.parent.name,
        "operation": out.name,
        "environment": environment,
        "gpu": torch.cuda.get_device_name(),
        "input_sha256": inputs,
        "raw_record_counts": counts,
        "recomputed_on_cuda": True,
        "exact_summary_agreement": True,
        "exact_per_style_agreement": True,
        "scope": "Replays the frozen statistics; raw identities and selection are checked by audit.py.",
    }
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("x") as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=list(read(HERE / "study.json")["primary_models"]), required=True
    )
    parser.add_argument("--operation", choices=["add", "multiply", "divide"], required=True)
    parser.add_argument(
        "--force", action="store_true", help="Recompute even if cached hashes match"
    )
    args = parser.parse_args()
    validate(RUNS / args.model / args.operation, force=args.force)
    print("SCALAR SUMMARY GPU AUDIT PASSED", args.model, args.operation, flush=True)


if __name__ == "__main__":
    main()
