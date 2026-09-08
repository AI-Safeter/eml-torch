"""Exercise packaging integrity on isolated fixtures and a real CUDA trace."""

import argparse
import gzip
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import bundle
import torch
from audit import execution_sources
from runtime import HERE, RUNS, configure


def reject(function, text):
    try:
        function()
    except AssertionError as error:
        assert text in str(error), error
    else:
        raise AssertionError(f"Failed to reject: {text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure("qwen17b")
    from analyze import ordinary
    from model_io import setup

    setup()
    source = RUNS / "qwen17b/divide/ordinary-validation-audit.json"
    raw = source.read_bytes()
    metrics = ordinary(json.loads(raw))
    checks = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixture = root / "fixture"
        study, runs = fixture / "eml-torch/robustness", fixture / "emltorch-robustness-runs"
        study.mkdir(parents=True)
        out = runs / "qwen17b/divide"
        out.mkdir(parents=True)
        trace = out / source.name
        trace.write_bytes(raw)
        checkpoint = out / "selected.pt"
        selected = json.loads((source.parent / "selection.json").read_text())
        shutil.copyfile(
            source.parent / selected["heads"]["heads-active-r32-g0.1/eml"]["checkpoint"],
            checkpoint,
        )
        guarded = study / "bundle.py"
        shutil.copyfile(HERE / "bundle.py", guarded)
        roster = {"primary_models": {"qwen17b": "Qwen3-1.7B"}, "superseded_models": {}}
        (study / "study.json").write_text(json.dumps(roster))
        # This deliberately minimal record tests the packaging guard only. It
        # is never published as a scientific audit or used by the running study.
        audit = {
            "status": "complete",
            "roster": roster,
            "scalar": {f"qwen17b/{op}": {} for op in ["add", "multiply", "divide"]},
            "execution_sources": execution_sources(),
        }
        with patch.multiple(bundle, HERE=study, RUNS=runs):
            audit["input_sha256"] = bundle.input_fingerprint(roster)
            (runs / "audit.json").write_text(json.dumps(audit))
            assert bundle.require_audit() == audit
            for path in [trace, checkpoint, guarded]:
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                reject(bundle.require_audit, "Evidence changed since audit")
                path.write_bytes(original)
                checks.append(f"changed {path.name} rejected")
            extra = out / "ordinary-extra.json"
            extra.write_bytes(raw)
            reject(bundle.require_audit, "Evidence changed since audit")
            extra.unlink()
            checks.append("new evidence file rejected")
            (out / "metric-audit.json").write_text('{"fixture": true}')
            assert bundle.require_audit() == audit
            checks.append("regenerable metric cache excluded from portable input identity")

            results = study / "results"
            results.mkdir()
            for name in [
                "REPORT.md",
                "summary.json",
                "primary.png",
                "primary.pdf",
                "explorer.html",
            ]:
                (results / name).write_text("Packaging fixture, not a research result\n")
            provenance = bundle.report_provenance(audit)
            (results / "provenance.json").write_text(json.dumps(provenance))
            report = results / "REPORT.md"
            original = report.read_bytes()
            report.write_bytes(original + b"changed")
            reject(lambda: bundle.create(root / "rejected-bundle"), "Report differs")
            report.write_bytes(original)
            checks.append("changed report rejected before packaging")
            assert bundle.require_audit() == audit

        relocated = root / "relocated"
        shutil.copytree(fixture, relocated)
        with patch.multiple(
            bundle,
            HERE=relocated / "eml-torch/robustness",
            RUNS=relocated / "emltorch-robustness-runs",
        ):
            assert bundle.require_audit() == audit
            assert bundle.report_provenance(audit) == provenance
        checks.append("relocation preserves input and report identities")

        packed = root / "packed"
        packed.mkdir()
        payload = packed / "trace.json.gz"
        payload.write_bytes(gzip.compress(raw, mtime=0))
        (packed / "manifest.json").write_text(
            json.dumps(
                {
                    "files": {
                        payload.name: {
                            "sha256": bundle.sha(payload),
                            "bytes": payload.stat().st_size,
                            "target": "trace.json",
                            "codec": "gzip",
                            "uncompressed_sha256": bundle.sha(source),
                        }
                    }
                }
            )
        )
        bundle.unpack(packed)
        bundle.unpack(packed)
        restored = packed / "trace.json"
        assert restored.read_bytes() == raw
        assert ordinary(json.loads(restored.read_text())) == metrics
        checks.append("roundtrip bytes and GPU metrics identical; repeated unpack idempotent")
        restored.write_bytes(b"different")
        reject(lambda: bundle.unpack(packed), "trace.json")
        checks.append("different existing unpacked file rejected")
    evidence = {
        "scope": "Packaging fixtures and a real CUDA validation trace; full study bundle pending",
        "passed": True,
        "checks": checks,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "trace_sha256": bundle.sha(source),
        "method_records": sum(map(len, json.loads(raw).values())),
        "roundtrip_gpu_metrics_identical": True,
        "source_sha256": {
            name: bundle.sha(HERE / name)
            for name in ["bundle.py", "audit.py", "report.py", "validate_bundle.py"]
        },
    }
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print("BUNDLE INTEGRITY VALIDATION PASSED", len(checks), "checks", flush=True)


if __name__ == "__main__":
    main()
