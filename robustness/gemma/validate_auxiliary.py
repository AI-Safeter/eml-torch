"""Compare isolated GPU validation outputs and exercise atomic publication guards."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma import adapter
from gemma.auxiliary_evaluate import digest, main_owns, publish, scientific


def publication_guards():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source, target = root / "source", root / "target"
        source.write_bytes(b"complete cohort\n")
        assert publish(source, target) == "published"
        assert source.stat().st_ino != target.stat().st_ino
        before = target.stat()
        assert publish(source, target) == "identical_existing"
        assert target.stat().st_ino == before.st_ino
        assert target.stat().st_mtime_ns == before.st_mtime_ns
        source.write_bytes(b"different cohort\n")
        try:
            publish(source, target)
        except AssertionError:
            pass
        else:
            raise AssertionError("Different existing output was accepted")
        assert target.read_bytes() == b"complete cohort\n"
        assert not list(root.glob(".auxiliary-*"))

        queue = root / "queue-gemma.json"
        queue.write_text(json.dumps({"status": "complete"}))
        assert main_owns("divide", root)
        queue.write_text(json.dumps({"status": "running", "child_pid": os.getpid()}))
        assert not main_owns("divide", root)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "divide"])
        try:
            assert main_owns("divide", root)
        finally:
            child.terminate()
            child.wait()
        queue.write_text(json.dumps({"status": "failed"}))
        try:
            main_owns("divide", root)
        except AssertionError:
            pass
        else:
            raise AssertionError("Failed main queue was accepted")
    return [
        "new output published atomically as an independent inode",
        "identical existing output preserved without modification",
        "different existing output rejected without modification",
        "publication temporary files removed",
        "complete or matching live main stage yields ownership",
        "failed main queue rejected",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--cached", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [
        "gemma/auxiliary_evaluate.py",
        "gemma/validate_auxiliary.py",
        "gemma/generation-execution-freeze.json",
    ]
    expected = {
        "ordinary-validation-audit.json": 144,
        "interventions-validation-audit.json": 864,
        "ordinary-all-tokens-validation.json": 144,
        "coordinate-validation.json": 1728,
    }
    records, jobs = {}, {}
    for engine, root in [("native", args.native), ("cached", args.cached)]:
        out = root / "divide"
        completion = json.loads((out / "evaluation-audit.json").read_text())
        assert completion["restore_equivalent"]
        jobs[engine] = json.loads((out / "evaluation-job.json").read_text())
        assert "H100" in jobs[engine]["gpu"]
        assert {p.name for p in out.iterdir() if scientific(p.name)} == set(expected)
        manifest = json.loads((root / "staged-manifest.json").read_text())
        for name, count in expected.items():
            path = out / name
            assert digest(path) == manifest[name]
            data = json.loads(path.read_text())
            assert len(data) == 12 and sum(map(len, data.values())) == count
            if engine == "native":
                records[name] = {"sha256": digest(path), "records": count}
            else:
                assert path.read_bytes() == (args.native / "divide" / name).read_bytes(), name
    proof = {
        "passed": True,
        "operation": "divide",
        "native_cached_file_bytes_identical": True,
        "records": sum(expected.values()),
        "files": records,
        "publication_guards_passed": True,
        "publication_checks": publication_guards(),
        "gpu_jobs": jobs,
        "source_sha256": {name: digest(adapter.STUDY / name) for name in sources},
    }
    args.output.write_text(json.dumps(proof, indent=2) + "\n")
    print("AUXILIARY VALIDATION PASSED", proof["records"], "GPU records", flush=True)


if __name__ == "__main__":
    main()
