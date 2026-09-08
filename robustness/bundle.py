"""Create and verify an auditable research bundle without adding raw traces to Git."""

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from runtime import HERE, RUNS


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside bundle: {relative}")
    return path


def evidence_paths(roster):
    """Evidence retained for an offline audit, excluding live-job metadata."""
    for model in [*roster["primary_models"], *roster["superseded_models"], "whole-block"]:
        for source in sorted((RUNS / model).rglob("*")):
            if not source.is_file() or source.suffix not in [".json", ".pt"]:
                continue
            assert source.resolve().is_relative_to(RUNS.resolve()), str(source)
            if source.name == "job.json" or source.name.endswith("-job.json"):
                continue
            if source.name.startswith("partial-"):
                continue
            if source.name in ["inputs-train.pt", "inputs-validation.pt"]:
                continue
            if source.name.startswith("activations-") and source.name.endswith("-train.pt"):
                continue
            yield source


def input_fingerprint(roster):
    """Portable identities for scientific inputs and the code that audits them."""
    paths = {
        "runs/" + str(p.relative_to(RUNS)): p
        for p in evidence_paths(roster)
        if p.name not in ["checkpoint-audit.json", "metric-audit.json"]
    }
    for root in [HERE, HERE.parent / "emltorch"]:
        for path in sorted(root.rglob("*")):
            if (
                not path.is_file()
                or path.suffix not in [".py", ".json"]
                or path.is_relative_to(HERE / "results")
            ):
                continue
            assert path.resolve().is_relative_to(HERE.parent.resolve()), path
            paths["source/" + str(path.relative_to(HERE.parent))] = path
    return {name: sha(path) for name, path in sorted(paths.items())}


def require_audit():
    from audit import execution_sources
    from verify_sources import main as verify_sources

    verify_sources()
    audit = json.loads((RUNS / "audit.json").read_text())
    roster = json.loads((HERE / "study.json").read_text())
    assert audit["status"] == "complete" and audit["roster"] == roster
    assert set(audit["scalar"]) == {
        f"{model}/{op}"
        for model in roster["primary_models"]
        for op in ["add", "multiply", "divide"]
    }
    assert audit["execution_sources"] == execution_sources()
    assert audit["input_sha256"] == input_fingerprint(roster), "Evidence changed since audit"
    return audit


def audit_identity(audit):
    return hashlib.sha256(json.dumps(audit["input_sha256"], sort_keys=True).encode()).hexdigest()


def report_provenance(audit):
    return {
        "audited_input_sha256": audit_identity(audit),
        "files": {
            name: sha(HERE / "results" / name)
            for name in ["REPORT.md", "summary.json", "primary.png", "primary.pdf", "explorer.html"]
        },
    }


def create(destination):
    audit = require_audit()
    roster = audit["roster"]
    provenance = json.loads((HERE / "results/provenance.json").read_text())
    assert provenance == report_provenance(audit), "Report differs from audited evidence"
    repo = HERE.parent
    assert not destination.is_relative_to(repo.resolve()) and not destination.is_relative_to(
        RUNS.resolve()
    ), "Bundle outside the source and run directories"
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    assert not dirty, "Commit the audited source and report before bundling"
    assert not destination.exists(), "Use a new bundle destination"
    destination.mkdir(parents=True)
    manifest = {}

    def copy(source, relative, compress=False):
        target = destination / (relative + ".gz" if compress else relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            with source.open("rb") as original, target.open("wb") as output:
                with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as stream:
                    shutil.copyfileobj(original, stream)
        else:
            shutil.copyfile(source, target)
        manifest[str(target.relative_to(destination))] = {
            "sha256": sha(target),
            "bytes": target.stat().st_size,
            "target": relative,
            "codec": "gzip" if compress else "identity",
            "uncompressed_sha256": sha(source),
        }

    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).decode().split("\0")
    for name in filter(None, tracked):
        assert (repo / name).resolve().is_relative_to(repo.resolve()), name
        copy(repo / name, "eml-torch/" + name)
    for source in evidence_paths(roster):
        copy(
            source,
            "emltorch-robustness-runs/" + str(source.relative_to(RUNS)),
            source.suffix == ".json",
        )
    copy(RUNS / "audit.json", "emltorch-robustness-runs/audit.json", True)
    instructions = """# EML component robustness bundle

Read `eml-torch/robustness/results/REPORT.md` and open its adjacent `explorer.html`.
The source and compact report are in Git; this bundle additionally contains raw
test traces, every candidate checkpoint, and the validation tensors needed for
an offline GPU checkpoint audit. Model weights and large training activations
are not bundled. Earlier exploratory evidence is preserved under
`eml-torch/research/`; its retired executables and complete original manifest
are available at Git commit 51cb441c80ef6eef5b2fe4480b1c6b45ec896cf5.

After extracting this ZIP, run from the bundle root:

```
python eml-torch/robustness/bundle.py verify --path .
python eml-torch/robustness/bundle.py unpack --path .
cd eml-torch/robustness
GEMMA_PYTHON=/path/to/gemma-env/bin/python CUDA_VISIBLE_DEVICES=0 python audit.py
```

Install the core package and the research dependencies documented in the study
README. The audit performs numerical validation on CUDA. For numerical
replay of Gemma checkpoints, it uses the separate pinned environment selected by
GEMMA_PYTHON. Run the main audit in the original Qwen environment. For a fresh training
reproduction, use a separate empty checkout/workspace and follow that README;
do not treat this populated evaluation bundle as a fresh training directory.

Compressed evidence JSON is expanded only when its bytes match the manifest.
Unpacking refuses to replace a different existing file. Validation audit caches
are reused only when their input hashes match; moving this bundle normally
causes the whole-block checkpoint audit to run again.
"""
    (destination / "README.md").write_text(instructions)
    manifest["README.md"] = {
        "sha256": sha(destination / "README.md"),
        "bytes": (destination / "README.md").stat().st_size,
        "target": "README.md",
        "codec": "identity",
        "uncompressed_sha256": sha(destination / "README.md"),
    }
    record = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "files": manifest,
    }
    (destination / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    assert require_audit() == audit, "Audit changed while packaging"
    assert report_provenance(audit) == provenance, "Report changed while packaging"
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True), (
        "Source checkout changed while packaging"
    )
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        == record["source_commit"]
    ), "Source commit changed while packaging"
    verify(destination)
    archive = destination.with_suffix(".zip")
    assert not archive.exists(), "Refusing to overwrite an existing archive"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(destination))
    delivery = {
        "archive": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": sha(archive),
        "source_commit": record["source_commit"],
        "manifest_files": len(manifest),
    }
    archive.with_suffix(".delivery.json").write_text(json.dumps(delivery, indent=2) + "\n")
    print(json.dumps(delivery, indent=2), flush=True)


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())["files"]
    for name, record in manifest.items():
        path = safe(root, name)
        assert path.stat().st_size == record["bytes"] and sha(path) == record["sha256"], name
    print("BUNDLE VERIFIED", len(manifest), flush=True)


def unpack(root):
    verify(root)
    manifest = json.loads((root / "manifest.json").read_text())["files"]
    for name, record in manifest.items():
        if record["codec"] != "gzip":
            continue
        target = safe(root, record["target"])
        if target.exists():
            assert sha(target) == record["uncompressed_sha256"], str(target)
            continue
        temporary = target.with_name(target.name + ".unpacking")
        assert not temporary.exists(), str(temporary)
        with gzip.open(safe(root, name), "rb") as source, temporary.open("xb") as output:
            shutil.copyfileobj(source, output)
        assert sha(temporary) == record["uncompressed_sha256"], name
        temporary.replace(target)
    print("EVIDENCE JSON EXPANDED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["create", "verify", "unpack"])
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()
    {"create": create, "verify": verify, "unpack": unpack}[args.command](args.path.resolve())
