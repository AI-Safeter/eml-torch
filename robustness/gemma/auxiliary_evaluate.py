"""Precompute one Gemma operation without taking ownership from the main queue."""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import runtime
from gemma import adapter, generation_execute, host_execute


class MainQueueOwnsOperation(Exception):
    pass


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scientific(name):
    return name.endswith(".json") and name.startswith(
        ("ordinary-", "interventions-", "coordinate-")
    )


def main_owns(operation, runs):
    state = json.loads((runs / "queue-gemma.json").read_text())
    if state["status"] == "complete":
        return True
    assert state["status"] == "running", state
    parent = Path("/proc") / str(state["child_pid"])
    assert parent.exists(), "Main Gemma orchestrator is missing"
    children = parent / "task" / parent.name / "children"
    for child in children.read_text().split():
        try:
            args = (Path("/proc") / child / "cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        if operation.encode() in args[2:]:
            return True
    return False


def publish(source, target):
    """Install a complete copy atomically; never overwrite an existing result."""
    expected = digest(source)
    if target.exists():
        assert digest(target) == expected, f"Different existing cohort: {target}"
        return "identical_existing"
    temporary = target.with_name(f".auxiliary-{os.getpid()}-{target.name}")
    assert not temporary.exists(), temporary
    try:
        with source.open("rb") as original, temporary.open("xb") as destination:
            shutil.copyfileobj(original, destination)
        assert digest(temporary) == expected
        try:
            os.link(temporary, target)
        except FileExistsError:
            assert digest(target) == expected, f"Different concurrent cohort: {target}"
            return "identical_concurrent"
        return "published"
    finally:
        temporary.unlink(missing_ok=True)


def verify_publication():
    path = adapter.HERE / "auxiliary-execution-freeze.json"
    record = json.loads(path.read_text())
    for name, expected in record["files"].items():
        assert digest(adapter.STUDY / name) == expected, name
    proof = json.loads((adapter.HERE / "auxiliary-validation.json").read_text())
    assert proof["native_cached_file_bytes_identical"]
    assert proof["records"] == 2880 and proof["publication_guards_passed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=["add", "multiply", "divide"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=["native", "cached"], default="cached")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    generation_execute.verify()
    runs = runtime.RUNS
    source = runs / "gemma" / args.operation
    root = args.output.resolve()
    assert not root.is_relative_to(runs) and not root.is_relative_to(adapter.STUDY)
    out = root / args.operation
    if args.publish:
        assert args.engine == "cached" and not args.validate_only
        verify_publication()
        if main_owns(args.operation, runs):
            print("AUXILIARY YIELDED BEFORE START", args.operation, flush=True)
            return
    out.mkdir(parents=True, exist_ok=True)
    selected = json.loads((source / "selection.json").read_text())
    names = {
        "selection.json",
        "problems.json",
        "extra-problems.json",
        "component.pt",
        "component-active.pt",
        *selected["directories"],
    }
    names.update(
        spec["checkpoint"]
        for category in ["sparse", "linear"]
        for spec in selected[category].values()
    )
    for name in names:
        target = out / name
        if target.exists():
            assert target.is_symlink() and target.resolve() == (source / name).resolve(), target
        else:
            target.symlink_to(source / name, target_is_directory=(source / name).is_dir())
    manifest_path = root / "staged-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for path in out.iterdir():
        if scientific(path.name):
            assert digest(path) == manifest[path.name], (
                f"Incomplete/unrecorded staged cohort: {path}"
            )

    import data

    assert runtime.HERE == adapter.STUDY
    native_save = data.save
    original_install = adapter.install

    def save(path, value):
        path = Path(path)
        assert path.resolve().is_relative_to(root), f"Write outside staging directory: {path}"
        native_save(path, value)
        if not scientific(path.name):
            return
        manifest[path.name] = digest(path)
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)
        if args.publish:
            if main_owns(args.operation, runs):
                raise MainQueueOwnsOperation()
            status = publish(path, source / path.name)
            print(
                "AUXILIARY COHORT",
                args.operation,
                path.name,
                status,
                manifest[path.name],
                flush=True,
            )

    def install():
        original_install()
        import evaluate_heads

        data.ROOT = evaluate_heads.ROOT = root
        data.save = evaluate_heads.save = save

    adapter.install = install
    sys.argv = ["auxiliary", "evaluate_heads", args.operation]
    if args.validate_only:
        sys.argv.append("--validate-only")
    try:
        (generation_execute if args.engine == "cached" else host_execute).main()
    except MainQueueOwnsOperation:
        print("AUXILIARY YIELDED AFTER COHORT", args.operation, flush=True)


if __name__ == "__main__":
    main()
