"""Evaluate the frozen division seed interventions on a smaller available GPU."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bundle import sha
from gemma import adapter, host_embeddings, host_execute
from gemma.auxiliary_evaluate import publish, verify_publication
from gemma.streamed_mlp import StreamedMLP, load
from runtime import RUNS


class YieldToMain(Exception):
    pass


def main_owns_cohort():
    state = json.loads((RUNS / "queue-gemma.json").read_text())
    if state["status"] == "complete":
        return True
    assert state["status"] == "running", state
    parent = Path("/proc") / str(state["child_pid"])
    assert parent.exists(), "Main Gemma orchestrator is missing"
    for child in (parent / "task" / parent.name / "children").read_text().split():
        try:
            args = (Path("/proc") / child / "cmdline").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        if b"divide" not in args[2:]:
            continue
        if b"seed_evaluate" in args:
            # Native seed evaluation generates answers before interventions.
            return (RUNS / "gemma/divide/ordinary-all-seeds.json").exists()
        if any(stage in args for stage in [b"geometry_evaluate", b"analyze", b"per_style"]):
            return b"--validation-only" not in args
    return False


def verify():
    verify_publication()
    frozen = json.loads((adapter.HERE / "streamed-execution-freeze.json").read_text())
    for name, expected in frozen["files"].items():
        assert sha(adapter.STUDY / name) == expected, name
    proof = json.loads((adapter.HERE / "streamed-validation.json").read_text())
    assert proof["passed"] and proof["raw_file_bytes_identical"]
    assert proof["method_records"] == 11904 and proof["peak_reserved_GiB"] <= 9


def seed_selection(out):
    """Mirror the frozen seed_evaluate selection, without selecting on test data."""
    from evaluate_heads import freeze

    directories = ["heads", "heads-active-r32-g0", "heads-active-r32-g0.1"]
    assert (out / "selection.json").exists()
    selected, heads = freeze(out, directories), {}
    for directory in directories:
        protocol = json.loads((out / directory / "protocol.json").read_text())
        records = json.loads((out / directory / "candidates.json").read_text())
        assert len(records) == 10 and all(row["status"] == "complete" for row in records)
        for row in records:
            checkpoint = f"{directory}/{row['name']}.pt"
            heads[f"{directory}/{row['kind']}-seed-{row['seed']}"] = {
                **row,
                "features": protocol["features"],
                "rank": protocol["rank"],
                "checkpoint": checkpoint,
                "sha256": sha(out / checkpoint),
            }
    return {**selected, "heads": heads, "sparse": {}, "linear": {}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=["native", "streamed"], default="streamed")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    assert not args.publish or (args.engine == "streamed" and not args.validation_only)
    assert args.reference is None or (args.engine == "streamed" and args.validation_only)
    host_execute.verify()
    adapter.install()
    from data import STYLES, save
    from evaluate_heads import Replacements, interventions
    from model_io import expand, setup
    from trace_audit import check

    setup()
    torch.cuda.reset_peak_memory_stats()
    out = RUNS / "gemma/divide"
    destination = args.output.resolve()
    assert not destination.is_relative_to(RUNS) and not destination.is_relative_to(
        adapter.STUDY.parent
    )
    assert not destination.exists(), "Use a new private output directory"
    if args.publish:
        verify()
        if main_owns_cohort() or (out / "interventions-all-seeds.json").exists():
            print("STREAMED COHORT ALREADY OWNED OR COMPLETE", flush=True)
            return
    destination.mkdir(parents=True)
    last_checked = 0.0

    def guard(force=False):
        nonlocal last_checked
        if args.publish and (force or time.monotonic() - last_checked >= 1):
            last_checked = time.monotonic()
            if main_owns_cohort():
                raise YieldToMain()

    sources = {
        name: sha(adapter.STUDY / name)
        for name in [
            "gemma/streamed_mlp.py",
            "gemma/streamed_cohort.py",
            "gemma/host-execution-freeze.json",
            "gemma/auxiliary-execution-freeze.json",
            "seed_evaluate.py",
        ]
    }
    selected = seed_selection(out)
    save(destination / "selection.json", selected)
    rep = Replacements(out, selected)
    rep.kinds = ["original", *rep.models]
    assert rep.layer == 33 and len(rep.kinds) == 31
    problems = json.loads((out / "problems.json").read_text())
    assert sha(out / "problems.json") == selected["problems_sha256"]
    rows = expand(problems["validation" if args.validation_only else "test"], STYLES)
    if args.validation_only:
        rows = rows[:64]
    save(
        destination / "job.json",
        {"pid": os.getpid(), "engine": args.engine, "gpu": torch.cuda.get_device_name()},
    )
    started = time.monotonic()
    try:
        guard(force=True)
        model, tok = host_embeddings.load() if args.engine == "native" else load(guard)
        host_execute.install_prefix(model, rep.layer, globals())
        with torch.inference_mode():
            result = interventions(model, tok, rep, rows)
        guard(force=True)
    except YieldToMain:
        save(destination / "yielded.json", {"main_owns_cohort": True})
        print("STREAMED COHORT YIELDED TO MAIN", flush=True)
        return
    trace = check(result, rows, rep.kinds, alphas=[0, 0.125, 0.375, 0.625, 0.875, 1])
    for name, expected in sources.items():
        assert sha(adapter.STUDY / name) == expected, name
    name = "interventions-all-seeds" + ("-validation" if args.validation_only else "") + ".json"
    temporary = destination / (name + ".tmp")
    save(temporary, result)
    temporary.replace(destination / name)
    record = {
        "engine": args.engine,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
        "trace": trace,
        "method_records": sum(map(len, result.values())),
        "source_sha256": sources,
        "selection_sha256": sha(destination / "selection.json"),
        "raw_sha256": sha(destination / name),
    }
    if args.engine == "streamed":
        modules = [module for module in model.modules() if isinstance(module, StreamedMLP)]
        assert len(modules) == 16 and all(module.calls > 0 for module in modules)
        assert all(
            p.device.type == "cpu" and p.is_pinned() for m in modules for p in m.parameters()
        )
        record["streamed_mlp_calls"] = [module.calls for module in modules]
    if args.reference is not None:
        reference = json.loads((args.reference / "completed.json").read_text())
        assert reference["engine"] == "native" and reference["source_sha256"] == sources
        assert reference["selection_sha256"] == record["selection_sha256"]
        assert (args.reference / name).read_bytes() == (destination / name).read_bytes()
        record.update(passed=True, raw_file_bytes_identical=True, native_reference=reference)
    if args.publish:
        if main_owns_cohort():
            save(destination / "yielded.json", {"main_owns_cohort": True})
            print("STREAMED COHORT YIELDED BEFORE PUBLICATION", flush=True)
            return
        record["publication"] = publish(destination / name, out / name)
    save(destination / "completed.json", record)
    print("STREAMED COHORT COMPLETE", json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
