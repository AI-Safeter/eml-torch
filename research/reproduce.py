"""Reproduce frozen analyses, model interventions, or training in a new directory."""

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = {
    "1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", ""),
    "0.6b": (
        "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca",
        "replication-0.6b",
    ),
}


def verify():
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    for row in manifest["files"]:
        path = HERE / row["path"]
        assert path.is_file(), row["path"]
        assert path.stat().st_size == row["bytes"], row["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], row["path"]
    print(f"Verified {len(manifest['files'])} files")


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".gz":
        destination = destination.with_suffix("")
        with gzip.open(source, "rb") as src, destination.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["verify", "unpack", "analyze", "evaluate", "train", "airfoil"]
    )
    parser.add_argument("--model", choices=MODELS, default="1.7b")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stop-after",
        choices=["features", "fit", "evaluate"],
        default="evaluate",
        help="For train mode, optionally stop after feature collection or fitting",
    )
    parser.add_argument(
        "--gpu", default="0", help="One visible GPU; separate runs can use other GPUs"
    )
    parser.add_argument(
        "--model-path", type=Path, help="Optional local pinned Hugging Face snapshot"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Also reproduce the fixed compact-head follow-up",
    )
    args = parser.parse_args()
    if args.mode == "verify":
        verify()
        return
    if args.output is None:
        parser.error("--output is required; published evidence is never overwritten")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be empty or absent")
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "unpack":
        if output == HERE or HERE in output.parents:
            parser.error("Unpack outside the published research directory")
        for path in sorted(HERE.rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and ".ruff_cache" not in path.parts
            ):
                copy(path, output / path.relative_to(HERE))
        manifest = []
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != "MANIFEST.json":
                manifest.append(
                    {
                        "path": str(path.relative_to(output)),
                        "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        (output / "MANIFEST.json").write_text(
            json.dumps({"files": manifest, "unpacked_from": str(HERE)}, indent=2) + "\n"
        )
        print("Unpacked evidence and source into", output)
        return
    model, revision, subfolder = MODELS[args.model]
    source = HERE / subfolder
    env = {
        **os.environ,
        "EMLTORCH_RESEARCH_ROOT": str(output),
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "EMLTORCH_MODEL_ID": model,
        "EMLTORCH_MODEL_REVISION": revision,
    }
    if args.model_path:
        env["EMLTORCH_MODEL_PATH"] = str(args.model_path.resolve())
    else:
        env.pop("EMLTORCH_MODEL_PATH", None)

    def run(script, *arguments):
        subprocess.run(
            [sys.executable, str(HERE / script), *map(str, arguments)],
            env=env,
            check=True,
        )

    if args.mode == "airfoil":
        shutil.copytree(HERE / "airfoil-replay/source", output / "airfoil-replay/source")
        run("distill_airfoil.py")
        run("export_airfoil.py")
        return
    copy(
        HERE / "historical-addition-pairs.json",
        output / "historical-addition-pairs.json",
    )
    if (source / "minimal-protocol.json").exists():
        copy(source / "minimal-protocol.json", output / "minimal-protocol.json")
    for op in ["add", "multiply", "divide"]:
        folder = source / op
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(folder)
            if args.mode == "analyze":
                keep = path.suffix in [".json", ".gz"]
            elif args.mode == "evaluate":
                keep = path.suffix in [".json", ".pt"] and not (
                    path.name.startswith(
                        (
                            "ordinary-",
                            "interventions-",
                            "linear-ordinary-",
                            "linear-interventions-",
                        )
                    )
                    or path.name
                    in [
                        "coordinate-interventions.json",
                        "linear-coordinate-interventions.json",
                        "results.json",
                        "outcome.json",
                    ]
                    or (len(relative.parts) == 1 and path.name.endswith("completed.json"))
                )
            else:
                keep = path.name in [
                    "problems.json",
                    "protocol.json",
                    "minimal-problems.json",
                ]
            if keep:
                copy(path, output / op / relative)
    if args.mode == "analyze":
        for op in ["add", "multiply", "divide"]:
            run("analyze_heads.py", op)
        return
    directories = (
        ["heads-active-r32-g0"]
        if args.model == "0.6b"
        else ["heads", "heads-active-r32-g0", "heads-active-r32-g0.1"]
    )
    if args.mode == "train":
        for op in ["add", "multiply", "divide"]:
            run("collect.py", op)
        run("active_features.py")
        if args.model == "1.7b":
            run("derivative_data.py")
        if args.stop_after == "features":
            return
        for op in ["add", "multiply", "divide"]:
            if args.model == "1.7b":
                run("train_heads.py", op, "--max-seconds", 86400)
                run(
                    "train_heads.py",
                    op,
                    "--features",
                    "active",
                    "--gradient-weight",
                    0.1,
                    "--max-seconds",
                    86400,
                )
            extra = ["--widths", 32, "--max-steps", 12000] if args.model == "0.6b" else []
            run(
                "train_heads.py",
                op,
                "--features",
                "active",
                "--max-seconds",
                86400,
                *extra,
            )
            run("sparse_neurons.py", op)
        run("linear_controls.py")
        if args.stop_after == "fit":
            return
    for op in ["add", "multiply", "divide"]:
        run("evaluate_heads.py", op, "--directories", *directories)
        if args.compact and (output / op / "minimal-problems.json").exists():
            if args.mode == "train":
                run("minimal_heads.py", op, "--max-seconds", 86400)
            run("evaluate_minimal.py", op)
        run("analyze_heads.py", op)
    if args.mode == "evaluate" and args.model == "0.6b":
        run("evaluate_linear.py")
        for op in ["add", "multiply", "divide"]:
            run("analyze_heads.py", op)


if __name__ == "__main__":
    main()
