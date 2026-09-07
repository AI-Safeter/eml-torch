"""Record source, data, and model artifact hashes without including runtime caches."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    files = {}
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or path.suffix == ".log"
            or path == ROOT / "manifest.json"
        ):
            continue
        files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    core = {
        str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT.parent / "emltorch").glob("*.py"))
    }
    versions = {}
    for name in ["torch", "numpy", "scipy", "transformers", "pysr", "sympy", "matplotlib"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    (ROOT / "manifest.json").write_text(
        json.dumps({"files": files, "core": core, "versions": versions}, indent=2) + "\n"
    )
    print("Manifest written:", len(files), "application artifacts")


if __name__ == "__main__":
    main()
