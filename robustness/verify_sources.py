"""Verify append-only study snapshots, including documented deployment repair."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    expected = {}
    for name in [
        "freeze.json",
        "extension-freeze.json",
        "evaluation-freeze.json",
        "deployment-freeze.json",
        "gemma/source-freeze.json",
        "gemma/execution-freeze.json",
        "prefix-freeze.json",
        "prefix-reference-freeze.json",
        "gemma/prefix-prototype-freeze.json",
    ]:
        record = json.loads((HERE / name).read_text())
        for filename, digest in record["files"].items():
            if filename in expected and expected[filename] != digest:
                assert record.get("supersedes", {}).get(filename) == expected[filename]
            expected[filename] = digest
    for name, digest in expected.items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, name
    print("ALL SOURCE AND DATA SNAPSHOTS VERIFIED", len(expected), flush=True)


if __name__ == "__main__":
    main()
