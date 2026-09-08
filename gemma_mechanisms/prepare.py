"""Freeze operand groups, domain-disjoint prose, and general QA before model runs."""

import argparse
import hashlib
import json
import random
from urllib.parse import urlparse

import torch

from .runtime import HERE, SPEC, digest, root, save, tokenizer


def key(op, a, b):
    return tuple(sorted((a, b))) if op != "divide" else (a, b)


def old_pairs():
    used = {op: set() for op in SPEC["arithmetic"]["operations"]}

    def visit(obj, op):
        if isinstance(obj, dict):
            if all(k in obj for k in ["a", "b"]):
                used[op].add(key(op, obj["a"], obj["b"]))
                if "c" in obj:
                    used[op].add(key(op, obj["c"], obj["b"]))
            for v in obj.values():
                visit(v, op)
        elif isinstance(obj, list):
            for v in obj:
                visit(v, op)

    for op in used:
        for path in (HERE.parent / "robustness/data" / op).glob("*problems.json"):
            visit(json.loads(path.read_text()), op)
    assert all(used.values()), "Historical operand exclusion must be available"
    return used


def variables(a, b):
    c1 = int(a % 10 + b % 10 >= 10)
    c2 = int(a // 10 % 10 + b // 10 % 10 + c1 >= 10)
    return {
        "a_units": a % 10,
        "b_units": b % 10,
        "a_tens": a // 10 % 10,
        "b_tens": b // 10 % 10,
        "carry_units": c1,
        "carry_tens": c2,
        "sum_units": (a + b) % 10,
        "sum_tens": (a + b) // 10 % 10,
    }


def arithmetic(out):
    rng = random.Random(SPEC["data_seed"])
    used = old_pairs()
    initial = {op: len(v) for op, v in used.items()}
    counts = SPEC["arithmetic"]["counts_per_operation"]
    for split, n in counts.items():
        rows = []
        for op in used:
            for j in range(n):
                while True:
                    if split == "shift":
                        a = rng.randint(1000, 99999)
                        b = rng.randint(1000, 99999) if op == "add" else rng.randint(100, 999)
                        if op == "add" and j % 2 == 0:
                            # Explicitly include chains beyond the 3-digit training range.
                            length = rng.choice([3, 4, 5])
                            a = rng.randint(1, 99) * 10**length + 10**length - 1
                            b = rng.randint(1, 99)
                    elif op == "add":
                        a, b = rng.randint(500, 999), rng.randint(500, 999)
                    elif op == "multiply":
                        a, b = rng.randint(300, 999), rng.randint(11, 99)
                    else:
                        a, b = rng.randint(500, 9999), rng.randint(51, 99)
                    k = key(op, a, b)
                    if k not in used[op]:
                        used[op].add(k)
                        break
                answer = a + b if op == "add" else a * b if op == "multiply" else a // b
                rows.append(
                    {
                        "id": f"{split}-{op}-{j:05d}",
                        "op": op,
                        "a": a,
                        "b": b,
                        "answer": answer,
                        "variables": variables(a, b) if op == "add" else {},
                        "remainder": a % b if op == "divide" else None,
                    }
                )
        save(out / f"arithmetic-{split}.json", rows)
    return {"historical_excluded": initial, "seed": SPEC["data_seed"]}


def content(row, style):
    a, b, op = row["a"], row["b"], row["op"]
    symbol = {"add": "+", "multiply": "*", "divide": "//"}[op]
    noun = {"add": "sum", "multiply": "product", "divide": "integer quotient"}[op]
    if style == "symbolic":
        question = f"{a} {symbol} {b} = ?"
    elif style == "prose":
        question = f"Calculate the {noun} of {a} and {b}."
    elif style == "reversed":
        question = f"The second operand is {b}; the first is {a}. Give their {noun}."
    elif style == "code":
        question = f"What integer does this Python expression evaluate to?\n{a} {symbol} {b}"
    else:
        raise ValueError(style)
    return question + " Return only the integer answer."


def fetch(spec, filename):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        spec["id"], filename, repo_type="dataset", revision=spec["revision"], token=False
    )


def language(out, tok):
    import pyarrow.parquet as pq

    spec = SPEC["language"]
    path = fetch(spec, spec["file"])
    parquet = pq.ParquetFile(path)
    data = {k: [] for k in spec["counts"]}
    manifests = {k: [] for k in data}
    seen, offset = set(), 0
    for batch in parquet.iter_batches(batch_size=256, columns=["text", "url", "id"]):
        rows = batch.to_pylist()
        for i, row in enumerate(rows):
            domain = urlparse(row["url"]).hostname
            if not domain:
                continue
            value = (
                int(hashlib.sha256(f"{SPEC['data_seed']}:{domain}".encode()).hexdigest(), 16) % 100
            )
            split = (
                "train"
                if value < 50
                else "selection"
                if value < 60
                else "gate"
                if value < 80
                else "final"
            )
            if len(data[split]) >= spec["counts"][split]:
                continue
            text_hash = hashlib.sha256(row["text"].encode()).hexdigest()
            if text_hash in seen:
                continue
            ids = tok.encode(row["text"], add_special_tokens=False)
            if len(ids) < spec["tokens_per_document"]:
                continue
            ids = ids[: spec["tokens_per_document"]]
            seen.add(text_hash)
            data[split].append(ids)
            manifests[split].append(
                {
                    "id": row["id"],
                    "row": offset + i,
                    "domain": domain,
                    "url": row["url"],
                    "sha256": text_hash,
                }
            )
        offset += len(rows)
        if all(len(data[k]) == spec["counts"][k] for k in data):
            break
    assert all(len(data[k]) == spec["counts"][k] for k in data)
    for split, ids in data.items():
        torch.save(torch.tensor(ids, dtype=torch.long), out / f"language-{split}.pt")
    save(out / "language-documents.json", manifests)
    return {
        "source_sha256": digest(path),
        "scanned_rows": offset,
        "counts": {k: len(v) for k, v in data.items()},
    }


def arc(out):
    import pyarrow.parquet as pq

    spec = SPEC["general_qa"]
    records = {"selection": [], "gate": [], "final": []}
    provenance = {}
    for split in ["validation", "test"]:
        path = fetch(spec, f"ARC-Easy/{split}-00000-of-00001.parquet")
        provenance[split] = digest(path)
        for row in pq.read_table(path).to_pylist():
            if split == "validation":
                target = "selection"
            else:
                v = int(hashlib.sha256(f"{SPEC['data_seed']}:{row['id']}".encode()).hexdigest(), 16)
                target = "gate" if v % 2 == 0 else "final"
            records[target].append(row)
    for split, rows in records.items():
        save(out / f"arc-{split}.json", rows)
    return {"source_sha256": provenance, "counts": {k: len(v) for k, v in records.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    out = root(args.output) / "data"
    out.mkdir(exist_ok=True)
    manifest = out / "freeze.json"
    if manifest.exists():
        record = json.loads(manifest.read_text())
        assert record["protocol_sha256"] == digest(HERE / "protocol.json")
        assert all(digest(out / p) == h for p, h in record["files"].items())
        print("PRESERVE FROZEN DATA", flush=True)
        return
    tok = tokenizer()
    record = {"protocol_sha256": digest(HERE / "protocol.json"), "arithmetic": arithmetic(out)}
    print("ARITHMETIC FROZEN", flush=True)
    record["language"] = language(out, tok)
    print("LANGUAGE FROZEN", flush=True)
    record["arc"] = arc(out)
    record["files"] = {p.name: digest(p) for p in sorted(out.iterdir()) if p != manifest}
    save(manifest, record)
    print("DATA FROZEN", json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
