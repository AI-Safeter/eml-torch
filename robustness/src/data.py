"""Freeze operation-specific operand groups and prompt formats before collection."""

import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(os.environ.get("EMLTORCH_RESEARCH_ROOT", Path(__file__).resolve().parent))
STYLES = ["prose", "symbolic", "code"]


def save(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def result(op, a, b):
    return a + b if op == "add" else a * b if op == "multiply" else a // b


def key(op, a, b):
    return tuple(sorted((a, b))) if op != "divide" else (a, b)


def old_addition_pairs():
    frozen = ROOT / "historical-addition-pairs.json"
    if not frozen.exists():
        frozen = Path(__file__).resolve().parent / "historical-addition-pairs.json"
    if frozen.exists():
        return {tuple(pair) for pair in json.loads(frozen.read_text())}
    raise FileNotFoundError("The bundled historical-addition-pairs.json is required")


def make_pairs(op, pool, n, used, rng):
    pool = list(pool)
    rng.shuffle(pool)
    by_b = {}
    for a, b in pool:
        by_b.setdefault(b, []).append(a)
    rows = []
    for a, b in pool:
        if len(rows) >= n:
            break
        if key(op, a, b) in used:
            continue
        s = str(result(op, a, b))
        alternatives = [
            c
            for c in by_b[b]
            if key(op, c, b) not in used
            and len(str(result(op, c, b))) == len(s)
            and str(result(op, c, b))[0] != s[0]
        ]
        if not alternatives:
            continue
        c = rng.choice(alternatives)
        used.update([key(op, a, b), key(op, c, b)])
        rows.append(
            {
                "a": a,
                "b": b,
                "c": c,
                "answer": result(op, a, b),
                "corrupt_answer": result(op, c, b),
                "op": op,
            }
        )
    return rows


def main():
    for op in ["add", "multiply", "divide"]:
        out = ROOT / op
        out.mkdir(exist_ok=True)
        if (out / "problems.json").exists():
            print("Preserving frozen data", op, flush=True)
            continue
        rng = random.Random(20260910)
        if op == "add":
            pool = [(a, b) for b in range(10, 300) for a in range(b, 300) if a % 10 + b % 10 < 17]
            shifted = [(a, b) for b in range(300, 500) for a in range(b, 500)]
            carry = [(a, b) for b in range(10, 300) for a in range(b, 300) if a % 10 + b % 10 >= 17]
            counts = {
                "train": 1024,
                "validation": 256,
                "test": 256,
                "shift": 256,
                "carry": 256,
            }
        elif op == "multiply":
            pool = [(a, b) for b in range(2, 100) for a in range(b, 100)]
            shifted = [(a, b) for b in range(100, 300) for a in range(b, 300)]
            carry = []
            counts = {"train": 1024, "validation": 256, "test": 256, "shift": 256}
        else:
            pool = [(q * b, b) for b in range(2, 30) for q in range(2, 100)]
            shifted = [(q * b, b) for b in range(30, 50) for q in range(100, 400)]
            carry = []
            counts = {"train": 768, "validation": 192, "test": 192, "shift": 192}
        used = old_addition_pairs() if op == "add" else set()
        initial_used = used.copy()
        splits = {}
        for split, n in counts.items():
            chosen = shifted if split == "shift" else carry if split == "carry" else pool
            splits[split] = make_pairs(op, chosen, n, used, rng)
            assert len(splits[split]) == n, (op, split, len(splits[split]), n)
        groups = {
            s: {key(op, r[t], r["b"]) for r in rows for t in ["a", "c"]}
            for s, rows in splits.items()
        }
        assert all(len(groups[s]) == 2 * len(rows) for s, rows in splits.items())
        assert all(not groups[a] & groups[b] for a in groups for b in groups if a != b)
        assert not initial_used & set.union(*groups.values())
        save(out / "problems.json", splits)
        save(
            out / "protocol.json",
            {
                "operation": op,
                "counts": counts,
                "seed": 20260910,
                "formats_training": STYLES,
                "formats_test": STYLES + ["instruction"],
                "grouping": "Clean and corrupted operand problems are unique and disjoint across splits; commutative operations canonicalized.",
                "prior_addition_pairs_excluded": len(initial_used),
                "test_model_outputs_unseen_at_freeze": True,
                "selection": "Training full-MLP patching selects three layers; validation selects one direction; all heads selected on validation only.",
                "success": {
                    "accuracy_drop_pp_max": 1,
                    "causal_correlation_min": 0.9,
                    "causal_nrmse_max": 0.2,
                },
                "model": "Qwen/Qwen3-1.7B",
                "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                "sha256": hashlib.sha256((out / "problems.json").read_bytes()).hexdigest(),
            },
        )
        print(op, counts, flush=True)


if __name__ == "__main__":
    main()
