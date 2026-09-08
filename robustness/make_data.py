"""Generate independent, disjoint arithmetic cohorts without querying a model."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from data import key, make_pairs, result, save  # noqa: E402

SEED = 20260923


def historical(op):
    old = HERE.parent / "research"
    used = set()
    sources = {}
    paths = sorted(old.rglob("*problems.json"))
    for path in paths:
        if path.parent.name != op:
            continue
        content = json.loads(path.read_text())
        for rows in content.values():
            for row in rows:
                used.add(key(op, row["a"], row["b"]))
                used.add(key(op, row["c"], row.get("corrupt_b", row["b"])))
        sources[str(path.relative_to(old))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if op == "add":
        path = old / "historical-addition-pairs.json"
        used.update(tuple(pair) for pair in json.loads(path.read_text()))
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return used, sources


def pools(op):
    if op == "add":
        base = [(a, b) for b in range(10, 500) for a in range(b, 500)]
        return (
            [(a, b) for a, b in base if a % 10 + b % 10 < 17],
            [(a, b) for b in range(500, 1500) for a in range(b, 1500)],
            [(a, b) for a, b in base if a % 10 + b % 10 >= 17],
        )
    if op == "multiply":
        return (
            [(a, b) for b in range(2, 200) for a in range(b, 200)],
            [(a, b) for b in range(200, 400) for a in range(b, 400)],
            [],
        )
    return (
        [(q * b, b) for b in range(2, 80) for q in range(2, 500)],
        [(q * b, b) for b in range(80, 130) for q in range(500, 1500)],
        [],
    )


def ordinary(op, pool, used, rng, n):
    candidates = [pair for pair in pool if key(op, *pair) not in used]
    rng.shuffle(candidates)
    rows = []
    for a, b in candidates[:n]:
        used.add(key(op, a, b))
        rows.append(
            {
                "a": a,
                "b": b,
                "c": a,
                "answer": result(op, a, b),
                "corrupt_answer": result(op, a, b),
                "op": op,
            }
        )
    assert len(rows) == n
    return rows


def both_operands(op, pool, used, rng, n):
    candidates = [pair for pair in pool if key(op, *pair) not in used]
    rng.shuffle(candidates)
    rows = []
    for a, b in candidates:
        if key(op, a, b) in used:
            continue
        answer = str(result(op, a, b))
        alternatives = [
            (c, d)
            for c, d in candidates
            if c != a
            and d != b
            and key(op, c, d) not in used
            and len(str(result(op, c, d))) == len(answer)
            and str(result(op, c, d))[0] != answer[0]
        ]
        if not alternatives:
            continue
        c, d = rng.choice(alternatives)
        used.update([key(op, a, b), key(op, c, d)])
        rows.append(
            {
                "a": a,
                "b": b,
                "c": c,
                "corrupt_b": d,
                "answer": int(answer),
                "corrupt_answer": result(op, c, d),
                "op": op,
            }
        )
        if len(rows) == n:
            return rows
    raise RuntimeError((op, "insufficient both-operand contrasts", len(rows)))


def generate(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for op in ["add", "multiply", "divide"]:
        rng = random.Random(SEED)
        used, sources = historical(op)
        excluded = used.copy()
        pool, shift, carry = pools(op)
        splits = {}
        counts = {"train": 2048, "validation": 512, "test": 1024, "shift": 512}
        if carry:
            counts["carry"] = 512
        for split, count in counts.items():
            chosen = shift if split == "shift" else carry if split == "carry" else pool
            splits[split] = make_pairs(op, chosen, count, used, rng)
            assert len(splits[split]) == count, (op, split, len(splits[split]))
        extra = {
            "ordinary": ordinary(op, pool, used, rng, 1024),
            "both_operands": both_operands(op, pool, used, rng, 256),
        }
        groups = {}
        for split, rows in {**splits, **extra}.items():
            pairs = [key(op, r["a"], r["b"]) for r in rows]
            if split != "ordinary":
                pairs += [key(op, r["c"], r.get("corrupt_b", r["b"])) for r in rows]
            assert len(pairs) == len(set(pairs)), (op, split, "duplicate")
            groups[split] = set(pairs)
            assert not groups[split] & excluded
            for r in rows:
                assert result(op, r["a"], r["b"]) == r["answer"]
                assert result(op, r["c"], r.get("corrupt_b", r["b"])) == r["corrupt_answer"]
        assert all(not groups[a] & groups[b] for a in groups for b in groups if a != b)
        out = destination / op
        out.mkdir(exist_ok=True)
        for name, value in [("problems.json", splits), ("extra-problems.json", extra)]:
            if (out / name).exists():
                assert json.loads((out / name).read_text()) == value, (
                    "Refusing to change frozen data"
                )
        save(out / "problems.json", splits)
        save(out / "extra-problems.json", extra)
        save(
            out / "protocol.json",
            {
                "seed": SEED,
                "counts": counts,
                "ordinary_count": 1024,
                "both_operand_count": 256,
                "historical_problem_count_excluded": len(excluded),
                "historical_sources_sha256": sources,
                "new_unique_problem_count": len(used - excluded),
                "model_outputs_unseen": True,
                "sha256": {
                    name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                    for name in ["problems.json", "extra-problems.json"]
                },
            },
        )
        print(op, counts, "historical exclusions", len(excluded), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "data")
    generate(parser.parse_args().output)
