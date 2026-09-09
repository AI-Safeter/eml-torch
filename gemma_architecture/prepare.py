"""Freeze reused development data and untouched confirmation before fitting."""

import argparse
import hashlib
import json
import random
import struct
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import torch

from gemma_mechanisms.prepare import fetch, key, old_pairs
from gemma_mechanisms.runtime import SPEC as OLD_SPEC

from .common import (
    HERE,
    SNAPSHOT,
    SPEC,
    accounting,
    digest,
    freeze,
    root,
    save,
    setup,
    sources,
    tokenizer,
)


def question_hash(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def identity(out):
    import transformers
    from transformers import AutoConfig, Gemma4ForConditionalGeneration

    assert transformers.__version__ == OLD_SPEC["model"]["transformers"]
    config = AutoConfig.from_pretrained(SNAPSHOT, local_files_only=True)
    assert config.model_type == "gemma4" and config.text_config.hidden_size == 1536
    assert config.text_config.num_hidden_layers == 35
    with torch.device("meta"):
        model = Gemma4ForConditionalGeneration(config)
    counts = accounting(model)
    native = accounting(model.model.language_model.layers[26].mlp)
    assert counts["parameters"] == 5104297504 and native["parameters"] == 56623104
    with (SNAPSHOT / "model.safetensors").open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    result = {
        "model": SPEC["model"],
        "architecture": config.architectures,
        "config_sha256": digest(SNAPSHOT / "config.json"),
        "safetensors_header_sha256": hashlib.sha256(
            json.dumps(header, sort_keys=True).encode()
        ).hexdigest(),
        "parameters": counts["parameters"],
        "buffers": counts["buffers"],
        "native_mlp_parameters": native["parameters"],
        "note": "Configuration and meta architecture reverified locally; full-model CUDA integration is validated separately. E2B is effective size, not stored parameter count.",
    }
    prior = json.loads((HERE.parent / "gemma_mechanisms/model-identity.json").read_text())
    for k in ["config_sha256", "safetensors_header_sha256", "parameters", "native_mlp_parameters"]:
        assert result[k] == prior[k]
    save(out / "model-identity.json", result)
    save(HERE / "model-identity.json", result)
    print("IDENTITY VERIFIED", SPEC["model"]["id"], counts["parameters"], flush=True)


def historical(previous):
    used = old_pairs()
    inputs = {}
    roots = [previous]
    if previous.name.endswith("-bos"):
        companion = previous.with_name(previous.name[:-4])
        if companion.exists():
            roots.append(companion)

    def visit(obj, operation=None):
        if isinstance(obj, dict):
            operation = obj.get("op", operation)
            if isinstance(obj.get("a"), int) and isinstance(obj.get("b"), int):
                ops = [operation] if operation in used else list(used)
                for op in ops:
                    for operand in ["a", "c", "donor_a"]:
                        if isinstance(obj.get(operand), int):
                            used[op].add(key(op, obj[operand], obj["b"]))
            for value in obj.values():
                visit(value, operation)
        elif isinstance(obj, list):
            for value in obj:
                visit(value, operation)

    for directory in roots:
        for path in sorted(directory.rglob("*.json")):
            value = json.loads(path.read_text())
            visit(value)
            # Exclusion is derived from all old records, including counterfactual donors.
            inputs[str(path)] = digest(path)
    return used, inputs


def arithmetic(out, previous, used, rng):
    development = {}
    for name, old_split, count in [
        ("ordinary", "final", SPEC["development"]["ordinary_groups_per_operation"]),
        ("shift", "shift", SPEC["development"]["shift_groups_per_operation"]),
    ]:
        rows = json.loads((previous / "data" / f"arithmetic-{old_split}.json").read_text())
        development[name] = []
        for op in used:
            values = [r for r in rows if r["op"] == op]
            rng.shuffle(values)
            development[name] += values[:count]
    save(out / "arithmetic-development.json", development)
    confirmation = {}
    for split, count in [
        ("ordinary", SPEC["confirmation"]["ordinary_groups_per_operation"]),
        ("shift", SPEC["confirmation"]["shift_groups_per_operation"]),
    ]:
        rows = []
        for op in used:
            for i in range(count):
                while True:
                    if split == "shift":
                        a = rng.randint(1000, 99999)
                        b = rng.randint(1000, 99999) if op == "add" else rng.randint(100, 999)
                        if op == "add" and i % 2 == 0:
                            length = rng.choice([3, 4, 5])
                            a, b = (
                                rng.randint(1, 99) * 10**length + 10**length - 1,
                                rng.randint(1, 99),
                            )
                    elif op == "add":
                        a, b = rng.randint(500, 999), rng.randint(500, 999)
                    elif op == "multiply":
                        a, b = rng.randint(300, 999), rng.randint(11, 99)
                    else:
                        a, b = rng.randint(500, 9999), rng.randint(51, 99)
                    pair = key(op, a, b)
                    if pair not in used[op]:
                        used[op].add(pair)
                        break
                answer = a + b if op == "add" else a * b if op == "multiply" else a // b
                rows.append(
                    {
                        "id": f"architecture-{split}-{op}-{i:05d}",
                        "op": op,
                        "a": a,
                        "b": b,
                        "answer": answer,
                        "variables": {},
                    }
                )
        confirmation[split] = rows
        for op in used:
            values = torch.tensor(
                [[r["a"], r["b"], r["answer"]] for r in rows if r["op"] == op], device="cuda"
            )
            a, b, answer = values.unbind(-1)
            expected = (
                a + b
                if op == "add"
                else a * b
                if op == "multiply"
                else torch.div(a, b, rounding_mode="floor")
            )
            assert torch.equal(answer, expected)
    save(out / "arithmetic-confirmation.json", confirmation)


def language(out, previous, rng):
    import pyarrow.parquet as pq

    manifests = json.loads((previous / "data/language-documents.json").read_text())
    excluded_hashes = {r["sha256"] for rows in manifests.values() for r in rows}
    excluded_domains = {r["domain"] for rows in manifests.values() for r in rows}
    indices = list(range(len(manifests["final"])))
    rng.shuffle(indices)
    indices = indices[: SPEC["development"]["language_documents"]]
    blocks = torch.load(previous / "data/language-final.pt", weights_only=True)
    torch.save(blocks[indices], out / "language-development.pt")
    development = [manifests["final"][i] for i in indices]
    path = fetch(OLD_SPEC["language"], OLD_SPEC["language"]["file"])
    tok = tokenizer()
    collected, documents, scanned = [], [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["text", "url", "id"]):
        rows = batch.to_pylist()
        for offset, row in enumerate(rows):
            domain = urlparse(row["url"]).hostname
            sha = hashlib.sha256(row["text"].encode()).hexdigest()
            if not domain or domain in excluded_domains or sha in excluded_hashes:
                continue
            ids = tok.encode(row["text"], add_special_tokens=False)
            if len(ids) < SPEC["confirmation"]["language_tokens"]:
                continue
            collected.append([tok.bos_token_id] + ids[: SPEC["confirmation"]["language_tokens"]])
            documents.append(
                {
                    "id": row["id"],
                    "row": scanned + offset,
                    "domain": domain,
                    "url": row["url"],
                    "sha256": sha,
                }
            )
            excluded_domains.add(domain)
            excluded_hashes.add(sha)
            if len(collected) == SPEC["confirmation"]["language_documents"]:
                break
        scanned += len(rows)
        if len(collected) == SPEC["confirmation"]["language_documents"]:
            break
    assert len(collected) == SPEC["confirmation"]["language_documents"]
    torch.save(torch.tensor(collected, dtype=torch.long), out / "language-confirmation.pt")
    save(out / "language-documents.json", {"development": development, "confirmation": documents})
    return {
        "source_sha256": digest(path),
        "scanned_rows": scanned,
        "new_hostname_count": len(documents),
    }


def questions(out, previous, rng):
    import pyarrow.parquet as pq

    old = [
        r
        for path in sorted((previous / "data").glob("arc-*.json"))
        for r in json.loads(path.read_text())
    ]
    used_ids = {r["id"] for r in old}
    used_hashes = {question_hash(r["question"]) for r in old}
    dev = json.loads((previous / "data/arc-final.json").read_text())
    rng.shuffle(dev)
    save(out / "arc-development.json", dev[: SPEC["development"]["arc_questions"]])
    path = fetch(OLD_SPEC["general_qa"], "ARC-Easy/train-00000-of-00001.parquet")
    rows = pq.read_table(path).to_pylist()
    rng.shuffle(rows)
    selected = []
    for row in rows:
        sha = question_hash(row["question"])
        if row["id"] in used_ids or sha in used_hashes:
            continue
        selected.append(row)
        used_ids.add(row["id"])
        used_hashes.add(sha)
        if len(selected) == SPEC["confirmation"]["arc_questions"]:
            break
    assert len(selected) == SPEC["confirmation"]["arc_questions"]
    save(out / "arc-confirmation.json", selected)
    return {
        "source_sha256": digest(path),
        "source_split": "train",
        "source_rows": len(rows),
        "selected": len(selected),
        "novelty_scope": "Unseen in this study's previous evaluations, not guaranteed absent from model pretraining",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--previous", default="/home/ubuntu/samuel/emltorch-gemma-replacement-runs-bos"
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    setup(SPEC["data_seed"])
    out, old = root(args.output), Path(args.previous).resolve()
    data = out / "data"
    data.mkdir(exist_ok=True)
    if (data / "freeze.json").exists():
        record = json.loads((data / "freeze.json").read_text())
        assert record["sources"] == sources("prepare.py", "common.py")
        assert all(digest(data / n) == sha for n, sha in record["files"].items())
        print("PRESERVE FROZEN DATA", flush=True)
        return
    assert not (out / "training").exists(), "Freeze data before training"
    identity(out)
    used, exclusions = historical(old)
    counts = {k: len(v) for k, v in used.items()}
    rng = random.Random(SPEC["data_seed"])
    arithmetic(data, old, used, rng)
    print("OPERANDS FROZEN", counts, flush=True)
    prose = language(data, old, rng)
    qa = questions(data, old, rng)
    record = {
        "previous_root": str(old),
        "previous_inputs": exclusions,
        "excluded_operand_pairs": counts,
        "sources": sources("prepare.py", "common.py"),
        "language": prose,
        "questions": qa,
        "files": {p.name: digest(p) for p in sorted(data.iterdir()) if p.name != "freeze.json"},
        "confirmation_outputs_observed": False,
        "development_scope": "All previously inspected splits are development. Only previous train activations supervise fitting; previous selection activations select checkpoints.",
    }
    freeze(data / "freeze.json", record)
    save(HERE / "data-freeze.json", record)
    print("DATA FROZEN", len(exclusions), "historical files", flush=True)


if __name__ == "__main__":
    main()
