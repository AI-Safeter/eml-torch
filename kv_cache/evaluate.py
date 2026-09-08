"""Held-out QA, distinct storage accounting, and complete cache-hit request timings."""

import gc
import json
import re
import string
import time
from collections import Counter

import numpy as np
import torch
from common import (
    HERE,
    RUN,
    SPEC,
    answer,
    ids,
    model,
    restore,
    save_json,
    sha,
    snapshot,
    storage_bytes,
    tokenizer,
)
from v_codecs import load_codecs, pack4, unpack4


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score(text, references):
    prediction = normalize(text)
    em = max(float(prediction == normalize(ref)) for ref in references)
    f1 = 0.0
    for ref in references:
        a, b = prediction.split(), normalize(ref).split()
        overlap = sum((Counter(a) & Counter(b)).values())
        value = 2 * overlap / (len(a) + len(b)) if a or b else 1.0
        f1 = max(f1, value)
    return em, f1


def pack(records, method, codecs):
    packed = []
    for i, row in enumerate(records):
        v = row["v"]
        code = v if method == "original" else pack4(v) if method == "uint4" else codecs[i].encode(v)
        packed.append({"k": row["k"], "code": code, "length": row["length"]})
    return packed


def tensors(packed):
    return [
        x
        for r in packed
        for x in (r["k"], *(r["code"] if isinstance(r["code"], tuple) else (r["code"],)))
    ]


def unpack(packed, method, codecs, config):
    rows = []
    for i, r in enumerate(packed):
        v = (
            r["code"]
            if method == "original"
            else unpack4(r["code"])
            if method == "uint4"
            else codecs[i].decode(r["code"])
        )
        rows.append({"k": r["k"], "v": v, "length": r["length"]})
    # Native DynamicCache.update concatenates tensors; it does not modify the bank.
    return restore(rows, config, clone=False)


def clock():
    torch.cuda.synchronize()
    return time.perf_counter()


def clean():
    gc.collect()
    torch.cuda.empty_cache()


def verify_freeze():
    frozen = json.loads((RUN / "freeze.json").read_text())
    for filename, digest in frozen["sources"].items():
        assert sha(HERE.parent / filename) == digest, f"Frozen source changed: {filename}"
    assert sha(RUN / "documents.json") == frozen["documents_sha256"]
    assert sha(RUN / "codecs.pt") == frozen["codecs_sha256"]


@torch.inference_mode()
def quality(native, tok, documents):
    rows, storage = [], []
    for method in SPEC["methods"]:
        path = RUN / f"qa-{method}.json"
        codecs = [] if method in ["original", "uint4"] else load_codecs(RUN / "codecs.pt", method)
        method_rows = []
        for i, doc in enumerate(documents):
            original = native(
                input_ids=ids(doc["prefix"]), use_cache=True, logits_to_keep=1
            ).past_key_values
            records = snapshot(original, "cuda")
            del original
            kv_bytes = storage_bytes([x for r in records for x in (r["k"], r["v"])])
            k_bytes = storage_bytes([r["k"] for r in records])
            packed = pack(records, method, codecs)
            if i == 0:
                weight_bytes = sum(c.bytes() for c in codecs)
                compressed_bytes = storage_bytes(tensors(packed))
                n = SPEC["storage_reference_documents"]
                storage.append(
                    {
                        "method": method,
                        "original_kv_bytes_per_document": kv_bytes,
                        "unchanged_k_bytes_per_document": k_bytes,
                        "packed_kv_bytes_per_document": compressed_bytes,
                        "encoder_decoder_bytes": weight_bytes,
                        "reference_documents": n,
                        "bank_total_bytes": n * compressed_bytes + weight_bytes,
                        "total_kv_reduction": n * kv_bytes / (n * compressed_bytes + weight_bytes),
                        "v_reduction": n
                        * (kv_bytes - k_bytes)
                        / (n * (compressed_bytes - k_bytes) + weight_bytes),
                    }
                )
            del records
            before = [t.clone() for t in tensors(packed)]
            for q in doc["questions"]:
                generated = answer(
                    native,
                    q["suffix"],
                    cache=unpack(packed, method, codecs, native.config.text_config),
                )
                text = tok.decode(generated, skip_special_tokens=True).strip()
                em, f1 = score(text, q["answers"])
                method_rows.append(
                    {
                        "title": doc["title"],
                        "question_id": q["id"],
                        "method": method,
                        "prediction": text,
                        "tokens": generated,
                        "references": q["answers"],
                        "em": em,
                        "f1": f1,
                    }
                )
            assert all(torch.equal(a, b) for a, b in zip(before, tensors(packed))), (
                "Generation mutated the stored prefix"
            )
            del packed, before
            print("QA", method, i + 1, len(documents), flush=True)
        save_json(path, method_rows)
        rows.extend(method_rows)
        del codecs
        clean()
    save_json(RUN / "storage.json", storage)
    return rows, storage


@torch.inference_mode()
def benchmark(native, doc):
    cache = native(input_ids=ids(doc["prefix"]), use_cache=True, logits_to_keep=1).past_key_values
    reference = snapshot(cache)
    del cache
    clean()
    records = []
    for repetition in range(-1, SPEC["latency_repetitions"]):
        methods = SPEC["methods"]
        offset = max(repetition, 0) % len(methods)
        for method in methods[offset:] + methods[:offset]:
            codecs = (
                [] if method in ["original", "uint4"] else load_codecs(RUN / "codecs.pt", method)
            )
            source = [{**r, "k": r["k"].cuda(), "v": r["v"].cuda()} for r in reference]
            clean()
            torch.cuda.reset_peak_memory_stats()
            start = clock()
            packed = pack(source, method, codecs)
            compression_ms = (clock() - start) * 1000
            compression_peak = torch.cuda.max_memory_allocated()
            del source
            clean()
            resident = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = clock()
            restored = unpack(packed, method, codecs, native.config.text_config)
            restored_at = clock()
            generated = answer(
                native,
                doc["questions"][0]["suffix"],
                cache=restored,
                fixed_tokens=SPEC["latency_decode_tokens"],
            )
            end = clock()
            if repetition >= 0:
                records.append(
                    {
                        "method": method,
                        "repetition": repetition,
                        "compression_ms": compression_ms,
                        "restoration_ms": (restored_at - start) * 1000,
                        "cache_hit_request_ms": (end - start) * 1000,
                        "request_plus_compression_ms": (end - start) * 1000 + compression_ms,
                        "resident_allocated_bytes": resident,
                        "compression_peak_allocated_bytes": compression_peak,
                        "request_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "request_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "generated_tokens": generated,
                    }
                )
            del packed, restored, codecs
            clean()
        print("TIMING", repetition, flush=True)
    # A full-prefill reference measures avoided work, separately from codec overhead.
    misses = []
    for _ in range(3):
        start = clock()
        answer(
            native,
            doc["questions"][0]["suffix"],
            prefix=doc["prefix"],
            fixed_tokens=SPEC["latency_decode_tokens"],
        )
        misses.append((clock() - start) * 1000)
    save_json(
        RUN / "latency.json",
        {
            "gpu": torch.cuda.get_device_name(),
            "shared_gpu": True,
            "fixed_decode_tokens": SPEC["latency_decode_tokens"],
            "rows": records,
            "uncached_request_ms": misses,
        },
    )
    return records


def summarize(rows, storage, timing):
    rng = np.random.default_rng(SPEC["seed"])
    methods = SPEC["methods"]
    data = {m: [r for r in rows if r["method"] == m] for m in methods}
    for m in methods:
        assert [r["question_id"] for r in data[m]] == [r["question_id"] for r in data["original"]]
    n = SPEC["test_documents"]
    resamples = rng.integers(n, size=(SPEC["bootstrap_draws"], n))
    results = []
    for m in methods:
        row = next(dict(s) for s in storage if s["method"] == m)
        for metric in ["em", "f1"]:
            measured = np.array([r[metric] for r in data[m]])
            original = np.array([r[metric] for r in data["original"]])
            losses = (original - measured).reshape(n, 2).mean(1)
            row[metric] = float(measured.mean())
            row[metric + "_loss_pp"] = float(losses.mean() * 100)
            row[metric + "_loss_95ci_pp"] = (
                np.quantile(losses[resamples].mean(1), [0.025, 0.975]) * 100
            ).tolist()
        current = [r for r in timing if r["method"] == m]
        for key in [
            "compression_ms",
            "restoration_ms",
            "cache_hit_request_ms",
            "request_plus_compression_ms",
            "resident_allocated_bytes",
            "compression_peak_allocated_bytes",
            "request_peak_allocated_bytes",
            "request_peak_reserved_bytes",
        ]:
            row[key + "_median"] = float(np.median([r[key] for r in current]))
        baseline = [r["cache_hit_request_ms"] for r in timing if r["method"] == "original"]
        ratios = np.array([r["cache_hit_request_ms"] for r in current]) / baseline
        samples = rng.integers(len(ratios), size=(SPEC["bootstrap_draws"], len(ratios)))
        row["paired_request_latency_ratio"] = float(ratios.mean())
        row["paired_request_latency_ratio_95ci"] = np.quantile(
            ratios[samples].mean(1), [0.025, 0.975]
        ).tolist()
        results.append(row)
    save_json(
        RUN / "results.json",
        {
            "scope": SPEC["scope"],
            "test_documents": n,
            "test_questions": 2 * n,
            "results": results,
            "freeze_sha256": sha(RUN / "freeze.json"),
        },
    )


def main():
    verify_freeze()
    assert not (RUN / "results.json").exists(), "Refusing to overwrite completed pilot"
    documents = json.loads((RUN / "documents.json").read_text())
    native, tok = model(), tokenizer()
    rows, storage = quality(native, tok, documents["test"])
    timing = benchmark(native, documents["validation"][0])
    summarize(rows, storage, timing)


if __name__ == "__main__":
    main()
