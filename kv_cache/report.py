"""Audit raw QA identities and publish a small evidence package, without checkpoints."""

import json
import shutil

import numpy as np
from common import HERE, RUN, SPEC, save_json, sha, tokenizer
from evaluate import normalize, score, verify_freeze


def main():
    verify_freeze()
    data = json.loads((RUN / "documents.json").read_text())
    results = json.loads((RUN / "results.json").read_text())["results"]
    bank = json.loads((RUN / "bank-memory.json").read_text())
    store = json.loads((RUN / "store-cost.json").read_text())
    tok = tokenizer()
    expected = [(d["title"], q) for d in data["test"] for q in d["questions"]]
    assert len(expected) == len({q["id"] for _, q in expected}) == 48
    for result in results:
        rows = json.loads((RUN / f"qa-{result['method']}.json").read_text())
        assert len(rows) == len(expected)
        for row, (title, q) in zip(rows, expected):
            assert row["title"] == title and row["question_id"] == q["id"]
            assert row["references"] == q["answers"]
            assert tok.decode(row["tokens"], skip_special_tokens=True).strip() == row["prediction"]
            assert score(row["prediction"], q["answers"]) == (row["em"], row["f1"])
            assert row["em"] == float(
                any(normalize(row["prediction"]) == normalize(a) for a in q["answers"])
            )
        for metric in ["em", "f1"]:
            assert abs(np.mean([row[metric] for row in rows]) - result[metric]) < 1e-12
        physical = next(
            r for r in bank["rows"] if r["method"] == result["method"] and r["documents"] == 128
        )
        assert (
            physical["distinct_bank_tensor_bytes"] + physical["encoder_decoder_bytes"]
            == result["bank_total_bytes"]
        )
    quantized = next(r for r in results if r["method"] == "uint4")
    for r in results:
        if r["method"] != "original":
            assert r["bank_total_bytes"] <= quantized["bank_total_bytes"]
    evidence = HERE / "evidence"
    evidence.mkdir(exist_ok=True)
    files = [
        "data-manifest",
        "identity-validation",
        "collection",
        "fit",
        "codec-validation",
        "freeze",
        "storage",
        "latency",
        "results",
        "bank-memory",
        "store-cost",
    ] + [f"qa-{m}" for m in SPEC["methods"]]
    for name in files:
        shutil.copyfile(RUN / f"{name}.json", evidence / f"{name}.json")
    save_json(
        evidence / "audit.json",
        {
            "status": "passed",
            "qa_records": 240,
            "methods": SPEC["methods"],
            "article_disjoint": True,
            "physical_bank_matches_storage_accounting": True,
            "heldout_prediction_and_label_audit": True,
            "sha256": {
                p.name: sha(p) for p in sorted(evidence.glob("*.json")) if p.name != "audit.json"
            },
        },
    )
    lines = [
        "# Gemma E2B V-cache compression pilot",
        "",
        "EML did not meet the proposed quality target in this pilot. Packed 4-bit V quantization was the strongest observed compression baseline. The small evaluation does not establish a quality improvement or certify a 1 percentage point loss bound.",
        "",
        "Gemma 4 E2B IT, BF16, H100: 24 training articles, 8 validation articles, 24 held-out articles with 2 questions each. Every cached document prefix has 1,024 tokens. All 15 independently stored V caches are compressed; K remains unchanged.",
        "",
        "| Method | Exact match (%) | Answer F1 (%) | F1 loss (pp), 95% interval | Total KV reduction |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        ci = r["f1_loss_95ci_pp"]
        lines.append(
            f"| {r['method']} | {100 * r['em']:.2f} | {100 * r['f1']:.2f} | {r['f1_loss_pp']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {r['total_kv_reduction']:.3f}× |"
        )
    lines += [
        "",
        "Intervals resample articles, preserving the two questions per article. They are exploratory percentile bootstrap intervals, without correction across methods. In particular, an observed zero-regression bootstrap bound cannot certify the absence of rare regressions. Even zero regressions in 24 independent articles would give a one-sided 95% event-probability bound of 11.73%, far above 1%.",
        "",
        "Storage includes K, packed values or latent codes, range metadata, and shared encoder/decoder weights amortized over 128 prefixes. All compressed methods fit under the same byte ceiling; integer ranks leave a little unused capacity. SiLU and EML have identical ranks and parameter counts. Their shared codec weights occupy 2,049,408 bytes each.",
        "",
        "The measured V reduction is about 3.9×; the total KV reduction is about 1.59×. With equally sized K and V and K unchanged, total KV compression cannot exceed 2×. Achieving 4× total storage reduction requires compressing K as well.",
        "",
        "| Method | Compression (ms) | Restoration (ms) | Cache-hit request (ms) | Paired request ratio, 95% interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        ci = r["paired_request_latency_ratio_95ci"]
        lines.append(
            f"| {r['method']} | {r['compression_ms_median']:.2f} | {r['restoration_ms_median']:.2f} | {r['cache_hit_request_ms_median']:.2f} | {r['paired_request_latency_ratio']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |"
        )
    lines += [
        "",
        "Wall-clock measurements synchronize CUDA. Twelve repetitions rotate method order after a warmup; generation always runs 16 tokens for timing. Cache-hit requests include restoration, question processing and generation. The compression column above starts from an already compacted cache. The separate store-phase measurement below includes sliding-window compaction as well. Native prefill is shared by every method and measured separately in the uncached-request reference in latency.json. Measurements use a shared GPU and eager PyTorch, with allocator cleanup between cases. Every compressed method's latency interval includes a slowdown. No speedup or absence of latency regression is established.",
        "",
        "| Method | Complete store phase, median (ms) | Store peak allocation (GiB) |",
        "|---|---:|---:|",
    ]
    for method in SPEC["methods"]:
        observed = [r for r in store["rows"] if r["method"] == method]
        assert len(observed) == SPEC["latency_repetitions"]
        lines.append(
            f"| {method} | {np.median([r['compact_and_compress_ms'] for r in observed]):.2f} | {max(r['peak_allocated_bytes'] for r in observed) / 2**30:.3f} |"
        )
    lines += [
        "",
        "| Method | 128-prefix bank + codec weights (MiB) | Resident GPU allocation (GiB) | Request peak allocation (GiB) | Peak reserved (GiB) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in bank["rows"]:
        if r["documents"] == 128:
            lines.append(
                f"| {r['method']} | {(r['distinct_bank_tensor_bytes'] + r['encoder_decoder_bytes']) / 2**20:.2f} | {r['resident_allocated_bytes'] / 2**30:.3f} | {r['request_peak_allocated_bytes'] / 2**30:.3f} | {r['request_peak_reserved_bytes'] / 2**30:.3f} |"
            )
    lines += [
        "",
        "The bank measurement allocates 128 distinct copies of one validation prefix payload and checks their actual backing storage. It measures physical memory, not distinct-document hit rate, request concurrency or throughput. Resident and peak figures include the full model and codec weights; peak allocation includes restoration and generation temporaries. The auxiliary bank and store-phase scripts were added after QA evaluation solely to complete memory and compaction-cost accounting, with no refitting or choice changes.",
        "",
        "Native sliding-window tensor views retained 18 MiB after prefill. Compacting their backing storage reduced the uncompressed cache to 11.99 MiB while preserving logits bit for bit. All reported compression gains use this already compacted baseline.",
        "",
        "This is a single-seed reconstruction pilot on 1,024-token document prefixes. It does not cover long-context scaling, multiple datasets, cache eviction, production kernels, or whole-KV compression. The proposed 4× storage / ≤1 pp loss / no latency regression target is not met. A useful next experiment would test a small EML correction of quantization residuals against an equally sized SiLU correction; replacing the whole V representation with this learned low-rank decoder is not supported by these results.",
        "",
    ]
    (HERE / "RESULTS.md").write_text("\n".join(lines))
    print("QA AND MEMORY AUDIT PASSED; report and evidence written", flush=True)


if __name__ == "__main__":
    main()
