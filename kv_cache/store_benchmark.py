"""Include sliding-window compaction in the cost of storing a new prefix."""

import json

import torch
from common import RUN, SPEC, ids, model, save_json, sha, snapshot
from evaluate import clean, clock, pack, verify_freeze
from v_codecs import load_codecs


@torch.inference_mode()
def main():
    verify_freeze()
    native = model()
    doc = json.loads((RUN / "documents.json").read_text())["validation"][0]
    cache = native(input_ids=ids(doc["prefix"]), use_cache=True, logits_to_keep=1).past_key_values
    rows = []
    for repetition in range(-1, SPEC["latency_repetitions"]):
        methods = SPEC["methods"]
        offset = max(repetition, 0) % len(methods)
        for method in methods[offset:] + methods[:offset]:
            codecs = (
                [] if method in ["original", "uint4"] else load_codecs(RUN / "codecs.pt", method)
            )
            clean()
            torch.cuda.reset_peak_memory_stats()
            start = clock()
            compact = snapshot(cache, "cuda")
            packed = pack(compact, method, codecs)
            end = clock()
            if repetition >= 0:
                rows.append(
                    {
                        "method": method,
                        "repetition": repetition,
                        "compact_and_compress_ms": (end - start) * 1000,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    }
                )
            del compact, packed, codecs
    save_json(
        RUN / "store-cost.json",
        {
            "scope": "Store phase after native prefill, including compaction, compression and temporary buffers; prefill measured separately in latency.json",
            "script_sha256": sha(__file__),
            "freeze_sha256": sha(RUN / "freeze.json"),
            "rows": rows,
        },
    )
    print("STORE COST INCLUDING COMPACTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
