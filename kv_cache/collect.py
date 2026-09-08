"""GPU cache identity checks, then question-independent train/validation V sampling."""

import json

import torch
from common import (
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


@torch.inference_mode()
def main():
    assert not (RUN / "values.pt").exists()
    documents = json.loads((RUN / "documents.json").read_text())
    native, tok = model(), tokenizer()
    doc = documents["validation"][0]
    cache = native(input_ids=ids(doc["prefix"]), use_cache=True, logits_to_keep=1).past_key_values
    raw_bytes = storage_bytes([x for layer in cache.layers for x in (layer.keys, layer.values)])
    records = snapshot(cache, "cuda")
    copied = restore(records, native.config.text_config)
    for a, b in zip(cache.layers, copied.layers):
        assert torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
        assert a.get_seq_length() == b.get_seq_length()
    suffix = doc["questions"][0]["suffix"]
    direct = native(
        input_ids=ids(suffix), past_key_values=cache, use_cache=True, logits_to_keep=1
    ).logits
    recovered = native(
        input_ids=ids(suffix), past_key_values=copied, use_cache=True, logits_to_keep=1
    ).logits
    assert torch.equal(direct, recovered), "Identity restoration changed native cached logits"
    full = answer(native, suffix, prefix=doc["prefix"])
    reused = answer(native, suffix, cache=restore(records, native.config.text_config))
    assert full == reused, "Uncompressed prefix reuse changed validation answer"
    stats = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "identity_logits_bitwise_equal": True,
        "native_full_and_cached_answer_equal": True,
        "answer": tok.decode(full, skip_special_tokens=True),
        "raw_cache_backing_bytes": raw_bytes,
        "compacted_kv_bytes": storage_bytes([x for r in records for x in (r["k"], r["v"])]),
        "layers": [
            {"index": i, "shape": list(r["v"].shape), "length": r["length"]}
            for i, r in enumerate(records)
        ],
    }
    save_json(RUN / "identity-validation.json", stats)
    del cache, records, copied, direct, recovered
    collected = {}
    for split in ["train", "validation"]:
        values = [[] for _ in range(15)]
        for i, doc in enumerate(documents[split]):
            cache = native(
                input_ids=ids(doc["prefix"]), use_cache=True, logits_to_keep=1
            ).past_key_values
            for j, layer in enumerate(cache.layers):
                v = layer.values.reshape(-1, layer.values.shape[-1])
                ix = torch.randperm(len(v), device="cuda")[
                    : SPEC["sample_tokens_per_layer_per_train_document"]
                ]
                values[j].append(v[ix].cpu())
            del cache
            print("COLLECT", split, i + 1, len(documents[split]), flush=True)
        collected[split] = [torch.cat(v) for v in values]
    torch.save(collected, RUN / "values.pt")
    save_json(
        RUN / "collection.json",
        {
            "values_sha256": sha(RUN / "values.pt"),
            "documents_sha256": sha(RUN / "documents.json"),
            "shapes": {s: [list(x.shape) for x in xs] for s, xs in collected.items()},
        },
    )


if __name__ == "__main__":
    main()
