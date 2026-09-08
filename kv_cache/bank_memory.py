"""Measure 128 distinct GPU allocations; this is not a cache-hit-rate simulation."""

import json

import torch
from common import RUN, SPEC, answer, ids, model, save_json, sha, snapshot, storage_bytes
from evaluate import clean, pack, tensors, unpack, verify_freeze
from v_codecs import load_codecs


@torch.inference_mode()
def main():
    verify_freeze()
    native = model()
    doc = json.loads((RUN / "documents.json").read_text())["validation"][0]
    cache = native(input_ids=ids(doc["prefix"]), use_cache=True, logits_to_keep=1).past_key_values
    reference = snapshot(cache)
    del cache
    clean()
    model_bytes = torch.cuda.memory_allocated()
    rows = []
    for method in SPEC["methods"]:
        codecs = [] if method in ["original", "uint4"] else load_codecs(RUN / "codecs.pt", method)
        source = [{**r, "k": r["k"].cuda(), "v": r["v"].cuda()} for r in reference]
        template = pack(source, method, codecs)
        # Preserve the payload on CPU; only the measured bank lives on the GPU.
        cpu = [
            {
                "k": r["k"].cpu(),
                "length": r["length"],
                "code": tuple(x.cpu() for x in r["code"])
                if isinstance(r["code"], tuple)
                else r["code"].cpu(),
            }
            for r in template
        ]
        per_document = storage_bytes(tensors(template))
        del source, template
        for count in [1, 16, SPEC["storage_reference_documents"]]:
            bank = [
                [
                    {
                        "k": r["k"].cuda(),
                        "length": r["length"],
                        "code": tuple(x.cuda() for x in r["code"])
                        if isinstance(r["code"], tuple)
                        else r["code"].cuda(),
                    }
                    for r in cpu
                ]
                for _ in range(count)
            ]
            distinct_bytes = storage_bytes([x for prefix in bank for x in tensors(prefix)])
            assert distinct_bytes == count * per_document
            clean()
            resident = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            answer(
                native,
                doc["questions"][0]["suffix"],
                cache=unpack(bank[0], method, codecs, native.config.text_config),
                fixed_tokens=16,
            )
            torch.cuda.synchronize()
            rows.append(
                {
                    "method": method,
                    "documents": count,
                    "distinct_bank_tensor_bytes": distinct_bytes,
                    "encoder_decoder_bytes": sum(c.bytes() for c in codecs),
                    "model_baseline_allocated_bytes": model_bytes,
                    "resident_allocated_bytes": resident,
                    "request_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "request_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )
            del bank
        del codecs
        clean()
        print("BANK", method, flush=True)
    save_json(
        RUN / "bank-memory.json",
        {
            "scope": "Distinct allocations of a repeated validation prefix payload; not distinct-document hit-rate or throughput evidence",
            "script_sha256": sha(__file__),
            "freeze_sha256": sha(RUN / "freeze.json"),
            "gpu": torch.cuda.get_device_name(),
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
