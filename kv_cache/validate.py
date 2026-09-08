"""H100 checks for packed bytes, deployed fits, budgets, and split integrity."""

import json

import torch
from common import HERE, RUN, SPEC, save_json, setup, sha, storage_bytes
from v_codecs import load_codecs, pack4, unpack4


@torch.inference_mode()
def main():
    setup()
    for d in [256, 512]:
        v = torch.cat(
            [torch.randn(21, d, device="cuda"), torch.ones(3, d, device="cuda")]
        ).bfloat16()
        code = pack4(v)
        restored = unpack4(code)
        assert code[0].dtype == torch.uint8 and code[0].shape[-1] == d // 2
        assert storage_bytes(code) == len(v) * (d // 2 + 4)
        assert torch.equal(restored[-3:], v[-3:])
        tolerance = code[2].float() / 2 + 0.02
        assert ((v.float() - restored.float()).abs() <= tolerance).all()
        assert torch.isfinite(restored).all()
    values = torch.load(RUN / "values.pt", weights_only=True)["validation"]
    fitted = json.loads((RUN / "fit.json").read_text())["rows"]
    checks = []
    for method in ["lowrank", "silu", "eml"]:
        codecs = load_codecs(RUN / "codecs.pt", method)
        for layer, (codec, value) in enumerate(zip(codecs, values)):
            value = value.cuda()
            code = codec.encode(value)
            mse = (codec.decode(code).float() - value.float()).square().mean().item()
            expected = next(r for r in fitted if r["method"] == method and r["layer"] == layer)
            assert abs(mse - expected["validation_mse"]) < 1e-7
            assert codec.bytes() == expected["parameter_bytes"]
            tokens = SPEC["prefix_tokens"] if layer % 5 == 4 else 511
            n = SPEC["storage_reference_documents"]
            used = n * tokens * code.shape[-1] * 2 + codec.bytes()
            budget = n * tokens * (value.shape[-1] // 2 + 4)
            assert used <= budget
            checks.append(
                {
                    "method": method,
                    "layer": layer,
                    "validation_mse": mse,
                    "bank_v_bytes": used,
                    "bank_v_budget_bytes": budget,
                }
            )
    documents = json.loads((RUN / "documents.json").read_text())
    titles = [d["title"] for split in documents.values() for d in split]
    assert len(titles) == len(set(titles)) == 56
    assert all(
        len(d["prefix"]) == 1024 and len(d["questions"]) == 2
        for ds in documents.values()
        for d in ds
    )
    save_json(
        RUN / "codec-validation.json",
        {
            "gpu": torch.cuda.get_device_name(),
            "checks": checks,
            "packed_quantization": "passed",
            "document_disjointness": "passed",
        },
    )
    sources = sorted(HERE.glob("*.py")) + [
        HERE / "protocol.json",
        HERE.parent / "emltorch/head.py",
        HERE.parent / "emltorch/operator.py",
    ]
    freeze = {
        "sources": {str(p.relative_to(HERE.parent)): sha(p) for p in sources},
        "documents_sha256": sha(RUN / "documents.json"),
        "codecs_sha256": sha(RUN / "codecs.pt"),
        "codec_validation_sha256": sha(RUN / "codec-validation.json"),
    }
    assert not (RUN / "freeze.json").exists(), "Refusing to replace a frozen experiment"
    save_json(RUN / "freeze.json", freeze)
    print(
        "GPU VALIDATION PASSED: packed storage, 45 deployed fits, byte budgets, article-disjoint splits",
        flush=True,
    )


if __name__ == "__main__":
    main()
