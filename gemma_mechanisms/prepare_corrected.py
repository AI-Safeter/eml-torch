"""Apply the preregistered BOS correction into a fresh, provenance-bound root."""

import argparse
import json
import shutil

import torch

from .runtime import HERE, digest, root, save, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, out = root(args.source), root(args.output)
    assert source != out and not (out / "data").exists()
    old = json.loads((source / "data/freeze.json").read_text())
    assert all(digest(source / "data" / p) == h for p, h in old["files"].items())
    assert not (source / "evaluation/gate/original.json").exists()
    data = out / "data"
    data.mkdir()
    bos = tokenizer().bos_token_id
    for name in old["files"]:
        path = source / "data" / name
        if name.startswith("language-") and name.endswith(".pt"):
            text = torch.load(path, weights_only=True)
            assert text.shape[1] == 256
            corrected = torch.cat([torch.full((len(text), 1), bos), text], dim=1)
            assert torch.equal(corrected[:, 1:], text)
            torch.save(corrected, data / name)
        else:
            shutil.copy2(path, data / name)
    amendment = {
        "original_data_freeze_sha256": digest(source / "data/freeze.json"),
        "source_sha256": digest(HERE / "prepare_corrected.py"),
        "amendment_sha256": digest(HERE / "BOS_AMENDMENT.md"),
        "bos_token_id": bos,
        "text_tokens_per_document": 256,
        "input_tokens_per_document": 257,
        "gate_and_final_model_outputs_unopened": True,
    }
    save(data / "bos-amendment.json", amendment)
    corrected_freeze = dict(old)
    corrected_freeze["bos_amendment"] = amendment
    corrected_freeze["files"] = {p.name: digest(p) for p in sorted(data.iterdir())}
    save(data / "freeze.json", corrected_freeze)
    collection = out / "collection"
    collection.mkdir()
    reused = {}
    for domain in ["arithmetic", "mechanism"]:
        for split in ["train", "selection"]:
            name = f"{domain}-{split}.pt"
            shutil.copy2(source / "collection" / name, collection / name)
            reused[name] = digest(collection / name)
    save(collection / "arithmetic-reuse.json", {
        "files": reused,
        "original_collection_freeze_sha256": digest(source / "collection/freeze.json"),
        "reason": "Arithmetic native chat tokenization already includes BOS; no fitted weights reused.",
    })
    print("CORRECTED DATA PREPARED", out, flush=True)


if __name__ == "__main__":
    main()
