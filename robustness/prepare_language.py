"""Pin WikiText source and freeze token blocks without running the model."""

import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DESTINATION = HERE.parent / ".artifacts/robustness/whole-block"


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    spec = json.loads((HERE / "models.json").read_text())["qwen17b"]
    tok = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"])
    record = {
        "dataset": "Salesforce/wikitext",
        "revision": REVISION,
        "subset": "wikitext-2-raw-v1",
        "tokenizer": spec,
        "seed": 20260924,
        "source": "https://huggingface.co/datasets/Salesforce/wikitext",
        "license": "CC BY-SA 3.0 / GFDL",
        "splits": {},
    }
    for split, count in [("train", 2048), ("validation", 128), ("test", 256)]:
        source = Path(
            hf_hub_download(
                "Salesforce/wikitext",
                repo_type="dataset",
                revision=REVISION,
                filename=f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
            )
        )
        texts = pq.read_table(source)["text"].to_pylist()
        joined = "\n\n".join(t for t in texts if t.strip())
        tokens = tok.encode(joined, add_special_tokens=False)
        available = len(tokens) // 256
        indices = list(range(available))
        random.Random(20260924).shuffle(indices)
        chosen = indices[:count]
        blocks = torch.tensor([tokens[i * 256 : (i + 1) * 256] for i in chosen], dtype=torch.int64)
        path = DESTINATION / f"language-{split}.pt"
        if path.exists():
            assert torch.equal(torch.load(path, weights_only=True), blocks)
        else:
            torch.save(blocks, path)
        record["splits"][split] = {
            "available_blocks": available,
            "selected_blocks": len(blocks),
            "block_indices": chosen,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "token_array_sha256": hashlib.sha256(blocks.numpy().tobytes()).hexdigest(),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print("LANGUAGE FROZEN", split, len(blocks), flush=True)
    path = HERE / "language-freeze.json"
    if path.exists():
        assert json.loads(path.read_text()) == record
    else:
        path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
