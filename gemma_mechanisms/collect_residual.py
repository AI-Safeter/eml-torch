"""Localize arithmetic in the residual stream after the MLP-input patch failed."""

import argparse
import json
import time

import torch

from .prepare import content
from .runtime import HERE, SPEC, digest, load, prompt, root, save, setup, text_layers, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    setup(SPEC["data_seed"])
    base = root(args.output)
    out = base / "residual"
    out.mkdir(exist_ok=True)
    if not (out / "data").exists():
        (out / "data").symlink_to(base / "data", target_is_directory=True)
    directory = out / "collection"
    directory.mkdir(exist_ok=True)
    freeze = {
        "site": "residual",
        "sources": {
            p: digest(HERE / p)
            for p in ["collect_residual.py", "runtime.py", "prepare.py", "protocol.json"]
        },
        "data_freeze_sha256": digest(base / "data/freeze.json"),
        "trigger": "MLP-input carry swaps and full-input donor controls did not mediate tens-digit responses on selection data.",
        "confirmatory_data_used": False,
    }
    frozen_path = directory / "site.json"
    if frozen_path.exists():
        assert json.loads(frozen_path.read_text()) == freeze
    else:
        save(frozen_path, freeze)
    model, tok = load(), tokenizer()
    layers = text_layers(model)
    started = time.perf_counter()
    with torch.inference_mode():
        for split in ["train", "selection"]:
            path = directory / f"mechanism-{split}.pt"
            if path.exists():
                continue
            rows = json.loads((base / "data" / f"arithmetic-{split}.json").read_text())
            sequences, metadata = [], []
            for row in rows:
                if row["op"] != "add":
                    continue
                for style in SPEC["arithmetic"]["formats"]:
                    text = prompt(tok, content(row, style))
                    prefix_ids = tok.encode(text, add_special_tokens=False)
                    full = tok.encode(text + str(row["answer"]), add_special_tokens=False)
                    assert full[: len(prefix_ids)] == prefix_ids
                    sequences.append(full)
                    metadata.append(
                        {
                            **row,
                            "style": style,
                            "prompt_tokens": len(prefix_ids),
                            "answer_token_ids": full[len(prefix_ids) :],
                        }
                    )
            features = {
                layer: {"input": [], "output": []}
                for layer in SPEC["mechanism"]["candidate_layers"]
            }
            records = []
            for start in range(0, len(sequences), 8):
                batch = sequences[start : start + 8]
                inp = tok.pad({"input_ids": batch}, padding=True, return_tensors="pt").to("cuda")
                bi, pi = [], []
                for i, row in enumerate(metadata[start : start + len(batch)]):
                    assert len(row["answer_token_ids"]) == 4
                    padding = inp.input_ids.shape[1] - len(batch[i])
                    for k in range(4):
                        bi.append(i)
                        pi.append(padding + row["prompt_tokens"] + k - 1)
                        records.append({**row, "answer_prefix_tokens": k})
                bi, pi = torch.tensor(bi, device="cuda"), torch.tensor(pi, device="cuda")
                hooks = []
                for layer in features:

                    def capture(module, inputs, output, layer=layer):
                        features[layer]["input"].append(inputs[0][bi, pi].cpu())
                        features[layer]["output"].append(output[bi, pi].cpu())

                    hooks.append(layers[layer].register_forward_hook(capture))
                try:
                    model.model.language_model(**inp, use_cache=False)
                finally:
                    for hook in hooks:
                        hook.remove()
                if start % 512 == 0:
                    print("RESIDUAL COLLECT", split, start, len(sequences), flush=True)
            states = {
                layer: {k: torch.cat(v) for k, v in tensors.items()}
                for layer, tensors in features.items()
            }
            assert all(
                torch.isfinite(t).all() for tensors in states.values() for t in tensors.values()
            )
            torch.save({"states": states, "rows": records}, path.with_suffix(".tmp"))
            path.with_suffix(".tmp").replace(path)
    save(
        directory / "complete.json",
        {
            "seconds": time.perf_counter() - started,
            "site": "residual",
            "freeze_sha256": digest(frozen_path),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "confirmatory_data_used": False,
        },
    )
    print("RESIDUAL COLLECTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
