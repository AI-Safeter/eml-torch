"""Capture full MLP targets and candidate arithmetic intermediates on CUDA."""

import argparse
import json
import time

import torch

from .prepare import content
from .runtime import (
    HERE,
    SPEC,
    digest,
    load,
    prompt,
    root,
    save,
    setup,
    sources,
    text_layers,
    tokenizer,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    setup(SPEC["data_seed"])
    out = root(args.output)
    data = out / "data"
    frozen = json.loads((data / "freeze.json").read_text())
    assert frozen["protocol_sha256"] == digest(HERE / "protocol.json")
    assert all(digest(data / p) == h for p, h in frozen["files"].items())
    collection = out / "collection"
    collection.mkdir(exist_ok=True)
    source_freeze = {
        name: digest(HERE / name)
        for name in ["collect.py", "prepare.py", "runtime.py", "protocol.json", "PROTOCOL.md"]
    }
    collection_freeze = collection / "freeze.json"
    if collection_freeze.exists():
        assert json.loads(collection_freeze.read_text())["sources"] == source_freeze
    else:
        save(
            collection_freeze,
            {"sources": source_freeze, "data_sha256": digest(data / "freeze.json")},
        )
    model, tok = load(), tokenizer()
    layer = SPEC["replacement"]["layer"]
    layers = text_layers(model)
    norm = layers[layer].post_feedforward_layernorm
    torch.save(
        {"weight": norm.weight.detach().float().cpu(), "eps": norm.eps},
        collection / "native-postnorm.pt",
    )
    started = time.perf_counter()
    summaries = {}
    with torch.inference_mode():
        for split in ["train", "selection"]:
            for domain in ["arithmetic", "language"]:
                path = collection / f"{domain}-{split}.pt"
                if path.exists():
                    print("PRESERVE", path.name, flush=True)
                    continue
                metadata = []
                if domain == "language":
                    sequences = torch.load(
                        data / f"language-{split}.pt", weights_only=True
                    ).tolist()
                else:
                    rows = json.loads((data / f"arithmetic-{split}.json").read_text())
                    sequences = []
                    for row in rows:
                        for style in SPEC["arithmetic"]["formats"]:
                            p = tok.encode(
                                prompt(tok, content(row, style)), add_special_tokens=False
                            )
                            full = tok.encode(
                                prompt(tok, content(row, style)) + str(row["answer"]),
                                add_special_tokens=False,
                            )
                            assert full[: len(p)] == p
                            sequences.append(full)
                            metadata.append(
                                {
                                    **row,
                                    "style": style,
                                    "prompt_tokens": len(p),
                                    "answer_token_ids": full[len(p) :],
                                }
                            )
                total = sum(map(len, sequences))
                generator = torch.Generator().manual_seed(SPEC["data_seed"])
                maximum = SPEC["replacement"]["tokens_per_domain"][split]
                indices = torch.randperm(total, generator=generator)[:maximum].sort().values
                captured = {k: [] for k in ["x", "y", "contribution", "sequence", "position"]}
                features = {
                    candidate_layer: {k: [] for k in ["input", "output"]}
                    for candidate_layer in SPEC["mechanism"]["candidate_layers"]
                }
                feature_metadata = []
                offset = 0
                for start in range(0, len(sequences), args.batch_size):
                    batch = sequences[start : start + args.batch_size]
                    inp = tok.pad({"input_ids": batch}, padding=True, return_tensors="pt").to(
                        "cuda"
                    )
                    mask = inp.attention_mask.bool()
                    real_count = sum(map(len, batch))
                    selected_cpu = (
                        indices[(indices >= offset) & (indices < offset + real_count)] - offset
                    )
                    selected = selected_cpu.cuda()
                    selected_coordinates = mask.nonzero()[selected]
                    captured["sequence"].append((selected_coordinates[:, 0] + start).cpu())
                    positions = inp.attention_mask.cumsum(-1) - 1
                    captured["position"].append(positions[mask][selected].cpu())
                    hooks = []

                    def mlp_hook(module, inputs, output):
                        captured["x"].append(inputs[0][mask][selected].cpu())
                        captured["y"].append(output[mask][selected].cpu())

                    def norm_hook(module, inputs, output):
                        captured["contribution"].append(output[mask][selected].cpu())

                    hooks.append(layers[layer].mlp.register_forward_hook(mlp_hook))
                    hooks.append(norm.register_forward_hook(norm_hook))
                    batch_ids, prefix_positions = [], []
                    if domain == "arithmetic":
                        for i, row in enumerate(metadata[start : start + len(batch)]):
                            if row["op"] != "add":
                                continue
                            assert len(row["answer_token_ids"]) == len(str(row["answer"]))
                            padding = inp.input_ids.shape[1] - len(batch[i])
                            for k in range(len(row["answer_token_ids"])):
                                batch_ids.append(i)
                                prefix_positions.append(padding + row["prompt_tokens"] + k - 1)
                                feature_metadata.append({**row, "answer_prefix_tokens": k})
                    if batch_ids:
                        bi = torch.tensor(batch_ids, device="cuda")
                        pi = torch.tensor(prefix_positions, device="cuda")
                        for candidate_layer in features:

                            def feature_hook(module, inputs, output, candidate_layer=candidate_layer):
                                features[candidate_layer]["input"].append(inputs[0][bi, pi].cpu())
                                features[candidate_layer]["output"].append(output[bi, pi].cpu())

                            hooks.append(layers[candidate_layer].mlp.register_forward_hook(feature_hook))
                    try:
                        model.model.language_model(**inp, use_cache=False)
                    finally:
                        for hook in hooks:
                            hook.remove()
                    offset += real_count
                    if start % 256 == 0:
                        print("COLLECT", split, domain, start, len(sequences), flush=True)
                assert offset == total
                record = {k: torch.cat(v) for k, v in captured.items()}
                assert len(record["x"]) == len(indices)
                assert all(torch.isfinite(record[k]).all() for k in ["x", "y", "contribution"])
                record.update(token_indices=indices, eligible_tokens=total, layer=layer)
                tmp = path.with_suffix(".tmp")
                torch.save(record, tmp)
                tmp.replace(path)
                if feature_metadata:
                    feature_record = {
                        "states": {
                            candidate_layer: {k: torch.cat(v) for k, v in tensors.items()}
                            for candidate_layer, tensors in features.items()
                        },
                        "rows": feature_metadata,
                    }
                    feature_path = collection / f"mechanism-{split}.pt"
                    torch.save(feature_record, feature_path.with_suffix(".tmp"))
                    feature_path.with_suffix(".tmp").replace(feature_path)
                summaries[f"{domain}-{split}"] = {
                    "sampled_tokens": len(indices),
                    "eligible_tokens": total,
                    "feature_observations": len(feature_metadata),
                    "file_sha256": digest(path),
                }
                save(
                    collection / "progress.json",
                    {"summaries": summaries, "elapsed_seconds": time.perf_counter() - started},
                )
                print("COLLECTION FILE COMPLETE", path.name, flush=True)
    save(
        collection / "complete.json",
        {
            "summaries": summaries,
            "sources": sources(),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "test_activations_collected": False,
        },
    )
    print("COLLECTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
