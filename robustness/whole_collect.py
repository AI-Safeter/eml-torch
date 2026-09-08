"""Collect uniformly sampled real-token MLP activations from both training domains."""

import json
import time

import torch
from runtime import HERE, RUNS, configure


def main():
    spec = configure("qwen17b")
    from data import STYLES, save
    from model_io import expand, load, setup, text

    setup(20260924)
    out = RUNS / "whole-block"
    component = torch.load(RUNS / "qwen17b/add/component.pt", weights_only=True)
    layer = component["layer"]
    started = time.time()
    model, tok = load()
    statistics = {}
    with torch.inference_mode():
        for split, count, maximum in [("train", 512, 131072), ("validation", 128, 16384)]:
            for domain in ["arithmetic", "language"]:
                destination = out / f"activations-{domain}-{split}.pt"
                if destination.exists():
                    print("PRESERVE", destination.name, flush=True)
                    continue
                if domain == "language":
                    sequences = torch.load(out / f"language-{split}.pt", weights_only=True).tolist()
                else:
                    rows = []
                    for op in ["add", "multiply", "divide"]:
                        problems = json.loads((HERE / "data" / op / "problems.json").read_text())
                        rows.extend(expand(problems[split][:count], STYLES))
                    sequences = [
                        tok.encode(text(tok, r) + str(r["answer"]), add_special_tokens=False)
                        for r in rows
                    ]
                total = sum(map(len, sequences))
                generator = torch.Generator().manual_seed(20260924)
                indices = torch.randperm(total, generator=generator)[:maximum].sort().values
                xx, yy = [], []
                offset = 0
                selected_count = 0
                for start in range(0, len(sequences), 8):
                    batch = sequences[start : start + 8]
                    inp = tok.pad({"input_ids": batch}, padding=True, return_tensors="pt").to(
                        "cuda"
                    )
                    mask = inp.attention_mask.bool()
                    real_count = sum(map(len, batch))
                    selected = (
                        indices[(indices >= offset) & (indices < offset + real_count)] - offset
                    )
                    captured = {}

                    def hook(module, args, output):
                        captured["x"] = args[0][mask][selected.cuda()].cpu()
                        captured["y"] = output[mask][selected.cuda()].cpu()

                    handle = model.model.layers[layer].mlp.register_forward_hook(hook)
                    try:
                        model.model(**inp, use_cache=False)
                    finally:
                        handle.remove()
                    xx.append(captured["x"])
                    yy.append(captured["y"])
                    offset += real_count
                    selected_count += len(selected)
                    if start % 256 == 0:
                        print("WHOLE COLLECT", domain, split, start, len(sequences), flush=True)
                assert offset == total and selected_count == len(indices)
                record = {
                    "x": torch.cat(xx),
                    "y": torch.cat(yy),
                    "token_indices": indices,
                    "eligible_tokens": total,
                    "layer": layer,
                }
                assert torch.isfinite(record["x"]).all() and torch.isfinite(record["y"]).all()
                torch.save(record, destination)
                statistics[f"{domain}/{split}"] = {
                    "eligible_tokens": total,
                    "sampled_tokens": len(indices),
                }
                del xx, yy, record
    save(
        out / "collection.json",
        {
            "model": spec,
            "layer": layer,
            "seconds": time.time() - started,
            "statistics": statistics,
            "gpu": torch.cuda.get_device_name(),
            "test_activations_collected": False,
        },
    )
    print("WHOLE COLLECTION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
