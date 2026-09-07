"""Identify a causal MLP output direction using train/validation pairs only."""

import json

import torch
from common import OUT, dump, first_targets, inputs, load, logits, margin, setup


def collect(model, tok, rows):
    cache = {"input": [], "output": [], "margin": [], "top": []}
    all_in = []
    all_out = []
    for is_corrupt in [False, True]:
        hi = []
        ho = []
        ms = []
        tops = []
        for start in range(0, len(rows), 16):
            batch = rows[start : start + 16]
            ins = {}
            outs = {}
            handles = []
            for layer_index, layer in enumerate(model.model.layers):

                def pre(mod, args, layer_index=layer_index):
                    ins[layer_index] = args[0][:, -1, :].detach()

                def post(mod, args, out, layer_index=layer_index):
                    outs[layer_index] = out[:, -1, :].detach()

                handles += [
                    layer.mlp.register_forward_pre_hook(pre),
                    layer.mlp.register_forward_hook(post),
                ]
            z = logits(model, inputs(tok, batch, is_corrupt))
            for h in handles:
                h.remove()
            hi.append(torch.stack([ins[layer_index] for layer_index in range(len(ins))], 1))
            ho.append(torch.stack([outs[layer_index] for layer_index in range(len(outs))], 1))
            ms.append(margin(z, first_targets(tok, batch), first_targets(tok, batch, True)))
            tops.append(z.argmax(1))
        all_in.append(torch.cat(hi))
        all_out.append(torch.cat(ho))
        cache["margin"].append(torch.cat(ms))
        cache["top"].append(torch.cat(tops))
    cache["input"] = all_in
    cache["output"] = all_out
    return cache


def patched(model, tok, rows, clean_outputs, layer, direction=None):
    values = []
    for start in range(0, len(rows), 16):
        batch = rows[start : start + 16]
        target = clean_outputs[start : start + 16]

        def hook(mod, args, out):
            result = out.clone()
            if direction is None:
                result[:, -1, :] = target
            else:
                delta = (
                    (target - out[:, -1, :]) @ direction
                ) / direction.double().square().sum().float()
                result[:, -1, :] += delta[:, None] * direction
            return result

        h = model.model.layers[layer].mlp.register_forward_hook(hook)
        z = logits(model, inputs(tok, batch, True))
        h.remove()
        values.append(margin(z, first_targets(tok, batch), first_targets(tok, batch, True)))
    return torch.cat(values)


def main():
    setup()
    model, tok = load()
    problems = json.loads((OUT / "problems.json").read_text())
    with torch.inference_mode():
        train = collect(model, tok, problems["train"])
        val = collect(model, tok, problems["validation"])
        print("COLLECTED", flush=True)
        rows = problems["train"][:96]
        floor = train["margin"][1][: len(rows)]
        ceiling = train["margin"][0][: len(rows)]
        full = []
        for layer_index in range(len(model.model.layers)):
            patched_margin = patched(
                model, tok, rows, train["output"][0][: len(rows), layer_index], layer_index
            )
            effect = float((patched_margin - floor).mean())
            item = {
                "layer": layer_index,
                "mean_margin_effect": effect,
                "fraction_full_clean_corrupt_gap": effect / float((ceiling - floor).mean()),
            }
            full.append(item)
            print("LAYER", json.dumps(item), flush=True)
        top = sorted(full, key=lambda r: r["mean_margin_effect"], reverse=True)[:3]
        directions = []
        candidates = []
        floor = val["margin"][1]
        gap = float((val["margin"][0] - floor).mean())
        for entry in top:
            layer_index = entry["layer"]
            y = torch.cat([train["output"][0][:, layer_index], train["output"][1][:, layer_index]])
            u, s, v = torch.linalg.svd(y - y.mean(0), full_matrices=False)
            for pc in range(4):
                d = v[pc]
                response = patched(
                    model,
                    tok,
                    problems["validation"],
                    val["output"][0][:, layer_index],
                    layer_index,
                    d,
                )
                effect = float((response - floor).mean())
                item = {
                    "layer": layer_index,
                    "pc": pc,
                    "mean_margin_effect": effect,
                    "fraction_full_clean_corrupt_gap": effect / gap,
                }
                candidates.append(item)
                directions.append(d)
                print("DIRECTION", json.dumps(item), flush=True)
        best = max(range(len(candidates)), key=lambda i: candidates[i]["mean_margin_effect"])
        selected = candidates[best]
        layer_index = selected["layer"]
        d = directions[best]
        # An equal-norm orthogonal random direction controls for arbitrary intervention.
        random_d = torch.randn_like(d)
        random_d -= d * (d @ random_d) / d.double().square().sum().float()
        random_d /= random_d.norm()
        xm = torch.cat([train["input"][0][:, layer_index], train["input"][1][:, layer_index]])
        xmean = xm.mean(0)
        u, s, v = torch.linalg.svd(xm - xmean, full_matrices=False)
        encoder = v[:3].T.contiguous()
        stats = {
            "layer": layer_index,
            "direction": d.cpu(),
            "random_direction": random_d.cpu(),
            "input_mean": xmean.cpu(),
            "encoder": encoder.cpu(),
            "input_variance_explained": float(s[:3].square().sum() / s.square().sum()),
            "component_mean": float(
                (
                    torch.cat(
                        [train["output"][0][:, layer_index], train["output"][1][:, layer_index]]
                    )
                    @ d
                ).mean()
            ),
        }
        torch.save(stats, OUT / "component.pt")
        data = {}
        for split, cache in [("train", train), ("validation", val)]:
            data[split] = {
                name: torch.stack([cache[name][role][:, layer_index] for role in [0, 1]], 1).cpu()
                for name in ["input", "output"]
            }
        torch.save(data, OUT / "component-data.pt")
        dump(
            "localization.json",
            {
                "full_layers": full,
                "directions": candidates,
                "selected": selected,
                "feature_variance_explained": stats["input_variance_explained"],
                "train_first_digit_accuracy": float(
                    (train["top"][0] == first_targets(tok, problems["train"])).float().mean()
                ),
                "validation_first_digit_accuracy": float(
                    (val["top"][0] == first_targets(tok, problems["validation"])).float().mean()
                ),
            },
        )
        print(
            "SELECTED",
            json.dumps(selected),
            "INPUT_VARIANCE",
            stats["input_variance_explained"],
            flush=True,
        )


if __name__ == "__main__":
    main()
