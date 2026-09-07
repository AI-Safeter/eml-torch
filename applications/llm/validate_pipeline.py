"""GPU audit of projection replacement, controls, and data partitions."""

import json

import torch
from common import OUT, dump, inputs, load, logits, setup
from replacement import Replacement


def main():
    setup()
    model, tok = load()
    rep = Replacement()
    data = json.loads((OUT / "problems.json").read_text())
    sets = {
        k: {tuple(sorted((r[a], r["b"]))) for r in rows for a in ["a", "c"]}
        for k, rows in data.items()
    }
    for a in sets:
        for b in sets:
            if a != b:
                assert not sets[a] & sets[b]
    rows = data["validation"][:16]
    inp = inputs(tok, rows)
    result = {}
    with torch.inference_mode():
        baseline = {}

        def record(module, args, out):
            baseline["out"] = out[:, -1, :].clone()

        h = model.model.layers[rep.layer].mlp.register_forward_hook(record)
        logits(model, inp)
        h.remove()
        base = baseline["out"]
        for kind in [
            "restore",
            "eml_observed",
            "eml_intervention",
            "cubic",
            "network",
            "mean",
            "zero",
            "shuffle",
            "matched_random",
        ]:
            handles, cache = rep.hooks(model, kind)
            after = {}

            def post(module, args, out):
                after["out"] = out[:, -1, :].clone()

            handle = model.model.layers[rep.layer].mlp.register_forward_hook(post)
            logits(model, inp)
            for h in handles:
                h.remove()
            handle.remove()
            d = rep.random_d if kind == "matched_random" else rep.d
            delta = after["out"] - base
            scale = rep.normsq.sqrt() if kind == "matched_random" else rep.normsq
            expected = (cache["predicted_coefficient"] - base @ rep.d)[:, None] * d / scale
            error = float((delta - expected).abs().max())
            orth = delta - (delta @ d)[:, None] * d / d.double().square().sum().float()
            maxorth = float(orth.abs().max())
            assert error < 1e-4 and maxorth < 1e-4, (kind, error, maxorth)
            result[kind] = {
                "max_replacement_error": error,
                "max_change_outside_declared_direction": maxorth,
            }
        # Both encoder and output directions are fixed, upstream-only tensors.
        assert rep.encoder.shape == (model.config.hidden_size, 3)
        assert float(rep.normsq) > 0
        assert abs(float(rep.d @ rep.random_d)) < 1e-5
        assert len(sets["train"] & sets["test"]) == 0
    dump(
        "pipeline-validation.json",
        {
            "disjoint_partitions": True,
            "normalized_projection_and_orthogonal_control": True,
            "replacement_checks": result,
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
