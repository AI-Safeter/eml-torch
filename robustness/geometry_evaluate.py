"""Predeclared ambient and active-nullspace diagnostics, separate from natural edits."""

import argparse
import json

import torch
from runtime import RUNS, configure


def evaluate(model, tok, rep, rows, geometry):
    from model_io import inputs, logits, margin, targets

    records = {kind: [] for kind in rep.kinds}
    generator = torch.Generator(device="cuda").manual_seed(20260925)
    q = rep.components["active"]["encoder"]
    for start in range(0, len(rows), 32):
        batch = rows[start : start + 32]
        inp = inputs(tok, batch, True)
        with rep.hook(model, "original") as cache:
            reference = logits(model, inp)
        h = cache["input"].clone()
        direction = torch.randn(h.shape, device="cuda", generator=generator)
        if geometry == "nullspace":
            direction -= (direction @ q) @ q.T
        direction /= direction.norm(dim=1, keepdim=True)
        scale = h.norm(dim=1, keepdim=True)
        ct, bt = targets(tok, batch), targets(tok, batch, True)
        for strength in [0.0, -0.1, -0.01, 0.01, 0.1]:
            mixed = h + strength * scale * direction
            if geometry == "nullspace":
                change = (mixed - h) @ q
                assert float(change.abs().max()) < 1e-3
            with rep.hook(model, "original", mixed) as base_cache:
                base = logits(model, inp)
            if strength == 0:
                torch.testing.assert_close(base, reference, rtol=1e-4, atol=1e-4)
            base_lp, base_p = base.log_softmax(-1), base.softmax(-1)
            for kind in rep.kinds:
                if kind == "original":
                    scores, current = base, base_cache
                else:
                    with rep.hook(model, kind, mixed) as current:
                        scores = logits(model, inp)
                if kind == "restore":
                    torch.testing.assert_close(scores, base, rtol=0, atol=0)
                assert torch.isfinite(scores).all()
                effect = margin(scores, ct, bt)
                kl = (base_p * (base_lp - scores.log_softmax(-1))).sum(1)
                for i, row in enumerate(batch):
                    records[kind].append(
                        {
                            **row,
                            "geometry": geometry,
                            "alpha": strength,
                            "margin": float(effect[i]),
                            "kl_vs_same_intervention_original": float(kl[i]),
                            "true_coefficient": float(current["true"][i]),
                            "predicted_coefficient": float(current["pred"][i]),
                        }
                    )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen17b", "qwen4b", "smollm", "gemma"])
    parser.add_argument("operation", choices=["add", "multiply", "divide"])
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    configure(args.model)
    from data import STYLES, save
    from evaluate_heads import Replacements, freeze
    from model_io import expand, load, setup

    setup()
    out = RUNS / args.model / args.operation
    selected = freeze(out, ["heads", "heads-active-r32-g0", "heads-active-r32-g0.1"])
    rep = Replacements(out, selected)
    model, tok = load()
    from prefix_execution import install

    install(model, rep.layer, globals())
    problems = json.loads((out / "problems.json").read_text())
    rows = expand(
        problems["validation"][:4] if args.validation_only else problems["test"][:32], STYLES
    )
    with torch.inference_mode():
        for geometry in ["ambient", "nullspace"]:
            name = f"geometry-{geometry}{'-validation' if args.validation_only else ''}.json"
            if not (out / name).exists():
                save(out / name, evaluate(model, tok, rep, rows, geometry))
    print("GEOMETRY EVALUATION COMPLETE", args.model, args.operation, flush=True)


if __name__ == "__main__":
    main()
