"""Localize one contribution per operation; train-only PLS across prompt formats."""

import argparse
import json
import os
import time

import torch

from data import ROOT, STYLES, save
from model_io import capture, corrupt_row, expand, load, margin, patch_scores, setup, targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("operation", choices=["add", "multiply", "divide"])
    args = p.parse_args()
    out = ROOT / args.operation
    setup()
    save(
        out / "collection-job.json",
        {
            "pid": os.getpid(),
            "started_epoch": time.time(),
            "gpu": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    problems = json.loads((out / "problems.json").read_text())
    start = time.perf_counter()
    model, tok = load()
    with torch.inference_mode():
        train = expand(problems["train"][:32], STYLES)
        corrupt = [corrupt_row(r) for r in train]
        all_layers = list(range(len(model.model.layers)))
        clean, cl = capture(model, tok, train, all_layers)
        bad, bl = capture(model, tok, corrupt, all_layers)
        ct, bt = targets(tok, train), targets(tok, train, True)
        base = margin(bl.cuda(), ct, bt)
        clean_margin = margin(cl.cuda(), ct, bt)
        effects = []
        for layer in all_layers:
            score = patch_scores(
                model, tok, train, layer, clean[layer]["output"], bad[layer]["output"]
            )
            effect = float((margin(score.cuda(), ct, bt) - base).mean())
            effects.append({"layer": layer, "mean_margin_restored": effect})
        top = sorted(effects, key=lambda r: r["mean_margin_restored"], reverse=True)[:3]
        print(args.operation, "top layers", top, flush=True)
        validation = expand(problems["validation"][:32], STYLES)
        valbad = [corrupt_row(r) for r in validation]
        vc, vcl = capture(model, tok, validation, [r["layer"] for r in top])
        vb, vbl = capture(model, tok, valbad, [r["layer"] for r in top])
        vct, vbt = targets(tok, validation), targets(tok, validation, True)
        vm = margin(vbl.cuda(), vct, vbt)
        choices = []
        directions = []
        for layer in [r["layer"] for r in top]:
            difference = (clean[layer]["output"] - bad[layer]["output"]).cuda().double()
            _, _, vh = torch.linalg.svd(difference - difference.mean(0), full_matrices=False)
            ds = [d.float() for d in vh[:4]]
            mean_difference = difference.mean(0).float()
            ds.append(mean_difference / mean_difference.norm())
            for index, d in enumerate(ds):
                scores = patch_scores(
                    model,
                    tok,
                    validation,
                    layer,
                    vc[layer]["output"],
                    vb[layer]["output"],
                    d,
                )
                effect = float((margin(scores.cuda(), vct, vbt) - vm).mean())
                choices.append(
                    {
                        "layer": layer,
                        "direction_index": index,
                        "validation_margin_restored": effect,
                    }
                )
                directions.append(d)
        which = max(range(len(choices)), key=lambda i: choices[i]["validation_margin_restored"])
        chosen = choices[which]
        layer, d = chosen["layer"], directions[which]
        restore = patch_scores(
            model, tok, validation, layer, vb[layer]["output"], vb[layer]["output"]
        )
        assert torch.allclose(restore, vbl, atol=1e-4, rtol=1e-4)
        save(
            out / "localization.json",
            {
                "full_layer_effects": effects,
                "direction_candidates": choices,
                "selected": chosen,
                "training_clean_corrupt_margin_gap": float((clean_margin - base).mean()),
                "restore_equivalent": True,
            },
        )
        print(args.operation, "selected", chosen, flush=True)
        del clean, bad, vc, vb, cl, bl, vcl, vbl
        raw = {}
        for split in ["train", "validation"]:
            rows = expand(problems[split], STYLES)
            print(args.operation, "collect", split, len(rows), flush=True)
            c, _ = capture(model, tok, rows, [layer], keep_scores=False)
            b, _ = capture(
                model,
                tok,
                [corrupt_row(r) for r in rows],
                [layer],
                keep_scores=False,
            )
            raw[split] = {"input": torch.stack([c[layer]["input"], b[layer]["input"]], 1)}
            del c, b
        input_mean = raw["train"]["input"].mean((0, 1)).cuda()
        xs, ys = {}, {}
        selected_layer = model.model.layers[layer].mlp
        for split, record in raw.items():
            h = record["input"].cuda()
            xx = []
            yy = []
            for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                mixed = h[:, 1] * (1 - alpha) + h[:, 0] * alpha
                xx.append(mixed - input_mean)
                yy.append(torch.cat([selected_layer(chunk) @ d for chunk in mixed.split(128)]))
            xs[split] = torch.cat(xx)
            ys[split] = torch.cat(yy)
        ym, ystd = ys["train"].mean(), ys["train"].std().clamp_min(1e-6)
        xr = xs["train"].double().clone()
        yr = ys["train"].double() - ys["train"].double().mean()
        ws, ps = [], []
        for _ in range(32):
            w = xr.T @ yr
            w /= w.norm()
            t = xr @ w
            p = xr.T @ t / (t @ t)
            q = yr @ t / (t @ t)
            xr -= t[:, None] * p
            yr -= t * q
            ws.append(w)
            ps.append(p)
        w, p = torch.stack(ws, 1), torch.stack(ps, 1)
        encoder = (w @ torch.linalg.inv(p.T @ w)).float()
        zz = {k: v @ encoder for k, v in xs.items()}
        zm, zs = zz["train"].mean(0), zz["train"].std(0).clamp_min(1e-6)
        features = {
            k: {
                "x": ((z - zm) / zs).cpu(),
                "y": ((ys[k] - ym) / ystd).cpu(),
                "base_pairs": len(problems[k]),
                "cases": len(raw[k]["input"]),
            }
            for k, z in zz.items()
        }
        torch.save(features, out / "features.pt")
        torch.save(
            {
                "layer": layer,
                "direction": d.cpu(),
                "input_mean": input_mean.cpu(),
                "encoder": encoder.cpu(),
                "zmean": zm.cpu(),
                "zstd": zs.cpu(),
                "ymean": ym.cpu(),
                "ystd": ystd.cpu(),
                "component_mean": ym.cpu(),
                "feature_rank": 32,
            },
            out / "component.pt",
        )
        # Raw inputs allow subsequent feature audits; not required for frozen-head evaluation.
        for split, record in raw.items():
            torch.save(record, out / f"inputs-{split}.pt")
    save(
        out / "collection.json",
        {
            "seconds": time.perf_counter() - start,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "dtype": "float32 model, float64 PLS",
            "selected": chosen,
            "train_samples": len(features["train"]["x"]),
            "validation_samples": len(features["validation"]["x"]),
            "test_activations_not_collected": True,
            "completed_epoch": time.time(),
        },
    )
    print("COLLECTION DONE", args.operation, flush=True)


if __name__ == "__main__":
    main()
