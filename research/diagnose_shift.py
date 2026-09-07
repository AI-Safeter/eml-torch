"""Post-result diagnostic of addition range-shift errors; no refitting or selection."""

import argparse
import json
import time

import torch

from active_features import gradients
from data import ROOT, STYLES, save
from evaluate_heads import Replacements
from model_io import MODEL, capture, expand, load, setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=["add", "multiply", "divide"], default="add")
    args = parser.parse_args()
    setup()
    out = ROOT / args.operation
    protocol = out / "shift-diagnostic-protocol.json"
    assert not (out / "shift-diagnostic.json").exists(), "Preserve the diagnostic"
    save(
        protocol,
        {
            "created_epoch": time.time(),
            "scope": "Post-result diagnosis only; no fitting or checkpoint selection",
            "pairs_per_split": 64,
            "splits": ["test", "shift"],
            "styles": STYLES,
            "strengths": [0, 0.5, 1],
            "model": MODEL,
        },
    )
    selection = json.loads((out / "selection.json").read_text())
    method = "heads-active-r32-g0.1/eml" if "1.7B" in MODEL else "heads-active-r32-g0/eml"
    rep = Replacements(out, selection)
    head = rep.models[method]
    c = rep.components["active"]
    q = torch.linalg.qr(c["encoder"].double()).Q.float()
    training = torch.load(out / "features-active.pt", weights_only=True)["train"]["x"].cuda()
    train_lo, train_hi = training.min(0).values, training.max(0).values
    model, tok = load()
    model.requires_grad_(False)
    layer = model.model.layers[rep.layer].mlp
    problems = json.loads((out / "problems.json").read_text())
    results = {}
    for split in ["test", "shift"]:
        rows = expand(problems[split][:64], STYLES)
        corrupt = [{**r, "a": r["c"]} for r in rows]
        with torch.inference_mode():
            clean, _ = capture(model, tok, rows, [rep.layer], keep_scores=False)
            bad, _ = capture(model, tok, corrupt, [rep.layer], keep_scores=False)
        clean_h, bad_h = clean[rep.layer]["input"].cuda(), bad[rep.layer]["input"].cuda()
        h = torch.cat([bad_h * (1 - a) + clean_h * a for a in [0, 0.5, 1]]).clone()
        g = gradients(layer, h, rep.d)
        z = (((h - c["input_mean"]) @ c["encoder"]) - c["zmean"]) / c["zstd"]
        with torch.no_grad():
            true = layer(h) @ rep.d
            predicted = rep.coefficient(h, method)
            left = head.left(z)
            right = head.right(z)
            nonlinear = head(z) - head.skip(z).squeeze(-1)
            response_true = true.reshape(3, len(rows))[1:] - true.reshape(3, len(rows))[:1]
            response_pred = (
                predicted.reshape(3, len(rows))[1:] - predicted.reshape(3, len(rows))[:1]
            )
            ratio = (
                (response_pred - response_true).square().mean() / response_true.square().mean()
            ).sqrt()
        results[split] = {
            "pairs": 64,
            "styles": 3,
            "mixed_inputs": len(h),
            "gradient_energy_captured": float((g @ q).square().sum() / g.square().sum()),
            "coefficient_response_nrmse": float(ratio),
            "coefficient_rmse_in_training_std": float(
                ((predicted - true) / c["ystd"]).square().mean().sqrt()
            ),
            "fraction_outside_any_training_coordinate_range": float(
                ((z < train_lo) | (z > train_hi)).any(1).float().mean()
            ),
            "maximum_absolute_standardized_coordinate": float(z.abs().max()),
            "left_argument_range": [float(left.min()), float(left.max())],
            "right_argument_range": [float(right.min()), float(right.max())],
            "nonlinear_term_range_in_training_output_std": [
                float(nonlinear.min()),
                float(nonlinear.max()),
            ],
            "numerical_guard_activated": bool(
                (left.abs() > 80).any() or (right.square() > 1e30).any()
            ),
        }
    save(
        out / "shift-diagnostic.json",
        {
            "completed_epoch": time.time(),
            "model": MODEL,
            "head": method,
            "results": results,
            "interpretation": "Exploratory 64-pair diagnostic after observing shift results. Coefficient NRMSE differs from the primary downstream margin metric. No refitting, head changes, or test-based selection.",
        },
    )
    print("SHIFT DIAGNOSTIC", MODEL, results, flush=True)


if __name__ == "__main__":
    main()
