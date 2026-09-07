"""Export frozen equations and upstream features for an offline intervention explorer."""

import json
import time

import torch

from data import ROOT, STYLES, save
from model_io import MODEL, REVISION, capture, expand, load, setup


def key(row):
    return row["a"], row["b"], row["c"], row["style"]


def build(model, tok, out, compact):
    suffix = "compact" if compact else "primary"
    path = out / f"demo-{suffix}.json"
    if path.exists():
        return
    selection_path = out / ("minimal-selection.json" if compact else "selection.json")
    normal_path = out / (
        "ordinary-minimal-fresh.json" if compact else "ordinary-test-known-formats.json"
    )
    causal_path = out / (
        "interventions-minimal-fresh.json" if compact else "interventions-test-known-formats.json"
    )
    if not all(p.exists() for p in [selection_path, normal_path, causal_path]):
        return
    metrics_path = out / "results.json"
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text())
    suite = "minimal-fresh" if compact else "test-known-formats"
    if suite not in metrics:
        return
    if not compact and not all(
        (out / f"{category}-test-new-format.json").exists()
        for category in ["ordinary", "interventions"]
    ):
        return
    selection = json.loads(selection_path.read_text())
    if compact:
        selection = selection["selection"]
        name = "minimal/eml"
    else:
        name = min(
            [k for k in selection["heads"] if k.endswith("/eml")],
            key=lambda k: selection["heads"][k]["validation_objective"],
        )
    spec = selection["heads"][name]
    neural_name = name.removesuffix("/eml") + "/neural"
    c = torch.load(
        out / ("component.pt" if spec["features"] == "pls" else "component-active.pt"),
        weights_only=True,
    )
    c = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in c.items()}
    problems = json.loads(
        (out / ("minimal-problems.json" if compact else "problems.json")).read_text()
    )["test"][:32]
    rows = expand(problems, STYLES if compact else STYLES + ["instruction"])
    ordinary = json.loads(normal_path.read_text())
    causal = json.loads(causal_path.read_text())
    if not compact:
        for category, raw in [("ordinary", ordinary), ("interventions", causal)]:
            extra = json.loads((out / f"{category}-test-new-format.json").read_text())
            for kind in raw:
                raw[kind].extend(extra[kind])
    normal_tables = {
        kind: {key(r): r for r in ordinary[kind]} for kind in ["original", name, neural_name]
    }
    causal_tables = {
        kind: {(key(r), r["alpha"]): r for r in causal[kind]}
        for kind in ["original", name, neural_name]
    }
    hs = []
    for corrupt in [False, True]:
        inputs = [{**row, "a": row["c"]} for row in rows] if corrupt else rows
        captured, _ = capture(model, tok, inputs, [c["layer"]], keep_scores=False)
        hs.append(captured[c["layer"]]["input"].cuda())
    rank = spec["rank"]
    z = [
        (((h - c["input_mean"]) @ c["encoder"][:, :rank]) - c["zmean"][:rank]) / c["zstd"][:rank]
        for h in hs
    ]
    state = torch.load(out / spec["checkpoint"], weights_only=True)
    alphas = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
    items = []
    for i, row in enumerate(rows):
        item = {
            **row,
            "clean_features": z[0][i].tolist(),
            "corrupt_features": z[1][i].tolist(),
            "curves": {},
            "answers": {},
        }
        for label, kind in [
            ("original", "original"),
            ("eml", name),
            ("neural", neural_name),
        ]:
            item["curves"][label] = {
                field: [causal_tables[kind][(key(row), a)][field] for a in alphas]
                for field in ["margin", "true_coefficient", "predicted_coefficient"]
            }
            raw = normal_tables[kind][key(row)]
            item["answers"][label] = {field: raw[field] for field in ["text", "correct"]}
        items.append(item)
    save(
        path,
        {
            "model": MODEL,
            "revision": REVISION,
            "operation": out.name,
            "mode": suffix,
            "layer_index": c["layer"],
            "features": spec["features"],
            "rank": rank,
            "projection_coefficients": c["encoder"].shape[0] * rank,
            "head": spec,
            "state": {k: v.tolist() for k, v in state.items()},
            "output_mean": float(c["ymean"]),
            "output_scale": float(c["ystd"]),
            "alphas": alphas,
            "cases": items,
            "metrics": {
                label: {
                    section: metrics[suite][section][kind]
                    for section in ["ordinary", "interventions"]
                }
                for label, kind in [
                    ("original", "original"),
                    ("mean", "mean"),
                    ("eml", name),
                    ("neural", neural_name),
                ]
            },
            "suite_metrics": {
                test: {
                    "ordinary": value.get("ordinary", {}).get(name),
                    "original_ordinary": value.get("ordinary", {}).get("original"),
                    "interventions": value["interventions"][name],
                }
                for test, value in metrics.items()
                if "/" not in test and name in value["interventions"]
            },
            "sampling": "First 32 frozen test pairs; no filtering by accuracy or fit quality",
            "scope": "One scalar contribution at the last prompt position; the rest of the original MLP and language model remain active",
            "feature_meaning": "Numerical activation coordinates; no claim that they correspond to named arithmetic concepts",
            "completed_epoch": time.time(),
        },
    )
    print("DEMO EXPORTED", MODEL, out.name, suffix, len(items), flush=True)


def main():
    setup()
    model, tok = load()
    with torch.inference_mode():
        for op in ["add", "multiply", "divide"]:
            for compact in [False, True]:
                build(model, tok, ROOT / op, compact)


if __name__ == "__main__":
    main()
