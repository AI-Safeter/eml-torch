"""Preserve all screen outcomes, paired uncertainty, and paired timing samples."""

import json

import torch

from gemma_mechanisms.quality_summary import bootstrap

from .common import SPEC, digest, root, save, setup, sources
from .statistics import accuracy, point


def paired_timing(samples, method, key):
    values = torch.tensor(
        [[p["original"][key], p[method][key]] for p in samples], device="cuda", dtype=torch.float64
    )
    assert len(values) == 30
    # Fixed before timing: resample six contiguous blocks of five paired repeats.
    blocks = values.reshape(6, 5, 2).mean(1)
    rng = torch.Generator(device="cuda").manual_seed(918)
    ids = torch.randint(6, (10000, 6), device="cuda", generator=rng)
    means = blocks[ids].mean(1)
    ratios = 1 - means[:, 1] / means[:, 0]
    return {
        "original_mean": float(values[:, 0].mean()),
        "replacement_mean": float(values[:, 1].mean()),
        "relative_reduction": float(1 - values[:, 1].mean() / values[:, 0].mean()),
        "relative_reduction_95_block_bootstrap": torch.quantile(
            ratios, torch.tensor([0.025, 0.975], device="cuda", dtype=torch.float64)
        ).tolist(),
        "pairs": 30,
        "block_length": 5,
    }


def main():
    setup(94)
    out = root()
    directory = out / "evaluation/development"
    rows = {
        p.stem: json.loads(p.read_text())
        for p in directory.glob("*.json")
        if not p.stem.endswith("-freeze")
    }
    original = rows["original"]["results"]
    quality = {}
    for method, record in rows.items():
        r = record["results"]
        entry = {"point": point(record), "relative_to_original": {}, "diagnostics": {}}
        for op in ["add", "multiply", "divide"]:
            entry["relative_to_original"][op] = accuracy(
                [v for v in original["arithmetic"] if v["op"] == op],
                [v for v in r["arithmetic"] if v["op"] == op],
            )
        entry["relative_to_original"]["arc"] = accuracy(original["arc"], r["arc"])
        entry["relative_to_original"]["ce"] = bootstrap(
            [a["ce_nats"] - b["ce_nats"] for a, b in zip(r["language"], original["language"])]
        )
        for op in ["add", "multiply", "divide"]:
            for style in ["code", "reversed"]:
                v = [x for x in r["arithmetic"] if x["op"] == op and x["style"] == style]
                entry["diagnostics"][op + "-" + style] = {
                    "n": len(v),
                    "accuracy": sum(x["correct"] for x in v) / len(v),
                    "parse_rate": sum(x["prediction"] is not None for x in v) / len(v),
                }
            v = [x for x in r["shift"] if x["op"] == op]
            entry["diagnostics"]["shift-" + op] = {
                "n": len(v),
                "accuracy": sum(x["correct"] for x in v) / len(v),
                "parse_rate": sum(x["prediction"] is not None for x in v) / len(v),
            }
        quality[method] = entry
    comparisons = {}
    for arch in SPEC["training"]["architectures"]:
        methods = {act: f"{arch}-{act}-d1-s1103-n12000" for act in ["eml", "silu"]}
        if not all(m in rows for m in methods.values()):
            continue
        e, s = [rows[methods[act]]["results"] for act in ["eml", "silu"]]
        comparisons[arch] = {
            op: accuracy(
                [v for v in s["arithmetic"] if v["op"] == op],
                [v for v in e["arithmetic"] if v["op"] == op],
            )
            for op in ["add", "multiply", "divide"]
        }
        comparisons[arch]["arc"] = accuracy(s["arc"], e["arc"])
    timings = {}
    for path in sorted((out / "benchmark").glob("*-b*-p*.json")):
        r = json.loads(path.read_text())
        method = r["method"]
        timings[path.stem] = {
            "method": method,
            "batch": r["batch"],
            "prefill_tokens": r["prefill_tokens"],
            "decode_steps": r["decode_steps"],
            "timing": {
                k: paired_timing(r["samples"], method, k)
                for k in [
                    "prefill_seconds",
                    "decode_seconds",
                    "end_to_end_seconds",
                    "prefill_gpu_ms",
                    "decode_gpu_ms",
                ]
            },
            "throughput_mean": {
                who: {
                    k: sum(p[who][k] for p in r["samples"]) / len(r["samples"])
                    for k in [
                        "prefill_tokens_per_second",
                        "decode_tokens_per_second",
                        "output_tokens_per_second",
                        "total_tokens_per_second",
                    ]
                }
                for who in ["original", method]
            },
            "isolated_memory": r["isolated_memory"],
            "original_accounting": r["original_accounting"],
            "student_accounting": r["student_module_accounting"],
            "device_uuid": r["device_uuid"],
            "telemetry_before": r["telemetry_before"],
            "telemetry_after": r["telemetry_after"],
            "raw_sha256": digest(path),
        }
    fits = {p.stem: json.loads(p.read_text()) for p in (out / "training").glob("*-n*.json")}
    diagnostic = {p.stem: json.loads(p.read_text()) for p in (out / "diagnostics").glob("*.json")}
    result = {
        "scope": "Development screening; no seed-replication or confirmation claim unless a separate confirmation summary exists. All sampled outcomes are unconditional.",
        "quality": quality,
        "eml_vs_silu_development": comparisons,
        "timings": timings,
        "fits": fits,
        "diagnostics": diagnostic,
        "capacity": json.loads((out / "training/initialization.json").read_text()),
        "decision": json.loads((out / "decision.json").read_text()),
        "sources": sources("summarize.py", "statistics.py"),
        "inputs": {
            str(p.relative_to(out)): digest(p)
            for sub in ["training", "evaluation", "benchmark", "diagnostics"]
            for p in sorted((out / sub).rglob("*.json"))
        },
    }
    save(out / "summary.json", result)
    print("SUMMARY COMPLETE", len(fits), len(timings), flush=True)


if __name__ == "__main__":
    main()
