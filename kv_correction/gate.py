"""Apply the frozen stop rule to untouched validation articles and paired GPU costs."""

import json
import time

import numpy as np
import torch
from correctors import Correction, baseline, encode_residual
from support import (
    RUN,
    SPEC,
    errors,
    freeze_sources,
    load_records,
    save_json,
    setup,
    sha,
    storage_bytes,
    unpack4,
    verify_design,
    verify_sources,
)


def load_candidate(kind, seed):
    candidate = Correction(kind, seed).cuda().bfloat16().eval().requires_grad_(False)
    candidate.load_state_dict(
        torch.load(RUN / f"{kind}-{seed}.pt", weights_only=True, map_location="cuda")
    )
    return candidate


def clock():
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def validate_fits(selection, projection, fits):
    checked = []
    denom = torch.stack([errors(unpack4(r["code"]), r, projection) for r in selection])
    for row in fits:
        kind, seed = row["method"], row["seed"]
        assert sha(RUN / f"{kind}-{seed}.pt") == row["checkpoint_sha256"]
        candidate = load_candidate(kind, seed)
        actual = torch.stack([errors(candidate(r["code"]), r, projection) for r in selection])
        objective = float((actual / denom).mean())
        assert abs(objective - row["selection_objective"]) < 1e-7
        best = min(row["history"], key=lambda x: x["objective"])
        assert best["step"] == row["selected_step"]
        assert candidate.bytes() == row["weight_bytes"] <= SPEC["correction_weight_bytes_ceiling"]
        assert all(v.dtype == torch.bfloat16 for v in candidate.state_dict().values())
        checked.append(
            {
                "method": kind,
                "seed": seed,
                "selected_step": row["selected_step"],
                "objective": objective,
            }
        )
    assert len(checked) == 15
    # Identity initialization, real nibble packing, and stored-residual accounting.
    sample = selection[0]
    decoded = unpack4(sample["code"])
    for kind in ["linear", "silu", "eml"]:
        identity = Correction(kind, 101).cuda().bfloat16()
        assert torch.equal(identity(sample["code"]), decoded)
    assert sample["code"][0].dtype == torch.uint8
    expected_payload = SPEC["prefix_tokens"] * (512 // 2 + 4)
    assert storage_bytes(sample["code"]) == expected_payload
    code, indices, residuals = encode_residual(sample["v"])
    assert storage_bytes([indices, residuals]) == 65 * 6
    assert SPEC["reference_prefix_bank"] * 65 * 6 <= SPEC["correction_weight_bytes_ceiling"]
    assert (
        errors(baseline("stored_residual", code, (indices, residuals)), sample, projection)[0]
        <= errors(decoded, sample, projection)[0]
    )
    save_json(
        RUN / "gpu-validation.json",
        {
            "gpu": torch.cuda.get_device_name(),
            "fits": checked,
            "identity": "bitwise equal",
            "packed_storage": "passed",
        },
    )


@torch.inference_mode()
def evaluate(records, projection):
    rows = []
    for kind in SPEC["methods"]:
        seeds = SPEC["seeds"] if kind in ["linear", "silu", "eml"] else [None]
        for seed in seeds:
            candidate = load_candidate(kind, seed) if seed is not None else None
            for record in records:
                if candidate is None:
                    stored = encode_residual(record["v"])[1:] if kind == "stored_residual" else None
                    value = baseline(kind, record["code"], stored)
                else:
                    value = candidate(record["code"])
                measured = errors(value, record, projection)
                assert torch.isfinite(measured).all()
                rows.append(
                    {
                        "method": kind,
                        "seed": seed,
                        "title": record["title"],
                        "v_mse": float(measured[0]),
                        "attention_mse": float(measured[1]),
                    }
                )
            print("GATE EVALUATION", kind, seed, flush=True)
    save_json(RUN / "gate-errors.json", rows)
    return rows


@torch.inference_mode()
def timing(sample):
    code = sample["code"]
    stored = encode_residual(sample["v"])[1:]
    rows = []
    for seed in SPEC["seeds"]:
        candidates = {kind: load_candidate(kind, seed) for kind in ["linear", "silu", "eml"]}
        for repetition in range(-SPEC["latency_warmup"], SPEC["latency_repetitions"]):
            methods = SPEC["methods"]
            offset = max(0, repetition) % len(methods)
            for kind in methods[offset:] + methods[:offset]:
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
                start = clock()
                value = (
                    candidates[kind](code) if kind in candidates else baseline(kind, code, stored)
                )
                elapsed = (clock() - start) * 1000
                if repetition >= 0:
                    rows.append(
                        {
                            "method": kind,
                            "seed": seed,
                            "repetition": repetition,
                            "decode_correct_ms": elapsed,
                            "temporary_peak_bytes": torch.cuda.max_memory_allocated() - before,
                        }
                    )
                del value
        print("GPU TIMING", seed, flush=True)
    save_json(
        RUN / "timing.json", {"gpu": torch.cuda.get_device_name(), "shared_gpu": True, "rows": rows}
    )
    return rows


def analyze(rows, times, fits):
    seeds = SPEC["seeds"]
    learned = ["linear", "silu", "eml"]
    arrays = {}
    for kind in SPEC["methods"]:
        values = [[r["v_mse"], r["attention_mse"]] for r in rows if r["method"] == kind]
        arrays[kind] = np.array(values).reshape(
            len(seeds) if kind in learned else 1, SPEC["gate_articles"], 2
        )
    means = {k: v.mean((0, 1)) for k, v in arrays.items()}
    rng = np.random.default_rng(SPEC["data_seed"])
    draws = SPEC["bootstrap_draws"]
    seed_draws = rng.integers(len(seeds), size=(draws, len(seeds)))
    doc_draws = rng.integers(SPEC["gate_articles"], size=(draws, SPEC["gate_articles"]))
    eml_samples = arrays["eml"][seed_draws[:, :, None], doc_draws[:, None, :]].mean((1, 2))
    silu_samples = arrays["silu"][seed_draws[:, :, None], doc_draws[:, None, :]].mean((1, 2))
    improvement = (means["silu"] - means["eml"]) / means["silu"]
    improvement_ci = np.quantile(
        (silu_samples - eml_samples) / silu_samples, [0.025, 0.975], axis=0
    )
    latencies = {
        k: np.array([r["decode_correct_ms"] for r in times if r["method"] == k]).reshape(
            len(seeds), SPEC["latency_repetitions"]
        )
        for k in SPEC["methods"]
    }
    ratios = latencies["eml"] / latencies["silu"]
    rep_draws = rng.integers(SPEC["latency_repetitions"], size=(draws, SPEC["latency_repetitions"]))
    ratio_ci = np.quantile(
        ratios[seed_draws[:, :, None], rep_draws[:, None, :]].mean((1, 2)), [0.025, 0.975]
    )
    criteria = {
        "every_seed_both_metrics": bool((arrays["eml"].mean(1) < arrays["silu"].mean(1)).all()),
        "at_least_one_percent_both_metrics": bool(
            (improvement >= SPEC["gate"]["minimum_pooled_relative_improvement_both_metrics"]).all()
        ),
        "positive_bootstrap_lower_both_metrics": bool((improvement_ci[0] > 0).all()),
        "beats_all_other_controls": all(
            bool((means["eml"] < means[k]).all())
            for k in ["none", "rms", "stored_residual", "linear"]
        ),
        "matched_eml_silu_weight_bytes": all(
            next(r["weight_bytes"] for r in fits if r["method"] == "eml" and r["seed"] == s)
            == next(r["weight_bytes"] for r in fits if r["method"] == "silu" and r["seed"] == s)
            for s in seeds
        ),
        "comparable_decode_correct_cost": bool(
            ratio_ci[1] <= SPEC["gate"]["eml_to_silu_decode_correct_latency_ratio_upper95_max"]
        ),
    }
    payload = SPEC["prefix_tokens"] * (512 // 2 + 4) * SPEC["reference_prefix_bank"]
    method_results = []
    for kind in SPEC["methods"]:
        overhead = (
            next(r["weight_bytes"] for r in fits if r["method"] == kind)
            if kind in learned
            else 65 * 6 * SPEC["reference_prefix_bank"]
            if kind == "stored_residual"
            else 0
        )
        assert overhead <= SPEC["correction_weight_bytes_ceiling"]
        method_results.append(
            {
                "method": kind,
                "v_mse": float(means[kind][0]),
                "attention_mse": float(means[kind][1]),
                "error_ratio_to_none": (means[kind] / means["none"]).tolist(),
                "bank_v_and_correction_bytes": payload + overhead,
                "correction_weight_or_residual_bytes": overhead,
                "decode_correct_ms_median": float(np.median(latencies[kind])),
                "per_seed_mean_errors": arrays[kind].mean(1).tolist(),
            }
        )
    result = {
        "gate_passed": all(criteria.values()),
        "decision": "advance_to_independent_native_validation"
        if all(criteria.values())
        else "stop_this_direction",
        "criteria": criteria,
        "eml_relative_improvement_over_silu": improvement.tolist(),
        "improvement_95ci": improvement_ci.tolist(),
        "eml_silu_latency_ratio": float(ratios.mean()),
        "latency_ratio_95ci": ratio_ci.tolist(),
        "methods": method_results,
        "confidence_method": "paired crossed bootstrap over seeds and articles; latency resamples seeds and repetitions",
        "scope": SPEC["scope"],
    }
    save_json(RUN / "decision.json", result)
    print(json.dumps(result["criteria"]), result["decision"], flush=True)


def main():
    setup()
    verify_design()
    assert not (RUN / "gate-freeze.json").exists(), "Refusing to overwrite a gate run"
    complete = json.loads((RUN / "fits-complete.json").read_text())
    assert sha(RUN / "fits.json") == complete["fits_sha256"]
    fit_freeze = json.loads((RUN / "fit-freeze.json").read_text())
    verify_sources(fit_freeze["sources"])
    collection = json.loads((RUN / "collection.json").read_text())
    for name, digest in collection["artifact_sha256"].items():
        assert sha(RUN / name) == digest
    fits = json.loads((RUN / "fits.json").read_text())
    projection = torch.load(RUN / "projection.pt", weights_only=True, map_location="cuda")
    selection = load_records("selection")
    validate_fits(selection, projection, fits)
    save_json(
        RUN / "gate-freeze.json",
        {
            "sources": freeze_sources(),
            "fits_sha256": sha(RUN / "fits.json"),
            "design_sha256": sha(RUN / "design.json"),
        },
    )
    records = load_records("gate")
    rows = evaluate(records, projection)
    times = timing(selection[0])
    analyze(rows, times, fits)


if __name__ == "__main__":
    main()
