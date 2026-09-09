"""Bounded, prospective source-feature intervention screen on the local Gemma."""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from .runtime import SNAPSHOT, digest, load, prompt, save, setup, text_layers, tokenizer

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "weekday-protocol.json"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def build_rows(spec, tok):
    rows = []
    for split, offsets, formats in [
        ("train", spec["train_offsets"], spec["training_formats"]),
        ("development", spec["development_offsets"], spec["development_formats"]),
    ]:
        for offset in offsets:
            for fmt in formats:
                for day in range(7):
                    content = spec["formats"][fmt].format(day=DAYS[day], offset=offset)
                    text = prompt(tok, content)
                    encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
                    start = text.index(DAYS[day])
                    end = start + len(DAYS[day])
                    hits = [
                        i
                        for i, (a, b) in enumerate(encoded["offset_mapping"])
                        if a < end and b > start
                    ]
                    assert len(hits) == 1, "Weekday must occupy one source token"
                    assert content.index(DAYS[day]) < content.index(str(offset))
                    rows.append(
                        {
                            "id": f"{split}-n{offset}-f{fmt}-d{day}",
                            "split": split,
                            "offset": offset,
                            "format": fmt,
                            "day": day,
                            "answer": (day + offset) % 7,
                            "text": text,
                            "ids": encoded["input_ids"],
                            "site": hits[0],
                        }
                    )
    return rows


def batch_inputs(rows, tok):
    size = max(len(r["ids"]) for r in rows)
    assert all(len(r["ids"]) == size for r in rows), (
        "Padded inference failed the pinned-runtime audit"
    )
    ids, masks, sites = [], [], []
    for row in rows:
        pad = size - len(row["ids"])
        ids.append([tok.pad_token_id] * pad + row["ids"])
        masks.append([0] * pad + [1] * len(row["ids"]))
        sites.append(pad + row["site"])
    return {
        "input_ids": torch.tensor(ids, device="cuda"),
        "attention_mask": torch.tensor(masks, device="cuda"),
    }, torch.tensor(sites, device="cuda")


def confidence(records, key):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["day"], row["offset"])].append(float(row[key]))
    values = torch.tensor(
        [sum(v) / len(v) for v in grouped.values()], device="cuda", dtype=torch.float64
    )
    gen = torch.Generator(device="cuda").manual_seed(4103)
    ids = torch.randint(len(values), (10000, len(values)), device="cuda", generator=gen)
    samples = values[ids].mean(-1)
    return {
        "mean": float(values.mean()),
        "groups": len(values),
        "exploratory_95": torch.quantile(
            samples, torch.tensor([0.025, 0.975], device="cuda", dtype=torch.float64)
        ).tolist(),
    }


def score(logits, row, answer_ids, tok, target=None):
    top = int(logits.argmax())
    target = row["answer"] if target is None else target
    names = logits[answer_ids]
    return {
        "top_token": tok.decode([top]),
        "top_id": top,
        "correct": top == answer_ids[target],
        "restricted_correct": int(names.argmax()) == target,
        "weekday_logits": names.tolist(),
        "weekday_probability_mass": float(
            torch.logsumexp(names, 0).sub(torch.logsumexp(logits, 0)).exp()
        ),
    }


def make_features(rows, states, spec):
    bases, diagnostics = {}, {}
    train = [i for i, r in enumerate(rows) if r["split"] == "train"]
    dev = [i for i, r in enumerate(rows) if r["split"] == "development"]
    angle = torch.arange(7, device="cuda", dtype=torch.float64) * (2 * math.pi / 7)
    fourier = torch.stack([fn(k * angle) for k in [1, 2, 3] for fn in [torch.cos, torch.sin]], -1)
    for layer in spec["layers"]:
        h = states[layer].cuda().double()
        means = torch.stack(
            [h[[i for i in train if rows[i]["day"] == day]].mean(0) for day in range(7)]
        )
        centered = means - means.mean(0)
        for rank in spec["ranks"]:
            contrast = centered.T @ fourier[:, :rank]
            u, triangular = torch.linalg.qr(contrast)
            assert torch.linalg.matrix_rank(triangular) == rank
            orth_error = float((u.T @ u - torch.eye(rank, device="cuda")).abs().max())
            assert orth_error < 1e-10
            key = f"l{layer}-r{rank}"
            bases[key] = u.float().cpu()
            dev_coords, prototypes = h[dev] @ u, means @ u
            inferred = torch.cdist(dev_coords, prototypes).argmin(-1)
            labels = torch.tensor([rows[i]["day"] for i in dev], device="cuda")
            error = centered - (centered @ u) @ u.T
            diagnostics[key] = {
                "stored_basis_coefficients": u.numel(),
                "stored_basis_bytes": u.numel() * 4,
                "orthogonality_max_error": orth_error,
                "training_weekday_contrast_energy_captured": float(
                    1 - error.square().sum() / centered.square().sum()
                ),
                "development_nearest_prototype_weekday_accuracy": float(
                    (inferred == labels).double().mean()
                ),
                "prototype_coefficients_diagnostic_only": 7 * rank,
            }
    return bases, diagnostics


def decide(records, baseline, spec):
    gate = spec["gate"]
    dev_base = [r for r in baseline if r["split"] == "development"]
    native = confidence(dev_base, "correct")
    summaries, passing = {}, []
    for layer in spec["layers"]:
        full = [r for r in records if r["layer"] == layer and r["kind"] == "full"]
        for rank in spec["ranks"]:
            semantic = [
                r
                for r in records
                if r["layer"] == layer and r["rank"] == rank and r["kind"] == "feature"
            ]
            random = [
                r
                for r in records
                if r["layer"] == layer and r["rank"] == rank and r["kind"] == "random"
            ]
            assert [(r["id"], r["shift"]) for r in semantic] == [
                (r["id"], r["shift"]) for r in random
            ]
            paired = [
                dict(s, difference=int(s["correct"]) - int(r["correct"]))
                for s, r in zip(semantic, random)
            ]
            correct_subset = [
                r for r in semantic if r["baseline_correct"] and r["donor_baseline_correct"]
            ]
            row = {
                "native_accuracy": native,
                "full_patch_target_accuracy": confidence(full, "correct"),
                "feature_target_accuracy": confidence(semantic, "correct"),
                "random_target_accuracy": confidence(random, "correct"),
                "feature_minus_random": confidence(paired, "difference"),
                "new_format_feature_target_accuracy": confidence(
                    [r for r in semantic if r["format"] == 2], "correct"
                ),
                "conditional_both_native_correct": confidence(correct_subset, "correct")
                if correct_subset
                else None,
                "conditional_examples": len(correct_subset),
                "unconditional_examples": len(semantic),
                "mean_relative_edit_norm": sum(r["relative_edit_norm"] for r in semantic)
                / len(semantic),
                "mean_margin_change": sum(r["margin_change"] for r in semantic) / len(semantic),
            }
            checks = {
                "native": native["mean"] >= gate["native_accuracy_min"],
                "positive_control": row["full_patch_target_accuracy"]["mean"]
                >= gate["full_patch_target_accuracy_min"],
                "feature": row["feature_target_accuracy"]["mean"]
                >= gate["feature_target_accuracy_min"],
                "new_format": row["new_format_feature_target_accuracy"]["mean"]
                >= gate["new_format_feature_target_accuracy_min"],
                "beats_random": row["feature_minus_random"]["mean"]
                >= gate["feature_minus_random_accuracy_min"],
                "paired_lower": row["feature_minus_random"]["exploratory_95"][0]
                > gate["feature_minus_random_bootstrap_lower_min"],
            }
            row["checks"], row["passed"] = checks, all(checks.values())
            summaries[f"l{layer}-r{rank}"] = row
            if row["passed"]:
                passing.append((rank, -row["new_format_feature_target_accuracy"]["mean"], layer))
    selected = None
    if passing:
        rank, _, layer = min(passing)
        selected = {"layer": layer, "rank": rank}
    return {
        "candidates": summaries,
        "selected": selected,
        "next_step": "freeze equation comparison"
        if selected
        else "stop: no validated source-feature interface",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=HERE.parent / ".artifacts/gemma-weekdays")
    args = parser.parse_args()
    started = time.perf_counter()
    spec = json.loads(PROTOCOL.read_text())
    out = args.run_dir.resolve()
    assert out.is_relative_to(HERE.parent / ".artifacts"), "Keep run output inside this repo"
    assert not out.exists(), "Use a fresh run directory; never overwrite observed results"
    setup(spec["seed"])
    assert "H100" in torch.cuda.get_device_name()
    assert digest(SNAPSHOT / "config.json") == spec["config_sha256"]
    assert SNAPSHOT.name == spec["revision"]
    config = json.loads((SNAPSHOT / "config.json").read_text())
    deadline = started + (spec["feasibility_gpu_hour_cap"] - spec.get("prior_gpu_hours", 0)) * 3600
    out.mkdir(parents=True)
    freeze = {
        "protocol": spec,
        "source_sha256": digest(__file__),
        "runtime_sha256": digest(HERE / "runtime.py"),
        "runtime_protocol_sha256": digest(HERE / "protocol.json"),
        "protocol_sha256": digest(PROTOCOL),
        "model_config_sha256": digest(SNAPSHOT / "config.json"),
        "architecture": config["architectures"],
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "created_unix": time.time(),
    }
    save(out / "freeze.json", freeze)
    tok = tokenizer()
    answer_tokens = [tok.encode(s, add_special_tokens=False) for s in DAYS]
    assert all(len(ids) == 1 for ids in answer_tokens)
    answer_ids = [ids[0] for ids in answer_tokens]
    rows = build_rows(spec, tok)
    save(out / "prompts.json", rows)
    native = load(host_ple=True)
    baseline, records = [], []
    states = {layer: [] for layer in spec["layers"]}
    validation = {
        "zero_patch_bitwise_equal": True,
        "non_site_tokens_bitwise_equal": True,
        "full_patch_equals_donor": True,
        "random_relative_norm_max_error_after_bf16": 0.0,
    }

    def check_budget():
        if time.perf_counter() >= deadline:
            raise TimeoutError("Frozen feasibility GPU-hour cap reached")

    try:
        with torch.inference_mode():
            for start in range(0, len(rows), spec["batch_size"]):
                check_budget()
                group = rows[start : start + spec["batch_size"]]
                inputs, sites = batch_inputs(group, tok)
                handles = []
                for layer in spec["layers"]:

                    def capture(module, args, hidden, layer=layer):
                        assert torch.is_tensor(hidden)
                        states[layer].append(
                            hidden[torch.arange(len(group), device="cuda"), sites].cpu()
                        )

                    handles.append(text_layers(native)[layer].register_forward_hook(capture))
                try:
                    logits = (
                        native(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                    )
                finally:
                    for handle in handles:
                        handle.remove()
                assert torch.isfinite(logits).all()
                for row, values in zip(group, logits):
                    baseline.append(
                        {k: row[k] for k in ["id", "split", "day", "offset", "format", "answer"]}
                        | score(values, row, answer_ids, tok)
                    )
                if start % 64 == 0:
                    print(
                        json.dumps(
                            {"phase": "baseline", "completed": len(baseline), "total": len(rows)}
                        ),
                        flush=True,
                    )
            states = {k: torch.cat(v) for k, v in states.items()}
            save(out / "baseline.json", baseline)
            bases, diagnostics = make_features(rows, states, spec)
            torch.save({"states": states, "bases": bases}, out / "features.pt")
            save(out / "feature-diagnostics.json", diagnostics)
            lookup = {
                (r["split"], r["offset"], r["format"], r["day"]): i for i, r in enumerate(rows)
            }
            cases = []
            for i, row in enumerate(rows):
                if row["split"] != "development":
                    continue
                for shift in spec["donor_shifts"]:
                    donor = lookup[
                        ("development", row["offset"], row["format"], (row["day"] + shift) % 7)
                    ]
                    cases.append((i, donor, shift))
            for layer in spec["layers"]:
                for kind, rank in [("zero", 0), ("full", 1536)] + [
                    (kind, rank) for rank in spec["ranks"] for kind in ["feature", "random"]
                ]:
                    source_cases = cases[: spec["batch_size"]] if kind == "zero" else cases
                    for start in range(0, len(source_cases), spec["batch_size"]):
                        check_budget()
                        batch = source_cases[start : start + spec["batch_size"]]
                        group = [rows[i] for i, _, _ in batch]
                        inputs, sites = batch_inputs(group, tok)
                        donor = states[layer][[d for _, d, _ in batch]].cuda().float()
                        if kind in ["feature", "random"]:
                            u = bases[f"l{layer}-r{rank}"].cuda()
                            if kind == "random":
                                vectors = []
                                for i, _, shift in batch:
                                    gen = torch.Generator(device="cuda").manual_seed(
                                        spec["seed"] + 10000 * layer + 100 * i + shift + rank
                                    )
                                    v = torch.randn(1536, device="cuda", generator=gen)
                                    v -= u @ (u.T @ v)
                                    vectors.append(v / v.norm())
                                random_direction = torch.stack(vectors)
                        actual = {}

                        def patch(module, args, hidden):
                            bi = torch.arange(len(group), device="cuda")
                            current = hidden[bi, sites].float()
                            delta = donor - current
                            if kind == "zero":
                                delta.zero_()
                            if kind in ["feature", "random"]:
                                delta = (delta @ u) @ u.T
                                semantic_realized = (current + delta).to(
                                    hidden.dtype
                                ).float() - current
                                if kind == "random":
                                    delta = random_direction * semantic_realized.norm(
                                        dim=-1, keepdim=True
                                    )
                            changed = hidden.clone()
                            changed[bi, sites] = (current + delta).to(hidden.dtype)
                            mask = torch.ones(hidden.shape[:2], device="cuda", dtype=torch.bool)
                            mask[bi, sites] = False
                            assert torch.equal(changed[mask], hidden[mask])
                            realized = changed[bi, sites].float() - hidden[bi, sites].float()
                            actual["relative_norm"] = (
                                realized.norm(dim=-1) / hidden[bi, sites].float().norm(dim=-1)
                            ).tolist()
                            if kind == "random":
                                norm_error = float(
                                    (
                                        (realized.norm(dim=-1) - semantic_realized.norm(dim=-1))
                                        / semantic_realized.norm(dim=-1).clamp_min(1e-8)
                                    )
                                    .abs()
                                    .max()
                                )
                                validation["random_relative_norm_max_error_after_bf16"] = max(
                                    validation["random_relative_norm_max_error_after_bf16"],
                                    norm_error,
                                )
                                assert norm_error < 0.02, (
                                    "BF16 random control norm differs by over 2%"
                                )
                            if kind == "full":
                                assert torch.equal(changed[bi, sites].float(), donor)
                            if kind == "zero":
                                assert torch.equal(hidden, changed)
                            return changed

                        handle = text_layers(native)[layer].register_forward_hook(patch)
                        try:
                            logits = (
                                native(**inputs, use_cache=False, logits_to_keep=1)
                                .logits[:, -1]
                                .float()
                            )
                        finally:
                            handle.remove()
                        assert torch.isfinite(logits).all()
                        if kind == "zero":
                            clean = (
                                native(**inputs, use_cache=False, logits_to_keep=1)
                                .logits[:, -1]
                                .float()
                            )
                            assert torch.equal(logits, clean)
                            continue
                        for j, ((i, d, shift), values) in enumerate(zip(batch, logits)):
                            row = rows[i]
                            target = rows[d]["answer"]
                            margin = float(
                                values[answer_ids[target]] - values[answer_ids[row["answer"]]]
                            )
                            base_margin = (
                                baseline[i]["weekday_logits"][target]
                                - baseline[i]["weekday_logits"][row["answer"]]
                            )
                            records.append(
                                {
                                    "id": row["id"],
                                    "day": row["day"],
                                    "offset": row["offset"],
                                    "format": row["format"],
                                    "shift": shift,
                                    "layer": layer,
                                    "kind": kind,
                                    "rank": rank,
                                    "target": target,
                                    "baseline_correct": baseline[i]["correct"],
                                    "donor_baseline_correct": baseline[d]["correct"],
                                    "margin_change": margin - base_margin,
                                    "relative_edit_norm": actual["relative_norm"][j],
                                    **score(values, row, answer_ids, tok, target),
                                }
                            )
                    save(out / "interventions.json", records)
                    print(
                        json.dumps(
                            {
                                "phase": "interventions",
                                "layer": layer,
                                "kind": kind,
                                "rank": rank,
                                "records": len(records),
                                "seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
            decision = decide(records, baseline, spec)
            torch.cuda.synchronize()
            result = {
                "complete": True,
                "freeze": freeze,
                "feature_diagnostics": diagnostics,
                "validation": validation,
                "decision": decision,
                "baseline": baseline,
                "interventions": records,
                "elapsed_seconds": time.perf_counter() - started,
                "gpu_hours": (time.perf_counter() - started) / 3600,
                "total_study_gpu_hours": (time.perf_counter() - started) / 3600
                + spec.get("prior_gpu_hours", 0),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "confirmation_opened": False,
                "eml_fits": 0,
            }
            save(out / "result.json", result)
            print(json.dumps({"decision": decision, "gpu_hours": result["gpu_hours"]}), flush=True)
    except Exception as exc:
        save(
            out / "incomplete.json",
            {
                "exception": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
                "baseline_rows": len(baseline),
                "intervention_rows": len(records),
            },
        )
        raise


if __name__ == "__main__":
    main()
