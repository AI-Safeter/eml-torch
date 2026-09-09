"""Post-screen CUDA diagnostics for padding and source-token causal paths."""

import argparse
import inspect
import json
import time
from pathlib import Path

import torch

from .runtime import digest, load, save, setup, text_layers, tokenizer
from .weekdays import batch_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.run_dir.resolve()
    assert out.is_relative_to(Path(__file__).resolve().parents[1] / ".artifacts")
    assert not (out / "audit-freeze.json").exists(), "Preserve completed diagnostics"
    started = time.perf_counter()
    screen = json.loads((out / "result.json").read_text())
    spec = screen["freeze"]["protocol"]
    setup(spec["seed"])
    failed_path = out / "audit-attempts.json"
    failed = json.loads(failed_path.read_text()) if failed_path.exists() else []
    prior = screen["total_study_gpu_hours"] + sum(r["charged_gpu_hours"] for r in failed)
    deadline = started + (spec["gpu_hour_cap"] - prior) * 3600
    assert deadline > started and "H100" in torch.cuda.get_device_name()
    rows = json.loads((out / "prompts.json").read_text())
    chosen = [r for r in rows if r["split"] == "development" and r["offset"] == 6 and r["day"] == 0]
    assert len(chosen) == 3
    lookup = {(r["offset"], r["format"], r["day"]): r for r in rows}
    plan = {
        "posthoc": True,
        "selection_role": "None; do not reopen the failed gate",
        "source_sha256": digest(__file__),
        "screen_sha256": digest(out / "result.json"),
        "backends": ["sdpa", "eager"],
        "rows": [r["id"] for r in chosen],
        "patch_layers": [13, 14, 17],
        "donor_weekday_shift": 3,
        "hypothesis": "A source-only block-output edit at or after the last KV writer cannot affect other token positions when all subsequent attention layers reuse earlier KV and remaining operations are tokenwise.",
        "checks": "Check nonzero source-state replacement and unchanged other positions. Require bitwise equality of all next-token logits after edits at layers 14 and 17, under both backends. Test mixed-length padding against an explicit causal mask and single prompts. These checks diagnose implementation and topology, not new semantic features.",
    }
    save(out / "audit-freeze.json", plan)
    tok, model = tokenizer(), load()
    writers = {
        layer.self_attn.layer_type: i
        for i, layer in enumerate(text_layers(model))
        if layer.self_attn.store_full_length_kv
    }
    topology = [
        {
            "layer": i,
            "type": layer.self_attn.layer_type,
            "shared": layer.self_attn.is_kv_shared_layer,
            "kv_source": writers[layer.self_attn.layer_type]
            if layer.self_attn.is_kv_shared_layer
            else None,
            "stores_kv": layer.self_attn.store_full_length_kv,
        }
        for i, layer in enumerate(text_layers(model))
    ]
    source_path = inspect.getfile(type(text_layers(model)[0].self_attn))
    assert {r["kv_source"] for r in topology if r["shared"]} == {13, 14}
    assert all(r["shared"] for r in topology[15:])
    records, padding, references = [], [], {}

    def forward(row, patch=None):
        assert time.perf_counter() < deadline
        inputs, site = batch_inputs([row], tok)
        states, edits, handles = {}, {}, []
        for layer in plan["patch_layers"]:

            def hook(module, args, hidden, layer=layer):
                if patch is not None and patch[0] == layer:
                    updated = hidden.clone()
                    updated[0, site[0]] = patch[1].to(hidden.dtype)
                    change = updated[0, site[0]].float() - hidden[0, site[0]].float()
                    assert change.norm() > 0
                    assert torch.equal(updated[:, : site[0]], hidden[:, : site[0]])
                    assert torch.equal(updated[:, site[0] + 1 :], hidden[:, site[0] + 1 :])
                    edits["relative_norm"] = float(
                        change.norm() / hidden[0, site[0]].float().norm()
                    )
                    hidden = updated
                states[layer] = hidden[0, site[0]].clone()
                return hidden

            handles.append(text_layers(model)[layer].register_forward_hook(hook))
        try:
            values = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        finally:
            for handle in handles:
                handle.remove()
        assert torch.isfinite(values).all()
        return values, states, edits

    with torch.inference_mode():
        for backend in plan["backends"]:
            model.set_attn_implementation(backend)
            for row in chosen:
                baseline, _, _ = forward(row)
                references[(backend, row["id"])] = baseline
                donor = lookup[(row["offset"], row["format"], 3)]
                _, states, _ = forward(donor)
                for layer in plan["patch_layers"]:
                    edited, _, info = forward(row, (layer, states[layer]))
                    equal = torch.equal(baseline, edited)
                    if layer >= 14:
                        assert equal, "Shared-KV no-path prediction failed"
                    records.append(
                        {
                            "backend": backend,
                            "id": row["id"],
                            "layer": layer,
                            "baseline_token": tok.decode([int(baseline.argmax())]),
                            "patched_token": tok.decode([int(edited.argmax())]),
                            "all_logits_bitwise_equal": equal,
                            "max_logit_change": float((edited - baseline).abs().max()),
                            **info,
                        }
                    )
            save(out / "audit-progress.json", {"source_edits": records, "padding": padding})
            pair = chosen[:2]
            length = max(len(r["ids"]) for r in pair)
            assert length < model.config.text_config.sliding_window
            ids = torch.tensor(
                [[tok.pad_token_id] * (length - len(r["ids"])) + r["ids"] for r in pair],
                device="cuda",
            )
            mask = ids != tok.pad_token_id
            positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
            allowed = (
                torch.ones(length, length, device="cuda", dtype=torch.bool).tril()[None, None]
                & mask[:, None, None]
            )
            # Give padding queries a finite self-attention row. Real queries still
            # cannot read padding keys; this avoids NaNs from an all-masked row.
            allowed |= (~mask)[:, None, :, None] & torch.eye(
                length, device="cuda", dtype=torch.bool
            )[None, None]
            additive = torch.zeros(allowed.shape, device="cuda", dtype=torch.bfloat16).masked_fill(
                ~allowed, float("-inf")
            )
            for kind, attention in [
                ("native_2d", mask.long()),
                ("explicit_causal_4d", {"full_attention": additive, "sliding_attention": additive}),
            ]:
                values = (
                    model(
                        input_ids=ids,
                        attention_mask=attention,
                        position_ids=positions,
                        use_cache=False,
                        logits_to_keep=1,
                    )
                    .logits[:, -1]
                    .float()
                )
                assert torch.isfinite(values).all(), f"Nonfinite {backend}/{kind} padded logits"
                for row, logits in zip(pair, values):
                    single = references[(backend, row["id"])]
                    padding.append(
                        {
                            "backend": backend,
                            "kind": kind,
                            "id": row["id"],
                            "single_token": tok.decode([int(single.argmax())]),
                            "batch_token": tok.decode([int(logits.argmax())]),
                            "max_logit_difference": float((single - logits).abs().max()),
                        }
                    )
                save(out / "audit-progress.json", {"source_edits": records, "padding": padding})
        comparisons = [
            {
                "id": row["id"],
                "sdpa_token": tok.decode([int(references[("sdpa", row["id"])].argmax())]),
                "eager_token": tok.decode([int(references[("eager", row["id"])].argmax())]),
                "max_logit_difference": float(
                    (references[("sdpa", row["id"])] - references[("eager", row["id"])]).abs().max()
                ),
            }
            for row in chosen
        ]
        torch.cuda.synchronize()
    result = {
        "plan": plan,
        "topology": topology,
        "topology_source_sha256": digest(source_path),
        "topology_source": source_path,
        "source_edits": records,
        "padding": padding,
        "backend_comparisons": comparisons,
        "passed_no_path_checks": True,
        "gpu_hours": (time.perf_counter() - started) / 3600,
        "previous_audit_attempts": failed,
        "total_study_gpu_hours": prior + (time.perf_counter() - started) / 3600,
    }
    save(out / "audit.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
