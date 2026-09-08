"""Prospective semantic-swap feasibility gate, before fitting any EML equations."""

import json
import statistics

import torch
from lens import HERE, RUN, SPEC, capture, model, save_json, sha, tokenizer, verify


@torch.inference_mode()
def main():
    verify()
    assert json.loads((RUN / "audit.json").read_text())["passed"]
    assert json.loads((RUN / "audit-fp32.json").read_text())["passed"]
    calibration = json.loads((RUN / "calibration.json").read_text())
    assert calibration["complete"]
    assert calibration["calibration_sha256"] == sha(RUN / "calibration.pt")
    target = RUN / "discovery.json"
    assert not target.exists(), "Refusing to replace observed feasibility results"
    save_json(
        RUN / "probe-freeze.json",
        {
            "source_sha256": sha(HERE / "probe.py"),
            "calibration_sha256": sha(RUN / "calibration.pt"),
            "protocol_sha256": sha(HERE / "protocol.json"),
            "audit_sha256": sha(RUN / "audit.json"),
            "audit_fp32_sha256": sha(RUN / "audit-fp32.json"),
        },
    )
    native, tok = model(), tokenizer()
    vectors = (
        torch.stack(torch.load(RUN / "calibration.pt", weights_only=True)["per_document"])
        .mean(0)
        .cuda()
    )
    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    concepts = list(SPEC["concepts"])
    digit_ids = [tok.encode(str(i), add_special_tokens=False) for i in range(10)]
    assert all(len(d) == 1 for d in digit_ids)
    digit_ids = [d[0] for d in digit_ids]
    rows = []
    for pair_index, (source, destination) in enumerate(SPEC["discovery_pairs"]):
        for format_index, template in enumerate(SPEC["discovery_formats"]):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": template.format(animal=source)}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)], device="cuda")
            baseline = (
                native(input_ids=ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            )
            source_answer, destination_answer = (
                SPEC["concepts"][source],
                SPEC["concepts"][destination],
            )
            baseline_margin = float(
                baseline[digit_ids[destination_answer]] - baseline[digit_ids[source_answer]]
            )
            for layer_index, layer in enumerate(SPEC["layers"]):
                pair = vectors[layer_index, [concepts.index(source), concepts.index(destination)]].T
                inverse = torch.linalg.pinv(pair)
                generator = torch.Generator(device="cuda").manual_seed(
                    91000 + 100 * pair_index + layer
                )
                random_direction = torch.randn(1536, generator=generator, device="cuda")
                random_direction -= pair @ (inverse @ random_direction)
                random_direction = torch.nn.functional.normalize(random_direction, dim=0)
                for alpha in SPEC["discovery_alphas"]:
                    for control in ["semantic"] if alpha == 0 else ["semantic", "random"]:
                        diagnostics = {}

                        def edit(hidden):
                            if alpha == 0:
                                return hidden
                            coordinates = hidden.float() @ inverse.T
                            delta = alpha * ((coordinates.flip(-1) - coordinates) @ pair.T)
                            if control == "random":
                                delta = delta.norm(dim=-1, keepdim=True) * random_direction
                            diagnostics["mean_relative_perturbation_norm"] = float(
                                (delta.norm(dim=-1) / hidden.float().norm(dim=-1)).mean()
                            )
                            diagnostics["last_position_coordinates"] = coordinates[0, -1].tolist()
                            return (hidden.float() + delta).to(hidden.dtype)

                        with capture(native, [layer], intervention=(layer, edit)):
                            output = (
                                native(input_ids=ids, use_cache=False, logits_to_keep=1)
                                .logits[0, -1]
                                .float()
                            )
                        if alpha == 0:
                            assert torch.equal(baseline, output)
                        margin = float(
                            output[digit_ids[destination_answer]] - output[digit_ids[source_answer]]
                        )
                        row = {
                            "source": source,
                            "destination": destination,
                            "format": format_index,
                            "layer": layer,
                            "alpha": alpha,
                            "control": control,
                            "baseline_top_token": tok.decode([int(baseline.argmax())]),
                            "baseline_correct": int(baseline.argmax()) == digit_ids[source_answer],
                            "top_token": tok.decode([int(output.argmax())]),
                            "target_correct": int(output.argmax()) == digit_ids[destination_answer],
                            "baseline_margin": baseline_margin,
                            "margin_change": margin - baseline_margin,
                            **diagnostics,
                        }
                        rows.append(row)
            print(
                json.dumps(
                    {
                        "pair": pair_index,
                        "format": format_index,
                        "baseline": tok.decode([int(baseline.argmax())]),
                    }
                ),
                flush=True,
            )
            save_json(RUN / "discovery-progress.json", rows)
    save_json(target, rows)
    summary = {}
    gate = SPEC["feasibility_gate"]
    for layer in SPEC["layers"]:
        semantic = [
            r
            for r in rows
            if r["layer"] == layer and r["alpha"] == 1 and r["control"] == "semantic"
        ]
        control = [
            r for r in rows if r["layer"] == layer and r["alpha"] == 1 and r["control"] == "random"
        ]
        result = {
            "median_split_half_direction_cosine": statistics.median(
                calibration["split_half_cosines"][str(layer)].values()
            ),
            "baseline_accuracy": statistics.mean(r["baseline_correct"] for r in semantic),
            "positive_target_margin_change_fraction": statistics.mean(
                r["margin_change"] > 0 for r in semantic
            ),
            "target_argmax_fraction": statistics.mean(r["target_correct"] for r in semantic),
            "semantic_mean_margin_change": statistics.mean(r["margin_change"] for r in semantic),
            "random_mean_margin_change": statistics.mean(r["margin_change"] for r in control),
        }
        result["passed"] = (
            result["median_split_half_direction_cosine"]
            >= gate["median_split_half_direction_cosine_min"]
            and result["baseline_accuracy"] >= gate["native_baseline_answer_accuracy_min"]
            and result["positive_target_margin_change_fraction"]
            >= gate["positive_target_logit_margin_change_fraction_min"]
            and result["target_argmax_fraction"] >= gate["target_digit_argmax_fraction_min"]
            and result["semantic_mean_margin_change"] > result["random_mean_margin_change"]
        )
        summary[str(layer)] = result
    passing = [int(layer) for layer, row in summary.items() if row["passed"]]
    selected = (
        max(passing, key=lambda layer: summary[str(layer)]["semantic_mean_margin_change"])
        if passing
        else None
    )
    report = {
        "layers": summary,
        "selected_layer": selected,
        "next_step": "freeze equation comparison"
        if passing
        else "stop: no validated semantic interface",
        "discovery_sha256": sha(target),
    }
    save_json(RUN / "feasibility.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
