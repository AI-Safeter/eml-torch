"""Donor carry swaps and mediator blocking in the original Gemma computation."""

import argparse
import json
import random
import time

import torch

from .prepare import content, key, variables
from .runtime import HERE, SPEC, digest, load, prompt, root, save, setup, text_layers, tokenizer


def prepare_pairs(out):
    path = out / "mechanism/donor-pairs.json"
    if path.exists():
        return json.loads(path.read_text())
    all_rows = {
        s: json.loads((out / "data" / f"arithmetic-{s}.json").read_text())
        for s in SPEC["arithmetic"]["counts_per_operation"]
    }
    used = {
        key("add", r["a"], r["b"]) for rows in all_rows.values() for r in rows if r["op"] == "add"
    }
    rng = random.Random(SPEC["data_seed"] + 813)
    pairs = {}
    for split in ["selection", "gate", "final", "shift"]:
        pairs[split] = []
        for row in all_rows[split]:
            if row["op"] != "add":
                continue
            a, b = row["a"], row["b"]
            carry = variables(a, b)["carry_units"]
            options = [
                c
                for c in range(a // 10 * 10, a // 10 * 10 + 10)
                if c != a and key("add", c, b) not in used
            ]
            flip = [c for c in options if variables(c, b)["carry_units"] != carry]
            same = [c for c in options if variables(c, b)["carry_units"] == carry]
            if not flip or not same:
                pairs[split].append(
                    {
                        "recipient": row,
                        "eligible": False,
                        "reason": "No unused carry-flip and same-carry donor with fixed upper digits",
                    }
                )
                continue
            donor = rng.choice(flip)
            control = min(same, key=lambda c: abs(abs(c - a) - abs(donor - a)))
            used.update([key("add", donor, b), key("add", control, b)])
            pairs[split].append(
                {
                    "recipient": row,
                    "eligible": True,
                    "donor_a": donor,
                    "same_carry_a": control,
                    "operand_delta": donor - a,
                    "control_operand_delta": control - a,
                }
            )
    save(path, pairs)
    return pairs


def directions(probe, layer):
    weight = probe["weight"].cuda().double()
    carry = weight[:, probe["blocks"]["carry_units"][0]]
    operands = weight[:, :40]
    u, s, _ = torch.linalg.svd(operands, full_matrices=False)
    u = u[:, s > s.max() * 1e-8]
    q = carry - u @ (u.T @ carry)
    assert q.norm() > carry.norm() * 1e-5, "Carry has no separable linear direction"
    q = q / q.norm()
    generator = torch.Generator(device="cuda").manual_seed(SPEC["data_seed"] + layer)
    random_q = torch.randn(len(q), device="cuda", generator=generator, dtype=torch.float64)
    random_q = random_q - u @ (u.T @ random_q)
    random_q = random_q - q * (q @ random_q)
    random_q /= random_q.norm()
    # Normalize leakage by the intended carry-score change, not arbitrary probe units.
    leakage = float((operands.T @ q).abs().max() / (carry @ q).abs())
    assert leakage < 1e-8
    return {
        "q": q.float(),
        "random": random_q.float(),
        "carry": carry.float(),
        "denominator": (carry @ q).float(),
        "bias": probe["bias"][probe["blocks"]["carry_units"][0]].cuda(),
        "weight": weight.float(),
        "operand_rank": u.shape[1],
        "construction_leakage": leakage,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument(
        "--split", choices=["selection", "gate", "final", "shift"], default="selection"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    assert args.batch_size >= 4 and args.batch_size % 2 == 0
    setup(SPEC["data_seed"])
    out = root(args.output)
    directory = out / "mechanism"
    assert (directory / "probes.json").exists()
    pairs = prepare_pairs(out)
    # Prefix 2 is immediately before the tens digit in the four-digit training answers.
    # Longer answers use the corresponding tens-digit position; no final model output is inspected here.
    prefix = 2
    if args.split != "selection":
        raise RuntimeError(
            "Confirmatory interventions require a frozen equation/selection artifact; only selection is enabled"
        )
    candidate_layers = SPEC["mechanism"]["candidate_layers"]
    source_layers = candidate_layers[:-1]
    probe_paths = {
        layer: directory / f"layer{layer}-prefix{prefix}.pt" for layer in candidate_layers
    }
    probes = {layer: torch.load(path, weights_only=True) for layer, path in probe_paths.items()}
    coords = {layer: directions(probe, layer) for layer, probe in probes.items()}
    frozen = {
        "sources": {
            p: digest(HERE / p)
            for p in ["causal.py", "runtime.py", "prepare.py", "protocol.json", "PROTOCOL.md"]
        },
        "donor_pairs_sha256": digest(directory / "donor-pairs.json"),
        "probes": {str(layer): digest(path) for layer, path in probe_paths.items()},
        "answer_prefix_tokens": prefix,
        "batch_size": args.batch_size,
        "construction": {
            str(layer): {k: coords[layer][k] for k in ["operand_rank", "construction_leakage"]}
            for layer in candidate_layers
        },
        "scope": "Selection only; original MLPs execute. Probe directions orthogonalized against operand digit readouts.",
    }
    freeze_path = directory / f"causal-{args.split}-freeze.json"
    if freeze_path.exists():
        assert json.loads(freeze_path.read_text()) == frozen
    else:
        save(freeze_path, frozen)
    result_path = directory / f"causal-{args.split}.json"
    if result_path.exists():
        print("PRESERVE COMPLETE CAUSAL COHORT", flush=True)
        return
    model, tok = load(), tokenizer()
    layers = text_layers(model)
    rows = [
        {**r, "style": style}
        for r in pairs[args.split]
        if r["eligible"]
        for style in SPEC["arithmetic"]["formats"]
    ]
    digit_ids = [tok.encode(str(d), add_special_tokens=False) for d in range(10)]
    assert all(len(ids) == 1 for ids in digit_ids)
    digit_ids = torch.tensor([ids[0] for ids in digit_ids], device="cuda")
    records = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            texts = {kind: [] for kind in ["recipient", "donor", "same"]}
            target_ids, original_ids = [], []
            for row in batch:
                recipient = row["recipient"]
                answer_prefix = str(recipient["answer"])[:prefix]
                for kind, a in [
                    ("recipient", recipient["a"]),
                    ("donor", row["donor_a"]),
                    ("same", row["same_carry_a"]),
                ]:
                    changed = {**recipient, "a": a, "answer": a + recipient["b"]}
                    texts[kind].append(prompt(tok, content(changed, row["style"])) + answer_prefix)
                target_ids.append((row["donor_a"] + recipient["b"]) // 10 % 10)
                original_ids.append(recipient["answer"] // 10 % 10)
            target_tokens, base_tokens = digit_ids[target_ids], digit_ids[original_ids]
            inputs = {
                k: tok(v, padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
                for k, v in texts.items()
            }

            def run(inp, edit=None, block=None):
                states, hooks = {}, []
                for layer in candidate_layers:

                    def prehook(module, values, layer=layer):
                        x = values[0]
                        if edit is not None and layer == edit[0]:
                            x = x.clone()
                            x[:, -1] = (x[:, -1].float() + edit[1]).to(x.dtype)
                        if block is not None and layer == block[0]:
                            c = coords[layer]
                            delta_score = (baseline_states[layer].float() - x[:, -1].float()) @ c[
                                "carry"
                            ]
                            direction = c["q"] if block[1] == "carry" else c["random"]
                            delta = delta_score[:, None] / c["denominator"] * direction
                            x = x.clone()
                            x[:, -1] = (x[:, -1].float() + delta).to(x.dtype)
                        states[layer] = x[:, -1].clone()
                        return (x,) + values[1:]

                    hooks.append(layers[layer].mlp.register_forward_pre_hook(prehook))
                try:
                    logits = model(**inp, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                finally:
                    for h in hooks:
                        h.remove()
                idx = torch.arange(len(batch), device="cuda")
                return {
                    "margin": logits[idx, target_tokens] - logits[idx, base_tokens],
                    "target_probability": logits.softmax(-1)[idx, target_tokens],
                    "top_token": logits.argmax(-1),
                    "target_correct": logits.argmax(-1) == target_tokens,
                    "base_correct": logits.argmax(-1) == base_tokens,
                }, states

            baseline, baseline_states = run(inputs["recipient"])
            donor, donor_states = run(inputs["donor"])
            same, same_states = run(inputs["same"])

            def record(kind, result, states, source=None, target=None, strength=None):
                carry_values = {
                    str(layer): (
                        states[layer].float() @ coords[layer]["carry"] + coords[layer]["bias"]
                    ).tolist()
                    for layer in candidate_layers
                }
                converted = {k: v.tolist() for k, v in result.items()}
                quantities = {}
                for layer in candidate_layers:
                    prediction = (
                        states[layer].float() @ coords[layer]["weight"]
                        + probes[layer]["bias"].cuda()
                    )
                    columns = []
                    for name in SPEC["mechanism"]["candidate_variables"]:
                        lo, hi = probes[layer]["blocks"][name]
                        value = prediction[:, lo:hi]
                        columns.append(
                            value[:, 0]
                            if hi - lo == 1
                            else value @ torch.arange(10, device="cuda", dtype=torch.float32)
                        )
                    quantities[str(layer)] = torch.stack(columns, -1).tolist()
                diagnostics = {}
                if source is not None:
                    delta = states[source].float() - baseline_states[source].float()
                    diagnostics = {
                        "actual_edit_norm": delta.norm(dim=-1).tolist(),
                        "relative_edit_norm": (
                            delta.norm(dim=-1)
                            / baseline_states[source].float().norm(dim=-1).clamp_min(1e-12)
                        ).tolist(),
                        "operand_readout_change": (delta @ coords[source]["weight"][:, :40])
                        .abs()
                        .amax(-1)
                        .tolist(),
                    }
                for i, row in enumerate(batch):
                    records.append(
                        {
                            "id": row["recipient"]["id"],
                            "style": row["style"],
                            "kind": kind,
                            "source": source,
                            "target": target,
                            "strength": strength,
                            "carry": {layer: values[i] for layer, values in carry_values.items()},
                            "quantities": {
                                layer: values[i] for layer, values in quantities.items()
                            },
                            **{k: v[i] for k, v in diagnostics.items()},
                            **{k: v[i] for k, v in converted.items()},
                        }
                    )

            record("recipient", baseline, baseline_states)
            record("natural_donor", donor, donor_states)
            record("natural_same_carry", same, same_states)
            for source in source_layers:
                c = coords[source]
                difference = donor_states[source].float() - baseline_states[source].float()
                amplitude = (difference @ c["carry"]) / c["denominator"]
                same_amplitude = (
                    (same_states[source].float() - baseline_states[source].float()) @ c["carry"]
                ) / c["denominator"]
                for strength in SPEC["mechanism"]["intervention_strengths"]:
                    for kind, direction in [("carry_swap", c["q"]), ("random_swap", c["random"])]:
                        delta = strength * amplitude[:, None] * direction
                        result, states = run(inputs["recipient"], (source, delta))
                        if strength == 0:
                            assert all(torch.equal(result[k], baseline[k]) for k in result)
                        record(kind, result, states, source=source, strength=strength)
                for kind, delta in [
                    ("same_carry_swap", same_amplitude[:, None] * c["q"]),
                    ("full_donor_swap", difference),
                    # Adjacent rows are two formats of one operand group.
                    ("shuffled_donor_swap", amplitude.roll(2)[:, None] * c["q"]),
                ]:
                    result, states = run(inputs["recipient"], (source, delta))
                    record(kind, result, states, source=source, strength=1.0)
                for target in candidate_layers:
                    if target <= source:
                        continue
                    for block_kind in ["carry", "random"]:
                        result, states = run(
                            inputs["recipient"],
                            (source, amplitude[:, None] * c["q"]),
                            (target, block_kind),
                        )
                        record(
                            "block_" + block_kind,
                            result,
                            states,
                            source=source,
                            target=target,
                            strength=1.0,
                        )
            print("CAUSAL SELECTION", min(start + len(batch), len(rows)), len(rows), flush=True)
    assert all(
        torch.isfinite(
            torch.tensor([r["margin"], r["target_probability"], *r["carry"].values()])
        ).all()
        for r in records
    )
    save(
        result_path,
        {
            "records": records,
            "cohort": pairs[args.split],
            "eligible_groups": sum(r["eligible"] for r in pairs[args.split]),
            "all_groups": len(pairs[args.split]),
            "formats": SPEC["arithmetic"]["formats"],
            "freeze_sha256": digest(freeze_path),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "scope": "Teacher-forced tens digit, selection only. Complete generation and held-out mediation remain untested.",
        },
    )
    print("CAUSAL SELECTION COMPLETE", len(records), flush=True)


if __name__ == "__main__":
    main()
