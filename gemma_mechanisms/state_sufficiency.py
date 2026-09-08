"""Test whether reused KV context explains failures of single-position equations."""

import argparse
import json
import time

import torch

from .prepare import content
from .runtime import HERE, digest, load, prompt, root, save, setup, text_layers, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["selection", "gate", "shift"], default="selection")
    parser.add_argument("--groups", type=int, default=64)
    parser.add_argument("--new-formats", action="store_true")
    args = parser.parse_args()
    setup(73)
    out = root()
    if args.split != "selection":
        assert (out / "training/selection.json").exists(), (
            "Freeze replacements before exposing shared gate operands"
        )
    pairs = json.loads((out / "residual/mechanism/donor-pairs.json").read_text())[args.split]
    eligible = [r for r in pairs if r["eligible"]][: args.groups]
    assert len(eligible) == args.groups
    styles = ["reversed", "code"] if args.new_formats else ["symbolic", "prose"]
    rows = [{**row, "style": style} for row in eligible for style in styles]
    directory = out / "state-sufficiency"
    directory.mkdir(exist_ok=True)
    name = f"{args.split}-{args.groups}-{'new' if args.new_formats else 'known'}"
    freeze = {
        "source_sha256": digest(HERE / "state_sufficiency.py"),
        "split": args.split,
        "groups": args.groups,
        "styles": styles,
        "source_layer": 26,
        "target_layer": 32,
        "variants": [
            "recipient",
            "donor",
            "same_carry",
            "hidden_only",
            "kv_only",
            "both",
            "sliding_kv",
            "global_kv",
            "same_carry_kv",
        ],
        "prediction": "Exactly matching the last-token residual and both reused KV tensors should restore donor logits; matching residual alone may not.",
        "pair_source_sha256": digest(out / "residual/mechanism/donor-pairs.json"),
    }
    frozen = directory / f"{name}-freeze.json"
    if frozen.exists():
        assert json.loads(frozen.read_text()) == freeze
    else:
        save(frozen, freeze)
    path = directory / f"{name}.json"
    if path.exists():
        assert json.loads(path.read_text())["freeze_sha256"] == digest(frozen)
        return
    model, tok = load(), tokenizer()
    layers = text_layers(model)
    records = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(rows), 8):
            batch = rows[start : start + 8]
            strings = {k: [] for k in ["recipient", "donor", "same_carry"]}
            target_tokens = []
            for row in batch:
                recipient = row["recipient"]
                answer_prefix = str(recipient["answer"])[:-2]
                for kind, a in [
                    ("recipient", recipient["a"]),
                    ("donor", row["donor_a"]),
                    ("same_carry", row["same_carry_a"]),
                ]:
                    changed = {**recipient, "a": a}
                    strings[kind].append(
                        prompt(tok, content(changed, row["style"])) + answer_prefix
                    )
                ids = tok.encode(
                    str((row["donor_a"] + recipient["b"]) // 10 % 10), add_special_tokens=False
                )
                assert len(ids) == 1
                target_tokens.append(ids[0])
            inputs = {
                k: tok(v, padding=True, return_tensors="pt", add_special_tokens=False).to("cuda")
                for k, v in strings.items()
            }
            assert all(
                torch.equal(inputs["recipient"].attention_mask, x.attention_mask)
                for x in inputs.values()
            )

            def run(inp, hidden=None, kv=None):
                captured = {}

                def source_hook(module, args, kwargs):
                    state = args[0]
                    if hidden is not None:
                        state = state.clone()
                        state[:, -1] = hidden
                        assert torch.equal(state[:, -1], hidden)
                    shared = kwargs["shared_kv_states"]
                    if kv is not None:
                        for key, tensors in kv.items():
                            shared[key] = tuple(t.clone() for t in tensors)
                    captured["hidden"] = state[:, -1].clone()
                    captured["kv"] = {
                        key: tuple(t.clone() for t in tensors) for key, tensors in shared.items()
                    }
                    return (state,) + args[1:], kwargs

                def target_hook(module, args):
                    captured["target"] = args[0][:, -1].clone()

                handles = [
                    layers[26].register_forward_pre_hook(source_hook, with_kwargs=True),
                    layers[32].register_forward_pre_hook(target_hook),
                ]
                try:
                    logits = model(**inp, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                finally:
                    for handle in handles:
                        handle.remove()
                return logits, captured

            native = {kind: run(inp) for kind, inp in inputs.items()}
            donor_logits, donor = native["donor"]
            assert set(donor["kv"]) == {"sliding_attention", "full_attention"}
            variants = dict(native)
            variants["hidden_only"] = run(inputs["recipient"], hidden=donor["hidden"])
            variants["kv_only"] = run(inputs["recipient"], kv=donor["kv"])
            variants["both"] = run(inputs["recipient"], hidden=donor["hidden"], kv=donor["kv"])
            variants["sliding_kv"] = run(
                inputs["recipient"], kv={"sliding_attention": donor["kv"]["sliding_attention"]}
            )
            variants["global_kv"] = run(
                inputs["recipient"], kv={"full_attention": donor["kv"]["full_attention"]}
            )
            variants["same_carry_kv"] = run(inputs["recipient"], kv=native["same_carry"][1]["kv"])
            assert torch.equal(variants["hidden_only"][1]["hidden"], donor["hidden"])
            assert torch.equal(variants["both"][1]["hidden"], donor["hidden"])
            # Record failures of the prospective sufficiency prediction; do not manufacture equality.
            targets = torch.tensor(target_tokens, device="cuda")
            idx = torch.arange(len(batch), device="cuda")
            for kind, (logits, state) in variants.items():
                same_logits = (logits == donor_logits).all(-1)
                hidden_error = (state["target"].float() - donor["target"].float()).square().mean(-1)
                for i, row in enumerate(batch):
                    records.append(
                        {
                            "id": row["recipient"]["id"],
                            "style": row["style"],
                            "kind": kind,
                            "target_digit_correct": bool(logits[i].argmax() == targets[i]),
                            "donor_top_token_match": bool(
                                logits[i].argmax() == donor_logits[i].argmax()
                            ),
                            "all_logits_bitwise_equal_to_donor": bool(same_logits[i]),
                            "max_logit_error": float((logits[i] - donor_logits[i]).abs().max()),
                            "downstream_state_mse_from_donor": float(hidden_error[i]),
                            "target_digit_log_probability": float(
                                logits.log_softmax(-1)[idx, targets][i]
                            ),
                            "source_hidden_bitwise_equal_to_donor": bool(
                                torch.equal(state["hidden"][i], donor["hidden"][i])
                            ),
                        }
                    )
            print("STATE SUFFICIENCY", min(start + len(batch), len(rows)), len(rows), flush=True)
    save(
        path,
        {
            "records": records,
            "freeze_sha256": digest(frozen),
            "all_original_groups": len(pairs),
            "eligible_original_groups": sum(r["eligible"] for r in pairs),
            "evaluated_groups": args.groups,
            "seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "scope": "Original-model causal routing under a fixed answer prefix; behavior fidelity is distinct from arithmetic correctness.",
        },
    )
    print("STATE SUFFICIENCY COMPLETE", flush=True)


if __name__ == "__main__":
    main()
