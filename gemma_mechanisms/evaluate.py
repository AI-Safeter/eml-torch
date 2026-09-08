"""Native, unconditional quality evaluation of physically installed MLP students."""

import argparse
import gc
import json
import re
import time

import torch

from .prepare import content
from .runtime import (
    HERE,
    SPEC,
    accounting,
    digest,
    load,
    prompt,
    root,
    save,
    setup,
    text_layers,
    tokenizer,
)
from .student import install, load_student


def roster(out):
    selection_path = out / "training/selection.json"
    selection = json.loads(selection_path.read_text())
    checkpoints = {
        name: sha
        for family in selection["families"].values()
        for name, sha in family.get("checkpoints", {}).items()
    }
    eml = selection["families"]["eml"]
    if "budget" in eml:
        for kind in ["silu", "linear"]:
            depth = eml["depth"] if kind == "silu" else 0
            for seed in SPEC["replacement"]["seeds"]:
                name = f"{kind}-b{eml['budget']}-d{depth}-s{seed}"
                record = json.loads((out / "training" / f"{name}.json").read_text())
                if record["status"] == "complete":
                    checkpoints[name] = record["checkpoint_sha256"]
    for name, sha in checkpoints.items():
        assert digest(out / "training" / f"{name}.pt") == sha
    return {"original": None, **dict(sorted(checkpoints.items()))}


def parse_answer(text):
    # Require an integer-only answer; prose, incomplete output, and refusals are errors.
    text = text.strip()
    return int(text.replace(",", "")) if re.fullmatch(r"-?\d+(?:,\d{3})*", text) else None


@torch.inference_mode()
def arithmetic(model, tok, rows, styles, batch_size=16):
    expanded = [(row, style) for row in rows for style in styles]
    records = []
    for start in range(0, len(expanded), batch_size):
        batch = expanded[start : start + batch_size]
        inp = tok(
            [prompt(tok, content(row, style)) for row, style in batch],
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")
        output = model.generate(
            **inp,
            do_sample=False,
            max_new_tokens=SPEC["arithmetic"]["max_new_tokens"],
            use_cache=True,
        )
        for (row, style), tokens in zip(batch, output[:, inp.input_ids.shape[1] :]):
            text = tok.decode(tokens, skip_special_tokens=True)
            answer = parse_answer(text)
            records.append(
                {
                    "id": row["id"],
                    "op": row["op"],
                    "style": style,
                    "a": row["a"],
                    "b": row["b"],
                    "answer": row["answer"],
                    "prediction": answer,
                    "text": text,
                    "correct": answer == row["answer"],
                    "variables": row["variables"],
                }
            )
        if start % 256 == 0:
            print("ARITHMETIC", start, len(expanded), flush=True)
    return records


@torch.inference_mode()
def language(model, blocks):
    records = []
    for i, block in enumerate(blocks):
        ids = block[None].cuda()
        logits = model(input_ids=ids, use_cache=False).logits[:, :-1].float()
        ce = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), ids[:, 1:].flatten(), reduction="none"
        )
        assert torch.isfinite(ce).all()
        records.append({"document": i, "ce_nats": float(ce.mean()), "tokens": len(block) - 1})
        if i % 128 == 0:
            print("LANGUAGE", i, len(blocks), flush=True)
    return records


@torch.inference_mode()
def arc(model, tok, rows):
    records = []
    for i, row in enumerate(rows):
        question = prompt(tok, row["question"] + " Answer the question concisely.")
        prefix_ids = tok.encode(question, add_special_tokens=False)
        choices = row["choices"]
        inputs, answer_lengths = [], []
        for answer in choices["text"]:
            ids = tok.encode(question + answer, add_special_tokens=False)
            assert ids[: len(prefix_ids)] == prefix_ids
            inputs.append(ids)
            answer_lengths.append(len(ids) - len(prefix_ids))
        assert min(answer_lengths) > 0
        inp = tok.pad({"input_ids": inputs}, padding=True, return_tensors="pt").to("cuda")
        maximum = max(answer_lengths)
        logits = model(**inp, use_cache=False, logits_to_keep=maximum + 1).logits[:, :-1].float()
        targets = inp.input_ids[:, -maximum:]
        losses = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), reduction="none"
        ).view(len(inputs), -1)
        scores = [-float(loss[-n:].mean()) for loss, n in zip(losses, answer_lengths)]
        assert torch.isfinite(torch.tensor(scores)).all()
        predicted = max(range(len(scores)), key=scores.__getitem__)
        correct = choices["label"][predicted] == row["answerKey"]
        records.append(
            {
                "id": row["id"],
                "scores": scores,
                "prediction": choices["label"][predicted],
                "answer": row["answerKey"],
                "correct": correct,
                "answer_lengths": answer_lengths,
            }
        )
        if i % 128 == 0:
            print("ARC", i, len(rows), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--split", choices=["selection", "gate", "final", "shift"], default="gate")
    parser.add_argument("--method", help="Optional member of the frozen evaluation roster")
    args = parser.parse_args()
    setup(SPEC["data_seed"])
    out = root(args.output)
    methods = roster(out)
    directory = out / "evaluation" / args.split
    directory.mkdir(parents=True, exist_ok=True)
    if args.split in ["final", "shift"]:
        assert (out / "evaluation/gate-summary.json").exists(), (
            "Finish the frozen gate before final evaluation"
        )
    if args.method:
        assert args.method in methods
    selected = [args.method] if args.method else list(methods)
    freeze = {
        "sources": {
            p: digest(HERE / p)
            for p in [
                "evaluate.py",
                "runtime.py",
                "student.py",
                "prepare.py",
                "protocol.json",
                "PROTOCOL.md",
            ]
        },
        "selection_sha256": digest(out / "training/selection.json"),
        "checkpoints": methods,
        "data_freeze_sha256": digest(out / "data/freeze.json"),
        "scope": "Native BF16 model; CPU PLE lookup in every method; original MLP absent for students",
    }
    frozen_path = directory / "freeze.json"
    if frozen_path.exists():
        assert json.loads(frozen_path.read_text()) == freeze
    else:
        save(frozen_path, freeze)
    source_data = json.loads((out / "data/freeze.json").read_text())
    for file, sha in source_data["files"].items():
        assert digest(out / "data" / file) == sha
    rows = json.loads((out / "data" / f"arithmetic-{args.split}.json").read_text())
    model, tok = load(), tokenizer()
    layer = SPEC["replacement"]["layer"]
    original = text_layers(model)[layer].mlp
    original_forward = original.forward
    original_ids = {id(p) for p in original.parameters()}

    def forbidden(*a, **kw):
        raise AssertionError("Removed original MLP executed during student evaluation")

    for method in selected:
        result_path = directory / f"{method}.json"
        if result_path.exists():
            assert json.loads(result_path.read_text())["freeze_sha256"] == digest(frozen_path)
            print("PRESERVE EVALUATION", method, flush=True)
            continue
        if method == "original":
            text_layers(model)[layer].mlp = original.cuda()
            original.forward = original_forward
        else:
            original.cpu()
            original.forward = forbidden
            student = load_student(out / "training" / f"{method}.pt")
            removed = install(model, student, layer)
            del removed
            assert not original_ids.intersection(id(p) for p in model.parameters())
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        results = {"arithmetic": arithmetic(model, tok, rows, SPEC["arithmetic"]["formats"])}
        if args.split in ["final", "shift"]:
            results["new_formats"] = arithmetic(model, tok, rows, SPEC["arithmetic"]["new_formats"])
        if args.split != "shift":
            blocks = torch.load(out / "data" / f"language-{args.split}.pt", weights_only=True)
            results["language"] = language(model, blocks)
            results["arc"] = arc(
                model, tok, json.loads((out / "data" / f"arc-{args.split}.json").read_text())
            )
        torch.cuda.synchronize()
        save(
            result_path,
            {
                "method": method,
                "split": args.split,
                "results": results,
                "freeze_sha256": digest(frozen_path),
                "accounting": accounting(model),
                "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "original_mlp_executed": method == "original",
            },
        )
        print("EVALUATION COMPLETE", args.split, method, flush=True)


if __name__ == "__main__":
    main()
