"""Evaluate genuinely replaced whole blocks after freezing validation choices."""

import argparse
import hashlib
import json
import time
from contextlib import contextmanager

import torch
from robust_statistics import loss_bound
from runtime import HERE, RUNS, configure
from whole_student import Student


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(out):
    from data import save

    path = out / "selection.json"
    if path.exists():
        selected = json.loads(path.read_text())
        for row in selected["students"].values():
            if "checkpoint" in row:
                assert digest(out / row["checkpoint"]) == row["sha256"]
        return selected
    status = json.loads((out / "fits-completed.json").read_text())
    assert status["candidates"] == status["expected"] == 27
    rows = json.loads((out / "candidates.json").read_text())
    specs = {}
    for kind in ["eml", "swiglu", "linear"]:
        candidates = [r for r in rows if r["kind"] == kind and r["status"] == "complete"]
        if not candidates:
            specs[kind] = {"status": "no_completed_candidate"}
            continue
        best = min(candidates, key=lambda r: r["validation_objective"])
        checkpoint = best["name"] + ".pt"
        specs[kind] = {**best, "checkpoint": checkpoint, "sha256": digest(out / checkpoint)}
    collection = json.loads((out / "collection.json").read_text())
    selected = {
        "layer": collection["layer"],
        "model": collection["model"],
        "students": specs,
        "frozen_epoch": time.time(),
        "protocol_sha256": digest(HERE / "WHOLE_BLOCK_PROTOCOL.md"),
        "language_freeze_sha256": digest(HERE / "language-freeze.json"),
        "test_outputs_unseen_at_selection": True,
    }
    save(path, selected)
    return selected


def students(out, selected):
    models = {}
    for kind, spec in selected["students"].items():
        if "checkpoint" not in spec:
            continue
        state = torch.load(out / spec["checkpoint"], weights_only=True)
        stats = {k: state[k] for k in ["xmean", "xstd", "ymean", "yscale"]}
        student = Student(len(state["xmean"]), spec["width"], kind, stats).cuda()
        student.load_state_dict(state)
        models[kind] = student.eval().requires_grad_(False)
    return models


@contextmanager
def replace(model, layer, student, check_finite=True):
    original = model.model.layers[layer].mlp
    if student is None:
        yield
        return
    calls = []
    guard = original.register_forward_hook(lambda *args: calls.append(True))

    def finite(module, args, output):
        assert torch.isfinite(output).all(), "Nonfinite replacement output"

    finite_guard = student.register_forward_hook(finite) if check_finite else None
    model.model.layers[layer].mlp = student
    try:
        yield
        assert not calls, "Teacher MLP executed in replacement path"
    finally:
        model.model.layers[layer].mlp = original
        guard.remove()
        if finite_guard is not None:
            finite_guard.remove()


def language(model, blocks, batch_size=2):
    records = []
    for start in range(0, len(blocks), batch_size):
        ids = blocks[start : start + batch_size].cuda()
        scores = model(input_ids=ids, use_cache=False).logits[:, :-1, :].float()
        losses = torch.nn.functional.cross_entropy(
            scores.transpose(1, 2), ids[:, 1:], reduction="none"
        )
        assert torch.isfinite(losses).all()
        records.extend(losses.mean(1).cpu().tolist())
    return records


def arithmetic(model, tok, rows, batch_size=16):
    from evaluate_heads import generation, parse_answer
    from model_io import inputs

    records = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        _, texts = generation(model, tok, inputs(tok, batch))
        records.extend(
            {**r, "text": text, "correct": parse_answer(text) == r["answer"]}
            for r, text in zip(batch, texts)
        )
        if start % 256 == 0:
            print("WHOLE ARITHMETIC", start, len(rows), flush=True)
    return records


def summarize(results, parameter_counts):
    output = {}
    baseline = torch.tensor(results["original"]["language"], dtype=torch.float64, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(20260927)
    indices = torch.randint(
        len(baseline), (10000, len(baseline)), device="cuda", generator=generator
    )
    for kind, record in results.items():
        ce = torch.tensor(record["language"], dtype=torch.float64, device="cuda")
        delta = ce - baseline
        upper = float(torch.quantile(delta[indices].mean(1), 0.95))
        operations = {}
        for name, rows in record["arithmetic"].items():
            teacher = results["original"]["arithmetic"][name]
            assert [(r["a"], r["b"], r["style"]) for r in rows] == [
                (r["a"], r["b"], r["style"]) for r in teacher
            ]
            base = torch.tensor([r["correct"] for r in teacher], device="cuda").reshape(-1, 3)
            actual = torch.tensor([r["correct"] for r in rows], device="cuda").reshape(-1, 3)
            operations[name] = {
                "original_accuracy": float(base.float().mean()),
                "replacement_accuracy": float(actual.float().mean()),
                **loss_bound(base, actual, cells=3),
            }
        reduction = parameter_counts["original"] / parameter_counts[kind]
        checks = {
            "stored_coefficient_reduction_at_least_4": reduction >= 4,
            "language_upper_increase_at_most_point02_nats": upper <= 0.02,
            "all_unconditioned_accuracy_bounds_pass": all(
                r["passes_one_percentage_point_bound"]
                for name, r in operations.items()
                if name.endswith("/unconditioned")
            ),
        }
        output[kind] = {
            "language_cross_entropy": float(ce.mean()),
            "language_increase": float(delta.mean()),
            "language_increase_upper95": upper,
            "stored_coefficients": parameter_counts[kind],
            "block_storage_reduction": reduction,
            "arithmetic": operations,
            "checks": checks,
            "all_utility_checks_pass": all(checks.values()),
        }
    return output


def main():
    configure("qwen17b")
    from data import STYLES, save
    from model_io import expand, load, setup

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    setup()
    out = RUNS / "whole-block"
    selected = select(out)
    replacements = students(out, selected)
    model, tok = load()
    split = "validation" if args.validation_only else "test"
    blocks = torch.load(out / f"language-{split}.pt", weights_only=True)
    if args.validation_only:
        blocks = blocks[:4]
    frozen_language = json.loads((HERE / "language-freeze.json").read_text())
    assert digest(out / f"language-{split}.pt") == frozen_language["splits"][split]["file_sha256"]
    suites = {}
    for op in ["add", "multiply", "divide"]:
        problems = json.loads((HERE / "data" / op / "problems.json").read_text())
        suites[f"{op}/{split}"] = expand(
            problems[split][:4] if args.validation_only else problems[split], STYLES
        )
        if not args.validation_only:
            extra = json.loads((HERE / "data" / op / "extra-problems.json").read_text())
            suites[f"{op}/unconditioned"] = expand(extra["ordinary"], STYLES)
    results, counts = (
        {},
        {
            "original": sum(
                p.numel() for p in model.model.layers[selected["layer"]].mlp.parameters()
            )
        },
    )
    with torch.inference_mode():
        for kind, student in {"original": None, **replacements}.items():
            destination = out / f"evaluation-{split}-{kind}.json"
            if student is not None:
                counts[kind] = student.stored_coefficients()
            if destination.exists():
                results[kind] = json.loads(destination.read_text())
                continue
            with replace(model, selected["layer"], student):
                ce = language(model, blocks)
                answers = {name: arithmetic(model, tok, rows) for name, rows in suites.items()}
            result = {"language": ce, "arithmetic": answers}
            save(destination, result)
            results[kind] = result
        if args.validation_only:
            restored = language(model, blocks)
            assert restored == results["original"]["language"]
        else:
            save(out / "utility-results.json", summarize(results, counts))
    print("WHOLE EVALUATION COMPLETE", split, flush=True)


if __name__ == "__main__":
    main()
