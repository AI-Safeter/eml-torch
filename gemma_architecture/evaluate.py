"""Development and sealed confirmation evaluation of the installed complete replacement."""

import argparse
import json
import time

import torch

from gemma_mechanisms.evaluate import arc, arithmetic, language

from .common import (
    accounting,
    digest,
    freeze,
    load,
    root,
    save,
    setup,
    sources,
    text_layers,
    tokenizer,
)
from .model import load_replacement


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["development", "confirmation"], default="development")
    p.add_argument("--method", required=True)
    a = p.parse_args()
    setup(73)
    out = root()
    directory = out / "evaluation" / a.split
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = None if a.method == "original" else out / "training" / f"{a.method}.pt"
    selection = None
    if a.split == "confirmation":
        selection = json.loads((out / "selection.json").read_text())
        assert selection["confirmation_opened"] is False
        if checkpoint:
            assert selection["checkpoints"][a.method] == digest(checkpoint)
    data_freeze = json.loads((out / "data/freeze.json").read_text())
    for file, sha in data_freeze["files"].items():
        assert digest(out / "data" / file) == sha
    binding = {
        "sources": sources(
            "evaluate.py",
            "model.py",
            "statistics.py",
            "../gemma_mechanisms/evaluate.py",
            "../gemma_mechanisms/prepare.py",
            "../gemma_mechanisms/student.py",
        ),
        "data_freeze_sha256": digest(out / "data/freeze.json"),
        "checkpoint_sha256": digest(checkpoint) if checkpoint else None,
        "selection_sha256": digest(out / "selection.json") if selection else None,
        "split": a.split,
        "method": a.method,
    }
    frozen = freeze(directory / f"{a.method}-freeze.json", binding)
    path = directory / f"{a.method}.json"
    assert not path.exists()
    rows = json.loads((out / "data" / f"arithmetic-{a.split}.json").read_text())
    model, tok = load(), tokenizer()
    base_count = accounting(model)
    original = text_layers(model)[26].mlp
    original_ids = {id(p) for p in original.parameters()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Removed original MLP executed")

    if checkpoint:
        original.cpu()
        original.forward = forbidden
        student = load_replacement(checkpoint, deployed=True)
        text_layers(model)[26].mlp = student
        assert not original_ids.intersection(id(p) for p in model.parameters())

    def finite(module, inputs, output):
        assert torch.isfinite(output.logits).all()

    model.register_forward_hook(finite)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    styles = ["code", "reversed"] if a.split == "development" else ["code"]
    results = {
        "arithmetic": arithmetic(model, tok, rows["ordinary"], styles),
        "shift": arithmetic(model, tok, rows["shift"], ["code"]),
    }
    if a.split == "confirmation":
        diagnostic = []
        for op in ["add", "multiply", "divide"]:
            diagnostic += [r for r in rows["ordinary"] if r["op"] == op][:256]
        results["reversed"] = arithmetic(model, tok, diagnostic, ["reversed"])
    results["language"] = language(
        model, torch.load(out / "data" / f"language-{a.split}.pt", weights_only=True)
    )
    results["arc"] = arc(model, tok, json.loads((out / "data" / f"arc-{a.split}.json").read_text()))
    torch.cuda.synchronize()
    save(
        path,
        {
            "method": a.method,
            "split": a.split,
            "results": results,
            "freeze_sha256": frozen,
            "original_accounting": base_count,
            "accounting": accounting(model),
            "student_accounting": accounting(student) if checkpoint else None,
            "elapsed_seconds": time.perf_counter() - start,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "original_mlp_executed": checkpoint is None,
            "teacher_activation_inference_inputs": False,
            "nonfinite_outputs": 0,
        },
    )
    print("EVALUATION COMPLETE", a.split, a.method, time.perf_counter() - start, flush=True)


if __name__ == "__main__":
    main()
