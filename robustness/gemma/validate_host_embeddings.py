"""GPU replay against native FP32 references before enabling CPU embedding lookup."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma import adapter
from gemma.host_embeddings import load


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    adapter.install()
    import evaluate_heads
    from data import ROOT, STYLES, save
    from model_io import capture, corrupt_row, expand, inputs, setup
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding

    setup()
    with torch.inference_mode(False):
        model, tok = load()
    table = model.native.model.language_model.embed_tokens_per_layer
    assert table.weight.device.type == "cpu"
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == 5104297504
    generator = torch.Generator().manual_seed(819)
    unique = torch.cat(
        (
            torch.tensor([0, table.weight.shape[0] - 1]),
            torch.randint(table.weight.shape[0], (254,), generator=generator),
        )
    ).unique()
    reference = (
        Gemma4TextScaledWordEmbedding(
            len(unique), table.weight.shape[1], None, table.embedding.scalar_embed_scale
        )
        .float()
        .cuda()
    )
    reference.weight.copy_(table.weight[unique])
    mapped = torch.randint(len(unique), (4, 128), generator=generator).cuda()
    torch.testing.assert_close(table(unique.cuda()[mapped]), reference(mapped), rtol=0, atol=0)
    del reference, mapped
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "model": adapter.SPEC,
        "host_table_bytes": table.weight.numel() * table.weight.element_size(),
        "gpu_parameter_bytes": sum(
            p.numel() * p.element_size() for p in model.parameters() if p.is_cuda
        ),
        "registered_parameter_count_unchanged": True,
        "sampled_native_gpu_lookup_bitwise_equal": True,
        "source_hashes": {
            name: digest(adapter.HERE / name)
            for name in ["host_embeddings.py", "validate_host_embeddings.py"]
        },
        "input_checks": [],
        "generation_checks": [],
        "cohort_checks": [],
        "native_gpu_lookup_logit_checks": [],
    }

    class NativeRowLookup(torch.nn.Module):
        """Run the original CUDA embedding kernel on needed vocabulary rows."""

        def forward(self, ids):
            unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
            with torch.device("meta"):
                reference = Gemma4TextScaledWordEmbedding(
                    len(unique), table.weight.shape[1], None, table.embedding.scalar_embed_scale
                )
            reference.weight = torch.nn.Parameter(
                table.weight[unique.cpu()].to(ids.device), requires_grad=False
            )
            reference.embed_scale = table.embedding.embed_scale.to(ids.device)
            return reference(inverse)

    def compare_logits(cases):
        inp = inputs(tok, cases)
        native = model.native
        actual = native(**inp, use_cache=False, logits_to_keep=1).logits
        native.model.language_model.embed_tokens_per_layer = NativeRowLookup()
        try:
            reference = native(**inp, use_cache=False, logits_to_keep=1).logits
        finally:
            native.model.language_model.embed_tokens_per_layer = table
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        return actual.numel()

    previous = json.loads((adapter.HERE / "adapter-validation.json").read_text())
    for operation in ["add", "multiply", "divide"]:
        out = ROOT / operation
        problems = json.loads((out / "problems.json").read_text())
        layer = torch.load(out / "component.pt", weights_only=True)["layer"]
        for split in ["train", "validation"]:
            rows = expand(problems[split][:32], STYLES)
            expected = torch.load(out / f"inputs-{split}.pt", weights_only=True)["input"][
                : len(rows)
            ].cuda()
            for index, cases in enumerate([rows, [corrupt_row(row) for row in rows]]):
                actual, _ = capture(model, tok, cases, [layer], keep_scores=False)
                torch.testing.assert_close(
                    actual[layer]["input"].cuda(), expected[:, index], rtol=0, atol=0
                )
            report["input_checks"].append(
                {
                    "operation": operation,
                    "split": split,
                    "layer": layer,
                    "cases": len(rows) * 2,
                    "bitwise_equal": True,
                    "reference_sha256": digest(out / f"inputs-{split}.pt"),
                }
            )
        _, generated = evaluate_heads.generation(model, tok, inputs(tok, problems["train"][:4]))
        reference = next(
            row for row in previous["training_checks"] if row["operation"] == operation
        )
        assert generated == reference["generated"]
        report["generation_checks"].append(
            {"operation": operation, "generated": generated, "identical": True}
        )
        count = compare_logits(expand(problems["validation"][:4], STYLES))
        report["native_gpu_lookup_logit_checks"].append(
            {"operation": operation, "logits_compared": count, "bitwise_equal": True}
        )
        print("HOST LOOKUP INPUT AND GENERATION REPLAY", operation, flush=True)

    out = ROOT / "add"
    selected = json.loads((out / "selection.json").read_text())
    with torch.inference_mode(False):
        rep = evaluate_heads.Replacements(out, selected)
    rows = expand(json.loads((out / "problems.json").read_text())["validation"][:4], STYLES)
    for name, function, kwargs in [
        ("ordinary-validation-audit.json", evaluate_heads.ordinary, {}),
        ("interventions-validation-audit.json", evaluate_heads.interventions, {}),
        ("ordinary-all-tokens-validation.json", evaluate_heads.ordinary, {"all_tokens": True}),
        ("coordinate-validation.json", evaluate_heads.coordinate_interventions, {}),
    ]:
        actual = function(model, tok, rep, rows, **kwargs)
        reference = json.loads((out / name).read_text())
        raw = args.output.with_suffix("")
        raw.mkdir(exist_ok=True)
        save(raw / name, actual)
        differences = []
        if actual != reference:
            differences = [
                {
                    "method": method,
                    "row": index,
                    "field": key,
                    "actual": value,
                    "reference": expected[key],
                }
                for method in actual
                for index, (observed, expected) in enumerate(zip(actual[method], reference[method]))
                for key, value in observed.items()
                if value != expected[key]
            ]
            save(raw / "differences.json", differences)
            print("REPLAY DIFFERENCES", json.dumps(differences[:5]), flush=True)
            raise AssertionError(f"Native reference differs: {name}")
        report["cohort_checks"].append(
            {
                "file": name,
                "all_records_exactly_equal": True,
                "reference_sha256": digest(out / name),
                "records": sum(len(records) for records in actual.values()),
            }
        )
        print("HOST LOOKUP COHORT REPLAY", name, flush=True)
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["passed"] = True
    save(args.output, report)
    print("HOST EMBEDDING GPU VALIDATION PASSED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
