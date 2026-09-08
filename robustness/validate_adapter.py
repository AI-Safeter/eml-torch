"""GPU adapter checks on training examples only, before any test evaluation."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen17b", "qwen4b", "smollm"])
    args = parser.parse_args()
    spec = json.loads((HERE / "models.json").read_text())[args.model]
    path = (
        Path.home()
        / ".cache/huggingface/hub"
        / ("models--" + spec["id"].replace("/", "--"))
        / "snapshots"
        / spec["revision"]
    )
    os.environ.update(
        EMLTORCH_MODEL_ID=spec["id"],
        EMLTORCH_MODEL_REVISION=spec["revision"],
        EMLTORCH_MODEL_PATH=str(path),
    )
    from data import save
    from evaluate_heads import generation
    from model_io import corrupt_row, inputs, load, load_mlp, logits, setup, targets, text

    setup()
    model, tok = load()
    layer_index = len(model.model.layers) // 2
    reference = model.model.layers[layer_index].mlp
    replica = load_mlp(layer_index)
    h = torch.randn(3, 2, model.config.hidden_size, device="cuda", requires_grad=True)
    expected, actual = reference(h), replica(h)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.randn_like(actual)
    g1 = torch.autograd.grad((expected * weights).sum(), h)[0]
    g2 = torch.autograd.grad((actual * weights).sum(), h)[0]
    torch.testing.assert_close(g1, g2, rtol=0, atol=0)
    del replica, h, expected, actual, weights, g1, g2
    records = []
    with torch.inference_mode():
        for op in ["add", "multiply", "divide"]:
            rows = json.loads((HERE / "data" / op / "problems.json").read_text())["train"][:4]
            assert (targets(tok, rows) != targets(tok, rows, True)).all()
            for row in rows:
                assert text(tok, row, True) == text(tok, corrupt_row(row))
            inp = inputs(tok, rows)
            baseline = logits(model, inp)
            handle = reference.register_forward_hook(lambda module, args, out: out.clone())
            try:
                restored = logits(model, inp)
            finally:
                handle.remove()
            torch.testing.assert_close(restored, baseline, rtol=0, atol=0)
            _, texts = generation(model, tok, inp)
            records.append(
                {
                    "operation": op,
                    "training_answers": [r["answer"] for r in rows],
                    "generated": texts,
                }
            )
    save(
        HERE / f"adapter-{args.model}.json",
        {
            "model": spec,
            "device": torch.cuda.get_device_name(),
            "standalone_mlp_output_and_gradient_bitwise_equal": True,
            "restoration_logits_bitwise_equal": True,
            "contrast_first_tokens_distinct": True,
            "test_outputs_collected": False,
            "training_smoke_examples": records,
        },
    )
    print("ADAPTER PASSED", args.model, flush=True)


if __name__ == "__main__":
    main()
