"""Validate native Gemma execution on training prompts before collection."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma import adapter


def main():
    adapter.install()
    from data import save
    from evaluate_heads import Replacements, generation
    from model_io import inputs, logits, setup, targets

    setup()
    model, tok = adapter.load()
    mlp_checks = []
    for layer in [0, len(model.model.layers) - 1]:
        reference = model.model.layers[layer].mlp
        replica = adapter.load_mlp(layer)
        h = torch.randn(3, 2, model.config.hidden_size, device="cuda", requires_grad=True)
        expected, actual = reference(h), replica(h)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        weight = torch.randn_like(actual)
        g1 = torch.autograd.grad((expected * weight).sum(), h)[0]
        g2 = torch.autograd.grad((actual * weight).sum(), h)[0]
        torch.testing.assert_close(g1, g2, rtol=0, atol=0)
        chosen = torch.tensor([0, 7, 25, 511], device="cuda")
        direction = torch.randn(model.config.hidden_size, device="cuda")
        readout = direction @ replica.down_proj.weight[:, chosen]
        sparse = object.__new__(Replacements)
        sparse.sparse = {
            "control": {
                "gate": replica.gate_proj.weight[chosen],
                "up": replica.up_proj.weight[chosen],
                "readout": readout,
                "bias": torch.zeros((), device="cuda"),
            }
        }
        flat = h.detach().flatten(0, 1)
        expected_sparse = (replica.act_fn(replica.gate_proj(flat)) * replica.up_proj(flat))[
            :, chosen
        ] @ readout
        torch.testing.assert_close(
            sparse.coefficient(flat, "control"), expected_sparse, rtol=2e-4, atol=1e-4
        )
        mlp_checks.append({"layer": layer, "intermediate_size": replica.intermediate_size})
        del replica, h, expected, actual, weight, g1, g2
    checks = []
    with torch.inference_mode():
        for op in ["add", "multiply", "divide"]:
            rows = json.loads((adapter.STUDY / "data" / op / "problems.json").read_text())["train"][
                :4
            ]
            assert (targets(tok, rows) != targets(tok, rows, True)).all()
            inp = inputs(tok, rows)
            expected = model.native(**inp, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            torch.testing.assert_close(logits(model, inp), expected, rtol=0, atol=0)
            body = model.model(**inp, use_cache=False).last_hidden_state[:, -1]
            projected = model.lm_head(body).float()
            cap = model.config.final_logit_softcapping
            projected = (projected / cap).tanh() * cap
            torch.testing.assert_close(projected, expected, rtol=0, atol=0)
            score, text = generation(model, tok, inp)
            assert torch.isfinite(score).all()
            calls = []

            def restore(module, args, out):
                calls.append(out.shape[1])
                return out.clone()

            handle = model.model.layers[-1].mlp.register_forward_hook(restore)
            try:
                restored_score, restored_text = generation(model, tok, inp)
            finally:
                handle.remove()
            torch.testing.assert_close(restored_score, score, rtol=0, atol=0)
            assert restored_text == text and len(calls) > 1
            checks.append(
                {
                    "operation": op,
                    "training_prompts": 4,
                    "generated": text,
                    "generation_hook_calls": len(calls),
                }
            )
    save(
        adapter.HERE / "adapter-validation.json",
        {
            "model": adapter.SPEC,
            "gpu": torch.cuda.get_device_name(),
            "loaded_parameters": sum(p.numel() for p in model.parameters()),
            "all_native_logits_body_and_restoration_checks_bitwise_equal": True,
            "standalone_mlp_outputs_and_input_gradients_bitwise_equal": True,
            "native_gelu_sparse_control_verified": True,
            "mlp_checks": mlp_checks,
            "training_checks": checks,
            "test_outputs_collected": False,
        },
    )
    print("GEMMA GPU ADAPTER VALIDATION PASSED", flush=True)


if __name__ == "__main__":
    main()
