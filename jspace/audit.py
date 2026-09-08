"""GPU checks of native forwarding, the projected estimator and causal masking."""

import argparse
import json

import torch
from lens import (
    HERE,
    RUN,
    SPEC,
    capture,
    gradients,
    model,
    save_json,
    sha,
    source_rows,
    tokenizer,
    verify,
)


def main(fp32_suffix=False):
    verify()
    native, tok = model(), tokenizer()
    document = json.loads((RUN / "documents.json").read_text())[0]
    ids = torch.tensor([document["ids"]], device="cuda")
    text = native.model.language_model
    rows, _ = source_rows(native, tok)
    report = {"gpu": torch.cuda.get_device_name(), "source_sha256": sha(HERE / "audit.py")}
    with torch.no_grad():
        reference = native(input_ids=ids, use_cache=False, logits_to_keep=1).logits
        hidden = text(input_ids=ids, use_cache=False).last_hidden_state[:, -1:]
        logits = native.lm_head(hidden)
        logits = 30 * torch.tanh(logits / 30)
        assert torch.equal(reference, logits)
        with capture(native, SPEC["layers"], intervention=(17, lambda h: h)):
            identity = native(input_ids=ids, use_cache=False, logits_to_keep=1).logits
        assert torch.equal(reference, identity)
    report["text_only_forward_bitwise_equal"] = True
    report["identity_intervention_bitwise_equal"] = True
    report["gradient_execution"] = (
        "FP32 suffix with BF16 checkpoint weights" if fp32_suffix else "native BF16"
    )
    if fp32_suffix:
        # Keep the large embedding tables and upstream computation unchanged.
        # Upcasting the differentiated suffix separates estimator correctness
        # from BF16 rounding and finite-step nonlinear response.
        report["unused_modules_offloaded_to_cpu"] = []
        for name, child in native.model.named_children():
            if name != "language_model":
                child.cpu()
                report["unused_modules_offloaded_to_cpu"].append(name)
        torch.cuda.empty_cache()
        for block in text.layers[11:]:
            block.float()
            block.register_forward_pre_hook(
                lambda module, args: tuple(x.float() if torch.is_tensor(x) else x for x in args)
            )
        text.layers[10].register_forward_hook(lambda module, args, output: output.float())
        torch.cuda.empty_cache()

    # Independently build a sparse cotangent from coordinate-basis VJPs.
    # This checks the direct projection against rows of the full-J estimator.
    generator = torch.Generator(device="cuda").manual_seed(812)
    dims = torch.randperm(1536, generator=generator, device="cuda")[:16]
    basis = torch.zeros(16, 1536, device="cuda")
    basis[torch.arange(16, device="cuda"), dims] = 1
    weights = torch.randn(16, generator=generator, device="cuda").to(torch.bfloat16).float()
    sparse_row = (weights[:, None] * basis).sum(0, keepdim=True)
    direct = gradients(native, ids, sparse_row)[:, 0]
    explicit = (gradients(native, ids, basis) * weights.cpu()[None, :, None]).sum(1)
    relative = (direct - explicit).norm(dim=-1) / explicit.norm(dim=-1)
    assert (relative < 0.03).all(), relative
    report["projected_vs_explicit_coordinate_jacobian_relative_error"] = relative.tolist()

    with capture(native, SPEC["layers"] + [34], start_graph=10) as h:
        text(input_ids=ids, use_cache=False)
        output = (h[34][0, 64].float() * rows[0]).sum()
        grads = torch.autograd.grad(output, [h[layer] for layer in SPEC["layers"]])
        leakage = [float(g[0, 65:].abs().max()) for g in grads]
        assert leakage == [0.0, 0.0, 0.0]
        report["future_source_to_earlier_target_max_abs_gradient"] = leakage
        report["earlier_source_to_later_target_gradient_norm"] = [
            float(g[0, :64].float().norm()) for g in grads
        ]
        # All layers after 14 reuse keys/values already computed by layer 14.
        # Their residual outputs cannot affect other positions in this forward.
        assert grads[0][0, :64].abs().max() > 0
        assert grads[1][0, :64].abs().max() == grads[2][0, :64].abs().max() == 0

    valid = slice(SPEC["calibration"]["skip_first"], -1)
    reduced = gradients(native, ids, rows[:1])
    finite_differences = []
    for index, layer in enumerate(SPEC["layers"]):
        direction = torch.nn.functional.normalize(reduced[index, 0].cuda(), dim=-1)
        predicted = float(reduced[index, 0].cuda() @ direction)
        with torch.no_grad(), capture(native, [layer]) as h:
            text(input_ids=ids, use_cache=False)
            norm = float(h[layer][0, valid].float().norm(dim=-1).mean())
        fractions = [0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
        if fp32_suffix:
            fractions = [0.000001, 0.00001, 0.000025, 0.00005] + fractions
        for fraction in fractions:
            step = norm * fraction
            values = []
            for sign in [-1, 1]:

                def edit(hidden, sign=sign):
                    changed = hidden.clone()
                    changed[:, valid] = (hidden[:, valid].float() + sign * step * direction).to(
                        hidden.dtype
                    )
                    return changed

                with torch.no_grad(), capture(native, [layer, 34], intervention=(layer, edit)) as h:
                    text(input_ids=ids, use_cache=False)
                    values.append(float((h[34][0, valid].float() @ rows[0]).mean()))
            observed = (values[1] - values[0]) / (2 * step)
            finite_differences.append(
                {
                    "layer": layer,
                    "step_fraction_of_residual_norm": fraction,
                    "residual_norm": norm,
                    "autograd": predicted,
                    "finite_difference": observed,
                    "relative_error": abs(observed - predicted) / abs(predicted),
                }
            )
    report["finite_difference_checks"] = finite_differences
    report["passed"] = all(
        min(r["relative_error"] for r in finite_differences if r["layer"] == layer)
        < (0.001 if fp32_suffix else 0.1)
        for layer in SPEC["layers"]
    )
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    save_json(RUN / ("audit-fp32.json" if fp32_suffix else "audit.json"), report)
    print(json.dumps(report), flush=True)
    assert report["passed"], "Finite differences did not validate the estimator"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-suffix", action="store_true")
    main(parser.parse_args().fp32_suffix)
