"""Verify the self-contained cache adapter against one saved training document."""

import json

import torch
from support import RUN, SPEC, ids, model, restore, save_json, sha, snapshot, verify_design


@torch.inference_mode()
def main():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    verify_design()
    doc = json.loads((RUN / "documents.json").read_text())["train"][0]
    expected = torch.load(RUN / "train.pt", weights_only=True)[0]
    native = model()
    target = native.model.language_model.layers[SPEC["layer"]].self_attn
    cache = native(input_ids=ids(doc["prefix"]), use_cache=True, logits_to_keep=1).past_key_values
    state = snapshot(cache)
    del cache
    assert torch.equal(state[SPEC["layer"]]["v"][0, 0], expected["v"])
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    captured = []

    def capture(module, q, k, v, mask, **kwargs):
        output = original(module, q, k, v, mask, **kwargs)
        if module is target:
            scores = q.float() @ k.float().transpose(-1, -2)
            scores = (
                scores.masked_fill(~mask, -torch.inf)
                if mask.dtype == torch.bool
                else scores + mask.float()
            )
            indices = (
                torch.linspace(
                    0, q.shape[-2] - 1, SPEC["query_positions_per_question"], device="cuda"
                )
                .round()
                .long()
                .unique()
            )
            captured.append(
                scores.softmax(-1)[0]
                .index_select(1, indices)[:, :, : SPEC["prefix_tokens"]]
                .transpose(0, 1)
                .cpu()
            )
        return output

    ALL_ATTENTION_FUNCTIONS.register("sdpa", capture)
    try:
        for question in doc["questions"]:
            native(
                input_ids=ids(question["suffix"]),
                past_key_values=restore(state, native.config.text_config),
                use_cache=True,
                logits_to_keep=1,
            )
    finally:
        ALL_ATTENTION_FUNCTIONS.register("sdpa", original)
    assert torch.equal(torch.cat(captured), expected["attention"])
    assert torch.equal(
        target.o_proj.weight.float().cpu(), torch.load(RUN / "projection.pt", weights_only=True)
    )
    save_json(
        RUN / "replay-validation.json",
        {
            "status": "bitwise equal",
            "gpu": torch.cuda.get_device_name(),
            "title": doc["title"],
            "checked": ["V", "attention weights", "output projection"],
            "script_sha256": sha(__file__),
            "collection_sha256": sha(RUN / "collection.json"),
        },
    )
    print("SELF-CONTAINED ADAPTER GPU REPLAY PASSED", flush=True)


if __name__ == "__main__":
    main()
