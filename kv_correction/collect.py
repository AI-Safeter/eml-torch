"""Fresh document split and native attention replay, without generated-answer evaluation."""

import json
import random

import torch
from support import (
    HERE,
    PILOT,
    RUN,
    SPEC,
    ids,
    model,
    restore,
    save_json,
    sha,
    snapshot,
    tokenizer,
    verify_design,
)


def documents():
    tok = tokenizer()
    old = json.loads((PILOT / "documents.json").read_text())
    excluded = {d["title"] for ds in old.values() for d in ds}
    source = PILOT / "squad-train.json"
    expected = json.loads((PILOT / "data-manifest.json").read_text())["sources"]["train"]["sha256"]
    assert sha(source) == expected
    articles = json.loads(source.read_text())["data"]
    rng = random.Random(SPEC["data_seed"])
    rng.shuffle(articles)
    prompt = tok.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Answer the question using only the document. Return only the shortest answer span, with no explanation.\n\nDocument:\nDOC_MARKER\n\nQuestion: QUESTION_MARKER",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    start, ending = prompt.split("DOC_MARKER")[0], prompt.split("QUESTION_MARKER")[1]
    chosen = []
    total = sum(SPEC[f"{s}_articles"] for s in ["train", "selection", "gate"])
    for article in articles:
        if article["title"] in excluded:
            continue
        text = "\n\n".join(p["context"] for p in article["paragraphs"])
        prefix = tok.encode(start + text, add_special_tokens=False)
        if len(prefix) < SPEC["prefix_tokens"]:
            continue
        prefix = prefix[: SPEC["prefix_tokens"]]
        visible = tok.decode(prefix)
        questions = []
        for paragraph in article["paragraphs"]:
            if paragraph["context"] not in visible:
                continue
            q = rng.choice(paragraph["qas"])
            questions.append(
                {
                    "id": q["id"],
                    "suffix": tok.encode(
                        "\n\nQuestion: " + q["question"] + ending, add_special_tokens=False
                    ),
                }
            )
        if len(questions) >= 2:
            chosen.append(
                {
                    "title": article["title"],
                    "prefix": prefix,
                    "questions": [questions[0], questions[-1]],
                }
            )
        if len(chosen) == total:
            break
    assert len(chosen) == len({d["title"] for d in chosen}) == total
    assert not excluded.intersection(d["title"] for d in chosen)
    result, offset = {}, 0
    for split in ["train", "selection", "gate"]:
        size = SPEC[f"{split}_articles"]
        result[split], offset = chosen[offset : offset + size], offset + size
    save_json(RUN / "documents.json", result)
    save_json(
        RUN / "design.json",
        {
            "protocol_sha256": sha(HERE / "protocol.json"),
            "documents_sha256": sha(RUN / "documents.json"),
            "source_sha256": expected,
            "excluded_titles": sorted(excluded),
            "titles": {s: [d["title"] for d in ds] for s, ds in result.items()},
        },
    )
    return result


@torch.inference_mode()
def main():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    assert not list(RUN.glob("*.pt")), "Refusing to replace collected tensors"
    if (RUN / "design.json").exists():
        verify_design()
        data = json.loads((RUN / "documents.json").read_text())
    else:
        data = documents()
    native = model()
    target = native.model.language_model.layers[SPEC["layer"]].self_attn
    assert target.layer_type == "full_attention" and target.head_dim == 512
    assert not target.is_kv_shared_layer and target.num_key_value_groups == 8
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    captured = []
    enabled = False
    checks = []

    def capture(module, query, key, value, attention_mask, **kwargs):
        output = original(module, query, key, value, attention_mask, **kwargs)
        if enabled and module is target:
            assert query.shape[-2] > 1 and attention_mask is not None
            assert kwargs["scaling"] == 1.0
            scores = query.float() @ key.float().transpose(-1, -2)
            if attention_mask.dtype == torch.bool:
                scores.masked_fill_(~attention_mask, -torch.inf)
            else:
                scores += attention_mask.float()
            weights = scores.softmax(-1)
            replay = (weights @ value.float()).transpose(1, 2)
            if not checks:
                # Validate against native SDPA and against an independent float32 SDPA call.
                ref = torch.nn.functional.scaled_dot_product_attention(
                    query.float(),
                    key.float().expand(-1, 8, -1, -1),
                    value.float().expand(-1, 8, -1, -1),
                    attn_mask=attention_mask,
                    scale=1.0,
                ).transpose(1, 2)
                relative = (replay - ref).square().mean().sqrt() / ref.square().mean().sqrt()
                assert relative < 1e-5
                native_relative = (
                    replay - output[0].float()
                ).square().mean().sqrt() / replay.square().mean().sqrt()
                assert native_relative < 0.01
                checks.append(
                    {
                        "float32_replay_relative_error": float(relative),
                        "bf16_sdpa_relative_error": float(native_relative),
                    }
                )
            ix = (
                torch.linspace(
                    0, query.shape[-2] - 1, SPEC["query_positions_per_question"], device="cuda"
                )
                .round()
                .long()
                .unique()
            )
            assert len(ix) == SPEC["query_positions_per_question"]
            captured.append(
                weights[0].index_select(1, ix)[:, :, : SPEC["prefix_tokens"]].transpose(0, 1).cpu()
            )
        return output

    # Retain the native backend name so Transformers builds its usual causal mask.
    ALL_ATTENTION_FUNCTIONS.register("sdpa", capture)
    try:
        for split, docs in data.items():
            records = []
            for i, doc in enumerate(docs):
                enabled = False
                cache = native(
                    input_ids=ids(doc["prefix"]), use_cache=True, logits_to_keep=1
                ).past_key_values
                state = snapshot(cache)
                del cache
                v = state[SPEC["layer"]]["v"][0, 0].contiguous()
                captured.clear()
                enabled = True
                for q in doc["questions"]:
                    native(
                        input_ids=ids(q["suffix"]),
                        past_key_values=restore(state, native.config.text_config),
                        use_cache=True,
                        logits_to_keep=1,
                    )
                records.append({"title": doc["title"], "v": v, "attention": torch.cat(captured)})
                print("COLLECT", split, i + 1, len(docs), flush=True)
            torch.save(records, RUN / f"{split}.pt")
        torch.save(target.o_proj.weight.detach().float().cpu(), RUN / "projection.pt")
    finally:
        ALL_ATTENTION_FUNCTIONS.register("sdpa", original)
    save_json(
        RUN / "collection.json",
        {
            "gpu": torch.cuda.get_device_name(),
            "checks": checks,
            "layer": SPEC["layer"],
            "artifact_sha256": {p.name: sha(p) for p in RUN.glob("*.pt")},
            "collector_sha256": sha(__file__),
        },
    )


if __name__ == "__main__":
    main()
