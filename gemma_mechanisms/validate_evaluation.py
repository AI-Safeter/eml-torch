"""Check native scorers and run a clearly labeled selection-only quality pilot."""

import gc
import json
import time

import torch

from .evaluate import arc, arithmetic, language
from .runtime import HERE, SPEC, accounting, digest, load, prompt, root, save, setup, tokenizer
from .student import install, load_student


def main():
    setup(73)
    out = root()
    tok = tokenizer()
    model = load(host_ple=False)
    host = load(host_ple=True)
    inp = tok(
        [prompt(tok, x) for x in ["What is 573 + 846?", "Describe a river."]],
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    checks = []
    with torch.inference_mode():
        native_result = model(**inp, use_cache=True, logits_to_keep=1)
        host_result = host(**inp, use_cache=True, logits_to_keep=1)
        assert torch.equal(native_result.logits, host_result.logits)
        mask = inp.attention_mask
        for _ in range(3):
            ids = native_result.logits[:, -1].argmax(-1, keepdim=True)
            mask = torch.cat([mask, torch.ones_like(ids)], dim=1)
            native_result = model(
                input_ids=ids,
                attention_mask=mask,
                past_key_values=native_result.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            host_result = host(
                input_ids=ids,
                attention_mask=mask,
                past_key_values=host_result.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            assert torch.equal(native_result.logits, host_result.logits)
    checks.append(
        "Host PLE and fully resident native BF16 model have bitwise equal prefill/three-step cached logits"
    )
    del host, native_result, host_result
    gc.collect()
    torch.cuda.empty_cache()
    blocks = torch.load(out / "data/language-selection.pt", weights_only=True)[:16]
    with torch.inference_mode():
        ours = language(model, blocks[:1])[0]["ce_nats"]
        ids = blocks[:1].cuda()
        reference = float(model(input_ids=ids, labels=ids, use_cache=False).loss)
        assert abs(ours - reference) < 1e-6, (ours, reference)
    checks.append("Language CE agrees with native label-loss implementation to <1e-6 nats/token")
    qa = json.loads((out / "data/arc-selection.json").read_text())[:16]
    scored = arc(model, tok, qa[:2])
    maximum_error = 0.0
    with torch.inference_mode():
        for row, result in zip(qa[:2], scored):
            question = prompt(tok, row["question"] + " Answer the question concisely.")
            prefix = tok.encode(question, add_special_tokens=False)
            for answer, score in zip(row["choices"]["text"], result["scores"]):
                full = tok.encode(question + answer, add_special_tokens=False)
                ids = torch.tensor([full], device="cuda")
                logits = model(input_ids=ids, use_cache=False).logits[0].float()
                target_logprob = logits[len(prefix) - 1 : -1].log_softmax(-1)
                indices = torch.tensor(full[len(prefix) :], device="cuda")
                reference = float(target_logprob.gather(-1, indices[:, None]).mean())
                maximum_error = max(maximum_error, abs(reference - score))
    assert maximum_error < 1e-5, maximum_error
    checks.append(
        "Single-choice native ARC scoring agrees with independent log-softmax scoring to <1e-5 nats/token"
    )
    raw_rows = json.loads((out / "data/arithmetic-selection.json").read_text())
    rows = [
        row
        for op in SPEC["arithmetic"]["operations"]
        for row in [r for r in raw_rows if r["op"] == op][:32]
    ]
    names = [
        "original",
        "eml-b3000000-d1-s1103",
        "silu-b3000000-d1-s1103",
        "linear-b3000000-d0-s1103",
    ]
    record = {}
    original = model.model.language_model.layers[26].mlp
    for name in names:
        if name != "original":
            original.cpu()

            def forbidden(*args, **kwargs):
                raise AssertionError("Original MLP executed in pilot replacement path")

            original.forward = forbidden
            removed = install(model, load_student(out / "training" / f"{name}.pt"))
            del removed
        started = time.perf_counter()
        a = arithmetic(model, tok, rows, SPEC["arithmetic"]["formats"])
        language_records = language(model, blocks)
        q = arc(model, tok, qa)
        record[name] = {
            "arithmetic": a,
            "language": language_records,
            "arc": q,
            "accounting": accounting(model),
            "seconds": time.perf_counter() - started,
            "checkpoint_sha256": None
            if name == "original"
            else digest(out / "training" / f"{name}.pt"),
        }
        print(
            "PILOT",
            name,
            {
                op: sum(r["correct"] for r in a if r["op"] == op) / sum(r["op"] == op for r in a)
                for op in SPEC["arithmetic"]["operations"]
            },
            "language CE",
            sum(r["ce_nats"] for r in language_records) / len(language_records),
            flush=True,
        )
    save(
        out / "selection-pilot.json",
        {
            "checks": checks,
            "arc_max_score_error": maximum_error,
            "records": record,
            "not_an_acceptance_evaluation": True,
            "limitations": "First seed, depth one, small selection subset. No gate/final data or latency claim.",
            "sources": {
                p: digest(HERE / p)
                for p in ["validate_evaluation.py", "evaluate.py", "student.py", "runtime.py"]
            },
        },
    )
    print("EVALUATION VALIDATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
