#!/usr/bin/env python3
"""H31 black-box runner — forward-only top-K logprobs per probe.

Strict black-box discipline (pre-reg locked):
- transformers.AutoModelForCausalLM.from_pretrained ONLY
- NO transformer_lens import
- NO output_attentions / output_hidden_states
- NO hooks registered anywhere
- Single accessed quantity: logits[:, -1, :].topk(K=50)

Usage:
  python h31_blackbox_runner.py MODEL_NAME GPU_ID OUT_TAG
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Import-time guard: forbid white-box libraries
for lib in ("transformer_lens", "sae_lens"):
    if lib in sys.modules:
        raise ImportError(f"Black-box discipline violation: {lib} already imported")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from _h31_common import OUT_DIR, assert_no_hooks  # noqa: E402

OUT_DIR.mkdir(parents=True, exist_ok=True)
PROBES_PATH = OUT_DIR / "probes.jsonl"


def load_probes() -> list[dict]:
    probes = []
    with PROBES_PATH.open() as f:
        for line in f:
            probes.append(json.loads(line))
    return probes


def find_target_token_id(tokenizer, target_str: str) -> tuple[int, str]:
    """Encode target string with the model's tokenizer, return first token id.

    Black-box: tokenizer access is allowed (tokenizers are public API for
    any LLM service). We forbid attention / hidden_state / gradient access
    on the model itself.
    """
    ids = tokenizer.encode(target_str, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError(f"Empty encoding for {target_str!r}")
    return ids[0], tokenizer.decode([ids[0]])


@torch.no_grad()
def run_blackbox(model_name: str, device: str, out_tag: str) -> None:
    print(f"[H31] Loading {model_name} on {device}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.train(mode=False)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # Audit: no hooks anywhere
    assert_no_hooks(model)
    print("  hook-free assertion PASSED", flush=True)

    probes = load_probes()
    print(f"[H31] Running {len(probes)} probes", flush=True)

    results = []
    for i, probe in enumerate(probes):
        prompt = probe["prompt"]
        target_str = probe["target_token_str"]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        target_id, target_decoded = find_target_token_id(tokenizer, target_str)

        outputs = model(
            **inputs,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
        )
        assert outputs.attentions is None, "attention output leaked"
        assert outputs.hidden_states is None, "hidden_states output leaked"

        logits = outputs.logits[:, -1, :]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        topk = torch.topk(logprobs, k=50, dim=-1)
        topk_ids = topk.indices[0].tolist()

        if target_id in topk_ids:
            idx = topk_ids.index(target_id)
            p_target = float(torch.exp(topk.values[0][idx]).item())
            rank = idx
        else:
            p_target = 0.0
            rank = -1

        top1_id = topk_ids[0]
        top1_str = tokenizer.decode([top1_id])
        top1_correct = top1_id == target_id

        p50 = torch.exp(topk.values[0])
        p50 = p50 / p50.sum()
        entropy_top50 = float(-(p50 * torch.log(p50 + 1e-30)).sum().item())

        results.append(
            {
                "probe_id": probe["probe_id"],
                "circuit": probe["circuit"],
                "subclass": probe["subclass"],
                "T": probe["T"],
                "L": probe.get("L", 0),
                "n_repeats": probe.get("n_repeats", 0),
                "target_token_id": target_id,
                "target_decoded": target_decoded,
                "p_target": p_target,
                "rank": rank,
                "top1_correct": bool(top1_correct),
                "top1_token_str": top1_str,
                "entropy_top50": entropy_top50,
            }
        )

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(probes)} done", flush=True)

        del outputs, logits, logprobs, topk
        if i % 100 == 0:
            torch.cuda.empty_cache()

    out_path = OUT_DIR / f"measurements_{out_tag}.jsonl"
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[H31] Wrote {len(results)} measurements → {out_path}", flush=True)

    from collections import defaultdict

    acc_by_circuit = defaultdict(list)
    for r in results:
        acc_by_circuit[r["circuit"]].append(r["top1_correct"])
    print(f"\n[H31] Top-1 accuracy by circuit ({out_tag}):")
    for c, lst in sorted(acc_by_circuit.items()):
        n = len(lst)
        n_corr = sum(lst)
        print(f"  {c}: {n_corr}/{n} = {n_corr/n:.1%}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: h31_blackbox_runner.py MODEL_NAME GPU_ID OUT_TAG", file=sys.stderr
        )
        sys.exit(1)
    model_name = sys.argv[1]
    gpu_id = int(sys.argv[2])
    out_tag = sys.argv[3]
    device = f"cuda:{gpu_id}"
    run_blackbox(model_name, device, out_tag)
