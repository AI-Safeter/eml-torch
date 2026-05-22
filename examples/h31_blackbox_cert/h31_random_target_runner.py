#!/usr/bin/env python3
"""H31 random-target control: re-run Qwen3.6 on factual prompts with
target_token replaced by a random non-target token (sampled from the
factual vocabulary pool). Tests whether the EML formula fit to true
P_target captures target-specific structure or prompt-statistics.

Same strict black-box discipline as h31_blackbox_runner.py.

Usage: h31_random_target_runner.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

for lib in ("transformer_lens", "sae_lens"):
    if lib in sys.modules:
        raise ImportError(f"Black-box discipline violation: {lib} already imported")

import random  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from _h31_common import OUT_DIR, assert_no_hooks  # noqa: E402

PROBES_PATH = OUT_DIR / "probes.jsonl"

SEED = 20260522 + 99


@torch.no_grad()
def main():
    rng = random.Random(SEED)

    probes = [json.loads(line) for line in PROBES_PATH.open()]
    factual = [p for p in probes if p["circuit"] == "factual"]
    print(f"[H31-rand] Loaded {len(factual)} factual probes")

    # Pool of all target tokens across factual probes — sample a random
    # non-self target per probe (i.e., for "France capital", pick a
    # random capital from another country)
    all_targets = list({p["target_token_str"] for p in factual})

    print(f"[H31-rand] Loading Qwen3.6-27B on cuda:0...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.6-27B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.train(mode=False)
    print(f"  loaded in {time.time()-t0:.1f}s")
    assert_no_hooks(model)

    results = []
    for i, p in enumerate(factual):
        prompt = p["prompt"]
        # Sample random non-self target
        actual_target = p["target_token_str"]
        candidates = [t for t in all_targets if t != actual_target]
        rand_target = rng.choice(candidates)

        rand_ids = tok.encode(rand_target, add_special_tokens=False)
        if len(rand_ids) == 0:
            continue
        rand_target_id = rand_ids[0]
        rand_target_decoded = tok.decode([rand_target_id])

        inputs = tok(prompt, return_tensors="pt").to("cuda:0")
        outputs = model(
            **inputs,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
        )
        assert outputs.attentions is None
        assert outputs.hidden_states is None

        logits = outputs.logits[:, -1, :]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        topk = torch.topk(logprobs, k=50, dim=-1)
        topk_ids = topk.indices[0].tolist()

        if rand_target_id in topk_ids:
            idx = topk_ids.index(rand_target_id)
            p_rand = float(torch.exp(topk.values[0][idx]).item())
            rank_rand = idx
        else:
            p_rand = 0.0
            rank_rand = -1

        p50 = torch.exp(topk.values[0])
        p50 = p50 / p50.sum()
        entropy = float(-(p50 * torch.log(p50 + 1e-30)).sum().item())

        results.append(
            {
                "probe_id": p["probe_id"],
                "circuit": "factual",
                "subclass": p["subclass"],
                "T": p["T"],
                "L": p["L"],
                "n_repeats": p["n_repeats"],
                "actual_target": actual_target,
                "random_target": rand_target,
                "target_token_id": rand_target_id,
                "target_decoded": rand_target_decoded,
                "p_target": p_rand,  # P(random_target | prompt)
                "rank": rank_rand,
                "top1_correct": False,
                "entropy_top50": entropy,
            }
        )

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(factual)} done")

        del outputs, logits, logprobs, topk
        if i % 20 == 0:
            torch.cuda.empty_cache()

    out_path = OUT_DIR / "measurements_qwen36_factual_RANDOMTARGET.jsonl"
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[H31-rand] Wrote {len(results)} measurements → {out_path}")

    import numpy as np

    ps = np.array([r["p_target"] for r in results])
    print(
        f"[H31-rand] P(random_target) stats: mean={ps.mean():.4f}, "
        f"std={ps.std():.4f}, max={ps.max():.4f}"
    )


if __name__ == "__main__":
    main()
