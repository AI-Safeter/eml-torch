"""Pinned model, arithmetic prompts, and GPU inference helpers."""

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MODEL_PATH = os.environ.get("EMLTORCH_MODEL_PATH", MODEL_ID)


def setup():
    torch.set_num_threads(2)
    torch.manual_seed(20260907)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()


def load():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, revision=REVISION, padding_side="left")
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, revision=REVISION, torch_dtype=torch.float32, attn_implementation="sdpa"
        )
        .eval()
        .cuda()
    )
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model, tok


def texts(tok, rows, corrupt=False, style="prose"):
    result = []
    for r in rows:
        a = r["c"] if corrupt else r["a"]
        b = r["b"]
        content = {
            "prose": f"What is {a} plus {b}? Give only the numerical result.",
            "symbolic": f"Calculate {a} + {b}. Give only the numerical result.",
            "code": f"What integer does this Python expression evaluate to: {a} + {b}? Give only the numerical result.",
        }[style]
        prompt = (
            tok.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + "The answer is "
        )
        result.append(prompt)
    return result


def inputs(tok, rows, corrupt=False, style="prose"):
    return tok(texts(tok, rows, corrupt, style), padding=True, return_tensors="pt").to("cuda")


def logits(model, inp):
    h = model.model(**inp, use_cache=False).last_hidden_state[:, -1, :]
    return model.lm_head(h).float()


def first_targets(tok, rows, corrupt=False):
    numbers = [r["c"] + r["b"] if corrupt else r["a"] + r["b"] for r in rows]
    ids = [tok.encode(str(s), add_special_tokens=False) for s in numbers]
    assert all(len(x) == 2 for x in ids)
    return torch.tensor([x[0] for x in ids], device="cuda")


def margin(logit, clean, corrupt):
    return logit.gather(1, clean[:, None]).squeeze(1) - logit.gather(1, corrupt[:, None]).squeeze(1)


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2))
