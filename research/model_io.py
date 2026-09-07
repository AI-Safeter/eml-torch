"""Operation-aware pinned model prompts and component patching."""

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("EMLTORCH_MODEL_ID", "Qwen/Qwen3-1.7B")
REVISION = os.environ.get("EMLTORCH_MODEL_REVISION", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")


def setup(seed=20260910):
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load():
    location = os.environ.get("EMLTORCH_MODEL_PATH", MODEL)
    tok = AutoTokenizer.from_pretrained(location, revision=REVISION, padding_side="left")
    model = (
        AutoModelForCausalLM.from_pretrained(
            location,
            revision=REVISION,
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        )
        .eval()
        .cuda()
    )
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model, tok


def load_mlp(layer):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

    location = os.environ.get("EMLTORCH_MODEL_PATH", MODEL)
    path = Path(location)
    if not path.is_dir():
        path = Path(snapshot_download(MODEL, revision=REVISION))
    config = AutoConfig.from_pretrained(path)
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"] if index_path.exists() else None
    state = {}
    for short in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]:
        key = f"model.layers.{layer}.mlp.{short}"
        with safe_open(
            path / (index[key] if index else "model.safetensors"),
            framework="pt",
            device="cpu",
        ) as file:
            state[short] = file.get_tensor(key)
    model = Qwen3MLP(config).float().cuda().eval().requires_grad_(False)
    model.load_state_dict(state)
    return model


def text(tok, row, corrupt=False):
    a, b = (row["c"] if corrupt else row["a"]), row["b"]
    op, style = row["op"], row.get("style", "prose")
    word = {"add": "plus", "multiply": "times", "divide": "divided by"}[op]
    symbol = {"add": "+", "multiply": "*", "divide": "//"}[op]
    prompt = {
        "prose": f"What is {a} {word} {b}? Give only the numerical result.",
        "symbolic": f"Calculate {a} {symbol} {b}. Give only the numerical result.",
        "code": f"What integer does this Python expression evaluate to: {a} {symbol} {b}? Give only the numerical result.",
        "instruction": f"Compute the exact integer result of {a} {word} {b}. Respond with digits and no explanation.",
    }[style]
    return (
        tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + "The answer is "
    )


def inputs(tok, rows, corrupt=False):
    return tok([text(tok, r, corrupt) for r in rows], padding=True, return_tensors="pt").to("cuda")


def logits(model, inp):
    h = model.model(**inp, use_cache=False).last_hidden_state[:, -1, :]
    return model.lm_head(h).float()


def targets(tok, rows, corrupt=False):
    field = "corrupt_answer" if corrupt else "answer"
    ids = [tok.encode(str(r[field]), add_special_tokens=False)[0] for r in rows]
    return torch.tensor(ids, device="cuda")


def margin(scores, clean, corrupt):
    return (scores.gather(1, clean[:, None]) - scores.gather(1, corrupt[:, None])).squeeze(1)


def expand(rows, styles):
    return [{**r, "style": style} for r in rows for style in styles]


def capture(model, tok, rows, layers, keep_scores=True):
    captured = {i: {"input": [], "output": []} for i in layers}
    scores = []
    for start in range(0, len(rows), 16):
        batch = rows[start : start + 16]
        handles = []
        for layer in layers:

            def hook(module, args, out, index=layer):
                captured[index]["input"].append(args[0][:, -1, :].detach().cpu())
                captured[index]["output"].append(out[:, -1, :].detach().cpu())

            handles.append(model.model.layers[layer].mlp.register_forward_hook(hook))
        try:
            if keep_scores:
                scores.append(logits(model, inputs(tok, batch)).cpu())
            else:
                model.model(**inputs(tok, batch), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
    return {i: {k: torch.cat(v) for k, v in d.items()} for i, d in captured.items()}, torch.cat(
        scores
    ) if scores else None


def patch_scores(model, tok, rows, layer, clean_out, corrupt_out, direction=None):
    outputs = []
    for start in range(0, len(rows), 16):
        batch = rows[start : start + 16]
        clean = clean_out[start : start + len(batch)].cuda()
        corrupt = corrupt_out[start : start + len(batch)].cuda()

        def hook(module, args, out, clean=clean, corrupt=corrupt):
            result = out.clone()
            if direction is None:
                result[:, -1, :] = clean
            else:
                d = direction.cuda()
                coefficient = (clean - corrupt) @ d / d.double().square().sum().float()
                result[:, -1, :] += coefficient[:, None] * d
            return result

        handle = model.model.layers[layer].mlp.register_forward_hook(hook)
        try:
            outputs.append(logits(model, inputs(tok, batch, True)).cpu())
        finally:
            handle.remove()
    return torch.cat(outputs)
