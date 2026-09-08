"""Pinned, text-only Gemma cache I/O for the compression pilot."""

import hashlib
import json
import os
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "protocol.json").read_text())
RUN = Path(os.environ.get("EML_KV_RUN", str(HERE.parent.parent / "emltorch-kv-runs")))
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
    / SPEC["revision"]
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def setup():
    import transformers

    assert torch.cuda.is_available(), "This experiment requires GPU validation and fitting"
    assert torch.__version__ == SPEC["torch"]
    assert transformers.__version__ == SPEC["transformers"]
    torch.set_num_threads(2)
    torch.manual_seed(SPEC["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)


def model():
    from transformers import Gemma4ForConditionalGeneration

    setup()
    return (
        Gemma4ForConditionalGeneration.from_pretrained(
            SNAPSHOT,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
            device_map={"": 0},
        )
        .eval()
        .requires_grad_(False)
    )


def ids(values):
    return torch.tensor([values], device="cuda", dtype=torch.long)


def storage_bytes(tensors):
    """Count distinct backing allocations, including retained views."""
    unique = {
        (str(x.device), x.untyped_storage().data_ptr()): x.untyped_storage().nbytes()
        for x in tensors
    }
    return sum(unique.values())


def snapshot(cache, device="cpu"):
    # Clone logical sliding windows so a small view cannot retain the full prefill.
    return [
        {
            "k": layer.keys.to(device).clone().contiguous(),
            "v": layer.values.to(device).clone().contiguous(),
            "length": int(layer.get_seq_length()),
        }
        for layer in cache.layers
    ]


def restore(records, config, clone=True):
    from transformers import DynamicCache

    cache = DynamicCache(config=config)
    assert len(records) == len(cache.layers) == 15
    for layer, item in zip(cache.layers, records):
        k, v = item["k"].cuda(), item["v"].cuda()
        layer.lazy_initialization(k, v)
        layer.keys, layer.values = (k.clone(), v.clone()) if clone else (k, v)
        if layer.is_sliding:
            layer.cumulative_length = item["length"]
        assert layer.get_seq_length() == item["length"]
    return cache


@torch.inference_mode()
def answer(native, question, cache=None, prefix=None, fixed_tokens=None):
    inp = ids(question if prefix is None else prefix + question)
    tokens = []
    eos = set(native.generation_config.eos_token_id)
    for _ in range(fixed_tokens or SPEC["max_new_tokens"]):
        output = native(input_ids=inp, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        token = int(output.logits[0, -1].argmax())
        tokens.append(token)
        if fixed_tokens is None and token in eos:
            break
        inp = ids([token])
    return tokens
