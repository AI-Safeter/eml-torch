"""Pinned model identity, CUDA accounting, and native Gemma loading."""

import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "protocol.json").read_text())
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
    / SPEC["model"]["revision"]
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def setup(seed=0):
    assert torch.cuda.is_available(), "Training and validation require CUDA"
    assert torch.__version__ == SPEC["model"]["torch"]
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, padding_side="left")


def accounting(module):
    params, buffers = list(module.named_parameters()), list(module.named_buffers())
    return {
        "parameters": sum(p.numel() for _, p in params),
        "parameter_bytes": sum(p.numel() * p.element_size() for _, p in params),
        "buffers": sum(p.numel() for _, p in buffers),
        "buffer_bytes": sum(p.numel() * p.element_size() for _, p in buffers),
        "coefficients": sum(p.numel() for _, p in params + buffers),
        "by_device": {
            device: sum(
                p.numel() * p.element_size() for _, p in params + buffers if p.device.type == device
            )
            for device in {p.device.type for _, p in params + buffers}
        },
    }


class HostEmbedding(nn.Module):
    """Keep the frozen PLE table on CPU, preserving native CUDA row scaling."""

    def __init__(self, embedding):
        super().__init__()
        assert embedding.weight.device.type == "cpu" and embedding.max_norm is None
        self.embedding = embedding.requires_grad_(False)

    def _apply(self, fn, recurse=True):
        return self

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, ids):
        rows = torch.nn.functional.embedding(ids.cpu(), self.weight, self.embedding.padding_idx).to(
            ids.device
        )
        return rows * self.embedding.embed_scale.to(device=rows.device, dtype=rows.dtype)


def load(host_ple=True):
    import transformers
    from transformers import Gemma4ForConditionalGeneration

    assert transformers.__version__ == SPEC["model"]["transformers"]
    assert not torch.is_inference_mode_enabled()
    started = time.perf_counter()
    model = (
        Gemma4ForConditionalGeneration.from_pretrained(
            SNAPSHOT,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .eval()
        .requires_grad_(False)
    )
    if host_ple:
        text = model.model.language_model
        text.embed_tokens_per_layer = HostEmbedding(text.embed_tokens_per_layer)
    model.cuda()
    for name in ["temperature", "top_p", "top_k"]:
        setattr(model.generation_config, name, None)
    print(
        "MODEL",
        json.dumps(
            {
                "load_seconds": time.perf_counter() - started,
                "host_ple": host_ple,
                **accounting(model),
            }
        ),
        flush=True,
    )
    return model


def prompt(tok, content):
    return tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def text_layers(model):
    return model.model.language_model.layers
