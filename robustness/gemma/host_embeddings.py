"""Keep Gemma's frozen per-layer lookup table on CPU; compute on CUDA in FP32."""

import torch
from torch import nn


class HostEmbedding(nn.Module):
    """Transfer selected embedding rows, preserving the native GPU scaling step.

    This module deliberately keeps its registered table and scale on CPU when
    the parent moves to CUDA. It is restricted to the frozen FP32 study model.
    """

    def __init__(self, embedding):
        super().__init__()
        assert embedding.weight.dtype == torch.float32
        assert embedding.weight.device.type == "cpu" and embedding.max_norm is None
        self.embedding = embedding.requires_grad_(False)

    def _apply(self, fn, recurse=True):
        return self

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, input_ids):
        assert input_ids.device.type == "cuda"
        rows = torch.nn.functional.embedding(
            input_ids.cpu(), self.embedding.weight, self.embedding.padding_idx
        ).to(input_ids.device)
        # Moving the lookup instead of calling the CPU module keeps this
        # multiplication on the same device as the original implementation.
        return rows * self.embedding.embed_scale.to(device=rows.device, dtype=rows.dtype)


def load():
    from gemma.adapter import SNAPSHOT, Model

    assert not torch.is_inference_mode_enabled(), "Construct the model outside inference_mode"
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(SNAPSHOT, padding_side="left")
    native = Gemma4ForConditionalGeneration.from_pretrained(
        SNAPSHOT, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    ).eval()
    text = native.model.language_model
    text.embed_tokens_per_layer = HostEmbedding(text.embed_tokens_per_layer)
    native.cuda()
    native.generation_config.temperature = None
    native.generation_config.top_p = None
    native.generation_config.top_k = None
    print(
        "GEMMA HOST EMBEDDING ENABLED",
        "table_bytes",
        text.embed_tokens_per_layer.weight.numel() * 4,
        "gpu_parameter_bytes",
        sum(p.numel() * p.element_size() for p in native.parameters() if p.is_cuda),
        flush=True,
    )
    return Model(native).eval(), tok
