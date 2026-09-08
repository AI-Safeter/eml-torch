"""Stream frozen prefix MLP weights; all arithmetic remains native FP32 CUDA."""

import torch
from gemma import adapter
from gemma.host_embeddings import HostEmbedding
from torch import nn


class StreamedMLP(nn.Module):
    def __init__(self, mlp, guard=None):
        super().__init__()
        self.native = mlp.requires_grad_(False)
        assert not list(mlp.buffers())
        self.slots, self.calls, self.guard = [], 0, guard
        seen = set()
        for module in mlp.modules():
            for name, parameter in list(module._parameters.items()):
                if parameter is None:
                    continue
                assert parameter.device.type == "cpu" and parameter.dtype == torch.float32
                assert id(parameter) not in seen, "Shared parameters are unsupported"
                seen.add(id(parameter))
                master = nn.Parameter(parameter.detach().pin_memory(), requires_grad=False)
                module._parameters[name] = master
                self.slots.append((module, name, master))

    def _apply(self, fn, recurse=True):
        return self

    def forward(self, x):
        assert not self.training and x.is_cuda and x.dtype == torch.float32
        assert torch.is_inference_mode_enabled()
        if self.guard is not None:
            self.guard()
        try:
            for module, name, master in self.slots:
                module._parameters[name] = nn.Parameter(
                    master.to(x.device, non_blocking=True), requires_grad=False
                )
            result = self.native(x)
            self.calls += 1
            return result
        finally:
            for module, name, master in self.slots:
                module._parameters[name] = master


def load(guard=None):
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    assert not torch.is_inference_mode_enabled()
    tok = AutoTokenizer.from_pretrained(adapter.SNAPSHOT, padding_side="left")
    native = Gemma4ForConditionalGeneration.from_pretrained(
        adapter.SNAPSHOT, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    ).eval()
    text = native.model.language_model
    text.embed_tokens_per_layer = HostEmbedding(text.embed_tokens_per_layer)
    # All streamed blocks precede the fixed division intervention at layer 33.
    for index in range(17, 33):
        text.layers[index].mlp = StreamedMLP(text.layers[index].mlp, guard).eval()
    native.cuda()
    for name in ["temperature", "top_p", "top_k"]:
        setattr(native.generation_config, name, None)
    print(
        "STREAMED GEMMA PREFIX MLPS",
        "resident_cuda_parameter_bytes",
        sum(p.numel() * p.element_size() for p in native.parameters() if p.is_cuda),
        flush=True,
    )
    return adapter.Model(native).eval(), tok
