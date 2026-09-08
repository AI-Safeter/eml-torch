"""Reuse the fixed Gemma study's unchanged layers during generation prefill.

This is restricted to the pinned, frozen text model and native greedy study
generation. Interventions must remain after the cached prefix. Decode forwards
and calls outside generate execute normally. It is not a serving cache API.
"""

import copy
from functools import partial

import torch


def same(left, right):
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and left.device == right.device
            and torch.equal(left, right)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    return left == right


class GenerationPrefix:
    def __init__(self, model, stop):
        from transformers import DynamicCache

        assert not model.training and model.config.model_type == "gemma4_text"
        assert 0 < stop < len(model.model.layers)
        assert all(layer.self_attn.is_kv_shared_layer for layer in model.model.layers[stop:])
        self.model, self.stop, self.cache_type = model, stop, DynamicCache
        self.layers = list(model.model.layers[:stop])
        assert not any(
            module._forward_hooks or module._forward_pre_hooks
            for layer in self.layers
            for module in layer.modules()
        ), "Only interventions after the cached prefix are supported"
        self.originals = [layer.forward for layer in self.layers]
        self.active = self.reuse = self.generating = self.closed = False
        self.enabled = True
        self.saved = self.snapshot = None
        self.outputs = {}
        self.hits = self.misses = 0
        for index, (layer, forward) in enumerate(zip(self.layers, self.originals)):
            layer.forward = partial(self.forward, index, forward)
        self.pre = model.native.register_forward_pre_hook(self.before, with_kwargs=True)
        self.post = model.native.register_forward_hook(
            self.after, with_kwargs=True, always_call=True
        )
        self.generate_original = model.generate
        model.generate = self.generate

    def clear(self):
        self.saved = self.snapshot = None
        self.outputs.clear()

    def generate(self, *args, **kwargs):
        assert not self.closed and not self.generating and not torch.is_grad_enabled()
        assert not args and not self.model.training
        fixed = {
            "max_new_tokens": 16,
            "do_sample": False,
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": True,
            "logits_to_keep": 1,
        }
        assert set(kwargs) == {*fixed, "input_ids", "attention_mask", "pad_token_id"}
        assert all(type(kwargs[k]) is type(v) and kwargs[k] == v for k, v in fixed.items())
        # Transformers fills an unset value from its native default (one beam).
        assert self.model.native.generation_config.num_beams in [None, 1]
        self.generating = True
        try:
            return self.generate_original(**kwargs)
        except BaseException:
            self.clear()
            raise
        finally:
            self.generating = self.active = False

    def before(self, module, args, kwargs):
        self.active = False
        if not self.enabled or not self.generating:
            return
        assert not args and not torch.is_grad_enabled() and not module.training
        cache = kwargs.get("past_key_values")
        assert type(cache) is self.cache_type and not cache.offloading
        if cache.get_seq_length() != 0:
            return
        assert all(not layer.is_initialized for layer in cache.layers)
        signature = {k: v for k, v in kwargs.items() if k != "past_key_values"}
        assert set(signature) == {
            "input_ids",
            "attention_mask",
            "position_ids",
            "logits_to_keep",
            "return_dict",
            "use_cache",
        }
        assert signature["use_cache"] is True and signature["return_dict"] is True
        for name in ["input_ids", "attention_mask", "position_ids"]:
            assert isinstance(signature[name], torch.Tensor) and signature[name].ndim == 2
        assert signature["input_ids"].shape[1] > 0
        self.reuse = self.saved is not None and same(signature, self.saved)
        if not self.reuse:
            self.clear()
            self.saved = copy.deepcopy(signature)
        self.active = True

    def after(self, module, args, kwargs, output):
        if self.active:
            if output is None:
                self.clear()
            else:
                assert len(self.outputs) == self.stop and self.snapshot is not None
        self.active = False

    def forward(self, index, original, *args, **kwargs):
        if not self.active:
            return original(*args, **kwargs)
        cache, shared = kwargs["past_key_values"], kwargs["shared_kv_states"]
        if self.reuse:
            if index == 0:
                # Native masks and positions were constructed with an empty cache.
                # Restore mutable state here, before any uncached layer executes.
                state, restored_shared = copy.deepcopy(self.snapshot)
                cache.__dict__.clear()
                cache.__dict__.update(state)
                shared.clear()
                shared.update(restored_shared)
                self.hits += 1
            return self.outputs[index].clone()
        output = original(*args, **kwargs)
        assert isinstance(output, torch.Tensor)
        self.outputs[index] = output.clone()
        if index == self.stop - 1:
            # One deepcopy preserves aliases between sliding cache and shared KV.
            # A fresh copy on every hit prevents decoding from changing this snapshot.
            self.snapshot = copy.deepcopy((cache.__dict__, shared))
            self.misses += 1
        return output

    def close(self):
        if self.closed:
            return
        self.pre.remove()
        self.post.remove()
        self.model.generate = self.generate_original
        for layer, original in zip(self.layers, self.originals):
            layer.forward = original
        self.clear()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
