"""Optional evaluation cache for layers strictly before a scalar intervention.

Use only with a fixed eval-mode model and interventions at or after `stop`.
Generation and calls outside `logits` execute the original layer forwards.
"""

from functools import partial

import torch


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone(item) for item in value)
    if value is None:
        return None
    raise TypeError(f"Unsupported cached layer output: {type(value)}")


class PrefixCache:
    def __init__(self, model, stop, logits):
        assert not model.training and 0 < stop < len(model.model.layers)
        self.model, self.original_logits = model, logits
        self.layers = list(model.model.layers[:stop])
        assert not any(
            module._forward_hooks or module._forward_pre_hooks
            for layer in self.layers
            for module in layer.modules()
        ), "Only interventions after the cached prefix are supported"
        self.originals = [layer.forward for layer in self.layers]
        self.saved_inputs = None
        self.outputs = {}
        self.active = self.reuse = False
        self.executions = self.hits = 0
        for index, (layer, forward) in enumerate(zip(self.layers, self.originals)):
            layer.forward = partial(self.forward, index, forward)

    def forward(self, index, original, *args, **kwargs):
        if not self.active:
            return original(*args, **kwargs)
        if self.reuse:
            self.hits += 1
            return clone(self.outputs[index])
        output = original(*args, **kwargs)
        self.outputs[index] = clone(output)
        self.executions += 1
        return output

    def logits(self, model, inputs):
        assert model is self.model and not model.training and not torch.is_grad_enabled()
        assert not self.active, "Reentrant cached forward"
        assert set(inputs) == {"input_ids", "attention_mask"}
        self.reuse = self.saved_inputs is not None and all(
            inputs[key].shape == saved.shape
            and inputs[key].dtype == saved.dtype
            and inputs[key].device == saved.device
            and torch.equal(inputs[key], saved)
            for key, saved in self.saved_inputs.items()
        )
        if not self.reuse:
            self.outputs.clear()
            self.saved_inputs = {key: value.clone() for key, value in inputs.items()}
        self.active = True
        try:
            result = self.original_logits(model, inputs)
            assert len(self.outputs) == len(self.layers)
            return result
        except BaseException:
            self.saved_inputs = None
            self.outputs.clear()
            raise
        finally:
            self.active = False

    def close(self):
        for layer, forward in zip(self.layers, self.originals):
            layer.forward = forward
        self.saved_inputs = None
        self.outputs.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
