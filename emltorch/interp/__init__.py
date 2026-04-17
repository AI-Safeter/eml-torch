"""
emltorch.interp - plumbing for transformer activations -> symbolic regression.

The typical workflow:

    1. Run a model on a list of prompts.
    2. Cache the residual-stream activations at some layer.
    3. Reduce the per-example activation vectors to a small set of features
       (usually via PCA).
    4. Call `emltorch.fit(features, target)` to discover a symbolic formula.

This module provides step (1)-(2) only. Dimensionality reduction and the
actual `fit()` call stay in user-space because the right reduction depends
on the interpretability question.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def _resolve_module(model: nn.Module, layer) -> nn.Module:
    """Resolve a layer identifier into the actual submodule.

    Accepts:
        - int: treat as index into `model.model.layers[i]` for HF LLMs.
          Falls back to scanning named_modules for a `.layers.{i}` entry.
        - str: dotted name looked up via getattr / split-on-dot, matching
          `dict(model.named_modules())`.
    """
    if isinstance(layer, int):
        # Preferred: HF causal-LM / decoder convention.
        try:
            return model.model.layers[layer]
        except (AttributeError, IndexError, TypeError):
            pass
        named = dict(model.named_modules())
        for suffix in (f"model.layers.{layer}", f"layers.{layer}",
                       f"transformer.h.{layer}"):
            if suffix in named:
                return named[suffix]
        raise ValueError(
            f"Could not resolve layer index {layer} - tried model.model.layers,"
            f" named_modules('model.layers.{layer}'), etc."
        )
    if isinstance(layer, str):
        # Dotted name - walk attributes.
        named = dict(model.named_modules())
        if layer in named:
            return named[layer]
        mod = model
        for part in layer.split("."):
            mod = getattr(mod, part)
        return mod
    raise TypeError(f"layer must be int or str, got {type(layer)}")


def _move_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device)
    return batch


def from_transformer_hook(
    model: nn.Module,
    layer,
    inputs,
    position=-1,
    target: str | Callable = "logits",
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `model` on `inputs`, cache activations at `layer`, return features and targets.

    Args:
        model:    an `nn.Module` (typically a HF causal LM).
        layer:    int layer index or full dotted module name.
        inputs:   either
                    - a list of dicts (HF-style: `{"input_ids": ..., "attention_mask": ...}`),
                      one per example; or
                    - a token tensor of shape (B, T) - treated as `input_ids`.
        position: which token position to cache. Accepts int or slice.
                  Defaults to -1 (last token).
        target:   "logits" - per-example max logit at `position`.
                  "probs"  - per-example max probability at `position`.
                  callable(activations, logits) -> (N,) tensor for custom
                  targets.
        device:   torch device string.

    Returns:
        activations: (N, D) tensor. One feature vector per input.
        targets:     (N,)  tensor. One scalar target per input.

    The returned tensors are ready to be fed into `emltorch.fit()` after
    an appropriate dimensionality reduction:

        >>> acts, tgt = emltorch.interp.from_transformer_hook(
        ...     model, layer=20, inputs=prompts
        ... )
        >>> # acts shape: (N, D_model). We need to reduce D_model -> scalar features.
        >>> # The typical workflow is: PCA the activations first, then fit EML.
        >>> from sklearn.decomposition import PCA
        >>> feats = torch.from_numpy(PCA(n_components=4).fit_transform(acts.cpu()))
        >>> result = emltorch.fit(feats.T.float(), tgt.cpu().float(), depth=3)
    """
    # Put model in inference mode (no dropout, no grad-tracking side effects).
    model.train(False)
    model.to(device)
    module = _resolve_module(model, layer)

    cache: list[torch.Tensor] = []

    def hook(_mod, _inp, out):
        # Modules can return a tensor or (tensor, ...) tuple. Take the first.
        t = out[0] if isinstance(out, tuple) else out
        cache.append(t.detach())

    handle = module.register_forward_hook(hook)

    # --- Normalize inputs into an iterable of batches ---
    if torch.is_tensor(inputs):
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        batches = [{"input_ids": inputs[i : i + 1]} for i in range(inputs.shape[0])]
    elif isinstance(inputs, list):
        batches = inputs
    else:
        raise TypeError(
            f"inputs must be a list of dicts or a (B, T) tensor, got {type(inputs)}"
        )

    activations_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    try:
        with torch.no_grad():
            for batch in batches:
                cache.clear()
                batch_on_device = _move_to_device(batch, device)
                if isinstance(batch_on_device, dict):
                    output = model(**batch_on_device)
                else:
                    output = model(batch_on_device)

                if not cache:
                    raise RuntimeError(
                        f"Forward hook on {layer!r} did not fire - is the "
                        f"module actually on the forward path?"
                    )
                act = cache[-1]  # (1, T, D) or (B, T, D)

                # Select token position(s).
                if isinstance(position, slice):
                    act_slice = act[:, position, :]
                    # Flatten selected positions into the batch dim.
                    feat = act_slice.reshape(-1, act_slice.shape[-1])
                else:
                    feat = act[:, position, :]

                # --- Compute target ---
                logits = getattr(output, "logits", None)
                if callable(target) and not isinstance(target, str):
                    tgt = target(feat, logits)
                elif target == "logits":
                    lg = logits[:, position, :]
                    tgt = lg.max(dim=-1).values
                elif target == "probs":
                    lg = logits[:, position, :]
                    tgt = torch.softmax(lg, dim=-1).max(dim=-1).values
                else:
                    raise ValueError(
                        f"target must be 'logits', 'probs', or callable; got {target!r}"
                    )

                activations_list.append(feat.reshape(-1, feat.shape[-1]).cpu())
                targets_list.append(tgt.reshape(-1).cpu())
    finally:
        handle.remove()

    activations = torch.cat(activations_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    return activations, targets


__all__ = ["from_transformer_hook"]
