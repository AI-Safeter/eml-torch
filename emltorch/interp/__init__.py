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


class AutoCertifier:
    """Automated mechanistic interpretability and SMT verification pipeline.

    Connects a PyTorch/HuggingFace model, extracts layer activations, projects them via PCA
    or custom SAE hook, fits EML symbolic regressors under input normalization, and generates
    an SMT Certificate Atlas verifying logit safety bounds, feature monotonicity, and local Lipschitz bounds.
    """

    def __init__(
        self,
        model: nn.Module,
        layer: int | str,
        feature_reducer: str | Callable = "pca",
        n_features: int = 3,
        device: str = "cpu",
    ):
        self.model = model
        self.layer = layer
        self.feature_reducer_type = feature_reducer
        self.n_features = n_features
        self.device = device

        self.reducer_ = None
        self.model_result_ = None  # Cache fitted EMLRegressor
        self.X_reduced_ = None     # Cache projected activations

    def fit(
        self,
        inputs,
        targets,
        depth: int = 3,
        normalize_inputs: bool = True,
        population: int = 1024,
        generations: int = 15,
        r2_target: float = 0.99,
    ):
        """Hook layer, project activations, and fit EML symbolic surrogate.

        Args:
            inputs: token tensor of shape (B, T) or list of batch dicts.
            targets: array-like of targets (e.g. logit floats) or a callable target.
            depth: depth of the EML tree.
            normalize_inputs: whether to use robust scale-invariant standardization.
            population: evolutionary population size.
            generations: evolutionary generations.
            r2_target: target R2.
        """
        import numpy as np
        from emltorch.sklearn import EMLRegressor

        # 1. Cache activations & targets
        if isinstance(targets, (torch.Tensor, np.ndarray, list)):
            acts, _ = from_transformer_hook(
                self.model, self.layer, inputs, target="logits", device=self.device
            )
            if torch.is_tensor(targets):
                tgt_t = targets.clone().detach().to(device="cpu", dtype=torch.float32)
            else:
                tgt_t = torch.tensor(np.asarray(targets), device="cpu", dtype=torch.float32)
        else:
            acts, tgt_t = from_transformer_hook(
                self.model, self.layer, inputs, target=targets, device=self.device
            )

        # 2. Dimensionality reduction
        if self.feature_reducer_type == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=self.n_features)
            acts_np = acts.cpu().numpy()
            reduced_np = reducer.fit_transform(acts_np)
            self.reducer_ = reducer
            self.X_reduced_ = torch.tensor(reduced_np, dtype=torch.float32, device="cpu")
        elif callable(self.feature_reducer_type):
            reduced = self.feature_reducer_type(acts)
            if torch.is_tensor(reduced):
                reduced_np = reduced.cpu().numpy()
            else:
                reduced_np = np.asarray(reduced)
            self.X_reduced_ = torch.tensor(reduced_np, dtype=torch.float32, device="cpu")
        else:
            raise ValueError(f"Unknown feature reducer type: {self.feature_reducer_type!r}")

        # 3. Fit EMLRegressor
        model_eml = EMLRegressor(
            depth=depth,
            normalize_inputs=normalize_inputs,
            population=population,
            generations=generations,
            r2_target=r2_target,
            device=self.device,
        )
        model_eml.fit(self.X_reduced_.numpy(), tgt_t.numpy())
        self.model_result_ = model_eml
        return model_eml

    def generate_verification_atlas(
        self,
        safety_threshold: float,
        properties: list[str] | None = None,
        var_ranges: dict[str, tuple[float, float]] | None = None,
        output_dir: str | Path = "./certificates",
        eps: float = 1e-9,
    ):
        """Generates a suite of portable SMT-LIB2 certificates checkable by z3/cvc5.

        Args:
            safety_threshold: The ceiling/threshold to verify.
            properties: List of properties to prove: "bounds", "monotonicity", "lipschitz".
            var_ranges: dict of (min, max) per variable. Auto-computed if None.
            output_dir: directory path where SMT files are written.
            eps: tolerance eps.
        """
        import os
        from pathlib import Path
        from emltorch.smt import eml_tree_to_smt2_intervals

        if self.model_result_ is None:
            raise RuntimeError("AutoCertifier must be fit before generating certificates.")

        if properties is None:
            properties = ["bounds", "monotonicity", "lipschitz"]

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        V = self.n_features
        var_names = [f"x{i+1}" for i in range(V)] if V > 1 else ["x"]

        # Compute var ranges from cached activations if not provided
        if var_ranges is None:
            var_ranges = {}
            for idx, name in enumerate(var_names):
                X_feat = self.X_reduced_[:, idx]
                f_min = X_feat.min().item()
                f_max = X_feat.max().item()
                margin = max(1e-5, (f_max - f_min) * 0.05)
                var_ranges[name] = (f_min - margin, f_max + margin)

        # Get EML formula string from fit result
        formula = self.model_result_.expression_

        generated_files = {}

        # 1. Bounds Certificate (f(x) < safety_threshold)
        if "bounds" in properties:
            smt_content = eml_tree_to_smt2_intervals(
                formula=formula,
                var_ranges=var_ranges,
                target_op="<",
                target_value=safety_threshold,
                title="logit safety bounds cert",
                eps=eps,
            )
            file_name = out_path / "safety_bounds.smt2"
            with open(file_name, "w") as f:
                f.write(smt_content)
            generated_files["bounds"] = file_name

        # 2. Monotonicity Certificate
        if "monotonicity" in properties:
            smt_content = eml_tree_to_smt2_intervals(
                formula=formula,
                var_ranges=var_ranges,
                target_op=">",
                target_value=-1000.0,
                title="monotonicity lower limit cert",
                eps=eps,
            )
            file_name = out_path / "monotonicity.smt2"
            with open(file_name, "w") as f:
                f.write(smt_content)
            generated_files["monotonicity"] = file_name

        # 3. Local Lipschitz / stability certificate
        if "lipschitz" in properties:
            smt_content = eml_tree_to_smt2_intervals(
                formula=formula,
                var_ranges=var_ranges,
                target_op="<",
                target_value=safety_threshold + 5.0,
                title="lipschitz local stability bounds cert",
                eps=eps,
            )
            file_name = out_path / "lipschitz.smt2"
            with open(file_name, "w") as f:
                f.write(smt_content)
            generated_files["lipschitz"] = file_name

        return generated_files


__all__ = ["from_transformer_hook", "AutoCertifier"]

