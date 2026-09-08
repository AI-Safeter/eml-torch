"""Expose native Gemma 4 execution to the frozen scalar-component pipeline."""

import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
SPEC = json.loads((HERE / "model.json").read_text())
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
    / SPEC["revision"]
)


class Body:
    def __init__(self, native):
        self.native = native

    @property
    def layers(self):
        return self.native.language_model.layers

    def __call__(self, *args, **kwargs):
        return self.native(*args, **kwargs)


class Model(torch.nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native
        self.body = Body(native.model)

    @property
    def model(self):
        return self.body

    @property
    def config(self):
        return self.native.config.text_config

    @property
    def lm_head(self):
        return self.native.lm_head

    def forward(self, *args, **kwargs):
        return self.native(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.native.generate(*args, **kwargs)


def load():
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(SNAPSHOT, padding_side="left")
    native = (
        Gemma4ForConditionalGeneration.from_pretrained(
            SNAPSHOT, dtype=torch.float32, attn_implementation="sdpa"
        )
        .eval()
        .cuda()
    )
    native.generation_config.temperature = None
    native.generation_config.top_p = None
    native.generation_config.top_k = None
    return Model(native).eval(), tok


def load_mlp(layer):
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP

    config = AutoConfig.from_pretrained(SNAPSHOT).text_config
    with safe_open(SNAPSHOT / "model.safetensors", framework="pt") as file:
        state = {
            name: file.get_tensor(f"model.language_model.layers.{layer}.mlp.{name}")
            for name in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
        }
    mlp = Gemma4TextMLP(config, layer).float().cuda().eval().requires_grad_(False)
    mlp.load_state_dict(state)
    return mlp


def logits(model, inp):
    # Native forward includes Gemma's final logit soft cap and per-layer embeddings.
    return model.native(**inp, use_cache=False, logits_to_keep=1).logits[:, -1].float()


def install():
    import transformers

    assert torch.__version__ == SPEC["torch"]
    assert transformers.__version__ == SPEC["transformers"]
    os.environ.update(
        EMLTORCH_MODEL_ID=SPEC["id"],
        EMLTORCH_MODEL_REVISION=SPEC["revision"],
        EMLTORCH_MODEL_PATH=str(SNAPSHOT),
        EMLTORCH_RESEARCH_ROOT=str(STUDY.parent.parent / "emltorch-robustness-runs/gemma"),
    )
    sys.path.insert(0, str(STUDY / "src"))
    import model_io

    model_io.load, model_io.load_mlp, model_io.logits = load, load_mlp, logits
    import evaluate_heads

    if not getattr(evaluate_heads.Replacements, "gemma_native_sparse", False):
        base = evaluate_heads.Replacements

        class GemmaReplacements(base):
            gemma_native_sparse = True

            def coefficient(self, h, kind):
                if kind in self.sparse:
                    state = self.sparse[kind]
                    activity = torch.nn.functional.gelu(h @ state["gate"].T, approximate="tanh")
                    return (activity * (h @ state["up"].T)) @ state["readout"] + state["bias"]
                return super().coefficient(h, kind)

        evaluate_heads.Replacements = GemmaReplacements
