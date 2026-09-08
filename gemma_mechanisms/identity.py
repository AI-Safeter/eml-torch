"""Count the configured model, distinguishing parameters from stored buffers."""

import json
import struct

import torch

from .runtime import HERE, SNAPSHOT, SPEC, accounting, digest, save


def main():
    from transformers import AutoConfig, Gemma4ForConditionalGeneration

    config = AutoConfig.from_pretrained(SNAPSHOT, local_files_only=True)
    assert config.model_type == "gemma4" and config.text_config.num_hidden_layers == 35
    with torch.device("meta"):
        model = Gemma4ForConditionalGeneration(config)
    counts = accounting(model)
    with (SNAPSHOT / "model.safetensors").open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    layer = model.model.language_model.layers[SPEC["replacement"]["layer"]]
    groups = {}
    for name, tensor in model.named_parameters():
        if ".layers." in name and name.startswith("model.language_model."):
            group = "text_mlp" if ".mlp." in name else "text_layer_other"
        elif name.startswith("model.language_model."):
            group = "text_embeddings_and_final_norm"
        else:
            group = "multimodal_and_other"
        groups[group] = groups.get(group, 0) + tensor.numel()
    record = {
        "model": SPEC["model"],
        "config_sha256": digest(SNAPSHOT / "config.json"),
        "safetensors_header_sha256": __import__("hashlib")
        .sha256(json.dumps(header, sort_keys=True).encode())
        .hexdigest(),
        "architecture": config.architectures,
        "hidden_size": config.text_config.hidden_size,
        "layers": config.text_config.num_hidden_layers,
        "kv_shared_layers": config.text_config.num_kv_shared_layers,
        "parameter_groups": groups,
        "parameters": counts["parameters"],
        "registered_buffers": counts["buffers"],
        "native_mlp_parameters": accounting(layer.mlp)["parameters"],
        "native_pre_post_norm_parameters": (
            accounting(layer.pre_feedforward_layernorm)["parameters"]
            + accounting(layer.post_feedforward_layernorm)["parameters"]
        ),
        "tie_word_embeddings": config.text_config.tie_word_embeddings,
        "notes": "Meta-device architecture count includes all modalities and embeddings; buffer count is separate.",
    }
    save(HERE / "model-identity.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
