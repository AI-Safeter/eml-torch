"""Count full-model parameters separately from a replaced block's reduction."""

import json
import os

import torch
from runtime import RUNS, configure
from transformers import AutoConfig, AutoModelForCausalLM


def main():
    configure("qwen17b")
    config = AutoConfig.from_pretrained(os.environ["EMLTORCH_MODEL_PATH"])
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    count = sum(p.numel() for p in model.parameters())
    out = RUNS / "whole-block"
    selected = json.loads((out / "selection.json").read_text())
    block = sum(p.numel() for p in model.model.layers[selected["layer"]].mlp.parameters())
    rows = {}
    for kind, spec in selected["students"].items():
        if "parameters" not in spec:
            continue
        deployed = spec["parameters"]
        rows[kind] = {
            "original_model_parameters": count,
            "original_block_parameters": block,
            "deployed_student_parameters": deployed,
            "replaced_model_parameters": count - block + deployed,
            "full_model_parameter_reduction_percent": 100 * (block - deployed) / count,
        }
    record = {
        "measurement": "Architecture parameter counts; not measured peak GPU memory. Paired runtime benchmarks keep both alternatives resident.",
        "students": rows,
    }
    (out / "model-storage.json").write_text(json.dumps(record, indent=2) + "\n")
    print("FULL-MODEL STORAGE COUNTED", rows, flush=True)


if __name__ == "__main__":
    main()
