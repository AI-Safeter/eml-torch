"""Download only pinned model snapshots needed for the confirmation study."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen17b", "qwen4b", "smollm"])
    args = parser.parse_args()
    spec = json.loads((ROOT / "models.json").read_text())[args.model]
    path = snapshot_download(
        spec["id"],
        revision=spec["revision"],
        max_workers=2,
        allow_patterns=["*.json", "*.safetensors", "tokenizer.model", "*.txt", "*.md", "LICENSE*"],
    )
    (ROOT / f"model-{args.model}-ready.json").write_text(
        json.dumps({**spec, "path": path}, indent=2) + "\n"
    )
    print("MODEL READY", args.model, path, flush=True)


if __name__ == "__main__":
    main()
