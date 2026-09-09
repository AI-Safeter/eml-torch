"""Generate with a standalone complete-MLP replacement and no activation dataset."""

import argparse

import torch

from gemma_mechanisms.runtime import accounting, load, prompt, setup, text_layers, tokenizer

from .export import load_export


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hybrid", action="store_true", help="Load a SiLU/EML hybrid-study checkpoint")
    p.add_argument("--prompt", default="Reply with only the integer value of 731 + 862.")
    p.add_argument("--tokens", type=int, default=64)
    a = p.parse_args()
    setup(73)
    model = load()
    tok = tokenizer()
    removed = text_layers(model)[26].mlp
    original_ids = {id(p) for p in removed.parameters()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Removed MLP executed")

    removed.forward = forbidden
    removed.cpu()
    if a.hybrid:
        from .hybrid_model import load_hybrid

        replacement = load_hybrid(a.checkpoint)
    else:
        replacement = load_export(a.checkpoint)
    text_layers(model)[26].mlp = replacement
    assert not original_ids.intersection(id(p) for p in model.parameters())
    del removed
    inputs = tok(prompt(tok, a.prompt), add_special_tokens=False, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=a.tokens)
    print(tok.decode(output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True))
    print("Installed replacement:", accounting(text_layers(model)[26].mlp))


if __name__ == "__main__":
    main()
