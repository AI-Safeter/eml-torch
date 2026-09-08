"""Fold affine statistics and install a teacher-free complete MLP replacement."""

import argparse
import copy
import gc
import json

import torch
from torch import nn

from .runtime import accounting, load, prompt, setup, tokenizer
from .student import install, load_student


class DeployedStudent(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.kind = student.kind
        self.training_rank = student.width
        self.stages = copy.deepcopy(student.stages)
        self.encoder = copy.deepcopy(student.encoder)
        self.decoder = copy.deepcopy(student.decoder)
        with torch.no_grad():
            ew = student.encoder.weight.double() / student.xstd.double()
            eb = student.encoder.bias.double() - ew @ student.xmean.double()
            dw = student.decoder.weight.double() * student.yscale.double()
            db = student.decoder.bias.double() * student.yscale.double() + student.ymean.double()
            self.encoder.weight.copy_(ew)
            self.encoder.bias.copy_(eb)
            self.decoder.weight.copy_(dw)
            self.decoder.bias.copy_(db)
            self.collapsed_linear = student.kind == "linear" and (
                student.dimension * student.dimension + student.dimension
                <= sum(p.numel() for p in self.encoder.parameters())
                + sum(p.numel() for p in self.decoder.parameters())
            )
            if self.collapsed_linear:
                affine = nn.Linear(student.dimension, student.dimension)
                affine.weight.copy_(dw @ ew)
                affine.bias.copy_(dw @ eb + db)
                self.encoder, self.decoder = nn.Identity(), affine

    def forward(self, x):
        value = self.encoder(x)
        for stage in self.stages:
            value = stage(value)
        return self.decoder(value)


def load_deployed(path, device="cuda", dtype=torch.bfloat16):
    trained = load_student(path, device="cpu", dtype=torch.float32)
    return DeployedStudent(trained).to(device=device, dtype=dtype).eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", default="What is 573 + 846? Return only the answer.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--resident", action="store_true", help="Keep the PLE table on GPU too")
    args = parser.parse_args()
    setup(73)
    model, tok = load(host_ple=not args.resident), tokenizer()
    original = install(model, load_deployed(args.checkpoint))
    removed_ids = {id(p) for p in original.parameters()}
    del original
    gc.collect()
    torch.cuda.empty_cache()
    assert not removed_ids.intersection(id(p) for p in model.parameters())
    print(json.dumps(accounting(model)), flush=True)
    inp = tok(prompt(tok, args.text), return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        result = model.generate(
            **inp, do_sample=False, max_new_tokens=args.max_new_tokens, use_cache=True
        )
    print(tok.decode(result[0, inp.input_ids.shape[1] :], skip_special_tokens=True))


if __name__ == "__main__":
    main()
