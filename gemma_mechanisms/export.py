"""Export only the deployed BF16 student, and verify a portable reload on CUDA."""

import argparse
import gc
import json

import torch

from .deploy import DeployedStudent, load_deployed
from .runtime import SPEC, accounting, digest, load, prompt, setup, tokenizer
from .student import Student, install


def load_export(path, device="cuda"):
    record = torch.load(path, map_location="cpu", weights_only=True)
    assert record["format"] == "eml-complete-mlp-v1"
    assert record["base_model"] == SPEC["model"]
    student = DeployedStudent(Student(**record["specification"]))
    student = student.to(dtype=torch.bfloat16)
    student.load_state_dict(record["state"], strict=True)
    return student.to(device).eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--validation-input")
    parser.add_argument(
        "--run", help="Run an exported student, physically removing the original MLP"
    )
    parser.add_argument("--text", default="What is 573 + 846? Return only the integer answer.")
    args = parser.parse_args()
    setup(73)
    if args.run:
        model, tok = load(), tokenizer()
        removed = install(model, load_export(args.run))
        ids = {id(p) for p in removed.parameters()}
        del removed
        gc.collect()
        torch.cuda.empty_cache()
        assert not ids.intersection(id(p) for p in model.parameters())
        inputs = tok(prompt(tok, args.text), return_tensors="pt", add_special_tokens=False).to(
            "cuda"
        )
        with torch.inference_mode():
            result = model.generate(**inputs, do_sample=False, max_new_tokens=24, use_cache=True)
        print(json.dumps(accounting(model)), flush=True)
        print(tok.decode(result[0, inputs.input_ids.shape[1] :], skip_special_tokens=True))
        return
    assert args.checkpoint and args.output and args.validation_input
    checkpoint = torch.load(args.checkpoint, weights_only=True)
    student = load_deployed(args.checkpoint)
    torch.save(
        {
            "format": "eml-complete-mlp-v1",
            "base_model": SPEC["model"],
            "layer": SPEC["replacement"]["layer"],
            "specification": checkpoint["specification"],
            "state": {k: v.cpu() for k, v in student.state_dict().items()},
            "source_checkpoint_sha256": digest(args.checkpoint),
            "scope": "Experimental complete-MLP replacement. Consult held-out quality results before use.",
        },
        args.output,
    )
    reloaded = load_export(args.output)
    data = torch.load(args.validation_input, weights_only=True)["x"]
    count = 0
    with torch.inference_mode():
        for block in data.split(512):
            x = block.cuda()
            assert torch.equal(
                student(x).contiguous().view(torch.uint8),
                reloaded(x).contiguous().view(torch.uint8),
            ), "Export changed deployed BF16 computation"
            count += len(x)
    print(
        json.dumps(
            {
                "export": args.output,
                "sha256": digest(args.output),
                "cuda_bitwise_validation_tokens": count,
                "accounting": accounting(reloaded),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
