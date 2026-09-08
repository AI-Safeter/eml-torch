"""Whole-output fidelity on real, unpadded held-out token activations."""

import json

import torch
from runtime import HERE, RUNS, configure
from whole_evaluate import select, students


def main():
    configure("qwen17b")
    from data import STYLES, save
    from model_io import expand, load, setup, text

    setup()
    out = RUNS / "whole-block"
    selected = select(out)
    replacements = students(out, selected)
    model, tok = load()
    results = {}
    with torch.inference_mode():
        for domain in ["language", "arithmetic"]:
            if domain == "language":
                sequences = torch.load(out / "language-test.pt", weights_only=True).tolist()
            else:
                rows = []
                for op in ["add", "multiply", "divide"]:
                    problems = json.loads((HERE / "data" / op / "problems.json").read_text())
                    rows.extend(expand(problems["test"], STYLES))
                sequences = [
                    tok.encode(text(tok, r) + str(r["answer"]), add_special_tokens=False)
                    for r in rows
                ]
            totals = {
                kind: torch.zeros(3, dtype=torch.float64, device="cuda") for kind in replacements
            }
            for start in range(0, len(sequences), 8):
                inp = tok.pad(
                    {"input_ids": sequences[start : start + 8]}, padding=True, return_tensors="pt"
                ).to("cuda")
                mask = inp.attention_mask.bool()

                def hook(module, args, output):
                    h, y = args[0][mask], output[mask]
                    for kind, student in replacements.items():
                        pred = student(h)
                        assert torch.isfinite(pred).all()
                        totals[kind] += torch.stack(
                            [
                                (pred - y).double().square().sum(),
                                y.double().square().sum(),
                                torch.tensor(y.numel(), device="cuda", dtype=torch.float64),
                            ]
                        )

                handle = model.model.layers[selected["layer"]].mlp.register_forward_hook(hook)
                try:
                    model.model(**inp, use_cache=False)
                finally:
                    handle.remove()
                if start % 512 == 0:
                    print("WHOLE FIDELITY", domain, start, len(sequences), flush=True)
            results[domain] = {
                kind: {
                    "output_nrmse": float((v[0] / v[1]).sqrt()),
                    "output_rmse": float((v[0] / v[2]).sqrt()),
                    "scalar_output_values": int(v[2]),
                    "normalization": "RMS original MLP output, including its mean",
                }
                for kind, v in totals.items()
            }
            save(out / "output-fidelity.json", results)
    print("WHOLE FIDELITY COMPLETE", flush=True)


if __name__ == "__main__":
    main()
