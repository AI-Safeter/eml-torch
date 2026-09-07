"""Test frozen local equations on unseen problems and upstream interventions."""

import hashlib
import json
import re
import time

import torch
from common import OUT, dump, first_targets, inputs, load, logits, margin, setup
from replacement import Replacement

KINDS = [
    "original",
    "restore",
    "eml_observed",
    "eml_intervention",
    "linear",
    "quadratic",
    "cubic",
    "fourier",
    "network",
    "mean",
    "zero",
    "shuffle",
    "matched_random",
]


def generation(model, tok, inp):
    gen = model.generate(
        **inp,
        max_new_tokens=4,
        do_sample=False,
        use_cache=True,
        pad_token_id=tok.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        logits_to_keep=1,
    )
    text = tok.batch_decode(gen.sequences[:, inp.input_ids.shape[1] :], skip_special_tokens=True)
    return gen.scores[0].float(), text


def answer(text):
    found = re.match(r"^\s*(\d+)\s*[.!]?\s*$", text)
    return int(found.group(1)) if found else None


def ordinary(model, tok, rep, rows, style):
    records = {k: [] for k in KINDS}
    for start in range(0, len(rows), 16):
        batch = rows[start : start + 16]
        inp = inputs(tok, batch, style=style)
        base, base_text = generation(model, tok, inp)
        base_lp = base.log_softmax(-1)
        base_p = base_lp.exp()
        for kind in KINDS:
            handles, cache = rep.hooks(model, kind)
            scores, text = generation(model, tok, inp)
            for h in handles:
                h.remove()
            assert torch.isfinite(scores).all(), (style, kind)
            lp = scores.log_softmax(-1)
            kl = (base_p * (base_lp - lp)).sum(1)
            if kind == "restore":
                assert torch.allclose(base, scores, atol=1e-4, rtol=1e-4)
            true = cache["true_coefficient"]
            pred = cache["predicted_coefficient"]
            for i, r in enumerate(batch):
                records[kind].append(
                    {
                        "a": r["a"],
                        "b": r["b"],
                        "c": r["c"],
                        "answer": r["sum"],
                        "text": text[i],
                        "base_text": base_text[i],
                        "correct": answer(text[i]) == r["sum"],
                        "base_correct": answer(base_text[i]) == r["sum"],
                        "answer_agreement": answer(text[i]) == answer(base_text[i]),
                        "first_digit_agreement": bool(scores[i].argmax() == base[i].argmax()),
                        "first_digit_correct": bool(
                            scores[i].argmax() == first_targets(tok, [r])[0]
                        ),
                        "kl": float(kl[i]),
                        "true_coefficient": float(true[i]),
                        "predicted_coefficient": float(pred[i]),
                    }
                )
        print("ORDINARY", style, start + len(batch), "/", len(rows), flush=True)
    return records


def interventions(model, tok, rep, rows, style):
    kinds = [
        "original",
        "restore",
        "eml_observed",
        "eml_intervention",
        "linear",
        "cubic",
        "network",
        "mean",
        "matched_random",
    ]
    result = {k: [] for k in kinds}
    alphas = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
    for start in range(0, len(rows), 16):
        batch = rows[start : start + 16]
        corrupted = inputs(tok, batch, True, style)
        hs = []
        for role in [False, True]:
            handles, cache = rep.hooks(model, "original")
            reference = logits(model, inputs(tok, batch, role, style))
            for h in handles:
                h.remove()
            hs.append(cache["input"].clone())
        clean_target = first_targets(tok, batch)
        corrupt_target = first_targets(tok, batch, True)
        for alpha in alphas:
            mixed = hs[1] * (1 - alpha) + hs[0] * alpha
            scores_by_kind = {}
            caches = {}
            for kind in kinds:
                handles, cache = rep.hooks(model, kind, mixed)
                score = logits(model, corrupted)
                for h in handles:
                    h.remove()
                scores_by_kind[kind] = score
                caches[kind] = cache
            base = scores_by_kind["original"]
            if alpha == 0.0:
                assert torch.allclose(base, reference, atol=1e-4, rtol=1e-4)
            lp = base.log_softmax(-1)
            p = lp.exp()
            assert torch.allclose(base, scores_by_kind["restore"], atol=1e-4, rtol=1e-4)
            for kind in kinds:
                score = scores_by_kind[kind]
                assert torch.isfinite(score).all(), kind
                kl = (p * (lp - score.log_softmax(-1))).sum(1)
                effect = margin(score, clean_target, corrupt_target)
                for i, r in enumerate(batch):
                    result[kind].append(
                        {
                            "a": r["a"],
                            "b": r["b"],
                            "c": r["c"],
                            "alpha": alpha,
                            "margin": float(effect[i]),
                            "kl_vs_same_intervention_original": float(kl[i]),
                            "true_coefficient": float(caches[kind]["true_coefficient"][i]),
                            "predicted_coefficient": float(
                                caches[kind]["predicted_coefficient"][i]
                            ),
                        }
                    )
        print("INTERVENTION", style, start + len(batch), "/", len(rows), flush=True)
    return result


def main():
    setup()
    model, tok = load()
    rep = Replacement()
    data = json.loads((OUT / "problems.json").read_text())
    # Deployment expression must match the core result before any test data is opened.
    fitdata = torch.load(OUT / "fit-data.pt", weights_only=True)["validation"]["z"].cuda()
    with torch.inference_mode():
        checks = {}
        for mode in ["observed", "intervention"]:
            fit = torch.load(OUT / f"eml-{mode}.pt", weights_only=False)
            expected = fit.predict(fitdata).cuda()
            actual = rep.formulas["eml_" + mode](fitdata)
            checks[mode] = float((expected - actual).abs().max())
            assert torch.allclose(expected, actual, atol=1e-4, rtol=1e-4), checks
        dump("deployment-checks.json", checks)
        suites = [
            ("test", "prose"),
            ("carry", "prose"),
            ("range", "prose"),
            ("test", "symbolic"),
            ("test", "code"),
        ]
        started = time.perf_counter()
        for split, style in suites:
            name = f"{split}-{style}"
            records = ordinary(model, tok, rep, data[split], style)
            dump(f"ordinary-{name}.json", records)
            # All pairs are evaluated; report dependence between repeated source problems.
            causal = interventions(model, tok, rep, data[split], style)
            dump(f"interventions-{name}.json", causal)
            print("SUITE COMPLETE", name, flush=True)
        dump(
            "run-metadata.json",
            {
                "model": "Qwen3-1.7B",
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "dtype": "float32",
                "tf32": False,
                "seconds": time.perf_counter() - started,
                "source_hashes": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*.py")
                },
            },
        )
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
