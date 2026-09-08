"""Freeze validation choices, then test scalar replacements and finite interventions."""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from data import ROOT, STYLES, save
from heads import Head
from model_io import expand, inputs, load, logits, margin, setup, targets

ALPHAS = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(out, directories):
    path = out / "selection.json"
    if path.exists():
        selected = json.loads(path.read_text())
        assert selected["directories"] == directories
        for spec in selected["heads"].values():
            assert digest(out / spec["checkpoint"]) == spec["sha256"]
        for spec in selected["sparse"].values():
            assert digest(out / spec["checkpoint"]) == spec["sha256"]
        for name, sha256 in selected["component_sha256"].items():
            assert digest(out / name) == sha256
        for spec in selected.get("linear", {}).values():
            assert digest(out / spec["checkpoint"]) == spec["sha256"]
        assert digest(out / "extra-problems.json") == selected["extra_problems_sha256"]
        return selected
    selected = {}
    searches = {}
    for directory in directories:
        folder = out / directory
        assert (folder / "completed.json").exists(), f"Search still running: {folder}"
        protocol = json.loads((folder / "protocol.json").read_text())
        rows = json.loads((folder / "candidates.json").read_text())
        searches[directory] = json.loads((folder / "completed.json").read_text())
        for family, prefix in [("eml", "eml"), ("neural", "silu")]:
            choices = [
                r
                for r in rows
                if r["status"] == "complete"
                and r["kind"].startswith(prefix)
                and r["response_weight"] == 4
            ]
            assert choices
            best = min(choices, key=lambda r: r["validation_objective"])
            checkpoint = f"{directory}/{best['name']}.pt"
            selected[f"{directory}/{family}"] = {
                **best,
                "checkpoint": checkpoint,
                "features": protocol.get("features", "pls"),
                "rank": protocol.get("rank", 32),
                "legacy_protocol_defaults": "features" not in protocol,
                "sha256": digest(out / checkpoint),
            }
    sparse = json.loads((out / "sparse-neurons.json").read_text())
    sparse_specs = {}
    for label, neurons in [
        ("sparse16", 16),
        ("sparse_selected", sparse["selected_neurons"]),
    ]:
        checkpoint = f"neurons-{neurons}.pt"
        row = next(r for r in sparse["candidates"] if r["neurons"] == neurons)
        sparse_specs[label] = {
            **row,
            "checkpoint": checkpoint,
            "sha256": digest(out / checkpoint),
        }
    linear = json.loads((out / "linear-controls.json").read_text())["selected"]
    linear_file = linear["name"] + ".pt"
    result = {
        "frozen_epoch": time.time(),
        "directories": directories,
        "heads": selected,
        "sparse": sparse_specs,
        "linear": {
            "linear": {
                **linear,
                "checkpoint": linear_file,
                "sha256": digest(out / linear_file),
            }
        },
        "component_sha256": {
            name: digest(out / name) for name in ["component.pt", "component-active.pt"]
        },
        "searches": searches,
        "problems_sha256": digest(out / "problems.json"),
        "extra_problems_sha256": digest(out / "extra-problems.json"),
        "selection": "Minimum validation absolute MSE + 4 times coefficient-response MSE within feature method and EML/neural family; tests not used.",
        "evaluation": {
            "alphas": ALPHAS,
            "primary_response_alphas": ALPHAS[1:-1],
            "generation_tokens": 16,
            "patch_position": "Last prompt position, initial prefill only",
            "patch": "Replace scalar d^T MLP(h) by f(h), preserving orthogonal output and all other components; divide by d^T d.",
            "intervention": "On a corrupted prompt, evaluate the scalar coefficient at an interpolation of corrupted and clean MLP inputs. Other contributions remain corrupted.",
            "primary": "Unseen operand pairs, all three training prompt formats, grouped uncertainty by operand pair",
            "stress": "Four held-out formats, shifted operands, carry holdout, both operands changed, and unconditioned ordinary problems",
            "coordinate_interventions": "First 32 frozen test pairs, three known formats; separately shift active feature 0, 1, 7, or 31 by -0.25 or +0.25 training standard deviations. Local off-manifold diagnostic; no model selection from these results.",
            "criteria": {
                "accuracy_drop_pp_max": 1,
                "response_correlation_min": 0.9,
                "response_nrmse_max": 0.2,
            },
            "limitations": "One scalar contribution; dense model still runs. No full-MLP replacement, LLM speedup, or identification of a human arithmetic concept.",
        },
    }
    save(path, result)
    return result


class Replacements:
    def __init__(self, out, selected):
        self.components = {
            kind: torch.load(out / name, weights_only=True)
            for kind, name in [
                ("pls", "component.pt"),
                ("active", "component-active.pt"),
            ]
        }
        for component in self.components.values():
            for key, value in component.items():
                if isinstance(value, torch.Tensor):
                    component[key] = value.cuda()
        c = self.components["pls"]
        self.layer, self.d = c["layer"], c["direction"]
        self.normsq = self.d.double().square().sum().float()
        self.specs = selected["heads"]
        self.models = {}
        for name, spec in self.specs.items():
            head = Head(spec["kind"], spec["rank"], spec["width"]).cuda().eval()
            head.load_state_dict(torch.load(out / spec["checkpoint"], weights_only=True))
            head.requires_grad_(False)
            self.models[name] = head
        self.sparse = {}
        for name, spec in selected.get("sparse", {}).items():
            self.sparse[name] = {
                k: v.cuda()
                for k, v in torch.load(out / spec["checkpoint"], weights_only=True).items()
            }
        self.linear = {
            name: {
                k: v.cuda()
                for k, v in torch.load(out / spec["checkpoint"], weights_only=True).items()
            }
            for name, spec in selected.get("linear", {}).items()
        }
        self.kinds = [
            "original",
            "restore",
            "mean",
            *self.models,
            *self.sparse,
            *self.linear,
        ]

    def coefficient(self, h, kind):
        if kind in self.linear:
            return h @ self.linear[kind]["weight"] + self.linear[kind]["bias"]
        if kind in self.sparse:
            state = self.sparse[kind]
            return (torch.nn.functional.silu(h @ state["gate"].T) * (h @ state["up"].T)) @ state[
                "readout"
            ] + state["bias"]
        if kind == "mean":
            return self.components["pls"]["component_mean"].expand(len(h))
        spec = self.specs[kind]
        c, rank = self.components[spec["features"]], spec["rank"]
        z = (((h - c["input_mean"]) @ c["encoder"][:, :rank]) - c["zmean"][:rank]) / c["zstd"][
            :rank
        ]
        return self.models[kind](z) * c["ystd"] + c["ymean"]

    @contextmanager
    def hook(self, model, kind, mixed=None, all_tokens=False):
        cache = {}
        calls = 0

        def post(module, args, out):
            nonlocal calls
            calls += 1
            if calls > 1 and not all_tokens:
                return out
            h = args[0][:, -1, :] if mixed is None else mixed
            base = out[:, -1, :] @ self.d
            true = base if mixed is None else module.forward(mixed) @ self.d
            pred = true if kind in ["original", "restore"] else self.coefficient(h, kind)
            if calls == 1:
                cache.update(input=h.detach(), true=true.detach(), pred=pred.detach())
            cache["patched_calls"] = calls
            if kind == "original" and mixed is None:
                return out
            result = out.clone()
            result[:, -1, :] += ((pred - base) / self.normsq)[:, None] * self.d
            return result

        handle = model.model.layers[self.layer].mlp.register_forward_hook(post)
        try:
            yield cache
        finally:
            handle.remove()


def generation(model, tok, inp):
    result = model.generate(
        **inp,
        max_new_tokens=16,
        do_sample=False,
        use_cache=True,
        pad_token_id=tok.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        logits_to_keep=1,
    )
    texts = tok.batch_decode(
        result.sequences[:, inp.input_ids.shape[1] :], skip_special_tokens=True
    )
    return result.scores[0].float(), texts


def parse_answer(text):
    found = re.fullmatch(r"\s*(\d+)\s*[.!]?\s*", text)
    return int(found[1]) if found else None


def row_metadata(row, index):
    return {**row, "row_index": index}


def ordinary(model, tok, rep, rows, batch_size=32, all_tokens=False):
    records = {kind: [] for kind in rep.kinds}
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        inp = inputs(tok, batch)
        with rep.hook(model, "original", all_tokens=all_tokens) as cache:
            base, base_text = generation(model, tok, inp)
        base_lp = base.log_softmax(-1)
        base_p = base_lp.exp()
        ct = targets(tok, batch)
        for kind in rep.kinds:
            if kind == "original":
                scores, texts, current = base, base_text, cache
            else:
                with rep.hook(model, kind, all_tokens=all_tokens) as current:
                    scores, texts = generation(model, tok, inp)
            assert torch.isfinite(scores).all(), kind
            if kind == "restore":
                assert torch.equal(base, scores), "Zero-delta restoration must be exact"
                assert texts == base_text
            kl = (base_p * (base_lp - scores.log_softmax(-1))).sum(1)
            for i, row in enumerate(batch):
                parsed, original_parsed = (
                    parse_answer(texts[i]),
                    parse_answer(base_text[i]),
                )
                records[kind].append(
                    {
                        **row_metadata(row, start + i),
                        "text": texts[i],
                        "base_text": base_text[i],
                        "correct": parsed == row["answer"],
                        "base_correct": original_parsed == row["answer"],
                        "answer_agreement": parsed is not None and parsed == original_parsed,
                        "exact_text_agreement": texts[i] == base_text[i],
                        "first_token_agreement": bool(scores[i].argmax() == base[i].argmax()),
                        "first_token_correct": bool(scores[i].argmax() == ct[i]),
                        "patched_calls": current["patched_calls"],
                        "kl": float(kl[i]),
                        "true_coefficient": float(current["true"][i]),
                        "predicted_coefficient": float(current["pred"][i]),
                    }
                )
        print("ORDINARY", start + len(batch), len(rows), flush=True)
    return records


def interventions(model, tok, rep, rows, batch_size=64, alphas=ALPHAS):
    records = {kind: [] for kind in rep.kinds}
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        corrupted = inputs(tok, batch, True)
        hs = []
        for corrupt in [False, True]:
            with rep.hook(model, "original") as cache:
                reference = logits(model, inputs(tok, batch, corrupt))
            hs.append(cache["input"].clone())
        ct, bt = targets(tok, batch), targets(tok, batch, True)
        assert (ct != bt).all(), "A contrast must have distinct first answer tokens"
        for alpha in alphas:
            mixed = (1 - alpha) * hs[1] + alpha * hs[0]
            with rep.hook(model, "original", mixed) as base_cache:
                base = logits(model, corrupted)
            if alpha == 0:
                assert torch.allclose(base, reference, atol=1e-4, rtol=1e-4)
            base_lp, base_p = base.log_softmax(-1), base.softmax(-1)
            for kind in rep.kinds:
                if kind == "original":
                    scores, cache = base, base_cache
                else:
                    with rep.hook(model, kind, mixed) as cache:
                        scores = logits(model, corrupted)
                assert torch.isfinite(scores).all(), kind
                if kind == "restore":
                    assert torch.equal(base, scores)
                effect = margin(scores, ct, bt)
                kl = (base_p * (base_lp - scores.log_softmax(-1))).sum(1)
                for i, row in enumerate(batch):
                    records[kind].append(
                        {
                            **row_metadata(row, start + i),
                            "alpha": alpha,
                            "margin": float(effect[i]),
                            "kl_vs_same_intervention_original": float(kl[i]),
                            "true_coefficient": float(cache["true"][i]),
                            "predicted_coefficient": float(cache["pred"][i]),
                        }
                    )
        print("INTERVENTION", start + len(batch), len(rows), flush=True)
    return records


def coordinate_interventions(model, tok, rep, rows, batch_size=16):
    records = {kind: [] for kind in rep.kinds}
    c = rep.components["active"]
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        inp = inputs(tok, batch, True)
        with rep.hook(model, "original") as captured:
            reference = logits(model, inp)
        h = captured["input"].clone()
        ct, bt = targets(tok, batch), targets(tok, batch, True)
        for feature in [0, 1, 7, 31]:
            for strength in [0.0, -0.25, 0.25]:
                mixed = h + strength * c["encoder"][:, feature] * c["zstd"][feature]
                changed = ((mixed - h) @ c["encoder"]) / c["zstd"]
                expected_change = torch.zeros_like(changed)
                expected_change[:, feature] = strength
                torch.testing.assert_close(changed, expected_change, atol=1e-3, rtol=1e-3)
                with rep.hook(model, "original", mixed) as base_cache:
                    base = logits(model, inp)
                if strength == 0:
                    assert torch.allclose(base, reference, atol=1e-4, rtol=1e-4)
                base_lp, base_p = base.log_softmax(-1), base.softmax(-1)
                for kind in rep.kinds:
                    if kind == "original":
                        scores, cache = base, base_cache
                    else:
                        with rep.hook(model, kind, mixed) as cache:
                            scores = logits(model, inp)
                    assert torch.isfinite(scores).all(), kind
                    if kind == "restore":
                        assert torch.equal(base, scores)
                    effect = margin(scores, ct, bt)
                    kl = (base_p * (base_lp - scores.log_softmax(-1))).sum(1)
                    for i, row in enumerate(batch):
                        records[kind].append(
                            {
                                **row_metadata(row, start + i),
                                "feature": feature,
                                "alpha": strength,
                                "margin": float(effect[i]),
                                "kl_vs_same_intervention_original": float(kl[i]),
                                "true_coefficient": float(cache["true"][i]),
                                "predicted_coefficient": float(cache["pred"][i]),
                            }
                        )
        print("COORDINATE INTERVENTION", start + len(batch), len(rows), flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["add", "multiply", "divide"])
    parser.add_argument(
        "--directories",
        nargs="+",
        default=["heads", "heads-active-r32-g0", "heads-active-r32-g0.1"],
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    setup()
    out = ROOT / args.operation
    selected = freeze(out, args.directories)
    rep = Replacements(out, selected)
    model, tok = load()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from prefix_execution import install

    install(model, rep.layer, globals())
    problems = json.loads((out / "problems.json").read_text())
    extra = json.loads((out / "extra-problems.json").read_text())
    assert digest(out / "problems.json") == selected["problems_sha256"]
    save(
        out / "evaluation-job.json",
        {
            "pid": os.getpid(),
            "start_epoch": time.time(),
            "gpu": torch.cuda.get_device_name(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    if args.validate_only:
        suites = [("validation-audit", expand(problems["validation"][:4], STYLES))]
    else:
        suites = [
            ("test-known-formats", expand(problems["test"], STYLES)),
            (
                "test-new-formats",
                expand(problems["test"], ["completion", "named", "reversed", "distractor"]),
            ),
            ("shift", expand(problems["shift"], STYLES)),
            ("both-operands", expand(extra["both_operands"], STYLES)),
        ]
        if "carry" in problems:
            suites.append(("carry", expand(problems["carry"], STYLES)))
    with torch.inference_mode():
        for suite, rows in suites:
            for category, function in [
                ("ordinary", ordinary),
                ("interventions", interventions),
            ]:
                target = out / f"{category}-{suite}.json"
                if target.exists():
                    print("PRESERVE", target.name, flush=True)
                    continue
                save(target, function(model, tok, rep, rows))
        if not args.validate_only:
            for suite, rows in [
                ("test-known-formats", expand(problems["test"], STYLES)),
                ("unconditioned", expand(extra["ordinary"], STYLES)),
            ]:
                for all_tokens in [False, True]:
                    category = "ordinary-all-tokens" if all_tokens else "ordinary"
                    target = out / f"{category}-{suite}.json"
                    if not target.exists():
                        save(target, ordinary(model, tok, rep, rows, all_tokens=all_tokens))
            target = out / "interventions-extrapolation.json"
            if not target.exists():
                save(
                    target,
                    interventions(
                        model,
                        tok,
                        rep,
                        expand(problems["test"][:128], STYLES),
                        alphas=[0.0, -0.25, 1.25],
                    ),
                )
            target = out / "coordinate-interventions.json"
            if not target.exists():
                save(
                    target,
                    coordinate_interventions(
                        model, tok, rep, expand(problems["test"][:32], STYLES)
                    ),
                )
        else:
            save(
                out / "ordinary-all-tokens-validation.json",
                ordinary(model, tok, rep, suites[0][1], all_tokens=True),
            )
            save(
                out / "coordinate-validation.json",
                coordinate_interventions(
                    model, tok, rep, expand(problems["validation"][:4], STYLES)
                ),
            )
    save(
        out / ("evaluation-audit.json" if args.validate_only else "evaluation-completed.json"),
        {
            "completed_epoch": time.time(),
            "suites": [s[0] for s in suites],
            "restore_equivalent": True,
        },
    )
    print("EVALUATION DONE", args.operation, flush=True)


if __name__ == "__main__":
    main()
