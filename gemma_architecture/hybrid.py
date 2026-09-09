"""Capped hybrid experiment. Run: python -m gemma_architecture.hybrid run --gpu 0 --eval-gpu 1."""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from gemma_mechanisms.evaluate import arc, arithmetic, language
from gemma_mechanisms.runtime import (
    SNAPSHOT,
    accounting,
    digest,
    load,
    save,
    setup,
    text_layers,
    tokenizer,
)
from gemma_mechanisms.student import contribution
from gemma_mechanisms.train import evaluate as activation_evaluate
from gemma_mechanisms.train import objective

from .benchmark import telemetry, workload
from .hybrid_model import KINDS, DeployedHybrid, Hybrid, initialize, load_hybrid

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SPEC = json.loads((HERE / "hybrid-protocol.json").read_text())


def paths():
    out = Path(os.environ.get("EML_HYBRID_RUNS", REPO / ".artifacts/gemma-hybrid")).resolve()
    prior = Path(
        os.environ.get("EML_HYBRID_INPUTS", REPO / ".artifacts/gemma-architecture-release-5c87a61")
    ).resolve()
    collection = Path(
        os.environ.get(
            "EML_HYBRID_COLLECTION", REPO / ".artifacts/gemma-replacement-bos/collection"
        )
    ).resolve()
    assert out.is_relative_to(REPO) and out != prior
    out.mkdir(parents=True, exist_ok=True)
    return out, prior, collection


def read(path):
    return json.loads(path.read_text())


def bind():
    out, prior, collection = paths()
    src = [
        HERE / n for n in ("hybrid.py", "hybrid_model.py", "hybrid-protocol.json", "benchmark.py")
    ]
    src += [
        REPO / "gemma_mechanisms" / n
        for n in (
            "runtime.py",
            "train.py",
            "student.py",
            "evaluate.py",
            "prepare.py",
            "protocol.json",
        )
    ]
    src += [REPO / "emltorch/operator.py"]
    inputs = [
        collection / f"{d}-{s}.pt"
        for d in ("arithmetic", "language")
        for s in ("train", "selection")
    ]
    inputs += [collection / "native-postnorm.pt", prior / "training/initialization.pt"]
    inputs += [
        prior / "data" / f"{d}-{s}.{ext}"
        for s in ("development", "confirmation")
        for d, ext in (("arithmetic", "json"), ("arc", "json"), ("language", "pt"))
    ]
    inputs += [prior / "data/language-documents.json", prior / "data/freeze.json"]
    # Verify relocation against the original frozen inputs, using names, not stale paths.
    old = read(prior / "training/freeze.json")["inputs"]
    for p in inputs[:5]:
        expected = {v for k, v in old.items() if Path(k).name == p.name}
        assert len(expected) == 1 and digest(p) in expected, p
    value = {
        "sources": {str(p.relative_to(REPO)): digest(p) for p in src},
        "inputs": {str(p.relative_to(REPO)): digest(p) for p in inputs},
        "model_config_sha256": digest(SNAPSHOT / "config.json"),
        "model": SPEC["model"],
        "development": SPEC["development"],
        "confirmation": SPEC["confirmation"],
    }
    assert value["model_config_sha256"] == SPEC["model"]["config_sha256"]
    config = read(SNAPSHOT / "config.json")
    assert config["architectures"] == ["Gemma4ForConditionalGeneration"]
    assert config["text_config"]["hidden_size"] == 1536
    assert config["text_config"]["hidden_activation"] == SPEC["model"]["native_activation"]
    path = out / "freeze.json"
    if path.exists():
        assert read(path) == value, "Frozen source or inputs changed; use a new run root"
    else:
        save(path, value)
    return out, prior, collection


def data(prior, collection):
    raw = {
        s: {
            d: torch.load(collection / f"{d}-{s}.pt", weights_only=True, mmap=True)
            for d in ("arithmetic", "language")
        }
        for s in ("train", "selection")
    }
    init = torch.load(prior / "training/initialization.pt", weights_only=True)
    norm = torch.load(collection / "native-postnorm.pt", weights_only=True)
    norm["weight"] = norm["weight"].cuda()
    return raw, init, norm


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def key(kind, seed):
    return f"{kind}-s{seed}"


def avg(row, term):
    return sum(d[term] for d in row.values()) / len(row)


@torch.no_grad()
def interval(values):
    """Percentile bootstrap of independent evaluation groups, computed on CUDA."""
    x = torch.as_tensor(values, device="cuda", dtype=torch.float64)
    assert x.ndim == 1 and len(x) > 1
    rng = torch.Generator(device="cuda").manual_seed(91073)
    ids = torch.randint(len(x), (4000, len(x)), device="cuda", generator=rng)
    means = x[ids].mean(-1)
    return {
        "mean": float(x.mean()),
        "ci95": torch.quantile(
            means, torch.tensor([0.025, 0.975], device="cuda", dtype=torch.float64)
        ).tolist(),
        "groups": len(x),
    }


def validate():
    out, prior, collection = bind()
    raw, init, _ = data(prior, collection)
    x = raw["selection"]["language"]["x"][:32].cuda().float()
    results, predictions = {}, []
    for kind in KINDS:
        setup(1103)
        model = Hybrid(kind, init["statistics"]).cuda()
        initialize(model, init)
        with torch.no_grad():
            predictions.append(model(x).cpu())
            # Exercise nonzero gates/readouts: folding a zero branch is insufficient.
            for branch in (model.main, model.correction):
                if branch is not None:
                    branch.readout.weight.normal_(std=0.001)
                    branch.readout.bias.normal_(std=0.001)
            if model.gate is not None:
                model.gate.fill_(0.37)
        loss = model(x).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        with torch.no_grad():
            reference = model(x)
            folded = DeployedHybrid(model).cuda()
            torch.testing.assert_close(folded(x), reference, atol=2e-4, rtol=2e-5)
            if model.gate is not None:
                ablated = DeployedHybrid(model, True)
                model.gate.zero_()
                torch.testing.assert_close(ablated(x), model(x), atol=2e-4, rtol=2e-5)
            bf = folded.bfloat16()(x.bfloat16())
            assert torch.isfinite(bf).all()
        results[kind] = {
            "training": accounting(model),
            "deployment": accounting(folded),
            "main_hidden": model.main.readout.in_features,
            "correction_hidden": model.correction.readout.in_features if model.correction else None,
        }
        del model, folded
    for pred in predictions[1:]:
        torch.testing.assert_close(pred, predictions[0], atol=1e-6, rtol=1e-6)
    save(
        out / "validation.json",
        {
            "passed": True,
            "methods": results,
            "device": torch.cuda.get_device_name(),
            "checks": [
                "identical initial affine predictions",
                "finite backward through every parameter",
                "nonzero branch/gate FP32 deployment equivalence",
                "physical branch-removal equivalence",
                "finite BF16 forward",
            ],
            "freeze_sha256": digest(out / "freeze.json"),
        },
    )
    print("VALIDATED", json.dumps(results), flush=True)


@torch.no_grad()
def diagnostics(model, raw, init, norm):
    folded = DeployedHybrid(model).cuda()
    bf = DeployedHybrid(model).cuda().bfloat16()
    ablated = DeployedHybrid(model, True).cuda()
    q = torch.linalg.qr(model.decoder.weight.double(), mode="reduced").Q.float()
    output = {}
    for domain, rows in raw["selection"].items():
        totals = torch.zeros(6, dtype=torch.float64, device="cuda")
        group_sums = {}
        for start in range(0, len(rows["x"]), 512):
            x, y, c = (
                rows[k][start : start + 512].cuda().float() for k in ("x", "y", "contribution")
            )
            pred = model(x)
            error = (pred - y) / model.yscale
            outside = error - (error @ q) @ q.T
            no = ablated(x)
            bpred = bf(x.bfloat16()).float()
            cscale = init["cscale"].cuda()
            metrics = torch.stack(
                [
                    error.square().mean(-1),
                    outside.square().mean(-1),
                    ((no - y) / model.yscale).square().mean(-1),
                    ((bpred - y) / model.yscale).square().mean(-1),
                    ((folded(x) - pred) / model.yscale).square().mean(-1),
                    ((contribution(bpred, norm["weight"], norm["eps"]) - c) / cscale)
                    .square()
                    .mean(-1),
                ],
                -1,
            )
            assert torch.isfinite(metrics).all()
            totals += metrics.double().sum(0)
            ids = rows["sequence"][start : start + 512].tolist()
            for ident, row in zip(ids, metrics[:, :3].cpu().tolist()):
                current = group_sums.setdefault(str(ident), [0.0, 0.0, 0.0, 0])
                for j in range(3):
                    current[j] += row[j]
                current[3] += 1
        values = (totals / len(rows["x"])).tolist()
        assert values[4] < 1e-10
        output[domain] = dict(
            zip(
                (
                    "raw_mse",
                    "outside_decoder_mse",
                    "ablated_raw_mse",
                    "bf16_raw_mse",
                    "folding_mse",
                    "bf16_contribution_mse",
                ),
                values,
            )
        )
        output[domain]["outside_fraction"] = values[1] / values[0]
        output[domain]["sequence_mean_errors"] = {
            k: [v[i] / v[3] for i in range(3)] for k, v in group_sums.items()
        }
    output["train_affine_plus_rank512_bound"] = float(
        init["residual_eigenvalues"][512:].sum() / model.dimension
    )
    output["gate"] = float(model.gate) if model.gate is not None else None
    output["limitation"] = (
        "Bounds apply to raw training MSE; decoder diagnostic is held-out error geometry, not causal attribution or downstream guarantee. Sequence means and token-weighted loss use different weighting."
    )
    return output


def train(kind, seed):
    out, prior, collection = bind()
    assert read(out / "validation.json")["passed"]
    if seed != SPEC["training"]["screen_seed"]:
        assert read(out / "decision.json")["passes"]
    name = key(kind, seed)
    target = out / "training" / f"{name}.json"
    assert not target.exists()
    raw, init, norm = data(prior, collection)
    setup(seed)
    m = Hybrid(kind, init["statistics"]).cuda()
    initialize(m, init)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=0.0001)
    rng = torch.Generator().manual_seed(seed + 100000)
    cscale = init["cscale"].cuda()
    best, state, selected, clips, curves = float("inf"), None, None, 0, []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    events[0].record()
    for step in range(1, SPEC["training"]["steps"] + 1):
        terms = []
        for rows in raw["train"].values():
            ids = torch.randint(len(rows["x"]), (256,), generator=rng)
            terms.append(
                objective(
                    m,
                    *(rows[k][ids].cuda().float() for k in ("x", "y", "contribution")),
                    norm,
                    cscale,
                )[0]
            )
        loss = sum(terms) / 2
        assert torch.isfinite(loss), "nonfinite training loss"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        assert torch.isfinite(grad), "nonfinite training gradient"
        clips += int(grad > 1)
        opt.step()
        if step % SPEC["training"]["validation_interval"] == 0:
            val = activation_evaluate(m, raw["selection"], norm, cscale)
            value = avg(val, "objective")
            if value < best:
                best, state, selected = value, cpu_state(m), step
            curves.append(
                {
                    "step": step,
                    "selection": val,
                    "train_batch_objective": float(loss),
                    "gradient_norm": float(grad),
                    "gate": float(m.gate) if m.gate is not None else None,
                }
            )
            print("FIT", name, step, value, round(time.perf_counter() - started, 2), flush=True)
    events[1].record()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    assert state is not None
    m.load_state_dict(state)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"specification": m.specification(), "state": state}, target.with_suffix(".pt"))
    result = {
        "kind": kind,
        "seed": seed,
        "status": "complete",
        "steps": step,
        "selected_step": selected,
        "selection_objective": best,
        "selection": activation_evaluate(m, raw["selection"], norm, cscale),
        "training_seconds": elapsed,
        "cuda_elapsed_ms": events[0].elapsed_time(events[1]),
        "training_tokens": step * 512,
        "peak_allocated_bytes": peak,
        "gradient_clipped_steps": clips,
        "curves": curves,
        "accounting": accounting(m),
        "deployment_accounting": accounting(DeployedHybrid(m)),
        "diagnostics": diagnostics(m, raw, init, norm),
        "checkpoint_sha256": digest(target.with_suffix(".pt")),
        "freeze_sha256": digest(out / "freeze.json"),
    }
    save(target, result)
    print("TRAINED", name, best, flush=True)


def samples(prior, split):
    spec = SPEC[split]
    data_dir = prior / "data"
    full = read(data_dir / f"arithmetic-{split}.json")
    groups = {
        s: sum(
            (
                [r for r in full[s] if r["op"] == op][: spec[f"{s}_per_operation"]]
                for op in ("add", "multiply", "divide")
            ),
            [],
        )
        for s in ("ordinary", "shift")
    }
    questions = read(data_dir / f"arc-{split}.json")[: spec["arc_questions"]]
    blocks = torch.load(data_dir / f"language-{split}.pt", weights_only=True)[
        : spec["language_documents"], : spec["language_tokens"]
    ]
    return groups, questions, blocks


def method_names(seed):
    return [key(k, seed) for k in KINDS] + [
        key(k, seed) + "-ablated" for k in ("silu_silu", "silu_eml")
    ]


def installed(model, original, original_forward, out, name):
    old = text_layers(model)[26].mlp
    old.cpu()
    if name == "original":
        original.forward = original_forward
        text_layers(model)[26].mlp = original.cuda()
    else:

        def forbidden(*args, **kwargs):
            raise AssertionError("Removed original MLP executed")

        original.forward = forbidden
        ablate = name.endswith("-ablated")
        checkpoint = out / "training" / f"{name.removesuffix('-ablated')}.pt"
        student = load_hybrid(checkpoint, ablate=ablate)
        text_layers(model)[26].mlp = student
        assert not {id(p) for p in original.parameters()}.intersection(
            id(p) for p in model.parameters()
        )
    del old
    gc.collect()
    torch.cuda.empty_cache()


def downstream(seed, split, method=None):
    out, prior, _ = bind()
    if split == "confirmation":
        assert read(out / "decision.json")["passes"]
        selection = {
            key(k, s): digest(out / "training" / f"{key(k, s)}.pt")
            for s in (1103, 2207, 3301)
            for k in KINDS
        }
        seal = out / "confirmation-selection.json"
        if seal.exists():
            assert read(seal) == selection
        else:
            save(seal, selection)
    groups, questions, blocks = samples(prior, split)
    model, tok = load(), tokenizer()
    original = text_layers(model)[26].mlp
    original_forward = original.forward
    directory = out / "evaluation" / split
    names = ["original"] + method_names(seed)
    if method is not None:
        assert method in names
        names = [method]
    for name in names:
        target = directory / f"{name}.json"
        if target.exists():
            continue
        installed(model, original, original_forward, out, name)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        print("EVALUATE", split, name, flush=True)
        result = {
            "arithmetic": arithmetic(
                model, tok, groups["ordinary"], SPEC[split]["styles"], batch_size=4
            ),
            "shift": arithmetic(model, tok, groups["shift"], ["code"], batch_size=4),
            "arc": arc(model, tok, questions),
            "language": language(model, blocks),
        }
        torch.cuda.synchronize()
        save(
            target,
            {
                "method": name,
                "results": result,
                "seconds": time.perf_counter() - start,
                "accounting": accounting(model),
                "original_mlp_executed": name == "original",
                "teacher_activation_inputs": False,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "freeze_sha256": digest(out / "freeze.json"),
            },
        )


def points(row):
    r = row["results"]

    def acc(rows):
        return sum(v["correct"] for v in rows) / len(rows)

    return {
        **{
            op: acc([v for v in r["arithmetic"] if v["op"] == op])
            for op in ("add", "multiply", "divide")
        },
        "pooled": acc(r["arithmetic"]),
        "shift": acc(r["shift"]),
        "arc": acc(r["arc"]),
        "ce": sum(v["ce_nats"] for v in r["language"]) / len(r["language"]),
    }


def decision():
    out, _, _ = bind()
    train = {k: read(out / "training" / f"{key(k, 1103)}.json") for k in KINDS}
    values = {
        k: points(read(out / "evaluation/development" / f"{key(k, 1103)}.json")) for k in KINDS
    }
    h = "silu_eml"
    checks = {
        "complete_fits": all(
            r["status"] == "complete" and r["steps"] == 12000 for r in train.values()
        )
    }
    improvements = {}
    for c in ("silu", "silu_silu"):
        improvements[c] = {
            metric: 1 - avg(train[h]["selection"], metric) / avg(train[c]["selection"], metric)
            for metric in ("raw_mse", "contribution_mse")
        }
        checks[c + "_activation"] = (
            min(improvements[c].values())
            >= SPEC["gate"]["relative_raw_and_contribution_improvement"]
        )
        checks[c + "_quality"] = all(
            values[h][metric] - values[c][metric] >= -SPEC["gate"]["max_accuracy_regression"]
            for metric in ("add", "multiply", "divide", "shift", "arc")
        )
        checks[c + "_language"] = (
            values[h]["ce"] - values[c]["ce"] <= SPEC["gate"]["max_ce_increase_nats"]
        )
        checks[c + "_gain"] = (
            values[h]["pooled"] - values[c]["pooled"]
            >= SPEC["gate"]["minimum_pooled_accuracy_gain"]
        )
    result = {
        "passes": all(checks.values()),
        "checks": checks,
        "activation_relative_improvements": improvements,
        "development_points": values,
        "fresh_confirmation_opened": False,
        "note": "Development selection only; a failed gate stops additional seeds and confirmation.",
    }
    save(out / "decision.json", result)
    print("DECISION", json.dumps(result), flush=True)


def benchmark(seed):
    out, prior, _ = bind()
    model = load()
    original = text_layers(model)[26].mlp
    native_forward = original.forward
    spec = SPEC["benchmark"]
    blocks = torch.load(prior / "data/language-development.pt", weights_only=True)
    ids = blocks[:1, :128].cuda()
    forced = blocks[1:2, 1:17].cuda()
    names = ["original"] + [key(k, seed) for k in KINDS]
    before = telemetry()
    records = {name: [] for name in names}
    for name in names:
        installed(model, original, native_forward, out, name)
        for _ in range(spec["warmup"]):
            workload(model, ids, forced)
    # Every repeat contains every method; reverse order on alternate repeats.
    for repeat in range(spec["paired_repeats"]):
        for name in names if repeat % 2 == 0 else names[::-1]:
            installed(model, original, native_forward, out, name)
            records[name].append(workload(model, ids, forced))
        print("BENCHMARK", repeat + 1, flush=True)
    summaries = {
        name: {metric: interval([r[metric] for r in rows]) for metric in rows[0]}
        for name, rows in records.items()
    }
    comparisons = {}
    for control in ("original", key("silu", seed), key("silu_silu", seed)):
        ratios = [
            a["end_to_end_seconds"] / b["end_to_end_seconds"]
            for a, b in zip(records[key("silu_eml", seed)], records[control])
        ]
        comparisons[control] = interval(ratios)
    save(
        out / f"benchmark-s{seed}.json",
        {
            "specification": spec,
            "rows": records,
            "summaries": summaries,
            "hybrid_end_to_end_ratio": comparisons,
            "telemetry_before": before,
            "telemetry_after": telemetry(),
            "note": "Shared H100 and host PLE lookup; intervals measure repeat variability, not uncontended serving performance. Native MLP raises if executed while a replacement is installed.",
        },
    )


def paired(a, b, field, operation=None):
    def groups(rows):
        result = {}
        for r in rows:
            if operation is not None and r.get("op") != operation:
                continue
            result.setdefault(str(r.get("id", r.get("document"))), []).append(
                float(r["ce_nats"] if field == "language" else r["correct"])
            )
        return {k: sum(v) / len(v) for k, v in result.items()}

    av, bv = groups(a["results"][field]), groups(b["results"][field])
    assert av.keys() == bv.keys()
    return interval([av[k] - bv[k] for k in sorted(av)])


def report():
    out, _, _ = bind()
    fits = {p.stem: read(p) for p in sorted((out / "training").glob("*.json"))}
    evaluations = {
        str(p.relative_to(out / "evaluation")): read(p)
        for p in sorted((out / "evaluation").rglob("*.json"))
    }
    comparisons = {}
    for split in ("development", "confirmation"):
        for seed in (1103, 2207, 3301):
            path = f"{split}/{key('silu_eml', seed)}.json"
            if path not in evaluations:
                continue
            hybrid = evaluations[path]
            controls = [
                "original",
                key("silu", seed),
                key("silu_silu", seed),
                key("silu_eml", seed) + "-ablated",
            ]
            comparisons[f"{split}-s{seed}"] = {
                c: {
                    **{
                        f: paired(hybrid, evaluations[f"{split}/{c}.json"], f)
                        for f in ("arithmetic", "shift", "arc", "language")
                    },
                    **{
                        op: paired(hybrid, evaluations[f"{split}/{c}.json"], "arithmetic", op)
                        for op in ("add", "multiply", "divide")
                    },
                }
                for c in controls
                if f"{split}/{c}.json" in evaluations
            }
    result = {
        "protocol": SPEC,
        "freeze": read(out / "freeze.json"),
        "validation": read(out / "validation.json"),
        "fits": fits,
        "evaluations": evaluations,
        "paired_differences_hybrid_minus_control": comparisons,
        "decision": read(out / "decision.json"),
        "benchmark": {p.stem: read(p) for p in out.glob("benchmark-*.json")},
        "budget": read(out / "budget.json"),
        "uncertainty": "Paired bootstrap over operand groups (formats clustered), ARC questions, and documents; 4000 CUDA resamples, unadjusted 95% intervals. Development results are exploratory; single-seed results cannot establish seed robustness. Hostname correlations are not modeled.",
        "confirmation_completed": all(
            (out / "evaluation/confirmation" / f"{key(k, s)}.json").exists()
            for k in KINDS
            for s in (1103, 2207, 3301)
        ),
    }
    save(out / "summary.json", result)
    print("SUMMARY", str(out / "summary.json"), flush=True)


class BudgetExhausted(RuntimeError):
    pass


def run_job(action, label, gpu, timeout, options=()):
    out, _, _ = paths()
    path = out / "budget.json"
    budget = (
        read(path)
        if path.exists()
        else {"limit_seconds": SPEC["budget"]["device_hours"] * 3600, "jobs": {}}
    )
    if label in budget["jobs"]:
        assert budget["jobs"][label]["status"] == "complete", (
            f"Preserve failed job {label}; investigate before retry"
        )
        return
    used = sum(v["seconds"] for v in budget["jobs"].values())
    reserve = SPEC["budget"]["reserve_seconds"] if action in ("train", "evaluate") else 0
    if used + timeout + reserve > budget["limit_seconds"]:
        raise BudgetExhausted(f"Cannot reserve {timeout}s plus {reserve}s for {label}")
    log = out / "logs" / f"{label}.log"
    log.parent.mkdir(exist_ok=True)
    row = {
        "action": action,
        "gpu": gpu,
        "status": "running",
        "seconds": timeout,
        "options": list(options),
        "started_unix": time.time(),
    }
    budget["jobs"][label] = row
    save(path, budget)
    started = time.perf_counter()
    command = [sys.executable, "-m", "gemma_architecture.hybrid", action, *options]
    print("START", label, "GPU", gpu, flush=True)
    try:
        with log.open("w") as handle:
            process = subprocess.Popen(
                command,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            try:
                code = process.wait(timeout=timeout)
                row["status"] = "complete" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                code = -9
                row["status"] = "timeout"
    finally:
        row["seconds"] = time.perf_counter() - started
        row["finished_unix"] = time.time()
        save(path, budget)
    print(row["status"].upper(), label, round(row["seconds"], 2), flush=True)
    assert code == 0, f"Inspect {log}"


def run(gpu, eval_gpu):
    run_job("validate", "validate", gpu, 180)
    for kind in KINDS:
        run_job("train", key(kind, 1103), gpu, 900, ("--kind", kind, "--seed", "1103"))
    run_job("evaluate", "development", eval_gpu, 900)
    run_job("decision", "decision", gpu, 60)
    out, _, _ = paths()
    if read(out / "decision.json")["passes"]:
        try:
            for seed in (2207, 3301):
                for kind in KINDS:
                    run_job(
                        "train", key(kind, seed), gpu, 900, ("--kind", kind, "--seed", str(seed))
                    )
            for seed in (1103, 2207, 3301):
                for method in (["original"] if seed == 1103 else []) + method_names(seed):
                    run_job(
                        "evaluate",
                        f"confirmation-{method}",
                        eval_gpu,
                        900,
                        ("--seed", str(seed), "--split", "confirmation", "--method", method),
                    )
        except BudgetExhausted as exc:
            save(out / "confirmation-incomplete.json", {"reason": str(exc)})
    run_job("benchmark", "benchmark", eval_gpu, 600)
    run_job("report", "report", gpu, 120)
    summary = read(out / "summary.json")
    summary["budget"] = read(out / "budget.json")
    save(out / "summary.json", summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=("run", "validate", "train", "evaluate", "decision", "benchmark", "report"),
    )
    p.add_argument("--gpu", default="0")
    p.add_argument("--eval-gpu", default="1")
    p.add_argument("--kind", choices=KINDS)
    p.add_argument("--method", help="One evaluation method, useful for budgeted confirmation")
    p.add_argument("--seed", type=int, choices=(1103, 2207, 3301), default=1103)
    p.add_argument("--split", choices=("development", "confirmation"), default="development")
    a = p.parse_args()
    if a.action == "run":
        run(a.gpu, a.eval_gpu)
        return
    setup(a.seed)
    if a.action == "train":
        assert a.kind
        train(a.kind, a.seed)
    elif a.action == "evaluate":
        downstream(a.seed, a.split, a.method)
    elif a.action == "benchmark":
        benchmark(a.seed)
    else:
        {"validate": validate, "decision": decision, "report": report}[a.action]()


if __name__ == "__main__":
    main()
