"""Validate complete result identities, scoring, deployment counts and frozen sources."""

import argparse
import json

import torch

from .evaluate import parse_answer, roster
from .runtime import HERE, SPEC, digest, root, save, setup
from .student import load_student
from .train import evaluate, moments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--mechanisms-output", required=True)
    args = parser.parse_args()
    setup(919)
    out, mechanism = root(args.output), root(args.mechanisms_output)
    checked, decoder_ranks = {}, {}

    def read(path):
        key = (
            ("replacement/" + str(path.relative_to(out)))
            if path.is_relative_to(out)
            else ("mechanisms/" + str(path.relative_to(mechanism)))
        )
        checked[key] = digest(path)
        return json.loads(path.read_text())

    def source_check(freeze):
        for name, expected in freeze["sources"].items():
            assert digest(HERE / name) == expected, ("Changed frozen source", name)

    # The initial GPU operator checks predate the fitted candidates. Their
    # broader historical source snapshot is retained rather than rewritten.
    for name in ["student-validation.json", "integration-validation.json"]:
        path = HERE / name
        checked[f"repository/{name}"] = digest(path)
        record = json.loads(path.read_text())
        assert (
            digest(HERE.parent / "emltorch/operator.py")
            == record["sources"]["../emltorch/operator.py"]
        )

    data = read(out / "data/freeze.json")
    for name, sha in data["files"].items():
        assert digest(out / "data" / name) == sha
    used = {op: set() for op in SPEC["arithmetic"]["operations"]}
    for split, count in SPEC["arithmetic"]["counts_per_operation"].items():
        rows = read(out / "data" / f"arithmetic-{split}.json")
        assert len(rows) == 3 * count and len({r["id"] for r in rows}) == len(rows)
        for op in used:
            selected = [r for r in rows if r["op"] == op]
            assert len(selected) == count
            numbers = torch.tensor([[r["a"], r["b"], r["answer"]] for r in selected], device="cuda")
            a, b, answer = numbers.unbind(-1)
            expected = (
                a + b
                if op == "add"
                else a * b
                if op == "multiply"
                else torch.div(a, b, rounding_mode="floor")
            )
            assert torch.equal(expected, answer)
            for row in selected:
                pair = (
                    (row["a"], row["b"]) if op == "divide" else tuple(sorted([row["a"], row["b"]]))
                )
                assert pair not in used[op], "Duplicate operands across fresh splits"
                used[op].add(pair)
    assert not {(573, 846), (582, 736)}.intersection(used["add"]), (
        "Smoke prompts must not expose held-out operand groups"
    )
    source_check(read(out / "collection/freeze.json"))
    training = read(out / "training/freeze.json")
    source_check(training)
    for name, sha in training["inputs"].items():
        assert digest(out / name) == sha
    fits = list((out / "training").glob("*-b*-d*-s*.json"))
    assert len(fits) == 42
    for path in fits:
        record = read(path)
        assert record["status"] == "complete" and record["failure"] is None
        assert record["training_freeze_sha256"] == digest(out / "training/freeze.json")
        assert digest(path.with_suffix(".pt")) == record["checkpoint_sha256"]
        audit = read(out / "fit-audits" / path.name)
        source_check(audit)
        assert audit["fit_record_sha256"] == digest(path)
        assert audit["checkpoint_sha256"] == record["checkpoint_sha256"]
        assert audit["objective_replay_exact"]
        assert audit["training_raw_mse"] + 1e-6 >= audit["training_raw_mse_lower_bound"]
    extension = read(out / "optimization-extension/freeze.json")
    source_check(extension)
    assert extension["selection_sha256"] == digest(out / "training/selection.json")
    assert extension["training_freeze_sha256"] == digest(out / "training/freeze.json")
    raw = {
        split: {
            domain: torch.load(out / "collection" / f"{domain}-{split}.pt", weights_only=True)
            for domain in ["arithmetic", "language"]
        }
        for split in ["train", "selection"]
    }
    _, cscale = moments(raw["train"])
    norm = torch.load(out / "collection/native-postnorm.pt", weights_only=True)
    norm["weight"] = norm["weight"].cuda()
    for seed in extension["seeds"]:
        for kind in extension["families"]:
            name = f"{kind}-b{extension['budget']}-d{extension['depth']}-s{seed}"
            path = out / "optimization-extension" / f"{name}.json"
            record = read(path)
            assert record["freeze_sha256"] == digest(out / "optimization-extension/freeze.json")
            assert digest(path.with_suffix(".pt")) == record["checkpoint_sha256"]
            assert record["prefix_replay"]["absolute_difference"] < 1e-7
            model = load_student(path.with_suffix(".pt"), dtype=torch.float32)
            validation = evaluate(model, raw["selection"], norm, cscale)
            assert (
                sum(v["objective"] for v in validation.values()) / 2
                == record["selection_objective"]
            )
            del model
    del raw, norm, cscale
    torch.cuda.empty_cache()
    print("AUDITED OPTIMIZATION EXTENSION", flush=True)
    decoder = out / "decoder-error"
    if decoder.exists():
        frozen = read(decoder / "freeze.json")
        source_check(frozen)
        for name, sha in frozen["inputs"].items():
            assert digest(out / name) == sha
        paths = list(decoder.glob("*-b*-s*.json"))
        assert len(paths) == len(frozen["checkpoints"]) == 12
        seen = set()
        for path in paths:
            record = read(path)
            assert record["freeze_sha256"] == digest(decoder / "freeze.json")
            checkpoint = f"{record['phase']}/{record['name']}.pt"
            assert checkpoint not in seen
            seen.add(checkpoint)
            assert digest(out / checkpoint) == frozen["checkpoints"][checkpoint]
            weight = (
                torch.load(out / checkpoint, weights_only=True)["state"]["decoder.weight"]
                .cuda()
                .double()
            )
            singular = torch.linalg.svdvals(weight)
            rank = int((singular > singular.max() * 1e-10).sum())
            assert rank == weight.shape[1] == record["rank"], (
                "Decoder decomposition needs full column rank"
            )
            decoder_ranks[checkpoint] = {
                "rank": rank,
                "smallest_singular_value": float(singular.min()),
                "largest_singular_value": float(singular.max()),
                "condition_number": float(singular.max() / singular.min()),
            }
            del weight, singular
            for split in record["results"].values():
                for values in split.values():
                    numeric = torch.tensor(
                        list(values.values()), device="cuda", dtype=torch.float64
                    )
                    assert torch.isfinite(numeric).all() and (numeric >= 0).all()
                    assert values["pythagorean_closure_error"] < 1e-6
            assert (
                record["balanced"]["train"]["outside_learned_affine_decoder_mse"] + 1e-6
                >= record["optimal_training_rank_floor"]
            )
        assert seen == set(frozen["checkpoints"])
    deployment = read(out / "deployment-validation.json")
    source_check(deployment)
    deployed = {r["name"]: r for r in deployment["records"]}
    names = roster(out)
    for split in ["gate", "final", "shift"]:
        directory = out / "evaluation" / split
        frozen = read(directory / "freeze.json")
        source_check(frozen)
        assert frozen["checkpoints"] == names
        expected_rows = {r["id"]: r for r in read(out / "data" / f"arithmetic-{split}.json")}
        for name in names:
            record = read(directory / f"{name}.json")
            assert record["freeze_sha256"] == digest(directory / "freeze.json")
            assert record["nonfinite_outputs"] == 0
            assert record["original_mlp_executed"] == (name == "original")
            expected_count = (
                5104297504
                if name == "original"
                else 5104297504 - 56623104 + deployed[name]["deployed"]["parameters"]
            )
            assert record["accounting"]["parameters"] == expected_count
            assert record["accounting"]["buffers"] == 2289
            result = record["results"]
            fields = {"arithmetic": SPEC["arithmetic"]["formats"]}
            if split != "gate":
                fields["new_formats"] = SPEC["arithmetic"]["new_formats"]
            for field, styles in fields.items():
                identities = [(r["id"], r["style"]) for r in result[field]]
                expected = {(identity, style) for identity in expected_rows for style in styles}
                assert len(identities) == len(expected) and set(identities) == expected
                for row in result[field]:
                    source = expected_rows[row["id"]]
                    assert all(row[k] == source[k] for k in ["a", "b", "answer", "op"])
                    predicted = parse_answer(row["text"])
                    assert predicted == row["prediction"]
                    assert row["correct"] == (predicted == source["answer"])
            if split != "shift":
                qa = {r["id"]: r for r in read(out / "data" / f"arc-{split}.json")}
                assert len(result["arc"]) == len(qa) and {r["id"] for r in result["arc"]} == set(qa)
                for row in result["arc"]:
                    source = qa[row["id"]]
                    score = torch.tensor(row["scores"], device="cuda", dtype=torch.float64)
                    assert torch.isfinite(score).all()
                    label = source["choices"]["label"][int(score.argmax())]
                    assert row["prediction"] == label and row["correct"] == (
                        label == source["answerKey"]
                    )
                blocks = torch.load(out / "data" / f"language-{split}.pt", weights_only=True)
                assert [r["document"] for r in result["language"]] == list(range(len(blocks)))
                assert all(r["tokens"] == blocks.shape[1] - 1 == 256 for r in result["language"])
                assert torch.isfinite(
                    torch.tensor([r["ce_nats"] for r in result["language"]], device="cuda")
                ).all()
        print("AUDITED QUALITY", split, len(names), flush=True)
    source_check(read(out / "benchmark/freeze.json"))
    benchmark_paths = list((out / "benchmark").glob("*-b*-p*.json"))
    assert len(benchmark_paths) == 4 * sum(name.endswith("-s1103") for name in names)
    for path in benchmark_paths:
        record = read(path)
        assert record["freeze_sha256"] == digest(out / "benchmark/freeze.json")
        assert set(record["student_module_accounting"]["by_device"]) == {"cuda"}
        assert set(record["original_accounting"]["by_device"]) == {"cuda"}
        assert len(record["samples"]) == 50
        values = [
            v
            for pair in record["samples"]
            for method in ["original", record["method"]]
            for v in pair[method].values()
        ]
        assert torch.isfinite(torch.tensor(values, device="cuda", dtype=torch.float64)).all()
        assert all(v > 0 for v in values)
    micro = read(out / "microbenchmark.json")
    assert micro["source_sha256"] == digest(HERE / "microbenchmark.py")
    assert micro["input_sha256"] == digest(out / "collection/language-selection.pt")
    assert micro["selection_sha256"] == digest(out / "training/selection.json")
    assert len(micro["records"]) == 5 * sum(name.endswith("-s1103") for name in names)
    for record in micro["records"]:
        assert len(record["samples"]) == 50
        values = [
            v for pair in record["samples"] for timing in pair.values() for v in timing.values()
        ]
        tensor = torch.tensor(values, device="cuda")
        assert torch.isfinite(tensor).all() and (tensor > 0).all()
    compiler = out / "compiler-diagnostic"
    if compiler.exists():
        source_check(read(compiler / "freeze.json"))
        paths = list(compiler.glob("*-b*-s*.json"))
        assert len(paths) == 5
        for path in paths:
            record = read(path)
            assert record["freeze_sha256"] == digest(compiler / "freeze.json")
            assert record["checkpoint_sha256"] == digest(
                out / "training" / path.with_suffix(".pt").name
            )
            assert len(record["records"]) == 2
            for result in record["records"]:
                assert result["status"] in {"complete", "failed"}
                if result["status"] == "complete":
                    assert len(result["samples"]) == 50
                    values = [
                        v
                        for pair in result["samples"]
                        for metrics in pair.values()
                        for v in metrics.values()
                    ]
                    values = torch.tensor(values, device="cuda")
                    assert torch.isfinite(values).all() and (values > 0).all()
                else:
                    assert result["error_type"] and result["error"]
    for split in ["gate", "shift"]:
        for style in ["known", "new"]:
            name = f"{split}-256-{style}"
            frozen_path = mechanism / "state-sufficiency/v3" / f"{name}-freeze.json"
            frozen = read(frozen_path)
            assert frozen["source_sha256"] == digest(HERE / "state_sufficiency.py")
            for source, sha in frozen["support_sources"].items():
                assert digest(HERE / source) == sha
            for checkpoint, sha in frozen["equations"].items():
                assert digest(mechanism / "residual/equations" / checkpoint) == sha
            data = read(frozen_path.with_name(f"{name}.json"))
            assert data["freeze_sha256"] == digest(frozen_path)
            assert data["evaluated_groups"] == 256
            identities = [(r["id"], r["style"], r["kind"]) for r in data["records"]]
            assert len(identities) == len(set(identities)) == 256 * 2 * len(frozen["variants"])
            equations = read(mechanism / "equation-responses" / f"{name}.json")
            assert equations["source_data_sha256"] == digest(frozen_path.with_name(f"{name}.json"))
            assert equations["source_sha256"] == digest(HERE / "equation_responses.py")
            for checkpoint, sha in equations["checkpoints"].items():
                assert digest(mechanism / "residual/equations" / checkpoint) == sha
    save(
        out / "release-audit.json",
        {
            "status": "complete",
            "scientific_inputs": checked,
            "decoder_rank_checks": decoder_ranks,
            "source_sha256": digest(HERE / "audit_release.py"),
            "scope": "GPU scoring/accounting validation and complete identity/source/checkpoint binding. Fit objectives were independently replayed on CUDA by audit_fits.",
        },
    )
    print("RELEASE AUDIT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
