"""CUDA release audit: accounting, paired statistics, update streams, and bound closure."""

import hashlib
import json
import platform
import subprocess

import torch

from gemma_mechanisms.quality_summary import accuracy as reference_accuracy

from .common import HERE, SPEC, accounting, digest, root, save, setup, sources
from .model import load_replacement
from .statistics import accuracy


def main():
    setup(93)
    out = root()
    checked = {}
    rngs = []
    for path in sorted((out / "training").glob("*-n*.json")):
        row = json.loads(path.read_text())
        checkpoint = path.with_suffix(".pt")
        assert row["status"] == "complete" and row["steps"] == int(path.stem.rsplit("n", 1)[1])
        assert digest(checkpoint) == row["checkpoint_sha256"]
        m = load_replacement(checkpoint, dtype=torch.float32)
        assert accounting(m) == row["accounting"]
        deployed = load_replacement(checkpoint, deployed=True)
        assert accounting(deployed) == row["deployed_accounting"]
        # Full registration accounting; individual components sum to the same totals.
        parts = {
            k: {
                "shape": list(v.shape),
                "coefficients": v.numel(),
                "bytes": v.numel() * v.element_size(),
            }
            for k, v in [*deployed.named_parameters(), *deployed.named_buffers()]
        }
        assert (
            sum(v["coefficients"] for v in parts.values()) == accounting(deployed)["coefficients"]
        )
        continuation = torch.load(
            path.with_name(path.stem + "-continuation.pt"), map_location="cpu", weights_only=True
        )
        rngs.append((row["seed"], row["steps"], continuation["rng"].cuda()))
        diagnostic = out / "diagnostics" / path.name
        if diagnostic.exists():
            d = json.loads(diagnostic.read_text())
            expected = sum(v["raw_mse"] for v in row["train"].values()) / 2
            assert abs(expected - d["train"]["raw_mse"]) < 1e-6
            if d["train"]["decomposition_closure_error"] is not None:
                assert d["train"]["decomposition_closure_error"] < 1e-10
        checked[path.stem] = {
            "training": accounting(m),
            "deployed": accounting(deployed),
            "deployed_tensors": parts,
            "checkpoint_sha256": digest(checkpoint),
            "fit_record_sha256": digest(path),
        }
        del m, deployed, continuation
    for seed, steps, rng in rngs:
        for other_seed, other_steps, other_rng in rngs:
            if (seed, steps) == (other_seed, other_steps):
                assert torch.equal(rng, other_rng)
    directory = out / "evaluation/development"
    original = json.loads((directory / "original.json").read_text())["results"]
    student = json.loads((directory / "bottleneck-eml-d1-s1103-n12000.json").read_text())["results"]
    for task in ["arithmetic", "arc"]:
        fast = accuracy(original[task], student[task])
        slow = reference_accuracy(original[task], student[task], 0.05 / 15)
        for k in [
            "teacher_accuracy",
            "student_accuracy",
            "net_loss",
            "regressing_groups",
            "improving_groups",
            "groups",
            "loss_upper_bound",
            "passes",
            "net_loss_bootstrap",
        ]:
            if isinstance(fast[k], dict):
                for field in fast[k]:
                    torch.testing.assert_close(
                        torch.tensor(fast[k][field], device="cuda", dtype=torch.float64),
                        torch.tensor(slow[k][field], device="cuda", dtype=torch.float64),
                        rtol=0,
                        atol=1e-12,
                    )
            elif isinstance(fast[k], float):
                torch.testing.assert_close(
                    torch.tensor(fast[k], device="cuda", dtype=torch.float64),
                    torch.tensor(slow[k], device="cuda", dtype=torch.float64),
                    rtol=0,
                    atol=1e-12,
                )
            else:
                assert fast[k] == slow[k], (k, fast[k], slow[k])
    bindings = {
        str(p.relative_to(out)): digest(p)
        for directory in ["training", "evaluation", "diagnostics", "exports", "benchmark"]
        for p in sorted((out / directory).rglob("*.json"))
    }
    for path in (out / "evaluation").rglob("*-freeze.json"):
        record = json.loads(path.read_text())
        for relative, sha in record["sources"].items():
            assert digest(HERE.parent / relative) == sha
    decision = json.loads((out / "decision.json").read_text())
    if decision["status"] == "stop":
        assert not (out / "selection.json").exists()
        assert not list((out / "evaluation/confirmation").glob("*.json"))
    ledger = json.loads((out / "budget.json").read_text())
    assert ledger["jobs"]["prepare-data"]["finished_unix"] < min(
        row["started_unix"] for key, row in ledger["jobs"].items() if key.startswith("fit-")
    )
    precision = json.loads((out / "precision-audit.json").read_text())
    for method, row in precision["methods"].items():
        assert row["checkpoint_sha256"] == checked[method]["checkpoint_sha256"]
        assert row["metrics"]["folded_fp32"]["prediction_mse_vs_fp32"] < 1e-10
    inherited = {}
    for relative in [
        "gemma_mechanisms/runtime.py",
        "gemma_mechanisms/student.py",
        "gemma_mechanisms/train.py",
        "gemma_mechanisms/evaluate.py",
        "gemma_mechanisms/prepare.py",
        "gemma_mechanisms/quality_summary.py",
        "gemma_mechanisms/protocol.json",
        "emltorch/operator.py",
    ]:
        original = subprocess.check_output(
            ["git", "show", SPEC["previous_commit"] + ":" + relative], cwd=HERE.parent
        )
        sha = hashlib.sha256(original).hexdigest()
        assert digest(HERE.parent / relative) == sha
        inherited[relative] = sha
    record = {
        "inherited_sources_unchanged_from_previous_commit": inherited,
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
        },
        "precision_audit_sha256": digest(out / "precision-audit.json"),
        "fresh_confirmation_unopened": decision["status"] == "stop",
        "data_frozen_before_training": True,
        "fits": checked,
        "matched_minibatch_rng_states": True,
        "paired_statistics_match_previous_reference": True,
        "paired_float_absolute_tolerance": 1e-12,
        "paired_integer_and_boolean_comparisons_exact": True,
        "raw_error_replay_and_decomposition_pass": True,
        "inputs": bindings,
        "sources": sources("audit.py", "model.py", "statistics.py"),
        "model_and_tensor_checks_on_cuda": True,
        "scalar_probability_quantiles": "SciPy beta inverse CDF on CPU; paired bootstrap on CUDA",
    }
    save(out / "release-audit.json", record)
    print("RELEASE AUDIT COMPLETE", len(checked), len(bindings), flush=True)


if __name__ == "__main__":
    main()
