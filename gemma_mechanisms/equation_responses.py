"""Evaluate frozen equations on original-model intervention responses, on CUDA."""

import argparse
import collections
import json

import torch

from .equations import symbolic
from .runtime import HERE, digest, root, save, setup
from .student import Student


def bootstrap_mean(values, groups, generator):
    grouped = torch.stack([values[ids].mean() for ids in groups])
    indices = torch.randint(len(grouped), (10000, len(grouped)), device="cuda", generator=generator)
    samples = grouped[indices].mean(-1)
    return torch.quantile(samples, torch.tensor([0.025, 0.975], device="cuda")).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True, help="E.g. gate-256-known")
    args = parser.parse_args()
    setup(831)
    out = root()
    directory = out / "residual/equations"
    source = out / "state-sufficiency/v3" / f"{args.cohort}.json"
    data = json.loads(source.read_text())
    by_kind = collections.defaultdict(dict)
    for row in data["records"]:
        key = (row["id"], row["style"])
        assert key not in by_kind[row["kind"]]
        by_kind[row["kind"]][key] = row
    keys = sorted(by_kind["recipient"])
    ids = sorted({k[0] for k in keys})
    groups = [
        torch.tensor([i for i, key in enumerate(keys) if key[0] == name], device="cuda")
        for name in ids
    ]

    def tensor(kind, layer):
        assert set(by_kind[kind]) == set(keys)
        return torch.tensor(
            [by_kind[kind][k]["quantities"][str(layer)] for k in keys], device="cuda"
        )

    linear = torch.load(directory / "linear.pt", weights_only=True)
    weight, scale = linear["weight"].cuda(), linear["scale"].cuda()
    models = {
        "linear": lambda x: torch.cat([x, torch.ones(len(x), 1, device="cuda")], -1) @ weight,
        "symbolic": symbolic,
        "identity": lambda x: x,
    }
    checkpoints = {}
    for path in sorted(directory.glob("*-d*-s*.pt")):
        checkpoint = torch.load(path, weights_only=True)
        stats = {k: checkpoint["state"][k] for k in ["xmean", "xstd", "ymean", "yscale"]}
        model = Student(**checkpoint["specification"], statistics=stats)
        model.load_state_dict(checkpoint["state"])
        models[path.stem] = model.cuda().eval().requires_grad_(False)
        checkpoints[path.name] = digest(path)
    generator = torch.Generator(device="cuda").manual_seed(831)
    results = {}
    with torch.inference_mode():
        bx, by = tensor("recipient", 26), tensor("recipient", 32)
        for name, model in models.items():
            base = model(bx)
            results[name] = {}
            for kind in by_kind:
                if kind.startswith("values_then_"):
                    continue  # The downstream equation is directly overridden.
                x, y = tensor(kind, 26), tensor(kind, 32)
                prediction = model(x)
                error = ((prediction - base - y + by) / scale).square().mean(-1)
                response = ((y - by) / scale).square().mean(-1)
                observation = ((prediction - y) / scale).square().mean(-1)
                results[name][kind] = {
                    "observational_normalized_mse": float(observation.mean()),
                    "response_normalized_mse": float(error.mean()),
                    "response_mse_95ci": bootstrap_mean(error, groups, generator),
                    "actual_response_energy": float(response.mean()),
                    "response_nrmse": float((error.mean() / response.mean()).sqrt())
                    if response.mean() > 1e-12
                    else None,
                    "input_identical_to_recipient_count": int((x == bx).all(-1).sum()),
                    "cases": len(keys),
                }
        hx, hy = tensor("hidden_only", 26), tensor("hidden_only", 32)
        fullx, fully = tensor("both", 26), tensor("both", 32)
        assert torch.equal(hx, fullx)
        lower = ((hy - fully) / scale).square().mean(-1) / 4
        collision = {
            "identical_upstream_quantity_vectors": len(keys),
            "distinct_downstream_quantity_vectors": int((hy != fully).any(-1).sum()),
            "minimum_average_normalized_mse_on_equiprobable_pair": float(lower.mean()),
            "interpretation": "Any deterministic map of this upstream vector makes the same prediction for both outcomes; the optimal shared prediction is their midpoint. This is an input-sufficiency bound, not an EML-specific bound.",
        }
    save(
        out / "equation-responses" / f"{args.cohort}.json",
        {
            "cohort": args.cohort,
            "results": results,
            "input_collision": collision,
            "source_data_sha256": digest(source),
            "source_sha256": digest(HERE / "equation_responses.py"),
            "checkpoints": checkpoints,
            "variables": json.loads((directory / "freeze.json").read_text())["variables"],
            "scope": "Original-model responses under fixed answer prefixes; no generated-answer or algorithm claim. All intervals cluster prompt formats by operand pair.",
        },
    )
    print("EQUATION RESPONSES COMPLETE", args.cohort, flush=True)


if __name__ == "__main__":
    main()
