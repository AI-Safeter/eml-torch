"""Five seeds with checkpoint selection on documents excluded from the gate."""

import copy
import json

import torch
from correctors import Correction
from support import (
    RUN,
    SPEC,
    errors,
    freeze_sources,
    load_records,
    save_json,
    setup,
    sha,
    verify_design,
)


@torch.inference_mode()
def validation(candidate, records, projection):
    deployed = copy.deepcopy(candidate).bfloat16().eval()
    measured = torch.stack([errors(deployed(r["code"]), r, projection) for r in records])
    normalized = measured / torch.stack([r["baseline"] for r in records])
    return float(normalized.mean()), normalized.mean(0).tolist(), deployed


def main():
    setup()
    verify_design()
    assert not (RUN / "fit-freeze.json").exists()
    collection = json.loads((RUN / "collection.json").read_text())
    for name, digest in collection["artifact_sha256"].items():
        assert sha(RUN / name) == digest
    save_json(
        RUN / "fit-freeze.json",
        {"sources": freeze_sources(), "collection_sha256": sha(RUN / "collection.json")},
    )
    projection = torch.load(RUN / "projection.pt", weights_only=True, map_location="cuda")
    train, selection = load_records("train"), load_records("selection")
    from support import unpack4

    for record in train + selection:
        record["baseline"] = errors(unpack4(record["code"]), record, projection)
        assert (record["baseline"] > 1e-12).all()
    completed = []
    for kind in ["linear", "silu", "eml"]:
        for seed in SPEC["seeds"]:
            candidate = Correction(kind, seed).cuda().float()
            optimizer = torch.optim.AdamW(
                candidate.parameters(), lr=SPEC["learning_rate"], weight_decay=SPEC["weight_decay"]
            )
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randint(len(train), (SPEC["fit_steps"],), generator=generator).tolist()
            history = []
            best, best_state, best_step = float("inf"), None, None
            for step in range(SPEC["fit_steps"] + 1):
                if step:
                    record = train[indices[step - 1]]
                    loss = (
                        errors(candidate(record["code"]), record, projection) / record["baseline"]
                    ).mean()
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
                    optimizer.step()
                if step % SPEC["validation_every"] == 0:
                    objective, components, deployed = validation(candidate, selection, projection)
                    history.append(
                        {
                            "step": step,
                            "objective": objective,
                            "v_error_ratio": components[0],
                            "attention_error_ratio": components[1],
                        }
                    )
                    if objective < best:
                        best, best_step = objective, step
                        best_state = {k: v.cpu().clone() for k, v in deployed.state_dict().items()}
                    del deployed
            path = RUN / f"{kind}-{seed}.pt"
            torch.save(best_state, path)
            row = {
                "method": kind,
                "seed": seed,
                "selected_step": best_step,
                "selection_objective": best,
                "checkpoint_sha256": sha(path),
                "weight_bytes": sum(v.numel() * v.element_size() for v in best_state.values()),
                "history": history,
            }
            assert row["weight_bytes"] <= SPEC["correction_weight_bytes_ceiling"]
            completed.append(row)
            save_json(RUN / "fits.json", completed)
            print("FIT", kind, seed, best_step, best, flush=True)
    save_json(
        RUN / "fits-complete.json",
        {
            "fits": 15,
            "fits_sha256": sha(RUN / "fits.json"),
            "fit_freeze_sha256": sha(RUN / "fit-freeze.json"),
        },
    )


if __name__ == "__main__":
    main()
