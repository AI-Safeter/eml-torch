"""Fixed-architecture, new-initialization replication of the addition head."""

import argparse
import time

import torch

from data import ROOT, save
from model_io import setup
from train_heads import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, choices=[59, 71, 83, 97])
    args = parser.parse_args()
    setup()
    out = ROOT / "add/seed-replication"
    out.mkdir(exist_ok=True)
    assert not (out / f"seed-{args.seed}.json").exists(), "Preserve completed seed fits"
    data = torch.load(out.parent / "features-active.pt", weights_only=True)
    gradient = torch.load(out.parent / "gradients-active.pt", weights_only=True)
    for split in data:
        data[split]["gradient"] = gradient[split]["gradient"]
    rows = []
    for kind in ["eml_square", "silu_two"]:
        model, record = train(kind, 32, args.seed, 4, data, max_steps=20000, gradient_weight=0.1)
        assert model is not None
        name = f"s{args.seed}-{kind}"
        record["name"] = name
        torch.save(model.state_dict(), out / f"{name}.pt")
        rows.append(record)
        save(out / f"seed-{args.seed}.json", rows)
        print("SEED REPLICATION", args.seed, kind, record, flush=True)
    save(out / f"completed-{args.seed}.json", {"completed_epoch": time.time()})


if __name__ == "__main__":
    main()
