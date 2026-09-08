"""Generate standardized training/validation derivative targets in active coordinates."""

import os
import time

import torch

from active_features import gradients
from data import ROOT, save
from model_io import load_mlp, setup


def main():
    setup()
    save(ROOT / "derivative-job.json", {"pid": os.getpid(), "start_epoch": time.time()})
    for op in ["add", "multiply", "divide"]:
        out = ROOT / op
        c = torch.load(out / "component-active.pt", weights_only=True)
        layer = load_mlp(c["layer"])
        d = c["direction"].cuda()
        features = torch.load(out / "features-active.pt", weights_only=True)
        derivative = {}
        for split in ["train", "validation"]:
            h = torch.load(out / f"inputs-{split}.pt", weights_only=True)["input"].cuda()
            mixed = torch.cat(
                [h[:, 1] * (1 - a) + h[:, 0] * a for a in [0.0, 0.25, 0.5, 0.75, 1.0]]
            )
            g = gradients(layer, mixed, d)
            projected = (g @ c["encoder"].cuda()) * c["zstd"].cuda() / c["ystd"].cuda()
            assert torch.isfinite(projected).all()
            derivative[split] = {"gradient": projected.cpu()}
            # Confirm ordering and target standardization by recomputing a small sample.
            with torch.no_grad():
                y = (layer(mixed[:32]) @ d - c["ymean"].cuda()) / c["ystd"].cuda()
            assert torch.allclose(y, features[split]["y"][:32].cuda(), atol=1e-4, rtol=1e-4)
        torch.save(derivative, out / "gradients-active.pt")
        print("DERIVATIVES DONE", op, flush=True)
        del layer, g, projected, mixed, h, derivative
        torch.cuda.empty_cache()
    save(ROOT / "derivative-completed.json", {"completed_epoch": time.time()})


if __name__ == "__main__":
    main()
