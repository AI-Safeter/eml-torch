"""Export the folded student; reload it without teacher activations."""

import argparse

import torch

from .common import accounting, digest, previous, root, save, setup, sources
from .model import Deployed, Replacement, load_replacement


def load_export(path, device="cuda", dtype=torch.bfloat16):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = Deployed(Replacement(**payload["specification"]))
    model.load_state_dict(payload["state"])
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    a = p.parse_args()
    setup(92)
    out = root()
    checkpoint = out / "training" / f"{a.method}.pt"
    trained = load_replacement(checkpoint, device="cpu", dtype=torch.float32)
    deployed = Deployed(trained).bfloat16().eval().requires_grad_(False)
    directory = out / "exports"
    directory.mkdir(exist_ok=True)
    path = directory / f"{a.method}.pt"
    torch.save({"specification": trained.specification(), "state": deployed.state_dict()}, path)
    restored = load_export(path)
    deployed.cuda()
    x = (
        torch.load(previous(out) / "collection/language-selection.pt", weights_only=True)["x"][:256]
        .cuda()
        .bfloat16()
    )
    with torch.inference_mode():
        assert torch.equal(deployed(x), restored(x))
    save(
        directory / f"{a.method}.json",
        {
            "checkpoint_sha256": digest(checkpoint),
            "export_sha256": digest(path),
            "export_file_bytes": path.stat().st_size,
            "runtime": accounting(restored),
            "reload_bitwise_equal": True,
            "teacher_activation_inputs": False,
            "sources": sources("export.py", "model.py"),
        },
    )
    print("EXPORT VERIFIED", a.method, flush=True)


if __name__ == "__main__":
    main()
