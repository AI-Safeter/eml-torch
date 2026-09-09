"""Load a folded replacement without training data or teacher activations."""

import torch

from .model import Deployed, Replacement


def load_export(path, device="cuda", dtype=torch.bfloat16):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = Deployed(Replacement(**payload["specification"]))
    model.load_state_dict(payload["state"])
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)
