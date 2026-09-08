"""V codecs with real packed storage and a fixed, amortized byte budget."""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import SPEC, storage_bytes

from emltorch import EMLHead


class SiLUHead(EMLHead):
    """Same affine branches, skip and parameter count as EML; gated SiLU features."""

    def forward(self, x):
        return self.skip(x) + self.out(torch.nn.functional.silu(self.left(x)) * self.right(x))


def pack4(v):
    # Per-token ranges, no floating-point tensor masquerading as packed int4.
    lo = v.amin(-1, keepdim=True)
    scale = ((v.float().amax(-1, keepdim=True) - lo.float()) / 15).clamp_min(1e-8).to(v.dtype)
    q = ((v.float() - lo.float()) / scale.float()).round().clamp(0, 15).to(torch.uint8)
    return (q[..., ::2] | (q[..., 1::2] << 4)).contiguous(), lo, scale


def unpack4(code):
    packed, lo, scale = code
    q = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    return (q.float() * scale.float() + lo.float()).to(lo.dtype)


def parameter_elements(method, d, r):
    if method == "lowrank":
        return d * r + d  # one shared basis for encoding and decoding
    h = SPEC["nonlinear_width"]
    return 2 * d * r + 2 * d + 2 * r * h + 2 * h + h * d


def rank_for_budget(method, d, tokens):
    bank = SPEC["storage_reference_documents"]
    budget = bank * tokens * (d // 2 + 4)  # packed V + two BF16 range values per token
    return max(
        r
        for r in range(1, d + 1)
        if 2 * bank * tokens * r + 2 * parameter_elements(method, d, r) <= budget
    )


class Codec(nn.Module):
    def __init__(self, method, mean, basis):
        super().__init__()
        self.method = method
        self.register_buffer("mean", mean)
        self.register_buffer("basis", basis)
        d, r = basis.shape
        if method != "lowrank":
            cls = EMLHead if method == "eml" else SiLUHead
            self.head = cls(r, SPEC["nonlinear_width"], d, dtype=torch.float32, device=basis.device)
            with torch.no_grad():
                self.head.skip.weight.copy_(basis)
                self.head.skip.bias.copy_(mean)
                self.head.out.weight.zero_()

    def encode(self, v):
        return (v - self.mean) @ self.basis

    def decode(self, z):
        if self.method == "lowrank":
            return z @ self.basis.T + self.mean
        return self.head(z)

    def bytes(self):
        return storage_bytes(list(self.state_dict().values()))


def load_codecs(path, method):
    saved = torch.load(path, map_location="cuda", weights_only=True)[method]
    result = []
    for state in saved:
        obj = Codec(method, state["mean"], state["basis"]).bfloat16().eval()
        obj.load_state_dict(state)
        result.append(obj)
    return result
