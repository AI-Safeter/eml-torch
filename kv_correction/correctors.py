"""Correction consumes packed cache data and shared weights; no original V input."""

import torch
from support import SPEC, pack4, storage_bytes, unpack4
from torch import nn

from emltorch import EMLHead


class GatedSiLU(EMLHead):
    def forward(self, x):
        return self.out(torch.nn.functional.silu(self.left(x)) * self.right(x))


class Correction(nn.Module):
    def __init__(self, kind, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.kind = kind
        if kind == "linear":
            self.head = nn.Sequential(
                nn.Linear(514, SPEC["linear_rank"]), nn.Linear(SPEC["linear_rank"], 512)
            )
            final = self.head[-1]
        else:
            cls = EMLHead if kind == "eml" else GatedSiLU
            self.head = cls(514, SPEC["nonlinear_width"], 512, linear_skip=False)
            final = self.head.out
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, code):
        _, minimum, step = code
        decoded = unpack4(code)
        features = torch.cat((decoded.float(), minimum.float() / 4, step.float() * 4), -1)
        dtype = next(self.parameters()).dtype
        correction = self.head(features.to(dtype)).float() * step.float()
        return (decoded.float() + correction).to(decoded.dtype)

    def bytes(self):
        return storage_bytes(list(self.state_dict().values()))


def encode_residual(v):
    code = pack4(v)
    residual = (v.float() - unpack4(code).float()).flatten()
    indices = residual.abs().topk(SPEC["stored_residual_elements_per_prefix"]).indices
    return code, indices.to(torch.int32), residual[indices].bfloat16()


def baseline(kind, code, stored=None):
    decoded = unpack4(code)
    if kind == "rms":
        x = decoded.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)).to(decoded.dtype)
    if kind == "stored_residual":
        indices, residual = stored
        x = decoded.float().flatten()
        x[indices.long()] += residual.float()
        return x.view_as(decoded).to(decoded.dtype)
    assert kind == "none"
    return decoded
