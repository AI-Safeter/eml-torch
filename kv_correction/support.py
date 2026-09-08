"""Shared I/O and frozen numerical definitions for the one-layer correction gate."""

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

SPEC = json.loads((HERE / "protocol.json").read_text())
PILOT = ROOT.parent / "emltorch-kv-runs"
RUN = Path(os.environ.get("EML_CORRECTION_RUN", str(ROOT.parent / "emltorch-correction-runs")))
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
    / SPEC["revision"]
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def setup():
    import transformers

    assert torch.cuda.is_available()
    assert torch.__version__ == "2.9.0+cu128" and transformers.__version__ == "5.16.1"
    torch.set_num_threads(2)
    torch.manual_seed(20260908)
    torch.backends.cuda.matmul.allow_tf32 = False


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)


def model():
    from transformers import Gemma4ForConditionalGeneration

    setup()
    return (
        Gemma4ForConditionalGeneration.from_pretrained(
            SNAPSHOT,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
            device_map={"": 0},
        )
        .eval()
        .requires_grad_(False)
    )


def ids(values):
    return torch.tensor([values], device="cuda", dtype=torch.long)


def storage_bytes(tensors):
    return sum(
        {
            (str(x.device), x.untyped_storage().data_ptr()): x.untyped_storage().nbytes()
            for x in tensors
        }.values()
    )


def snapshot(cache):
    return [
        {
            "k": layer.keys.cpu().clone().contiguous(),
            "v": layer.values.cpu().clone().contiguous(),
            "length": int(layer.get_seq_length()),
        }
        for layer in cache.layers
    ]


def restore(records, config):
    from transformers import DynamicCache

    cache = DynamicCache(config=config)
    assert len(records) == len(cache.layers) == 15
    for layer, item in zip(cache.layers, records):
        k, v = item["k"].cuda(), item["v"].cuda()
        layer.lazy_initialization(k, v)
        layer.keys, layer.values = k.clone(), v.clone()
        if layer.is_sliding:
            layer.cumulative_length = item["length"]
        assert layer.get_seq_length() == item["length"]
    return cache


def pack4(v):
    lo = v.amin(-1, keepdim=True)
    scale = ((v.float().amax(-1, keepdim=True) - lo.float()) / 15).clamp_min(1e-8).to(v.dtype)
    q = ((v.float() - lo.float()) / scale.float()).round().clamp(0, 15).to(torch.uint8)
    return (q[..., ::2] | (q[..., 1::2] << 4)).contiguous(), lo, scale


def unpack4(code):
    packed, lo, scale = code
    q = torch.stack((packed & 15, packed >> 4), -1).flatten(-2)
    return (q.float() * scale.float() + lo.float()).to(lo.dtype)


def projected_error(delta, attention, projection):
    heads = torch.einsum("qht,td->qhd", attention, delta.float())
    return heads.flatten(1) @ projection.T


def errors(value, record, projection):
    delta = value.float() - record["v"].float()
    return torch.stack(
        (
            delta.square().mean(),
            projected_error(delta, record["attention"], projection).square().mean(),
        )
    )


def load_records(split):
    items = torch.load(RUN / f"{split}.pt", weights_only=True, map_location="cuda")
    for row in items:
        row["code"] = pack4(row["v"])
    return items


def verify_design():
    design = json.loads((RUN / "design.json").read_text())
    assert sha(HERE / "protocol.json") == design["protocol_sha256"]
    assert sha(RUN / "documents.json") == design["documents_sha256"]


def freeze_sources():
    paths = sorted(HERE.glob("*.py")) + [HERE / "protocol.json"]
    paths += [
        ROOT / p
        for p in [
            "emltorch/head.py",
            "emltorch/operator.py",
            "emltorch/_validation.py",
        ]
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def verify_sources(freeze):
    for name, digest in freeze.items():
        assert sha(ROOT / name) == digest, f"Source changed: {name}"
