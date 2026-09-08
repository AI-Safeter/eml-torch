"""Selected rows of the corpus-averaged Jacobian lens on native Gemma.

The estimator follows Anthropic's published target-sum/source-mean reduction.
Direct vector-Jacobian products avoid materializing the full 1536 x 1536 map.
This is a limited dictionary; it does not compute sparse J-space coordinates.
"""

import argparse
import contextlib
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "protocol.json").read_text())
RUN = Path(os.environ.get("EML_JSPACE_RUN", str(HERE.parent.parent / "emltorch-jspace-runs")))
SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots"
    / SPEC["revision"]
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_tensors(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)


def model():
    import transformers

    assert torch.cuda.is_available(), "This experiment requires CUDA"
    assert torch.__version__ == "2.9.0+cu128" and transformers.__version__ == "5.16.1"
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    return (
        transformers.Gemma4ForConditionalGeneration.from_pretrained(
            SNAPSHOT,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
            device_map={"": 0},
        )
        .eval()
        .requires_grad_(False)
    )


@contextlib.contextmanager
def capture(native, layers, start_graph=None, intervention=None):
    """Record block outputs while preserving native PLE, masking and shared KV."""
    records, handles = {}, []

    def hook(index):
        def apply(module, inputs, output):
            assert torch.is_tensor(output)
            if intervention is not None and index == intervention[0]:
                output = intervention[1](output)
            if index == start_graph:
                output.requires_grad_(True)
            records[index] = output
            return output

        return apply

    try:
        for layer in sorted(set(layers)):
            handles.append(
                native.model.language_model.layers[layer].register_forward_hook(hook(layer))
            )
        yield records
    finally:
        for handle in handles:
            handle.remove()


def source_rows(native, tok):
    token_ids = [tok.encode(" " + word, add_special_tokens=False) for word in SPEC["concepts"]]
    assert all(len(ids) == 1 for ids in token_ids)
    token_ids = [ids[0] for ids in token_ids]
    rows = native.lm_head.weight[token_ids].float()
    rows *= native.model.language_model.norm.weight.float()
    return torch.nn.functional.normalize(rows, dim=-1), token_ids


def make_documents(tok):
    source = Path(
        os.environ.get(
            "EML_SQUAD_TRAIN", str(HERE.parent.parent / "emltorch-kv-runs/squad-train.json")
        )
    )
    assert sha(source) == SPEC["calibration"]["source_sha256"]
    articles = json.loads(source.read_text())["data"]
    random.Random(SPEC["calibration"]["seed"]).shuffle(articles)
    documents = []
    for article in articles:
        text = article["paragraphs"][0]["context"]
        ids = tok.encode(text, add_special_tokens=False)
        ids = [tok.bos_token_id] + ids
        if len(ids) < SPEC["calibration"]["tokens"]:
            continue
        documents.append(
            {"title": article["title"], "text": text, "ids": ids[: SPEC["calibration"]["tokens"]]}
        )
        if len(documents) == SPEC["calibration"]["articles"]:
            break
    assert len(documents) == SPEC["calibration"]["articles"]
    save_json(RUN / "documents.json", documents)
    save_json(
        RUN / "freeze.json",
        {
            "protocol_sha256": sha(HERE / "protocol.json"),
            "documents_sha256": sha(RUN / "documents.json"),
            "lens_source_sha256": sha(HERE / "lens.py"),
            "model_revision": SPEC["revision"],
        },
    )
    return documents


def verify():
    freeze = json.loads((RUN / "freeze.json").read_text())
    for name, path in [
        ("protocol", HERE / "protocol.json"),
        ("documents", RUN / "documents.json"),
        ("lens_source", HERE / "lens.py"),
    ]:
        assert sha(path) == freeze[name + "_sha256"], f"Changed {name} after calibration began"


def gradients(native, ids, rows):
    """Return [layer, concept, input_dimension] reduced Jacobian rows."""
    layers = SPEC["layers"]
    count = SPEC["calibration"]["batch"]
    valid = slice(SPEC["calibration"]["skip_first"], -1)
    chunks = []
    # One retained graph for all cotangents, with independent batch replicas.
    with torch.enable_grad(), capture(native, layers + [34], start_graph=min(layers)) as h:
        native.model.language_model(input_ids=ids.expand(count, -1), use_cache=False)
        for offset in range(0, len(rows), count):
            take = min(count, len(rows) - offset)
            cotangent = torch.zeros_like(h[34])
            cotangent[:take, valid] = rows[offset : offset + take, None].to(cotangent.dtype)
            grads = torch.autograd.grad(
                h[34], [h[i] for i in layers], cotangent, retain_graph=offset + count < len(rows)
            )
            chunks.append(torch.stack([g[:take, valid].float().mean(1).cpu() for g in grads]))
    return torch.cat(chunks, dim=1)


def calibrate(limit=None):
    RUN.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    if (RUN / "freeze.json").exists():
        verify()
        documents = json.loads((RUN / "documents.json").read_text())
    else:
        documents = make_documents(tok)
    native = model()
    rows, token_ids = source_rows(native, tok)
    checkpoint = RUN / "calibration.pt"
    collected = (
        torch.load(checkpoint, weights_only=True)["per_document"] if checkpoint.exists() else []
    )
    for index in range(len(collected), min(limit or len(documents), len(documents))):
        started = time.monotonic()
        ids = torch.tensor([documents[index]["ids"]], device="cuda")
        value = gradients(native, ids, rows)
        assert torch.isfinite(value).all()
        collected.append(value)
        save_tensors(checkpoint, {"per_document": collected, "token_ids": token_ids})
        print(
            json.dumps(
                {
                    "document": index + 1,
                    "seconds": time.monotonic() - started,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
            ),
            flush=True,
        )
    per_document = torch.stack(collected)
    split = len(per_document) // 2
    report = {
        "documents": len(collected),
        "complete": len(collected) == len(documents),
        "gpu": torch.cuda.get_device_name(),
        "token_ids": token_ids,
        "calibration_sha256": sha(checkpoint),
    }
    if split:
        cosine = torch.nn.functional.cosine_similarity(
            per_document[:split].mean(0), per_document[split:].mean(0), dim=-1
        )
        report["split_half_cosines"] = {
            str(layer): dict(zip(SPEC["concepts"], cosine[i].tolist()))
            for i, layer in enumerate(SPEC["layers"])
        }
    save_json(RUN / "calibration.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    calibrate(parser.parse_args().limit)
