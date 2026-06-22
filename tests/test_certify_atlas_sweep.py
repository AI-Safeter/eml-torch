"""Sweep-mode acceptance: AttentionCertAtlas certifies every (layer, head) of a
real model on one prompt via ONE forward pass, using the honest softmax_interval
form. The canonical induction head L5.H5 must certify (radius > 0 at tau=0.5)
and attend to the induction target " drift".

Loads GPT-2 (CPU). Run from emltorch/:
    HF_HOME=~/.cache/huggingface CUDA_VISIBLE_DEVICES="" \
        pytest tests/test_certify_atlas_sweep.py -q
"""

from __future__ import annotations

import json

import pytest

from emltorch.certify.atlas import AttentionCertAtlas, AtlasResult

# Same prompt as test_certify_atlas_gpt2.py.
INDUCTION_PROMPT = (
    "vase comet lunar drift ember vase comet lunar drift ember vase comet lunar"
)


@pytest.fixture(scope="module")
def atlas_result():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    atlas = AttentionCertAtlas("gpt2", prompt=INDUCTION_PROMPT, tau=0.5, device="cpu")
    return atlas.run()


def test_sweep_returns_all_heads(atlas_result):
    res = atlas_result
    assert isinstance(res, AtlasResult)
    # GPT-2 small: 12 layers x 12 heads = 144 records.
    assert len(res.records) == 12 * 12
    layers = {r["layer"] for r in res.records}
    heads = {r["head"] for r in res.records}
    assert layers == set(range(12))
    assert heads == set(range(12))


def test_l5h5_certifies_and_attends_to_drift(atlas_result):
    res = atlas_result
    rec = next(r for r in res.records if r["layer"] == 5 and r["head"] == 5)
    assert rec["attends_to_token"] == " drift"
    assert rec["certified_radius"] > 0.0
    # honesty: target_prob recorded so vacuity is impossible to hide
    assert 0.45 < rec["target_prob"] < 0.85


def test_to_json_round_trips(atlas_result):
    res = atlas_result
    blob = res.to_json()
    s = json.dumps(blob)
    back = json.loads(s)
    assert back["tau"] == 0.5
    assert back["model_name"] == "gpt2"
    assert len(back["records"]) == 144
