"""Acceptance anchor: the honest cert tool, end-to-end on a real model.

GPT-2-small's canonical induction head L5.H5 (Olsson 2022) on a repeated-token
prompt attends to the induction target, and softmax_interval certifies its
concentration up to the head's real margin -- and SOUNDLY refuses to certify
above it. This is the non-vacuous behavior the vacuity audit (2026-06-19)
demanded after the log-prob v3 certs were found vacuous.

Loads GPT-2 (CPU). Run from emltorch/:
    HF_HOME=~/.cache/huggingface CUDA_VISIBLE_DEVICES="" \
        pytest tests/test_certify_atlas_gpt2.py -q
"""

from __future__ import annotations

import pytest

from emltorch.certify.extract import extract_head_logprob_scores
from emltorch.certify.atlas import certified_radius

# A sequence that repeats so the induction head has a prior occurrence to copy.
INDUCTION_PROMPT = (
    "vase comet lunar drift ember vase comet lunar drift ember vase comet lunar"
)


@pytest.fixture(scope="module")
def gpt2():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "gpt2", attn_implementation="eager", dtype=torch.float32
    ).eval()
    return model, tok


def test_l5h5_is_an_induction_head(gpt2):
    model, tok = gpt2
    hs = extract_head_logprob_scores(model, tok, INDUCTION_PROMPT, layer=5, head=5)
    # It attends to the token that followed the previous occurrence (induction).
    assert hs.tokens[hs.argmax_idx] == " drift"
    assert 0.45 < hs.target_prob < 0.85  # genuinely but not perfectly concentrated


def test_l5h5_certifies_concentration_below_its_margin(gpt2):
    model, tok = gpt2
    hs = extract_head_logprob_scores(model, tok, INDUCTION_PROMPT, layer=5, head=5)
    # tau well below the head's ~0.63 mass: a positive dual-UNSAT radius exists.
    cr = certified_radius(hs.scores, hs.argmax_idx, tau=0.5)
    assert cr.radius > 0.0


def test_l5h5_soundly_refuses_to_certify_above_its_margin(gpt2):
    model, tok = gpt2
    hs = extract_head_logprob_scores(model, tok, INDUCTION_PROMPT, layer=5, head=5)
    # tau ABOVE the head's actual concentration (~0.63): must NOT certify.
    # This is the non-vacuity guarantee the v3 form failed.
    cr = certified_radius(hs.scores, hs.argmax_idx, tau=0.9)
    assert cr.radius == 0.0


def test_diffuse_head_has_zero_certified_radius_at_high_tau(gpt2):
    model, tok = gpt2
    hs = extract_head_logprob_scores(model, tok, INDUCTION_PROMPT, layer=0, head=0)
    # A diffuse early head (~0.12 mass) certifies nothing at tau=0.5.
    cr = certified_radius(hs.scores, hs.argmax_idx, tau=0.5)
    assert cr.radius == 0.0
