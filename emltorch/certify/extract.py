"""Attention-score extraction from any HuggingFace causal LM.

For the honest, shift-invariant ``softmax_interval`` cert form, log-prob scores
(from ``output_attentions``) are sufficient -- softmax concentration is
shift-invariant, so we do NOT need raw pre-softmax logits for that form. (Raw
logits are only required for the EML-in-body ``v3`` artifact, which is
vacuity-prone on log-probs; see concentration.py.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HeadScores:
    scores: np.ndarray  # log(attn_prob) row at the query position, length = query_pos+1
    argmax_idx: int  # key the head attends to most (natural cert target)
    target_prob: (
        float  # softmax mass on argmax_idx (concentration the cert must respect)
    )
    tokens: list[str]
    layer: int
    head: int


def extract_head_logprob_scores(
    model,
    tokenizer,
    prompt: str,
    layer: int,
    head: int,
    device: str = "cpu",
    query_pos: int = -1,
) -> HeadScores:
    """Run one forward pass and return the (log-prob) attention score row for a
    single (layer, head) at the query position.

    Only causal keys (indices 0..query_pos) are returned, so the row is a valid
    distribution to certify concentration over.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    T = int(input_ids.shape[1])
    qpos = query_pos if query_pos >= 0 else T + query_pos

    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True)

    # out.attentions[layer]: (batch, n_heads, T_q, T_k) post-softmax probabilities
    probs_row = out.attentions[layer][0, head, qpos, : qpos + 1].float().cpu().numpy()
    probs_row = np.clip(probs_row, 1e-30, None)
    scores = np.log(probs_row)
    argmax_idx = int(np.argmax(probs_row))
    tokens = [tokenizer.decode([t]) for t in input_ids[0, : qpos + 1].tolist()]
    return HeadScores(
        scores=scores,
        argmax_idx=argmax_idx,
        target_prob=float(probs_row[argmax_idx] / probs_row.sum()),
        tokens=tokens,
        layer=layer,
        head=head,
    )


def _head_scores_from_probs(
    probs_row: np.ndarray, tokens: list[str], layer: int, head: int
) -> HeadScores:
    """Build a HeadScores row from a single causal attention-probability row."""
    probs_row = np.clip(probs_row.astype(np.float64), 1e-30, None)
    scores = np.log(probs_row)
    argmax_idx = int(np.argmax(probs_row))
    return HeadScores(
        scores=scores,
        argmax_idx=argmax_idx,
        target_prob=float(probs_row[argmax_idx] / probs_row.sum()),
        tokens=tokens,
        layer=layer,
        head=head,
    )


def extract_all_heads_logprob_scores(
    model,
    tokenizer,
    prompt: str,
    device: str = "cpu",
    query_pos: int = -1,
) -> list[HeadScores]:
    """Run the model forward ONCE with ``output_attentions=True`` and return a
    HeadScores row for every (layer, head) at the query position.

    This is the efficient sweep primitive: one forward pass produces all
    ``n_layers * n_heads`` rows, instead of re-running the model per head.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    T = int(input_ids.shape[1])
    qpos = query_pos if query_pos >= 0 else T + query_pos

    with torch.no_grad():
        out = model(input_ids=input_ids, output_attentions=True)

    tokens = [tokenizer.decode([t]) for t in input_ids[0, : qpos + 1].tolist()]
    results: list[HeadScores] = []
    for layer, attn in enumerate(out.attentions):
        # attn: (batch, n_heads, T_q, T_k)
        n_heads = int(attn.shape[1])
        rows = attn[0, :, qpos, : qpos + 1].float().cpu().numpy()  # (n_heads, qpos+1)
        for head in range(n_heads):
            results.append(_head_scores_from_probs(rows[head], tokens, layer, head))
    return results
