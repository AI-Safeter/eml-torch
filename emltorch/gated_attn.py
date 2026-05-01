"""Effective-weight extraction for Gated DeltaNet linear-attention layers.

Derived from torch_recurrent_gated_delta_rule (transformers/models/
qwen3_5/modeling_qwen3_5.py:339-350). The recurrence

    S_t = g_t * S_{t-1} + k_t (x) delta_t,    delta_t = beta_t (v_t - g_t S_{t-1} k_t)
    out_t = q_t^T S_t

unrolls to:

    out_t = sum_{j=1..t} a[t, j] * delta_j,
    a[t, j] = (prod_{i=j+1..t} g_i) * <q_t, k_j>

with q, k L2-normalized in-kernel (use_qk_l2norm_in_kernel=True).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def extract_gated_effective_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    log_g: torch.Tensor,
    beta: torch.Tensor,
    num_v_heads: int,
) -> torch.Tensor:
    """Compute the closed-form effective weight a[b, h_v, t, j].

    Args:
        query: (B, T, H_k, D_k)   - pre-l2norm query
        key:   (B, T, H_k, D_k)   - pre-l2norm key
        log_g: (B, T, H_v)        - log of decay scalar (g = exp(log_g) in (0,1])
        beta:  (B, T, H_v)        - kept for API symmetry (used in delta-rule
                                    correction; not needed for the kernel weight)
        num_v_heads: int          - H_v >= H_k (GQA broadcast)

    Returns:
        a: (B, H_v, T, T) lower-triangular signed weights in [-1, 1].
    """
    del beta  # unused in the simple kernel weight path
    B, T, H_k, D_k = query.shape
    H_v = num_v_heads
    assert H_v % H_k == 0, f"H_v={H_v} must be a multiple of H_k={H_k}"

    q_n = F.normalize(query, dim=-1, eps=1e-6)
    k_n = F.normalize(key, dim=-1, eps=1e-6)
    # Match the in-kernel scale = 1 / sqrt(D_k) applied to query AFTER l2norm
    # (modeling_qwen3_5.py:263, 328). Without this, |a| is sqrt(D_k)x too large.
    q_n = q_n * (D_k**-0.5)
    qk_inner = torch.einsum("bthd,bjhd->bhtj", q_n, k_n)  # (B, H_k, T, T)
    if H_v != H_k:
        qk_inner = qk_inner.repeat_interleave(H_v // H_k, dim=1)  # (B, H_v, T, T)

    log_g_cum = log_g.cumsum(dim=1)  # (B, T, H_v)
    # gamma[b, t, j, h] = exp(log_g_cum[b, t, h] - log_g_cum[b, j, h])
    log_gamma = log_g_cum[:, :, None, :] - log_g_cum[:, None, :, :]  # (B, T, T, H_v)
    log_gamma = log_gamma.permute(0, 3, 1, 2)  # (B, H_v, T, T)
    gamma = log_gamma.exp().tril()

    a = gamma * qk_inner
    return a.tril()  # j <= t only (lower triangular)


def compute_delta_rule_deltas(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_g: torch.Tensor,
    beta: torch.Tensor,
    num_v_heads: int,
) -> torch.Tensor:
    """Compute per-position delta_j from the explicit DeltaNet recurrence.

        S_t = g_t * S_{t-1} + k_t (x) delta_t,   delta_t = beta_t (v_t - g_t S_{t-1} k_t)

    Q, K are L2-normalized (matching kernel use_qk_l2norm_in_kernel=True);
    Q is additionally scaled by 1/sqrt(D_k) (matching kernel q_scale).

    Args:
        query: (B, T, H_k, D_k)
        key:   (B, T, H_k, D_k)
        value: (B, T, H_v, D_v)
        log_g: (B, T, H_v)
        beta:  (B, T, H_v)
        num_v_heads: int

    Returns:
        delta: (B, T, H_v, D_v) per-position effective write
    """
    B, T, H_k, D_k = query.shape
    H_v = num_v_heads
    D_v = value.shape[-1]
    assert H_v % H_k == 0
    q_n = F.normalize(query, dim=-1, eps=1e-6) * (D_k**-0.5)
    k_n = F.normalize(key, dim=-1, eps=1e-6)
    if H_v != H_k:
        q_n = q_n.repeat_interleave(H_v // H_k, dim=2)
        k_n = k_n.repeat_interleave(H_v // H_k, dim=2)
    g = log_g.exp()  # (B, T, H_v)

    S = torch.zeros(B, H_v, D_k, D_v, dtype=value.dtype, device=value.device)
    deltas = torch.zeros(B, T, H_v, D_v, dtype=value.dtype, device=value.device)
    for t in range(T):
        k_t = k_n[:, t]  # (B, H_v, D_k)
        v_t = value[:, t]  # (B, H_v, D_v)
        g_t = g[:, t, :, None, None]  # (B, H_v, 1, 1)
        beta_t = beta[:, t, :, None]  # (B, H_v, 1)
        S = S * g_t
        kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)  # (B, H_v, D_v)
        delta = (v_t - kv_mem) * beta_t
        deltas[:, t] = delta
        S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    return deltas


def extract_gated_contribution_log_magnitudes(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_g: torch.Tensor,
    beta: torch.Tensor,
    num_v_heads: int,
    eps: float = 1e-30,
) -> torch.Tensor:
    """Delta-corrected per-(t, j) contribution log-magnitude:

        log_w[t, j] = log( |a[t, j]| * ||delta_j|| + eps )

    This is the right input to the cert filter's eligibility / cert
    template for Gated DeltaNet linear attention because the simple
    kernel weight |a[t, j]| alone underestimates the contribution
    (validated: rel_err(B vs A) ~ 7.98 on real Qwen3.6-27B L0,
    forcing the delta-corrected promotion rule).

    Returns:
        log_w: (B, H_v, T, T)  lower-triangular; j > t entries == log(eps).
    """
    a = extract_gated_effective_weights(query, key, log_g, beta, num_v_heads)
    deltas = compute_delta_rule_deltas(query, key, value, log_g, beta, num_v_heads)
    # delta_norm[b, j, h] = ||delta[b, j, h, :]||
    delta_norm = deltas.norm(dim=-1)  # (B, T, H_v)
    delta_norm = delta_norm.permute(0, 2, 1)  # (B, H_v, T)  (positional axis = T at j)
    # Broadcast over t: contribution[b, h, t, j] = |a[b, h, t, j]| * delta_norm[b, h, j]
    contrib = a.abs() * delta_norm.unsqueeze(-2)  # (B, H_v, T, T)
    return torch.log(contrib.clamp_min(eps))
