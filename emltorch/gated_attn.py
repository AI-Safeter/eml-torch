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
