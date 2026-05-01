"""Unit tests for emltorch.gated_attn.extract_gated_effective_weights.

Validates the per-(t, j) effective-weight formula derived from
torch_recurrent_gated_delta_rule (transformers/models/qwen3_5/
modeling_qwen3_5.py:339-350):

    a[t, j] = (prod_{i=j+1..t} g_i) * <q_t, k_j>

with q, k L2-normalized in-kernel and g_i = exp(log_g_i) in (0, 1].
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from emltorch.gated_attn import extract_gated_effective_weights


def test_cumsum_decay_correctness():
    """gamma[t, j] should equal exp(sum_{i=j+1..t} log_g_i)."""
    torch.manual_seed(0)
    B, T, H_k, H_v, D_k, D_v = 1, 5, 1, 1, 4, 4
    q = torch.randn(B, T, H_k, D_k)
    k = torch.randn(B, T, H_k, D_k)
    log_g = torch.full((B, T, H_v), -0.1)  # constant decay; gamma[t,j]=exp(-0.1*(t-j))
    beta = torch.full((B, T, H_v), 0.5)

    a = extract_gated_effective_weights(q, k, log_g, beta, num_v_heads=H_v)

    q_n = F.normalize(q, dim=-1, eps=1e-6)
    k_n = F.normalize(k, dim=-1, eps=1e-6)
    for t in range(T):
        for j in range(t + 1):
            gamma_ref = math.exp(-0.1 * (t - j))
            qk_ref = float((q_n[0, t, 0] * k_n[0, j, 0]).sum())
            a_ref = gamma_ref * qk_ref
            assert (
                abs(float(a[0, 0, t, j]) - a_ref) < 1e-5
            ), f"a[{t}, {j}]={float(a[0, 0, t, j])}  ref={a_ref}"
