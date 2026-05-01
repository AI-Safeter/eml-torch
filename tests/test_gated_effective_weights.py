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


def test_inner_product_in_unit_interval():
    """L2-normalized Q, K give |<q, k>| <= 1; with log_g=0, gamma=1
    so |a[t, j]| <= 1."""
    torch.manual_seed(1)
    B, T, H, D = 2, 8, 4, 16
    q = torch.randn(B, T, H, D) * 5.0
    k = torch.randn(B, T, H, D) * 5.0
    log_g = torch.zeros(B, T, H)
    beta = torch.zeros(B, T, H)

    a = extract_gated_effective_weights(q, k, log_g, beta, num_v_heads=H)
    diag = a.diagonal(dim1=-2, dim2=-1)
    assert diag.abs().max() <= 1.0 + 1e-6
    assert a.abs().max() <= 1.0 + 1e-6


def test_gqa_broadcast():
    """H_v=8, H_k=2 means each K head feeds 4 V heads."""
    torch.manual_seed(2)
    B, T, H_k, H_v, D = 1, 6, 2, 8, 8
    q = torch.randn(B, T, H_k, D)
    k = torch.randn(B, T, H_k, D)
    log_g = torch.zeros(B, T, H_v)
    beta = torch.zeros(B, T, H_v)

    a = extract_gated_effective_weights(q, k, log_g, beta, num_v_heads=H_v)
    assert a.shape == (B, H_v, T, T)
    q_n = F.normalize(q, dim=-1, eps=1e-6)
    k_n = F.normalize(k, dim=-1, eps=1e-6)
    for hv in range(H_v):
        h_k_src = hv * H_k // H_v
        qk_ref = torch.einsum(
            "btd,bjd->btj", q_n[:, :, h_k_src], k_n[:, :, h_k_src]
        ).tril()
        assert torch.allclose(
            a[0, hv], qk_ref[0], atol=1e-5
        ), f"GQA head {hv} (k-src {h_k_src}) mismatch"


def test_lower_triangular_only():
    """j > t entries must be zero (causal)."""
    torch.manual_seed(3)
    q = torch.randn(1, 5, 1, 4)
    k = torch.randn(1, 5, 1, 4)
    log_g = torch.randn(1, 5, 1)
    beta = torch.zeros(1, 5, 1)
    a = extract_gated_effective_weights(q, k, log_g, beta, num_v_heads=1)
    for t in range(5):
        for j in range(t + 1, 5):
            assert (
                float(a[0, 0, t, j]) == 0.0
            ), f"a[{t}, {j}] = {float(a[0, 0, t, j])} (should be 0)"
