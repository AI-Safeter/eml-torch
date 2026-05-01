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
    scale = D_k**-0.5  # in-kernel q_scale (modeling_qwen3_5.py:263)
    for t in range(T):
        for j in range(t + 1):
            gamma_ref = math.exp(-0.1 * (t - j))
            qk_ref = float((q_n[0, t, 0] * k_n[0, j, 0]).sum()) * scale
            a_ref = gamma_ref * qk_ref
            assert (
                abs(float(a[0, 0, t, j]) - a_ref) < 1e-5
            ), f"a[{t}, {j}]={float(a[0, 0, t, j])}  ref={a_ref}"


def test_inner_product_in_unit_interval():
    """L2-normalized Q, K give |<q, k>| <= 1; with log_g=0, gamma=1.
    With in-kernel scale 1/sqrt(D), |a[t, j]| <= 1/sqrt(D)."""
    torch.manual_seed(1)
    B, T, H, D = 2, 8, 4, 16
    q = torch.randn(B, T, H, D) * 5.0
    k = torch.randn(B, T, H, D) * 5.0
    log_g = torch.zeros(B, T, H)
    beta = torch.zeros(B, T, H)

    a = extract_gated_effective_weights(q, k, log_g, beta, num_v_heads=H)
    bound = (D**-0.5) + 1e-6
    diag = a.diagonal(dim1=-2, dim2=-1)
    assert diag.abs().max() <= bound, f"diag max {float(diag.abs().max())} > {bound}"
    assert a.abs().max() <= bound, f"a max {float(a.abs().max())} > {bound}"


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
    scale = D**-0.5
    for hv in range(H_v):
        h_k_src = hv * H_k // H_v
        qk_ref = (
            torch.einsum("btd,bjd->btj", q_n[:, :, h_k_src], k_n[:, :, h_k_src]).tril()
            * scale
        )
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


def test_single_key_reconstruction_against_recurrence():
    """For a small synthetic step, verify the unrolled formula
    out_t = sum_{j} a[t, j] * delta_j matches the explicit recurrence
    out_t = q_t^T S_t computed step-by-step.
    """
    torch.manual_seed(4)
    B, T, H_k, H_v, D_k, D_v = 1, 6, 1, 1, 4, 4
    q = torch.randn(B, T, H_k, D_k)
    k = torch.randn(B, T, H_k, D_k)
    v = torch.randn(B, T, H_v, D_v)
    log_g = torch.full((B, T, H_v), -0.05)
    beta = torch.full((B, T, H_v), 0.7)

    q_n = F.normalize(q, dim=-1, eps=1e-6) * (D_k**-0.5)  # in-kernel scale
    k_n = F.normalize(k, dim=-1, eps=1e-6)
    g = log_g.exp()

    S = torch.zeros(B, H_v, D_k, D_v)
    out_ref = torch.zeros(B, T, H_v, D_v)
    deltas = []
    for t in range(T):
        q_t = q_n[:, t]
        k_t = k_n[:, t]
        v_t = v[:, t]
        g_t = g[:, t, :, None, None]
        beta_t = beta[:, t, :, None]

        S = S * g_t
        kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        deltas.append(delta)
        S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out_ref[:, t] = (S * q_t.unsqueeze(-1)).sum(dim=-2)

    a = extract_gated_effective_weights(q, k, log_g, beta, num_v_heads=H_v)
    deltas_stack = torch.stack(deltas, dim=1)  # (B, T, H_v, D_v)
    out_unrolled = torch.einsum("bhtj,bjhd->bthd", a, deltas_stack)

    diff = float((out_unrolled - out_ref).abs().max())
    assert torch.allclose(out_unrolled, out_ref, atol=1e-4), f"max |diff| = {diff}"


def test_public_api_export():
    """extract_gated_effective_weights is exported at the top level."""
    import emltorch

    assert hasattr(emltorch, "extract_gated_effective_weights")


def test_delta_rule_recurrence_helper():
    """compute_delta_rule_deltas should match the same recurrence used
    in test_single_key_reconstruction_against_recurrence."""
    torch.manual_seed(11)
    B, T, H_k, H_v, D_k, D_v = 1, 7, 1, 1, 4, 4
    q = torch.randn(B, T, H_k, D_k)
    k = torch.randn(B, T, H_k, D_k)
    v = torch.randn(B, T, H_v, D_v)
    log_g = torch.full((B, T, H_v), -0.05)
    beta = torch.full((B, T, H_v), 0.7)

    from emltorch.gated_attn import compute_delta_rule_deltas

    deltas_helper = compute_delta_rule_deltas(q, k, v, log_g, beta, num_v_heads=H_v)
    a = extract_gated_effective_weights(q, k, log_g, beta, num_v_heads=H_v)
    out_unrolled = torch.einsum("bhtj,bjhd->bthd", a, deltas_helper)

    # Reproduce ground-truth recurrence in-line
    q_n = F.normalize(q, dim=-1, eps=1e-6) * (D_k ** -0.5)
    k_n = F.normalize(k, dim=-1, eps=1e-6)
    g = log_g.exp()
    S = torch.zeros(B, H_v, D_k, D_v)
    out_ref = torch.zeros(B, T, H_v, D_v)
    for t in range(T):
        q_t = q_n[:, t]
        k_t = k_n[:, t]
        v_t = v[:, t]
        g_t = g[:, t, :, None, None]
        beta_t = beta[:, t, :, None]
        S = S * g_t
        kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out_ref[:, t] = (S * q_t.unsqueeze(-1)).sum(dim=-2)

    assert torch.allclose(out_unrolled, out_ref, atol=1e-4)


def test_contribution_log_magnitudes_dominant_target():
    """If we deliberately set position j=4 to have large |a| and large
    ||delta||, contribution_log_magnitudes should pick j=4 at the last_q
    argmax for some head."""
    from emltorch.gated_attn import extract_gated_contribution_log_magnitudes

    torch.manual_seed(12)
    B, T, H_k, H_v, D = 1, 8, 1, 2, 4
    q = torch.randn(B, T, H_k, D)
    k = torch.randn(B, T, H_k, D)
    v = torch.randn(B, T, H_v, D)
    log_g = torch.full((B, T, H_v), -0.05)
    beta = torch.full((B, T, H_v), 0.5)

    # Boost position 4: align q_last with k_4 AND v_4 with large magnitude
    q[0, T - 1] = k[0, 4] * 5.0
    v[0, 4] = torch.randn(H_v, D) * 5.0

    log_w = extract_gated_contribution_log_magnitudes(
        q, k, v, log_g, beta, num_v_heads=H_v
    )
    last_q = T - 1
    targets = log_w[0, :, last_q, :].argmax(dim=-1).tolist()
    assert 4 in targets, f"Expected at least one head argmax at j=4; got {targets}"
