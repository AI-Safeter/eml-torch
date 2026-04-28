"""Soundness tests for the attention-block Lipschitz primitive.

Adoption of arxiv:2507.07814 (Yudin et al. 2025).  These tests gate the
downstream Headline-15 work on Qwen3-8B: a transcription bug here would
silently corrupt every multi-layer cert that consumes the bound.

Test coverage:
  T1  softmax_jacobian_g1 ≤ 1/2 always (Corollary 1 tightness)
  T2  g_1 ≈ 1/2 at the (1/2, 1/2, 0, ...) extremal point
  T3  g_1 ≈ 0 at peaked one-hot distributions (induction-search heads)
  T4  Theorem 3 bound ≥ torch-autograd ‖J_Attn‖_2 on a tiny T=4 block
       across 5 random seeds (the load-bearing soundness check)
  T5  attention_block_lipschitz_interval is monotone non-decreasing in
       delta_l2 and matches clean bound at delta_l2 = 0 (within slack)
  T6  emit_attention_lipschitz_smt2_block produces parseable SMT-LIB2
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emltorch.smt import (  # noqa: E402
    softmax_jacobian_g1,
    softmax_jacobian_g1_max,
    attention_block_lipschitz_clean,
    attention_block_lipschitz_interval,
    emit_attention_lipschitz_smt2_block,
)


# ─── T1, T2, T3: softmax-Jacobian Corollary 1 sanity ──────────────────────


def test_g1_le_half_random():
    """Corollary 1: g_1(p) ≤ 1/2 for any probability vector."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        T = int(rng.integers(2, 32))
        z = rng.standard_normal(T)
        p = np.exp(z - z.max())
        p = p / p.sum()
        g1 = softmax_jacobian_g1(p)
        assert 0.0 <= g1 <= 0.5 + 1e-12, f"g_1 = {g1} violates Corollary 1"


def test_g1_extremal_half_half():
    """g_1 = 0.5 - 0.25 = 0.25 at p = (0.5, 0.5)? Check the formula form."""
    # g_1(p) = p_(1) * (1 - p_(1) + p_(2))  with p_(1)=p_(2)=0.5
    # = 0.5 * (1 - 0.5 + 0.5) = 0.5 * 1 = 0.5
    p = np.array([0.5, 0.5])
    g1 = softmax_jacobian_g1(p)
    assert (
        abs(g1 - 0.5) < 1e-12
    ), f"g_1((0.5,0.5)) = {g1}, expected 0.5 (Corollary 1 extremum)"


def test_g1_peaked_distribution():
    """g_1 → 0 as attention becomes peaked (one position dominates).
    Induction-search heads in Qwen3-8B are highly peaked → g_1 ≈ 0
    → attention-block bound becomes tight."""
    p = np.array([0.99, 0.005, 0.003, 0.002])
    g1 = softmax_jacobian_g1(p)
    # g_1 = 0.99 * (1 - 0.99 + 0.005) = 0.99 * 0.015 ≈ 0.01485
    assert g1 < 0.02, f"g_1 = {g1} should be near 0 for peaked distribution"


def test_g1_max_picks_worst_row():
    """softmax_jacobian_g1_max returns max over rows."""
    P = np.array(
        [
            [0.5, 0.5, 0.0],  # row-g1 = 0.5
            [0.99, 0.005, 0.005],  # row-g1 ≈ 0.0149
        ]
    )
    g1_max = softmax_jacobian_g1_max(P)
    assert abs(g1_max - 0.5) < 1e-12


# ─── T4: Theorem 3 soundness check vs torch autograd ‖J‖_2 ───────────────


def _toy_attention_jacobian_torch(W_Q, W_K, W_V, X, d_head):
    """Compute the spectral norm of the per-head attention Jacobian via
    torch.autograd.functional.jacobian on a tiny (T, d_model) input.

    Returns (‖J‖_2, P, X_t) for cross-checking with our analytic bound.

    Single-head, no out-projection, no LN: matches Theorem 3 scope.
    """
    X_t = torch.as_tensor(X, dtype=torch.float64)
    W_Q_t = torch.as_tensor(W_Q, dtype=torch.float64)
    W_K_t = torch.as_tensor(W_K, dtype=torch.float64)
    W_V_t = torch.as_tensor(W_V, dtype=torch.float64)
    inv_sqrt_d = 1.0 / math.sqrt(d_head)

    def attn_fn(x_flat):
        T, d_model = X_t.shape
        x = x_flat.reshape(T, d_model)
        q = x @ W_Q_t  # (T, d_head)
        k = x @ W_K_t  # (T, d_head)
        v = x @ W_V_t  # (T, d_head)
        scores = q @ k.T * inv_sqrt_d  # (T, T)
        p = torch.softmax(scores, dim=-1)
        out = p @ v  # (T, d_head)
        return out.reshape(-1)

    x_flat = X_t.reshape(-1).clone().requires_grad_(True)
    J = torch.autograd.functional.jacobian(attn_fn, x_flat, create_graph=False)
    # J has shape (T*d_head, T*d_model)
    sigma_max = float(torch.linalg.svdvals(J)[0].item())

    with torch.no_grad():
        T, d_model = X_t.shape
        q = X_t @ W_Q_t
        k = X_t @ W_K_t
        scores = q @ k.T * inv_sqrt_d
        P = np.array(torch.softmax(scores, dim=-1).detach().cpu().tolist())
    return sigma_max, P


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_theorem3_bound_dominates_autograd_jacobian(seed):
    """Theorem 3 ‖J_Attn‖_2 bound ≥ torch.autograd ‖J‖_2 — implementation
    soundness check.  This is the load-bearing test for the primitive.

    Tiny T=4, d_model=8, d_head=4 attention block; 5 random seeds.
    Theorem 3's bound must dominate the autograd-computed spectral norm
    for every seed.  A transcription error (e.g. forgetting the 2× factor,
    swapping ‖X‖² for ‖X‖) would fail this with high probability.
    """
    rng = np.random.default_rng(seed)
    T = 4
    d_model = 8
    d_head = 4
    # Small weight scale so attention isn't pathologically peaked.
    W_Q = rng.standard_normal((d_model, d_head)) * 0.3
    W_K = rng.standard_normal((d_model, d_head)) * 0.3
    W_V = rng.standard_normal((d_model, d_head)) * 0.3
    X = rng.standard_normal((T, d_model)) * 0.5

    sigma_max, P = _toy_attention_jacobian_torch(W_Q, W_K, W_V, X, d_head)
    bound = attention_block_lipschitz_clean(P, W_Q, W_K, W_V, X, d_head)

    L_th3 = bound["L"]
    assert L_th3 >= sigma_max - 1e-9, (
        f"seed={seed}: Theorem-3 bound L={L_th3:.4f} < autograd ‖J‖_2={sigma_max:.4f}; "
        f"transcription bug or unsoundness.  Components: {bound['components']}"
    )
    # Also report tightness: bound / sigma_max should be modest (not 100×)
    ratio = L_th3 / max(sigma_max, 1e-9)
    assert (
        1.0 <= ratio < 100.0
    ), f"seed={seed}: bound is {ratio:.2f}× looser than autograd — sanity check"


# ─── T5: interval bound monotonicity + clean-limit ────────────────────────


def test_interval_bound_monotone_in_delta():
    """attention_block_lipschitz_interval is non-decreasing in delta_l2."""
    rng = np.random.default_rng(7)
    T, d_model, d_head = 4, 8, 4
    W_Q = rng.standard_normal((d_model, d_head)) * 0.3
    W_K = rng.standard_normal((d_model, d_head)) * 0.3
    W_V = rng.standard_normal((d_model, d_head)) * 0.3
    X = rng.standard_normal((T, d_model)) * 0.5
    P = np.exp(np.random.default_rng(0).standard_normal((T, T)))
    P = P / P.sum(axis=-1, keepdims=True)

    deltas = [0.0, 0.01, 0.05, 0.1, 0.5]
    L_uppers = []
    for delta in deltas:
        out = attention_block_lipschitz_interval(P, W_Q, W_K, W_V, X, d_head, delta)
        L_uppers.append(out["L_upper"])
    for i in range(len(L_uppers) - 1):
        assert L_uppers[i] <= L_uppers[i + 1] + 1e-9, (
            f"non-monotone at delta={deltas[i]} → {deltas[i+1]}: "
            f"{L_uppers[i]:.4f} > {L_uppers[i+1]:.4f}"
        )


def test_interval_bound_dominates_clean_at_zero_delta():
    """At delta_l2 = 0, interval bound ≥ clean-evaluated bound: the interval
    version uses worst-case g1_max=0.5 and P_norm ceiling, so it can be
    strictly looser even at delta=0."""
    rng = np.random.default_rng(11)
    T, d_model, d_head = 4, 8, 4
    W_Q = rng.standard_normal((d_model, d_head)) * 0.3
    W_K = rng.standard_normal((d_model, d_head)) * 0.3
    W_V = rng.standard_normal((d_model, d_head)) * 0.3
    X = rng.standard_normal((T, d_model)) * 0.5
    P = np.exp(rng.standard_normal((T, T)))
    P = P / P.sum(axis=-1, keepdims=True)

    clean = attention_block_lipschitz_clean(P, W_Q, W_K, W_V, X, d_head)
    intv = attention_block_lipschitz_interval(P, W_Q, W_K, W_V, X, d_head, 0.0)
    assert intv["L_upper"] >= clean["L"] - 1e-9


# ─── T6: SMT block emission ───────────────────────────────────────────────


def test_emit_attention_lipschitz_smt2_block_format():
    """Emitted block is non-empty, contains required declarations."""
    block = emit_attention_lipschitz_smt2_block(
        name="attn_L7H4",
        L_upper=2.5,
        delta_l2_upper=0.1,
    )
    assert "(declare-const attn_L7H4_perturb_norm Real)" in block
    assert "Theorem 3" in block
    assert "2507.07814" in block
    # bound = 2.5 * 0.1 = 0.25
    assert "0.25" in block
