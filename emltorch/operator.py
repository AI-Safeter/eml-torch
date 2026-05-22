"""
Core EML (Exp-Minus-Log) operator with numerical safety.

The EML operator eml(x, y) = exp(x) - ln(y) is a universal binary operator
that, combined with the constant 1, can represent ALL elementary functions.

Reference: "All elementary functions from a single binary operator"
           Andrzej Odrzywołek, arXiv:2603.21852 (March 2026)

Key constructions:
    exp(x) = eml(x, 1)                          depth 1
    e       = eml(1, 1)                          depth 1
    ln(z)   = eml(1, eml(eml(1, z), 1))         depth 3
    e - x   = eml(1, eml(x, 1))                 depth 2
"""

import torch
from torch import Tensor


def safe_eml(
    left: Tensor, right: Tensor, clamp_val: float = 80.0, log_eps: float = 1e-6
) -> Tensor:
    """
    Compute eml(x, y) = exp(x) - ln(y), numerically stabilized.

    Both exp overflow and log-of-zero are guarded:
    - exp argument clamped to [-clamp_val, clamp_val]
    - log argument magnitude clamped to [log_eps, +inf]

    Args:
        left:  Tensor of any shape (real or complex).
        right: Tensor of same shape (real or complex).
        clamp_val: Clamp bound for exp argument (default 80.0).
        log_eps: Minimum magnitude for log argument (default 1e-6).

    Returns:
        exp(clamp(left)) - log(clamp(right)), same shape and dtype.
    """
    # Clamp exp argument
    if left.is_complex():
        left_safe = torch.complex(
            torch.nan_to_num(left.real, nan=0.0, posinf=clamp_val, neginf=-clamp_val).clamp(-clamp_val, clamp_val),
            torch.nan_to_num(left.imag, nan=0.0, posinf=clamp_val, neginf=-clamp_val),
        )
    else:
        left_safe = torch.nan_to_num(left, nan=0.0, posinf=clamp_val, neginf=-clamp_val).clamp(-clamp_val, clamp_val)

    # Clamp log argument magnitude away from zero and protect upper bound
    if right.is_complex():
        right_num = torch.nan_to_num(right, nan=1.0, posinf=1e30, neginf=-1e30)
        mag = right_num.abs().clamp(min=log_eps, max=1e30)
        phase = right_num / (right_num.abs() + 1e-30)  # unit phasor, avoid div-by-zero
        right_safe = mag * phase
    else:
        right_safe = torch.nan_to_num(right, nan=1.0, posinf=1e30, neginf=-1e30).clamp(min=log_eps, max=1e30)

    out = torch.exp(left_safe) - torch.log(right_safe)
    return torch.nan_to_num(out, nan=0.0, posinf=1e30, neginf=-1e30)


def safe_eml_param(
    left: Tensor,
    right: Tensor,
    alpha: Tensor,
    beta: Tensor,
    clamp_val: float = 80.0,
    log_eps: float = 1e-6,
) -> Tensor:
    """
    Parameterized EML: eml_param(x, y; α, β) = α·exp(x) - β·ln(y).

    The H22h experiment showed that depth-4 standard EML on cross-model
    attention α coefficient saturates at HELDOUT R² 0.78, slightly below
    poly K=5's 0.82.  Adding per-node learnable α, β scaling factors
    gives the operator extra capacity without changing tree topology.

    With α=β=1, this reduces to safe_eml(x, y) — backward compatible.

    Args:
        left, right: same as safe_eml.
        alpha: scalar Tensor (broadcastable to left/right shape) — exp scale.
        beta:  scalar Tensor (broadcastable to left/right shape) — ln scale.
        clamp_val, log_eps: same as safe_eml.

    Returns:
        α · exp(clamp(left)) - β · log(clamp(right)).
    """
    if left.is_complex():
        left_safe = torch.complex(
            torch.nan_to_num(left.real, nan=0.0, posinf=clamp_val, neginf=-clamp_val).clamp(-clamp_val, clamp_val),
            torch.nan_to_num(left.imag, nan=0.0, posinf=clamp_val, neginf=-clamp_val),
        )
    else:
        left_safe = torch.nan_to_num(left, nan=0.0, posinf=clamp_val, neginf=-clamp_val).clamp(-clamp_val, clamp_val)

    if right.is_complex():
        right_num = torch.nan_to_num(right, nan=1.0, posinf=1e30, neginf=-1e30)
        mag = right_num.abs().clamp(min=log_eps, max=1e30)
        phase = right_num / (right_num.abs() + 1e-30)
        right_safe = mag * phase
    else:
        right_safe = torch.nan_to_num(right, nan=1.0, posinf=1e30, neginf=-1e30).clamp(min=log_eps, max=1e30)

    out = alpha * torch.exp(left_safe) - beta * torch.log(right_safe)
    return torch.nan_to_num(out, nan=0.0, posinf=1e30, neginf=-1e30)
