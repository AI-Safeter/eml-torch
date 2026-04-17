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


def safe_eml(left: Tensor, right: Tensor, clamp_val: float = 80.0,
             log_eps: float = 1e-6) -> Tensor:
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
            left.real.clamp(-clamp_val, clamp_val),
            left.imag,
        )
    else:
        left_safe = left.clamp(-clamp_val, clamp_val)

    # Clamp log argument magnitude away from zero
    if right.is_complex():
        mag = right.abs().clamp(min=log_eps)
        phase = right / (right.abs() + 1e-30)  # unit phasor, avoid div-by-zero
        right_safe = mag * phase
    else:
        right_safe = right.clamp(min=log_eps)

    return torch.exp(left_safe) - torch.log(right_safe)
