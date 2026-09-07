"""Numerically stabilized exp(left) - log(right)."""

import torch
from torch import Tensor


def safe_eml(left: Tensor, right: Tensor, clamp_val: float = 80.0, log_eps: float = 1e-6) -> Tensor:
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
        Float16 computations use float32 intermediates and saturate to the
        finite float16 range on return.
    """
    if left.dtype == torch.float16 or right.dtype == torch.float16:
        output_dtype = torch.promote_types(left.dtype, right.dtype)
        working_dtype = torch.promote_types(output_dtype, torch.float32)
        result = safe_eml(left.to(working_dtype), right.to(working_dtype), clamp_val, log_eps)
        if output_dtype == torch.float16:
            limit = torch.finfo(output_dtype).max
            result = result.clamp(-limit, limit)
        return result.to(output_dtype)

    # Clamp exp argument
    if left.is_complex():
        left_safe = torch.complex(
            torch.nan_to_num(left.real, nan=0.0, posinf=clamp_val, neginf=-clamp_val).clamp(
                -clamp_val, clamp_val
            ),
            torch.nan_to_num(left.imag, nan=0.0, posinf=clamp_val, neginf=-clamp_val),
        )
    else:
        left_safe = torch.nan_to_num(left, nan=0.0, posinf=clamp_val, neginf=-clamp_val).clamp(
            -clamp_val, clamp_val
        )

    # Clamp log argument magnitude away from zero and protect upper bound
    if right.is_complex():
        right_num = torch.complex(
            torch.nan_to_num(right.real, nan=1.0, posinf=1e30, neginf=-1e30),
            torch.nan_to_num(right.imag, nan=0.0, posinf=1e30, neginf=-1e30),
        )
        mag = right_num.abs().clamp(min=log_eps, max=1e30)
        phase = torch.where(
            right_num.abs() > 0,
            right_num / right_num.abs().clamp(min=1e-30),
            torch.ones_like(right_num),
        )
        right_safe = mag * phase
    else:
        right_safe = torch.nan_to_num(right, nan=1.0, posinf=1e30, neginf=-1e30).clamp(
            min=log_eps, max=1e30
        )

    out = torch.exp(left_safe) - torch.log(right_safe)
    if out.is_complex():
        return torch.complex(
            torch.nan_to_num(out.real, nan=0.0, posinf=1e30, neginf=-1e30),
            torch.nan_to_num(out.imag, nan=0.0, posinf=1e30, neginf=-1e30),
        )
    return torch.nan_to_num(out, nan=0.0, posinf=1e30, neginf=-1e30)
