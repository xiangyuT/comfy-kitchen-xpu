"""Adaptive LayerNorm adapter using omni's native XPU LayerNorm."""

import torch
from omni_xpu_kernel import norm

from comfy_kitchen.backends.eager.adaln import adaln as eager_adaln


def adaln(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize with ESIMD when supported, preserving Kitchen broadcasting."""
    hidden = x.shape[-1]
    if hidden % 32 or hidden > 8192 or hidden == 0:
        return eager_adaln(x, scale, shift, eps)

    shape = x.shape
    x_2d = x.reshape(-1, hidden).contiguous()
    mapping = _modulation_mapping(x, scale, shift)
    native = norm._get_native()
    if mapping is not None and hasattr(native, "fused_adaln"):
        scale_2d, shift_2d, row_repeat = mapping
        return norm.fused_adaln(x_2d, scale_2d, shift_2d, row_repeat, eps).reshape(shape)
    normalized = norm.layer_norm(x_2d, None, None, eps).reshape(shape)
    return (normalized * (1 + scale) + shift).contiguous()


def _modulation_mapping(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """Map common broadcast layouts to native flattened modulation rows."""
    if scale.shape != shift.shape or scale.dtype != x.dtype or shift.dtype != x.dtype:
        return None
    if scale.device != x.device or shift.device != x.device or scale.shape[-1] != x.shape[-1]:
        return None
    if scale.dim() > x.dim():
        return None
    padded = (1,) * (x.dim() - scale.dim()) + tuple(scale.shape)
    prefix = padded[:-1]
    x_prefix = tuple(x.shape[:-1])
    rows = x.numel() // x.shape[-1]
    if all(dim == 1 for dim in prefix):
        repeat = rows
    elif prefix == x_prefix:
        repeat = 1
    elif prefix[-1] == 1 and prefix[:-1] == x_prefix[:-1]:
        repeat = x_prefix[-1]
    else:
        return None
    return scale.reshape(-1, x.shape[-1]).contiguous(), shift.reshape(
        -1, x.shape[-1]
    ).contiguous(), repeat


__all__ = ["adaln"]
