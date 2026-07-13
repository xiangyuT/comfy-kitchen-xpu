"""Kitchen-compatible arbitrary-matrix RoPE adapters for Intel XPU."""

import torch
from omni_xpu_kernel import rotary


def apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    return rotary.apply_kitchen_rope1(x, freqs_cis)


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return rotary.apply_kitchen_rope(xq, xk, freqs_cis)


def apply_rope_split_half1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    return rotary.apply_kitchen_rope_split_half1(x, freqs_cis)


def apply_rope_split_half(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return rotary.apply_kitchen_rope_split_half(xq, xk, freqs_cis)


__all__ = ["apply_rope", "apply_rope1", "apply_rope_split_half", "apply_rope_split_half1"]
