"""Decode Kitchen's grouped 4/6-bit storage for the native XPU INT8 linear."""

from __future__ import annotations

import torch

from comfy_kitchen.allocation import allocation_context
from comfy_kitchen.backends.eager.quantization import quantize_and_rotate_rowwise
from comfy_kitchen.backends.eager.w4a8_int8 import (
    _dequant_int4_grouped_to_int8,
    validate_w4a8_operands,
)
from comfy_kitchen.constraints import ValidationResult
from comfy_kitchen.tensor.int8_utils import _build_hadamard


def native_call_rule(kwargs):
    if kwargs.get("correction") is not None:
        return ValidationResult.fail("correction", "asymmetric W4A8 uses the eager route")
    return ValidationResult.ok()


def w4a8_int8_linear(
    x, qdata, s_rel, s_channel, codebook=None, correction=None, bias=None,
    group_size=16, convrot_groupsize=256, out_dtype=torch.bfloat16,
):
    from . import _int8

    _, k, _ = validate_w4a8_operands(
        qdata, s_rel, s_channel, codebook, correction, group_size, convrot_groupsize,
    )
    if x.shape[-1] != k:
        raise ValueError(f"Input K={x.shape[-1]} does not match qdata K={k}")
    if correction is not None:
        raise ValueError("asymmetric W4A8 correction requires the eager route")
    weight = _dequant_int4_grouped_to_int8(qdata, s_rel, codebook, group_size)
    with allocation_context():
        h = _build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
    quantized, scale = quantize_and_rotate_rowwise(x, h, convrot_groupsize)
    return _int8.int8_linear_prequantized(
        quantized, scale, weight, s_channel.float(), bias=bias, out_dtype=out_dtype,
    )
