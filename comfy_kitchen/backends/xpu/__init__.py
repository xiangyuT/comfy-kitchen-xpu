# SPDX-License-Identifier: Apache-2.0
"""Optional SYCL kernels on the calling PyTorch XPU stream."""
from functools import lru_cache

import torch

from comfy_kitchen.constraints import ExactDims, FunctionConstraints, ParamConstraint, ValidationResult
from comfy_kitchen.registry import registry

__all__ = ["dequantize_int8_convrot_weight", "dequantize_int8_convrot_weight_dtype"]

try:
    from . import _C
except (ImportError, OSError) as exc:
    _C = None
    registry.mark_unavailable("xpu", f"SYCL extension unavailable: {exc}")


@lru_cache
def _supported_device(device):
    stream = torch.xpu.current_stream(device)
    queue = getattr(stream, "sycl_queue", None)
    return queue is not None and _C.supports_device(queue)


def _convrot_rule(kwargs):
    q, scale = kwargs["q"], kwargs["scale"]
    if kwargs.get("group_size") != 256 or q.shape[1] == 0 or q.shape[1] % 256:
        return ValidationResult.fail("group_size", "requires group 256 dividing a nonempty row")
    if not q.is_contiguous() or not scale.is_contiguous() or scale.device != q.device:
        return ValidationResult.fail("scale", "requires contiguous weights/scales on one XPU")
    if not (scale.numel() == 1 or (scale.ndim == 2 and scale.shape == (q.shape[0], 1))):
        return ValidationResult.fail("scale", "requires scalar or [N,1] scale")
    if scale.requires_grad:
        return ValidationResult.fail("scale", "native dequantization is inference-only")
    if kwargs.get("output_dtype_code", 0) not in (0, 1, 2):
        return ValidationResult.fail("output_dtype_code", "requires FP32, FP16 or BF16")
    if not _supported_device(q.device):
        return ValidationResult.fail("q", "requires a SYCL device with 16-lane subgroups")
    return ValidationResult.ok()


def dequantize_int8_convrot_weight_dtype(q, scale, group_size, output_dtype_code):
    output = torch.empty_like(q, dtype=(torch.float32, torch.float16, torch.bfloat16)[output_dtype_code])
    if q.numel():
        queue = torch.xpu.current_stream(q.device).sycl_queue
        _C.dequantize_convrot(q, scale, output, queue)
    return output


def dequantize_int8_convrot_weight(q, scale, group_size):
    return dequantize_int8_convrot_weight_dtype(q, scale, group_size, 0)


if _C is not None:
    constraints = FunctionConstraints(
        params={
            "q": ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)),
            "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
            "group_size": ParamConstraint(dtypes=frozenset({int})),
            "output_dtype_code": ParamConstraint(dtypes=frozenset({int})),
        },
        default_devices=frozenset({"xpu"}),
        call_rules=(_convrot_rule,),
    )
    registry.register("xpu", __import__(__name__, fromlist=__all__),
        {name: constraints for name in __all__})
