"""Intel XPU backend powered by the optional omni_xpu_kernel package."""

from __future__ import annotations

import sys

import torch

from comfy_kitchen.constraints import ExactDims, FunctionConstraints, ParamConstraint
from comfy_kitchen.registry import registry

__all__ = [
    "adaln",
    "apply_rope",
    "apply_rope1",
    "apply_rope_split_half",
    "apply_rope_split_half1",
    "convrot_w4a4_linear",
    "dequantize_convrot_w4a4_weight",
    "dequantize_gguf",
    "dequantize_per_tensor_fp8",
    "dequantize_int8_convrot_weight",
    "dequantize_int8_convrot_weight_dtype",
    "dequantize_int8_simple",
    "dequantize_int8_simple_dtype",
    "int8_linear",
    "mm_int8",
    "prepare_int4_weight_for_int8_linear",
    "quantize_and_rotate_rowwise",
    "quantize_int8_convrot_weight",
    "quantize_int8_rowwise",
    "quantize_int8_tensorwise",
    "quantize_convrot_w4a4_weight",
    "quantize_per_tensor_fp8",
    "quantize_svdquant_w4a4",
    "scaled_mm_svdquant_w4a4",
    "svdquant_w4a16_linear",
    "stochastic_rounding_fp8",
]

_AVAILABLE = False
_ERROR = None
_NATIVE_CAPABILITIES = frozenset()
_INT8_AVAILABLE = False
_INT8_ERROR = None
_SVDQ_AVAILABLE = False
_SVDQ_W4A16_AVAILABLE = False
_NORM_AVAILABLE = False
_FP8_AVAILABLE = False
_FP8_QDQ_AVAILABLE = False
_ROPE_AVAILABLE = False
_CONVROT_NATIVE_AVAILABLE = False
_GGUF_AVAILABLE = False

_REQUIRED_NATIVE_INT8_OPS = frozenset(
    {
        "dequantize_int8_simple",
        "dequantize_int8_simple_dtype",
        "int8_linear",
        "mm_int8",
        "quantize_int8_rowwise",
        "quantize_int8_tensorwise",
    }
)

try:
    import omni_xpu_kernel
    from omni_xpu_kernel import int8 as _int8

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        _ERROR = "PyTorch XPU is not available on this system"
    elif not omni_xpu_kernel.is_available():
        _ERROR = "omni_xpu_kernel native extension is not available"
    else:
        _extension = omni_xpu_kernel._load_extension()
        _AVAILABLE = True
        _native_int8 = getattr(_extension, "int8", None)
        _NATIVE_CAPABILITIES = frozenset(
            name
            for name in _REQUIRED_NATIVE_INT8_OPS
            if _native_int8 is not None and hasattr(_native_int8, name)
        )
        missing = _REQUIRED_NATIVE_INT8_OPS - _NATIVE_CAPABILITIES
        if missing:
            _INT8_ERROR = "omni_xpu_kernel INT8 extension is missing: " + ", ".join(
                sorted(missing)
            )
        else:
            _INT8_AVAILABLE = True

        _native_svdq = getattr(_extension, "svdq", None)
        _SVDQ_AVAILABLE = _native_svdq is not None and all(
            hasattr(_native_svdq, name)
            for name in (
                "dequantize_svdq_w4",
                "quantize_svdq_act_int4",
                "onednn_int4_gemm",
            )
        )
        _SVDQ_W4A16_AVAILABLE = _native_svdq is not None and all(
            hasattr(_native_svdq, name)
            for name in (
                "fused_smooth_mul_convert",
                "onednn_int4_gemm_add_to_output",
                "onednn_int4_gemm_preconverted",
            )
        )
        _native_norm = getattr(_extension, "norm", None)
        _NORM_AVAILABLE = _native_norm is not None and hasattr(
            _native_norm, "layer_norm"
        )
        _native_linear = getattr(_extension, "linear", None)
        _FP8_AVAILABLE = _native_linear is not None and hasattr(
            _native_linear, "onednn_w8a16_fp8"
        )
        _native_fp8 = getattr(_extension, "fp8", None)
        _FP8_QDQ_AVAILABLE = _native_fp8 is not None and all(
            hasattr(_native_fp8, name)
            for name in (
                "dequantize_per_tensor",
                "quantize_per_tensor",
                "stochastic_rounding",
            )
        )
        _native_rotary = getattr(_extension, "rotary", None)
        _ROPE_AVAILABLE = _native_rotary is not None and all(
            hasattr(_native_rotary, name)
            for name in (
                "apply_kitchen_rope",
                "apply_kitchen_rope1",
                "apply_kitchen_rope_split_half",
                "apply_kitchen_rope_split_half1",
            )
        )
        _CONVROT_NATIVE_AVAILABLE = _native_int8 is not None and all(
            hasattr(_native_int8, name)
            for name in (
                "dequantize_int8_convrot_weight",
                "quantize_int8_convrot_weight",
                "rotate_convrot",
            )
        )
        _native_gguf = getattr(_extension, "gguf", None)
        _GGUF_AVAILABLE = _native_gguf is not None and all(
            hasattr(_native_gguf, name)
            for name in (
                "dequantize_q4_0",
                "dequantize_q4_1",
                "dequantize_q8_0",
                "dequantize_q4_k",
                "dequantize_q6_k",
            )
        )
except (ImportError, OSError, RuntimeError) as exc:
    _ERROR = f"{type(exc).__name__}: {exc}"


if _AVAILABLE:
    if _INT8_AVAILABLE:
        quantize_int8_tensorwise = _int8.quantize_int8_tensorwise
        quantize_int8_rowwise = _int8.quantize_int8_rowwise
        dequantize_int8_simple = _int8.dequantize_int8_simple
        int8_linear = _int8.int8_linear
        mm_int8 = _int8.mm_int8
        quantize_int8_convrot_weight = _int8.quantize_int8_convrot_weight
        dequantize_int8_convrot_weight = _int8.dequantize_int8_convrot_weight

    if _NORM_AVAILABLE:
        from .adaln import adaln

    if _SVDQ_AVAILABLE:
        from .svdquant import quantize_svdquant_w4a4, scaled_mm_svdquant_w4a4

    if _SVDQ_W4A16_AVAILABLE:
        from .svdquant_w4a16 import svdquant_w4a16_linear

    if _FP8_QDQ_AVAILABLE:
        from .fp8 import (
            dequantize_per_tensor_fp8,
            quantize_per_tensor_fp8,
            stochastic_rounding_fp8,
        )

    if _ROPE_AVAILABLE:
        from .rope import apply_rope, apply_rope1, apply_rope_split_half, apply_rope_split_half1

    if _CONVROT_NATIVE_AVAILABLE and _SVDQ_AVAILABLE:
        from .convrot_w4a4 import (
            convrot_w4a4_linear,
            dequantize_convrot_w4a4_weight,
            prepare_int4_weight_for_int8_linear,
            quantize_and_rotate_rowwise,
            quantize_convrot_w4a4_weight,
        )

    if _GGUF_AVAILABLE:
        from .gguf import dequantize_gguf

    if _INT8_AVAILABLE:

        def dequantize_int8_simple_dtype(
            q: torch.Tensor,
            scale: torch.Tensor,
            output_dtype_code: int,
        ) -> torch.Tensor:
            """Adapt Kitchen's dtype-code ABI to omni's torch.dtype API."""
            out_dtype = _CODE_TO_DTYPE[output_dtype_code]
            return _int8.dequantize_int8_simple_dtype(q, scale, out_dtype)

        def dequantize_int8_convrot_weight_dtype(
            q: torch.Tensor,
            scale: torch.Tensor,
            group_size: int,
            output_dtype_code: int,
        ) -> torch.Tensor:
            """Dequantize ConvRot weights and convert to the requested Kitchen dtype."""
            return _int8.dequantize_int8_convrot_weight(
                q, scale, group_size
            ).to(_CODE_TO_DTYPE[output_dtype_code])


_CODE_TO_DTYPE = {
    0: torch.float32,
    1: torch.float16,
    2: torch.bfloat16,
}


def _build_constraints() -> dict[str, FunctionConstraints]:
    xpu = frozenset({"xpu"})
    floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    int8_2d = ParamConstraint(dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),))

    capabilities = {
        "quantize_int8_tensorwise": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=xpu,
        ),
        "quantize_int8_rowwise": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats),
                "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=xpu,
        ),
        "dequantize_int8_simple": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=frozenset({torch.int8})),
                "scale": ParamConstraint(dtypes=floats),
            },
            default_devices=xpu,
        ),
        "dequantize_int8_simple_dtype": FunctionConstraints(
            params={
                "q": ParamConstraint(dtypes=frozenset({torch.int8})),
                "scale": ParamConstraint(dtypes=floats),
                "output_dtype_code": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=xpu,
        ),
        "int8_linear": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=frozenset({torch.float16, torch.bfloat16})),
                "weight": int8_2d,
                "weight_scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "bias": ParamConstraint(dtypes=floats),
                "out_dtype": ParamConstraint(dtypes=floats),
                "convrot": ParamConstraint(dtypes=frozenset({bool})),
                "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=xpu,
        ),
        "mm_int8": FunctionConstraints(
            params={"a": int8_2d, "b": int8_2d},
            default_devices=xpu,
        ),
        "quantize_int8_convrot_weight": FunctionConstraints(
            params={
                "weight": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=xpu,
        ),
        "dequantize_int8_convrot_weight": FunctionConstraints(
            params={
                "q": int8_2d,
                "scale": ParamConstraint(dtypes=floats),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=xpu,
        ),
        "dequantize_int8_convrot_weight_dtype": FunctionConstraints(
            params={
                "q": int8_2d,
                "scale": ParamConstraint(dtypes=floats),
                "group_size": ParamConstraint(dtypes=frozenset({int})),
                "output_dtype_code": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=xpu,
        ),
    }
    if not _INT8_AVAILABLE:
        capabilities.clear()
    if _SVDQ_AVAILABLE:
        capabilities.update(
            {
                "quantize_svdquant_w4a4": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                        "smooth": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(1),)),
                        "lora_down": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                        "pad_size": ParamConstraint(dtypes=frozenset({int})),
                        "act_unsigned": ParamConstraint(dtypes=frozenset({bool})),
                        "lora_x": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                    },
                    default_devices=xpu,
                ),
                "scaled_mm_svdquant_w4a4": FunctionConstraints(
                    params={
                        "act": ParamConstraint(
                            dtypes=frozenset({torch.int8, torch.uint8}), shape_rules=(ExactDims(2),)
                        ),
                        "wgt": ParamConstraint(dtypes=frozenset({torch.int8, torch.uint8})),
                        "ascales": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                        "wscales": ParamConstraint(dtypes=floats),
                        "lora_act_in": ParamConstraint(
                            dtypes=frozenset({torch.float32}), shape_rules=(ExactDims(2),)
                        ),
                        "lora_up": ParamConstraint(dtypes=floats),
                        "bias": ParamConstraint(dtypes=floats),
                        "act_unsigned": ParamConstraint(dtypes=frozenset({bool})),
                    },
                    default_devices=xpu,
                ),
            }
        )
    if _SVDQ_W4A16_AVAILABLE:
        capabilities["svdquant_w4a16_linear"] = FunctionConstraints(
            params={
                "x": ParamConstraint(
                    dtypes=frozenset({torch.bfloat16}),
                    shape_rules=(ExactDims(2),),
                ),
                "packed_u4": ParamConstraint(
                    dtypes=frozenset({torch.uint8}),
                    shape_rules=(ExactDims(2),),
                ),
                "scales_f16": ParamConstraint(
                    dtypes=frozenset({torch.float16}),
                    shape_rules=(ExactDims(2),),
                ),
                "rcp_smooth_f16": ParamConstraint(
                    dtypes=frozenset({torch.float16}),
                    shape_rules=(ExactDims(1),),
                ),
                "lora_down": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(2),),
                ),
                "lora_up": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(2),),
                ),
                "bias": ParamConstraint(
                    dtypes=frozenset({torch.float16, torch.bfloat16}),
                    shape_rules=(ExactDims(1),),
                ),
                "output_dtype_code": ParamConstraint(
                    dtypes=frozenset({int}),
                ),
            },
            default_devices=xpu,
        )
    if _NORM_AVAILABLE:
        capabilities["adaln"] = FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats),
                "scale": ParamConstraint(dtypes=floats),
                "shift": ParamConstraint(dtypes=floats),
            },
            default_devices=xpu,
        )
    if _FP8_QDQ_AVAILABLE:
        fp8_dtypes = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})
        capabilities.update(
            {
                "quantize_per_tensor_fp8": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=floats),
                        "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                        "output_type": ParamConstraint(dtypes=fp8_dtypes),
                    },
                    default_devices=xpu,
                ),
                "dequantize_per_tensor_fp8": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=fp8_dtypes),
                        "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                        "output_type": ParamConstraint(dtypes=floats),
                    },
                    default_devices=xpu,
                ),
                "stochastic_rounding_fp8": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=floats),
                        "rng": ParamConstraint(dtypes=frozenset({torch.uint8})),
                        "output_type": ParamConstraint(dtypes=fp8_dtypes),
                    },
                    default_devices=xpu,
                ),
            }
        )
    if _GGUF_AVAILABLE:
        capabilities["dequantize_gguf"] = FunctionConstraints(
            params={
                "data": ParamConstraint(dtypes=frozenset({torch.uint8})),
                "quant_type_code": ParamConstraint(dtypes=frozenset({int})),
                "output_dtype_code": ParamConstraint(dtypes=frozenset({int})),
                "layout_code": ParamConstraint(dtypes=frozenset({int})),
            },
            default_devices=xpu,
        )
    if _ROPE_AVAILABLE:
        rope_input = ParamConstraint(dtypes=floats, shape_rules=(ExactDims(4),))
        rope_freqs = ParamConstraint(dtypes=floats, shape_rules=(ExactDims(6),))
        capabilities.update(
            {
                "apply_rope1": FunctionConstraints(
                    params={"x": rope_input, "freqs_cis": rope_freqs},
                    default_devices=xpu,
                ),
                "apply_rope": FunctionConstraints(
                    params={"xq": rope_input, "xk": rope_input, "freqs_cis": rope_freqs},
                    default_devices=xpu,
                ),
                "apply_rope_split_half1": FunctionConstraints(
                    params={"x": rope_input, "freqs_cis": rope_freqs},
                    default_devices=xpu,
                ),
                "apply_rope_split_half": FunctionConstraints(
                    params={"xq": rope_input, "xk": rope_input, "freqs_cis": rope_freqs},
                    default_devices=xpu,
                ),
            }
        )
    if _CONVROT_NATIVE_AVAILABLE and _SVDQ_AVAILABLE:
        capabilities.update(
            {
                "quantize_convrot_w4a4_weight": FunctionConstraints(
                    params={
                        "weight": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(2),)),
                        "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                        "quant_group_size": ParamConstraint(dtypes=frozenset({int})),
                        "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
                    },
                    default_devices=xpu,
                ),
                "dequantize_convrot_w4a4_weight": FunctionConstraints(
                    params={
                        "qdata": ParamConstraint(
                            dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                        ),
                        "scales": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(1),)),
                        "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                        "quant_group_size": ParamConstraint(dtypes=frozenset({int})),
                        "output_dtype": ParamConstraint(dtypes=floats),
                    },
                    default_devices=xpu,
                ),
                "convrot_w4a4_linear": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=floats),
                        "qweight": ParamConstraint(
                            dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                        ),
                        "wscales": ParamConstraint(dtypes=floats, shape_rules=(ExactDims(1),)),
                        "bias": ParamConstraint(dtypes=floats),
                        "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
                        "quant_group_size": ParamConstraint(dtypes=frozenset({int})),
                        "linear_dtype": ParamConstraint(dtypes=frozenset({str})),
                    },
                    default_devices=xpu,
                ),
                "prepare_int4_weight_for_int8_linear": FunctionConstraints(
                    params={
                        "weight": ParamConstraint(
                            dtypes=frozenset({torch.int8}), shape_rules=(ExactDims(2),)
                        )
                    },
                    default_devices=xpu,
                ),
                "quantize_and_rotate_rowwise": FunctionConstraints(
                    params={
                        "x": ParamConstraint(dtypes=floats),
                        "H": ParamConstraint(dtypes=floats),
                        "group_size": ParamConstraint(dtypes=frozenset({int})),
                        "stochastic_rounding": ParamConstraint(dtypes=frozenset({int})),
                    },
                    default_devices=xpu,
                ),
            }
        )
    return capabilities


def _register() -> None:
    if not _AVAILABLE:
        registry.mark_unavailable("xpu", _ERROR or "omni_xpu_kernel is not available")
        return
    capabilities = _build_constraints()
    if not capabilities:
        registry.mark_unavailable(
            "xpu",
            _INT8_ERROR or "omni_xpu_kernel exposes no supported Kitchen capabilities",
        )
        return
    registry.register(
        name="xpu",
        module=sys.modules[__name__],
        capabilities=capabilities,
    )


_register()
