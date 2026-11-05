"""CPU-only XPU routing checks; native calls are stubs, not numerical tests."""

from types import SimpleNamespace

import pytest
import torch

from comfy_kitchen.backends import xpu
from comfy_kitchen.backends.xpu import w4a8_int8 as backend


@pytest.fixture
def dispatch(monkeypatch):
    calls = []
    result = torch.empty(0)

    def packed(*args, **kwargs):
        calls.append("packed")
        return result

    def legacy(*args, **kwargs):
        calls.append("legacy")
        return result

    native = SimpleNamespace(
        supports_w4a8_linear_prequantized=lambda: True,
        w4a8_linear_prequantized=packed,
        int8_linear_prequantized=legacy,
    )

    def activation(value, *args):
        calls.append("activation")
        return value[..., :256]

    def decode(*args):
        calls.append("dequantize_weight")
        return torch.zeros(7, 256, dtype=torch.int8)

    def rotation(*args, **kwargs):
        calls.append("rotation")
        return torch.empty(0)

    def quantize(value, *args):
        calls.append("quantize")
        return value.to(torch.int8), torch.ones(*value.shape[:-1], 1)

    monkeypatch.setattr(xpu, "_int8", native)
    monkeypatch.setattr(xpu, "_GATED_QUANTIZE_AVAILABLE", False)
    monkeypatch.setattr(backend, "apply_input_act", activation)
    monkeypatch.setattr(backend, "_dequant_int4_grouped_to_int8", decode)
    monkeypatch.setattr(backend, "_build_hadamard", rotation)
    monkeypatch.setattr(backend, "quantize_and_rotate_rowwise", quantize)
    return SimpleNamespace(native=native, calls=calls, result=result)


def operands(shape=(1, 256), bits=4, scale_dtype=torch.float32):
    return {
        "x": torch.zeros(shape, dtype=torch.bfloat16),
        "qdata": torch.zeros(7, 256 * bits // 8, dtype=torch.int8),
        "s_rel": torch.ones(7, 16).to(scale_dtype),
        "s_channel": torch.ones(7),
    }


@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float8_e4m3fn])
@pytest.mark.parametrize("operand", ["qdata", "s_rel"])
def test_invalid_decode_layout_raises_without_preparation(dispatch, rows, scale_dtype, operand):
    args = operands((rows, 256), scale_dtype=scale_dtype)
    original = args[operand]
    args[operand] = original.t().contiguous().t()
    assert args[operand].shape == original.shape
    assert not args[operand].is_contiguous()
    assert args["s_rel" if operand == "qdata" else "qdata"].is_contiguous()
    with pytest.raises(ValueError, match="contiguous " + operand):
        backend.w4a8_int8_linear(**args)
    # No activation, quantization, full weight expansion or either GEMM.
    assert dispatch.calls == []


@pytest.mark.parametrize("missing", ["query", "capability", "entrypoint"])
def test_decode_requires_packed_companion(dispatch, missing):
    if missing == "query":
        del dispatch.native.supports_w4a8_linear_prequantized
    elif missing == "capability":
        dispatch.native.supports_w4a8_linear_prequantized = lambda: False
    else:
        del dispatch.native.w4a8_linear_prequantized
    with pytest.raises(RuntimeError, match="matching omni_xpu_kernel"):
        backend.w4a8_int8_linear(**operands())
    assert dispatch.calls == []


def test_decode_rejects_unsupported_group_scale_dtype(dispatch):
    with pytest.raises(TypeError, match="FP32 or FP8 E4M3 s_rel"):
        backend.w4a8_int8_linear(**operands(scale_dtype=torch.float16))
    assert dispatch.calls == []


@pytest.mark.parametrize("shape,activation", [
    ((1, 256), None), ((8, 256), None), ((2, 3, 256), "rms_norm"),
    ((1, 512), "swiglu"), ((2, 4, 512), "situ"),
])
def test_decode_counts_leading_rows_before_activation(dispatch, shape, activation):
    actual = backend.w4a8_int8_linear(**operands(shape), input_act=activation)
    assert actual is dispatch.result
    assert dispatch.calls == ["activation", "rotation", "quantize", "packed"]


@pytest.mark.parametrize("shape,bits", [((9, 256), 4), ((1, 256), 6), ((0, 256), 4)])
def test_other_workloads_keep_their_route_without_packed_capability(dispatch, shape, bits):
    def forbidden_query():
        raise AssertionError("non-decode workload queried packed capability")

    dispatch.native.supports_w4a8_linear_prequantized = forbidden_query
    del dispatch.native.w4a8_linear_prequantized
    actual = backend.w4a8_int8_linear(**operands(shape, bits))
    assert actual is dispatch.result
    assert dispatch.calls == ["activation", "dequantize_weight", "rotation", "quantize", "legacy"]
