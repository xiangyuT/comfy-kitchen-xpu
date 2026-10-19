"""Check the XPU adapter's new API using an independent integer GEMM stub.

These CPU tests establish Python-side preparation/epilogue semantics only.
Device numeric and native dispatch checks live in test_w4a8_xpu.py.
"""

from types import SimpleNamespace

import pytest
import torch

from comfy_kitchen.backends import xpu
from comfy_kitchen.backends.eager import w4a8_int8 as eager
from comfy_kitchen.backends.xpu import w4a8_int8 as adapter
from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor
from comfy_kitchen.tensor.w4a8_packed import (
    default_w4a8_codebook,
    unpack_w4a8_mma_weight,
    w4a8_mma_record_bytes,
)


@pytest.mark.parametrize("bits", [4, 6])
@pytest.mark.parametrize("activation", [None, "swiglu", "situ", "rms_norm"])
@pytest.mark.parametrize("with_residual", [False, True])
def test_new_linear_keywords_keep_native_gemm(bits, activation, with_residual, monkeypatch):
    calls = []

    def integer_gemm(qx, sx, qw, sw, bias, out_dtype):
        calls.append((qx.dtype, qw.dtype))
        shape = (*qx.shape[:-1], qw.shape[0])
        result = qx.reshape(-1, qx.shape[-1]).int() @ qw.int().t()
        out = (result.float() * (sx.reshape(-1, 1) * sw.reshape(1, -1))).to(out_dtype)
        if bias is not None:
            out = out + bias.to(out_dtype)
        return out.reshape(shape)

    monkeypatch.setattr(xpu, "_int8", SimpleNamespace(int8_linear_prequantized=integer_gemm), raising=False)
    torch.manual_seed(131)
    weight = torch.randn(32, 256, dtype=torch.bfloat16) * 0.02
    q, scale, channel, correction, codebook = eager.quantize_w4a8_int8_weight(
        weight, bits=bits, scale_dtype=torch.float32, scale_search=False,
    )
    assert correction is None
    raw_x = torch.randn(2, 3, 512 if activation in ("swiglu", "situ") else 256, dtype=weight.dtype)
    original = raw_x.clone()
    norm = torch.rand(256, dtype=weight.dtype)
    bias = torch.randn(32, dtype=weight.dtype)
    residual = torch.randn(2, 3, 32, dtype=weight.dtype) if with_residual else None
    residual_scale = torch.linspace(-1, 1, 32, dtype=weight.dtype) if with_residual else None
    kwargs = dict(codebook=codebook, bias=bias, out_dtype=weight.dtype,
                  input_act=activation, input_act_weight=norm, input_act_eps=1e-6,
                  residual=residual, residual_scale=residual_scale)
    actual = adapter.w4a8_int8_linear(raw_x, q, scale, channel, **kwargs)
    expected = eager.w4a8_int8_linear(raw_x, q, scale, channel, **kwargs)
    assert calls == [(torch.int8, torch.int8)]
    assert torch.equal(raw_x, original)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bits", [4, 6])
def test_packed_cuda_records_normalize_for_cpu(bits):
    n, k = 16, 256
    packed = (torch.arange(n * k * w4a8_mma_record_bytes(bits) // 512) % 127).to(torch.int8)
    params = AsymW4A8Int8Layout.Params(
        scale=torch.empty(0, dtype=torch.float8_e4m3fn),
        orig_dtype=torch.bfloat16, orig_shape=(n, k), s_channel=torch.ones(n),
        codebook=default_w4a8_codebook() if bits == 4 else None, mma_packed=True,
    )
    tensor = QuantizedTensor(packed, "AsymW4A8Int8Layout", params)
    expected_q, expected_scale = unpack_w4a8_mma_weight(packed, n, k)
    qdata, moved_params = AsymW4A8Int8Layout.normalize_device_storage(packed, params, torch.device("cpu"))
    moved = QuantizedTensor(qdata, "AsymW4A8Int8Layout", moved_params)
    assert not moved._params.mma_packed
    assert tensor._params.mma_packed
    assert torch.equal(moved._qdata, expected_q)
    assert torch.equal(moved._params.scale.view(torch.uint8), expected_scale.view(torch.uint8))
    assert moved._params.orig_shape == tensor._params.orig_shape
    torch.testing.assert_close(moved.dequantize(), tensor.dequantize(), rtol=0, atol=0)
