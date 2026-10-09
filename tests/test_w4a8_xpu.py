"""Grouped 4/6-bit storage through native XPU linear and QuantizedTensor."""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.eager import w4a8_int8 as eager
from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor


pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not ck.list_backends()["xpu"]["available"],
    reason="native XPU backend required",
)


@pytest.mark.parametrize("bits", [4, 6])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_grouped_storage_native_linear(bits, dtype, with_bias, monkeypatch):
    from comfy_kitchen.backends import xpu

    torch.manual_seed(31)
    weight = torch.randn(64, 256, device="xpu", dtype=dtype) * 0.02
    x = torch.randn(2, 3, 256, device="xpu", dtype=dtype)
    bias = torch.randn(64, device="xpu", dtype=dtype) if with_bias else None
    q, scale, channel, correction, codebook = eager.quantize_w4a8_int8_weight(
        weight, bits=bits, scale_dtype=torch.float32, scale_search=False,
    )
    calls = []
    original = xpu._native_int8.int8_linear_prequantized

    def native(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(xpu._native_int8, "int8_linear_prequantized", native)
    with ck.use_backend("xpu"):
        actual = ck.w4a8_int8_linear(
            x, q, scale, channel, codebook=codebook, bias=bias, out_dtype=dtype,
        )
    reference = eager.w4a8_int8_linear(
        x, q, scale, channel, codebook=codebook, bias=bias, out_dtype=dtype,
    )
    assert calls == [True]
    assert actual.shape == (2, 3, 64) and actual.dtype == dtype
    assert actual.device == x.device
    torch.testing.assert_close(actual, reference, rtol=0.025, atol=0.005)


@pytest.mark.parametrize("bits", [4, 6])
def test_grouped_quantized_tensor_serialization_and_linear(bits):
    torch.manual_seed(37)
    weight = torch.randn(64, 256, device="xpu", dtype=torch.bfloat16) * 0.02
    x = torch.randn(3, 256, device="xpu", dtype=torch.bfloat16)
    tensor = QuantizedTensor.from_float(
        weight, "AsymW4A8Int8Layout", bits=bits, scale_search=False,
        scale_dtype=torch.float32,
    )
    packed, params = tensor._qdata, tensor._params
    assert packed.shape == (64, 256 * bits // 8)
    stored = AsymW4A8Int8Layout.state_dict_tensors(packed, params)
    assert torch.equal(stored[""], packed)
    with ck.use_backend("xpu"):
        actual = torch.nn.functional.linear(x, tensor)
    reference = eager.w4a8_int8_linear(
        x, packed, params.scale, params.s_channel, codebook=params.codebook,
        out_dtype=x.dtype,
    )
    torch.testing.assert_close(actual, reference, rtol=0.025, atol=0.005)


def test_asymmetric_correction_uses_eager_backend():
    torch.manual_seed(41)
    weight = torch.randn(64, 256, device="xpu", dtype=torch.float16)
    x = torch.randn(3, 256, device="xpu", dtype=torch.float16)
    q, scale, channel, correction, _ = eager.quantize_w4a8_int8_weight(
        weight, symmetric=False, codebook=False, scale_dtype=torch.float32,
    )
    kwargs = dict(x=x, qdata=q, s_rel=scale, s_channel=channel,
                  correction=correction, codebook=None, bias=None,
                  group_size=16, convrot_groupsize=256, out_dtype=x.dtype)
    backend = ck.registry.get_capable_backend("w4a8_int8_linear", kwargs=kwargs)
    assert backend == "eager"
    actual = ck.w4a8_int8_linear(**kwargs)
    reference = eager.w4a8_int8_linear(**kwargs)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("bits", [4, 6])
def test_grouped_linear_fullgraph_capture(bits):
    weight = torch.randn(64, 256, device="xpu", dtype=torch.bfloat16) * 0.02
    x = torch.randn(3, 256, device="xpu", dtype=torch.bfloat16)
    q, scale, channel, _, codebook = eager.quantize_w4a8_int8_weight(
        weight, bits=bits, scale_dtype=torch.float32, scale_search=False,
    )

    def run(inp):
        return ck.w4a8_int8_linear(inp, q, scale, channel, codebook=codebook)

    with ck.use_backend("xpu"):
        expected = run(x)
        actual = torch.compile(run, backend="eager", fullgraph=True)(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_rotation_cache_uses_kitchen_allocation_context(monkeypatch):
    from comfy_kitchen import allocation
    from comfy_kitchen.backends.xpu import w4a8_int8 as backend

    entered = []

    class Context:
        active = False

        def __enter__(self):
            self.active = True

        def __exit__(self, *args):
            self.active = False

    context = Context()
    original = backend._build_hadamard

    def build(*args, **kwargs):
        entered.append(context.active)
        return original(*args, **kwargs)

    monkeypatch.setattr(backend, "_build_hadamard", build)
    weight = torch.randn(64, 256, device="xpu", dtype=torch.float16)
    x = torch.randn(3, 256, device="xpu", dtype=torch.float16)
    q, scale, channel, _, codebook = eager.quantize_w4a8_int8_weight(
        weight, scale_dtype=torch.float32,
    )
    previous = allocation.allocation_context()
    try:
        ck.set_allocation_context(context)
        with ck.use_backend("xpu"):
            ck.w4a8_int8_linear(x, q, scale, channel, codebook=codebook)
    finally:
        ck.set_allocation_context(previous)
    assert entered == [True] and not context.active
