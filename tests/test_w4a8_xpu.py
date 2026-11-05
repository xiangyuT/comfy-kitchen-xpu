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
@pytest.mark.parametrize("rows", [1, 4, 8])
@pytest.mark.parametrize("activation", ["rms_norm", "swiglu"])
def test_quantized_linear_input_act_and_residual(bits, rows, activation, monkeypatch):
    from comfy_kitchen.backends import xpu

    torch.manual_seed(127)
    weight = QuantizedTensor.from_float(
        torch.randn(32, 256, device="xpu", dtype=torch.bfloat16) * 0.02,
        "AsymW4A8Int8Layout", bits=bits, scale_search=False, scale_dtype=torch.float32,
    )
    x = torch.randn(rows, 512 if activation == "swiglu" else 256, device="xpu", dtype=torch.bfloat16)
    norm = torch.rand(256, device="xpu", dtype=x.dtype)
    bias = torch.randn(32, device="xpu", dtype=x.dtype)
    residual = torch.randn(rows, 32, device="xpu", dtype=x.dtype)
    scale = torch.linspace(-1, 1, 32, device="xpu", dtype=x.dtype)
    activated = (
        torch.nn.functional.rms_norm(x, (256,), norm, 1e-6)
        if activation == "rms_norm"
        else torch.nn.functional.silu(x[:, :256]) * x[:, 256:]
    )
    # Compare API fusion with the same native GEMM composed explicitly. Eager's
    # bias rounding differs by a BF16 ULP; residual cancellation can amplify
    # that pre-existing difference. The adapter itself must be bit-identical.
    with ck.use_backend("xpu"):
        expected = torch.addcmul(residual, torch.nn.functional.linear(activated, weight, bias), scale)
    calls = []
    name = "w4a8_linear_prequantized" if bits == 4 else "int8_linear_prequantized"
    native = getattr(xpu._native_int8, name)

    def tracked(*args, **kwargs):
        calls.append(True)
        return native(*args, **kwargs)

    monkeypatch.setattr(xpu._native_int8, name, tracked)
    with ck.use_backend("xpu"):
        actual = weight.linear_input_act(x, bias, activation, norm, 1e-6, residual, scale)
    assert calls == [True]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bits", [4, 6])
def test_packed_records_move_to_canonical_xpu_storage(bits):
    from comfy_kitchen.tensor.w4a8_packed import default_w4a8_codebook, w4a8_mma_record_bytes

    n, k = 16, 256
    packed = (torch.arange(n * k * w4a8_mma_record_bytes(bits) // 512) % 127).to(torch.int8)
    params = AsymW4A8Int8Layout.Params(
        scale=torch.empty(0, dtype=torch.float8_e4m3fn), orig_dtype=torch.bfloat16,
        orig_shape=(n, k), s_channel=torch.ones(n), mma_packed=True,
        codebook=default_w4a8_codebook() if bits == 4 else None,
    )
    source = QuantizedTensor(packed, "AsymW4A8Int8Layout", params)
    moved = source.to("xpu")
    assert source._params.mma_packed and not moved._params.mma_packed
    assert moved._qdata.device.type == "xpu"
    assert moved._qdata.shape == (n, k * bits // 8)
    torch.testing.assert_close(moved.dequantize().cpu(), source.dequantize(), rtol=0, atol=0)


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
    name = "w4a8_linear_prequantized" if bits == 4 else "int8_linear_prequantized"
    original = getattr(xpu._native_int8, name)

    def native(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(xpu._native_int8, name, native)
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


@pytest.mark.parametrize("group_size", [4, 16, 64, 256])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float8_e4m3fn])
def test_packed_decode_avoids_full_weight_materialization(group_size, scale_dtype, monkeypatch):
    from comfy_kitchen.backends import xpu
    from comfy_kitchen.backends.xpu import w4a8_int8 as backend
    if not getattr(xpu._int8, "supports_w4a8_linear_prequantized", lambda: False)():
        pytest.skip("packed W4A8 companion core required")
    torch.manual_seed(811)
    x=torch.randn(1,512,device="xpu",dtype=torch.bfloat16)
    q,scales,channel,_,codebook=eager.quantize_w4a8_int8_weight(
        torch.randn(71,512,device="xpu",dtype=x.dtype)*.02,
        group_size=group_size,scale_dtype=scale_dtype,scale_search=False)
    reference=eager.w4a8_int8_linear(x,q,scales,channel,codebook=codebook,
                                   group_size=group_size,out_dtype=x.dtype)
    def forbidden(*args,**kwargs):
        raise AssertionError("packed decode materialized a complete INT8 weight")
    monkeypatch.setattr(backend,"_dequant_int4_grouped_to_int8",forbidden)
    with ck.use_backend("xpu"):
        actual=ck.w4a8_int8_linear(x,q,scales,channel,codebook=codebook,
                                  group_size=group_size,out_dtype=x.dtype)
    torch.testing.assert_close(actual,reference,rtol=.025,atol=.005)


@pytest.mark.parametrize("rows,bits", [(9,4),(1,6)])
def test_existing_routes_remain_available(rows,bits,monkeypatch):
    from comfy_kitchen.backends import xpu
    from comfy_kitchen.backends.xpu import w4a8_int8 as backend
    torch.manual_seed(813)
    x=torch.randn(rows,256,device="xpu",dtype=torch.bfloat16)
    q,scales,channel,_,codebook=eager.quantize_w4a8_int8_weight(
        torch.randn(49,256,device="xpu",dtype=x.dtype)*.02,
        bits=bits,scale_search=False,scale_dtype=torch.float32)
    decode=backend._dequant_int4_grouped_to_int8;calls=[]
    def observed(*args,**kwargs):
        calls.append(True);return decode(*args,**kwargs)
    monkeypatch.setattr(backend,"_dequant_int4_grouped_to_int8",observed)
    with ck.use_backend("xpu"):
        actual=ck.w4a8_int8_linear(x,q,scales,channel,codebook=codebook,out_dtype=x.dtype)
    assert calls==[True]
    reference=eager.w4a8_int8_linear(x,q,scales,channel,codebook=codebook,out_dtype=x.dtype)
    torch.testing.assert_close(actual,reference,rtol=.025,atol=.005)


@pytest.mark.parametrize("operand", ["qdata", "s_rel"])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float8_e4m3fn])
def test_decode_strided_storage_fails_before_preparation(operand, scale_dtype, monkeypatch):
    from comfy_kitchen.backends import xpu
    from comfy_kitchen.backends.xpu import w4a8_int8 as backend

    x = torch.zeros(8, 256, device="xpu", dtype=torch.bfloat16)
    operands = {
        "qdata": torch.zeros(7, 128, device="xpu", dtype=torch.int8),
        "s_rel": torch.ones(7, 16, device="xpu").to(scale_dtype),
    }
    original = operands[operand]
    operands[operand] = original.t().contiguous().t()
    assert operands[operand].shape == original.shape
    assert not operands[operand].is_contiguous()
    assert operands["s_rel" if operand == "qdata" else "qdata"].is_contiguous()

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid packed decode reached preparation or a GEMM")

    monkeypatch.setattr(xpu._int8, "supports_w4a8_linear_prequantized", lambda: True, raising=False)
    monkeypatch.setattr(xpu._int8, "w4a8_linear_prequantized", forbidden, raising=False)
    monkeypatch.setattr(xpu._int8, "int8_linear_prequantized", forbidden)
    monkeypatch.setattr(backend, "_dequant_int4_grouped_to_int8", forbidden)
    monkeypatch.setattr(backend, "_build_hadamard", forbidden)
    monkeypatch.setattr(backend, "apply_input_act", forbidden)
    with ck.use_backend("xpu"), pytest.raises(ValueError, match="contiguous " + operand):
        ck.w4a8_int8_linear(x, **operands, s_channel=torch.ones(7, device="xpu"))


def test_decode_missing_packed_companion_fails_explicitly(monkeypatch):
    from comfy_kitchen.backends import xpu
    from comfy_kitchen.backends.xpu import w4a8_int8 as backend

    def forbidden(*args, **kwargs):
        raise AssertionError("missing packed companion reached preparation or legacy GEMM")

    monkeypatch.setattr(xpu._int8, "supports_w4a8_linear_prequantized", lambda: False, raising=False)
    monkeypatch.setattr(xpu._int8, "int8_linear_prequantized", forbidden)
    monkeypatch.setattr(backend, "_dequant_int4_grouped_to_int8", forbidden)
    monkeypatch.setattr(backend, "apply_input_act", forbidden)
    with ck.use_backend("xpu"), pytest.raises(RuntimeError, match="matching omni_xpu_kernel"):
        ck.w4a8_int8_linear(
            torch.zeros(1, 256, device="xpu", dtype=torch.bfloat16),
            torch.zeros(7, 128, device="xpu", dtype=torch.int8),
            torch.ones(7, 16, device="xpu"), torch.ones(7, device="xpu"),
        )
