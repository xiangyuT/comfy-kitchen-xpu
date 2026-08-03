"""Native smoke and parity tests for the optional omni XPU backend.

Executable contract baseline: Comfy Kitchen v0.2.26 at
255a43879fe57bbcbecfdb273b46d772b00c5a90.  XPU-specific ports live here so
upstream's generic CUDA/Triton/eager tests remain easy to resync.  In
particular, the RMS-RoPE cases mirror ``TestPartialRotary`` in
``test_rms_rope.py`` and the activation cases mirror ``TestInputActQuantizer``,
``TestInt8LinearInputAct``, and ``TestSwiGLUInputAct`` in
``test_int8_input_act.py``.
"""

import pytest
import torch

import comfy_kitchen as ck


def _xpu_available() -> bool:
    return ck.list_backends()["xpu"]["available"]


pytestmark = [
    pytest.mark.xpu,
    pytest.mark.skipif(not _xpu_available(), reason="omni XPU backend is unavailable"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_xpu_adaln_matches_reference(dtype):
    x = torch.randn(2, 16, 128, device="xpu", dtype=dtype)
    scale = torch.randn(2, 1, 128, device="xpu", dtype=dtype) * 0.1
    shift = torch.randn(2, 1, 128, device="xpu", dtype=dtype) * 0.1

    with ck.use_backend("xpu"):
        actual = ck.adaln(x, scale, shift)
    expected = torch.nn.functional.layer_norm(x.float(), (128,), eps=1e-6)
    expected = (expected * (1 + scale.float()) + shift.float()).to(dtype)

    tolerance = 2e-2 if dtype != torch.float32 else 2e-4
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert actual.is_contiguous()


@pytest.mark.parametrize("shape", [(2, 16, 64), (2, 256, 768), (1, 64, 3072)])
def test_xpu_adaln_original_shape_matrix(shape):
    x = torch.randn(shape, device="xpu", dtype=torch.bfloat16)
    scale = torch.randn(shape[0], 1, shape[-1], device="xpu", dtype=torch.bfloat16)
    shift = torch.randn_like(scale)
    with ck.use_backend("xpu"):
        actual = ck.adaln(x, scale, shift)
    expected = torch.nn.functional.layer_norm(x.float(), (shape[-1],), eps=1e-6)
    expected = (expected * (1 + scale.float()) + shift.float()).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)


def test_xpu_adaln_unsupported_hidden_size_falls_back_safely():
    x = torch.randn(2, 4, 63, device="xpu", dtype=torch.float32)
    scale = torch.randn(2, 1, 63, device="xpu", dtype=torch.float32)
    shift = torch.randn(2, 1, 63, device="xpu", dtype=torch.float32)

    with ck.use_backend("xpu"):
        actual = ck.adaln(x, scale, shift)
    expected = torch.nn.functional.layer_norm(x, (63,), eps=1e-6)
    expected = expected * (1 + scale) + shift
    torch.testing.assert_close(actual, expected)


def test_xpu_rms_adaln_matches_reference():
    x = torch.randn(2, 16, 128, device="xpu", dtype=torch.bfloat16)
    scale = torch.randn(2, 1, 128, device="xpu", dtype=torch.bfloat16) * 0.1
    shift = torch.randn_like(scale) * 0.1

    with ck.use_backend("xpu"):
        actual = ck.rms_adaln(x, scale, shift)
    expected = torch.nn.functional.rms_norm(x.float(), (128,), eps=1e-6)
    expected = (expected * (1 + scale.float()) + shift.float()).to(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)


@pytest.mark.parametrize("split_half", [False, True])
@pytest.mark.parametrize("layout", ["BHND", "BNHD"])
def test_xpu_rope_arbitrary_matrix_pair_semantics(split_half, layout):
    if layout == "BHND":
        q = torch.randn(2, 4, 19, 64, device="xpu", dtype=torch.bfloat16)
        k = torch.randn(2, 2, 19, 64, device="xpu", dtype=torch.bfloat16)
        freqs = torch.randn(2, 1, 19, 32, 2, 2, device="xpu", dtype=torch.float32)
    else:
        q = torch.randn(2, 19, 4, 64, device="xpu", dtype=torch.float16)
        k = torch.randn(2, 19, 2, 64, device="xpu", dtype=torch.float16)
        freqs = torch.randn(1, 19, 1, 32, 2, 2, device="xpu", dtype=torch.bfloat16)
    operation = ck.apply_rope_split_half if split_half else ck.apply_rope
    with ck.use_backend("xpu"):
        actual_q, actual_k = operation(q, k, freqs)
    with ck.use_backend("eager"):
        expected_q, expected_k = operation(q, k, freqs)
    if q.dtype == torch.bfloat16:
        # The native expression and eager addcmul can differ by one BF16 ULP.
        rtol, atol = 8e-3, 4e-3
    else:
        rtol = atol = 0.07 if freqs.dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(actual_q, expected_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_k, expected_k, rtol=rtol, atol=atol)


def test_xpu_h3_packed_qkv_partial_rms_rope_inplace():
    sequence, heads, head_dim, rot_dim = 37, 56, 128, 96
    inner = heads * head_dim
    packed = torch.randn(
        sequence,
        3 * inner,
        device="xpu",
        dtype=torch.bfloat16,
    )
    q = packed[:, :inner].view(1, sequence, heads, head_dim)
    k = packed[:, inner : 2 * inner].view(1, sequence, heads, head_dim)
    q_reference = q.clone()
    k_reference = k.clone()
    q_scale = torch.randn(head_dim, device="xpu", dtype=torch.float32)
    k_scale = torch.randn(head_dim, device="xpu", dtype=torch.float32)
    freqs = torch.randn(
        1,
        sequence,
        1,
        rot_dim // 2,
        2,
        2,
        device="xpu",
        dtype=torch.bfloat16,
    )
    q_pointer, k_pointer = q.data_ptr(), k.data_ptr()

    with ck.use_backend("xpu"):
        q_out, k_out = ck.rms_rope_split_half_(
            q,
            k,
            freqs,
            q_scale,
            k_scale,
            epsilon=1e-5,
            rot_dim=rot_dim,
        )
    with ck.use_backend("eager"):
        q_expected, k_expected = ck.rms_rope_split_half(
            q_reference,
            k_reference,
            freqs,
            q_scale,
            k_scale,
            epsilon=1e-5,
            rot_dim=rot_dim,
        )

    assert not q.is_contiguous()
    assert q_out.data_ptr() == q_pointer
    assert k_out.data_ptr() == k_pointer
    torch.testing.assert_close(q_out, q_expected, rtol=0.02, atol=0.02)
    torch.testing.assert_close(k_out, k_expected, rtol=0.02, atol=0.02)


def test_xpu_rope_compile_fullgraph():
    freqs = torch.randn(1, 1, 17, 32, 2, 2, device="xpu", dtype=torch.float32)

    @torch.compile(backend="eager", fullgraph=True)
    def compiled(x):
        return ck.apply_rope1(x, freqs)

    x = torch.randn(1, 3, 17, 64, device="xpu", dtype=torch.bfloat16)
    with ck.use_backend("xpu"):
        torch.testing.assert_close(compiled(x), ck.apply_rope1(x, freqs), rtol=0, atol=0)


def test_xpu_backend_respects_non_default_stream():
    """Kitchen dispatch and the native kernel share the active XPU queue."""
    stream = torch.xpu.Stream()
    x_cpu = torch.randn(1, 4, 257, 64, dtype=torch.bfloat16)
    freqs_cpu = torch.randn(1, 1, 257, 32, 2, 2, dtype=torch.float32)

    with torch.xpu.stream(stream), ck.use_backend("xpu"):
        x = x_cpu.to("xpu")
        freqs = freqs_cpu.to("xpu")
        actual = ck.apply_rope1(x, freqs)
    stream.synchronize()

    with ck.use_backend("eager"):
        expected = ck.apply_rope1(x, freqs)
    # This test checks producer/consumer ordering on the active queue. The
    # native SYCL expression may differ from eager addcmul by one BF16 rounding
    # step, which is covered separately by the RoPE parity matrix above.
    torch.testing.assert_close(actual, expected, rtol=8e-3, atol=4e-3)


def test_xpu_fp8_weight_only_linear_matches_dequantized_reference():
    from omni_xpu_kernel import linear

    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout

    m, k, n = 8, 2048, 512
    x = torch.randn(m, k, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="xpu", dtype=torch.bfloat16)
    scale = weight.abs().max().float() / torch.finfo(torch.float8_e4m3fn).max
    qweight = (
        (weight / scale)
        .clamp(
            torch.finfo(torch.float8_e4m3fn).min,
            torch.finfo(torch.float8_e4m3fn).max,
        )
        .to(torch.float8_e4m3fn)
    )
    params = TensorCoreFP8Layout.Params(
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=(n, k),
    )
    quantized_weight = QuantizedTensor(qweight, "TensorCoreFP8Layout", params)

    linear.fp8_cache_clear()
    actual = torch.nn.functional.linear(x, quantized_weight)
    expected = torch.nn.functional.linear(x.float(), (qweight.float() * scale).float()).to(
        torch.bfloat16
    )
    error = (actual.float() - expected.float()).abs()
    assert error.mean().item() < 0.2
    assert error.max().item() <= 1.5
    assert linear.fp8_cache_stats()["misses"] >= 1


def test_xpu_fp8_weight_only_bias_and_high_dimensional_input():
    from omni_xpu_kernel import linear

    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout

    batch, tokens, k, n = 2, 4, 2048, 512
    x = torch.randn(batch, tokens, k, device="xpu", dtype=torch.float16)
    qweight = torch.randn(n, k, device="xpu").to(torch.float8_e4m3fn)
    scale = torch.tensor(0.01, device="xpu", dtype=torch.float32)
    bias = torch.randn(n, device="xpu", dtype=torch.float16)
    params = TensorCoreFP8Layout.Params(scale=scale, orig_dtype=torch.float16, orig_shape=(n, k))
    weight = QuantizedTensor(qweight, "TensorCoreFP8Layout", params)

    linear.fp8_cache_clear()
    actual = torch.nn.functional.linear(x, weight, bias)
    expected = torch.nn.functional.linear(x.float(), qweight.float() * scale, bias.float()).to(
        torch.float16
    )
    error = (actual.float() - expected.float()).abs()
    assert actual.shape == (batch, tokens, n)
    assert actual.dtype == torch.float16
    assert error.mean().item() < 0.2
    assert error.max().item() < 2.0
    assert linear.fp8_cache_stats()["misses"] >= 1


def test_xpu_fp8_unsupported_shape_uses_safe_fallback():
    from omni_xpu_kernel import linear

    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout

    x = torch.randn(4, 128, device="xpu", dtype=torch.bfloat16)
    qweight = torch.randn(64, 128, device="xpu").to(torch.float8_e4m3fn)
    scale = torch.tensor(0.01, device="xpu", dtype=torch.float32)
    params = TensorCoreFP8Layout.Params(
        scale=scale, orig_dtype=torch.bfloat16, orig_shape=(64, 128)
    )
    weight = QuantizedTensor(qweight, "TensorCoreFP8Layout", params)
    linear.fp8_cache_clear()
    actual = torch.nn.functional.linear(x, weight)
    expected = torch.nn.functional.linear(x, (qweight.float() * scale).to(torch.bfloat16))
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=1e-3)
    assert linear.fp8_cache_stats()["size"] == 0


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_xpu_fp8_qdq_matches_eager(input_dtype, fp8_dtype):
    limit = torch.finfo(fp8_dtype).max
    values = torch.tensor(
        [-limit * 2, -7.25, -0.0, 0.0, 0.125, 9.5, limit * 2],
        device="xpu",
        dtype=input_dtype,
    )
    scale = torch.tensor(0.5, device="xpu", dtype=torch.float32)

    with ck.use_backend("xpu"):
        quantized = ck.quantize_per_tensor_fp8(values, scale, fp8_dtype)
    with ck.use_backend("eager"):
        expected_quantized = ck.quantize_per_tensor_fp8(values, scale, fp8_dtype)
    assert torch.equal(quantized.view(torch.uint8), expected_quantized.view(torch.uint8))

    for output_dtype in (torch.float32, torch.float16, torch.bfloat16):
        with ck.use_backend("xpu"):
            restored = ck.dequantize_per_tensor_fp8(quantized, scale, output_dtype)
        with ck.use_backend("eager"):
            expected_restored = ck.dequantize_per_tensor_fp8(
                expected_quantized, scale, output_dtype
            )
        assert restored.dtype == output_dtype
        torch.testing.assert_close(restored, expected_restored, rtol=0, atol=0)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_xpu_fp8_stochastic_rounding_matches_eager(fp8_dtype):
    x = torch.linspace(-12, 12, 4096, device="xpu", dtype=torch.float16)
    rng = torch.arange(256, device="xpu", dtype=torch.uint8).repeat(16)
    rng_before = rng.clone()

    with ck.use_backend("xpu"):
        actual = ck.stochastic_rounding_fp8(x, rng, fp8_dtype)
    with ck.use_backend("eager"):
        expected = ck.stochastic_rounding_fp8(x, rng_before, fp8_dtype)

    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(rng, rng_before)


def test_xpu_fp8_qdq_compile_fullgraph():
    scale = torch.tensor(0.25, device="xpu", dtype=torch.float32)

    @torch.compile(backend="eager", fullgraph=True)
    def compiled(inp):
        quantized = ck.quantize_per_tensor_fp8(inp, scale, torch.float8_e4m3fn)
        return ck.dequantize_per_tensor_fp8(quantized, scale, torch.bfloat16)

    x = torch.randn(16, 64, device="xpu", dtype=torch.bfloat16)
    with ck.use_backend("xpu"):
        actual = compiled(x)
        expected = ck.dequantize_per_tensor_fp8(
            ck.quantize_per_tensor_fp8(x, scale, torch.float8_e4m3fn),
            scale,
            torch.bfloat16,
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_xpu_int8_quantize_dequantize_roundtrip():
    x = torch.randn(32, 128, device="xpu", dtype=torch.bfloat16)

    with ck.use_backend("xpu"):
        q, scale = ck.quantize_int8_tensorwise(x)
        restored = torch.ops.comfy_kitchen.dequantize_int8_simple_dtype(q, scale, 2)

    assert q.dtype == torch.int8
    assert scale.dtype == torch.float32
    assert restored.dtype == torch.bfloat16
    assert restored.shape == x.shape
    assert (restored.float() - x.float()).abs().mean().item() < 0.02


def test_xpu_int8_stochastic_rounding_is_seeded():
    x = torch.full((32, 128), 0.5, device="xpu", dtype=torch.float16)
    x[:, 0] = 127.0
    with ck.use_backend("xpu"):
        q1, s1 = ck.quantize_int8_rowwise(x, stochastic_rounding=123)
        q2, s2 = ck.quantize_int8_rowwise(x, stochastic_rounding=123)
        q3, _ = ck.quantize_int8_rowwise(x, stochastic_rounding=124)
    assert torch.equal(q1, q2)
    assert torch.equal(s1, s2)
    assert not torch.equal(q1, q3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_xpu_int8_linear_single_row_bias_and_dtype(dtype):
    x = torch.randn(1, 256, device="xpu", dtype=dtype)
    weight = torch.randn(96, 256, device="xpu", dtype=dtype)
    bias = torch.randn(96, device="xpu", dtype=dtype)
    with ck.use_backend("xpu"):
        qweight, scale = ck.quantize_int8_tensorwise(weight)
        actual = ck.int8_linear(x, qweight, scale, bias, dtype)
    with ck.use_backend("eager"):
        expected = ck.int8_linear(x, qweight, scale, bias, dtype)
    error = (actual.float() - expected.float()).abs()
    assert actual.shape == (1, 96)
    assert actual.dtype == dtype
    assert error.mean().item() < 0.15
    assert error.max().item() < 0.75


def test_xpu_int8_linear_swiglu_input_act():
    rows, hidden, output = 37, 256, 96
    x = torch.randn(rows, 2 * hidden, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(output, hidden, device="xpu", dtype=torch.bfloat16)
    with ck.use_backend("xpu"):
        qweight, scale = ck.quantize_int8_tensorwise(weight)
        actual = ck.int8_linear(
            x,
            qweight,
            scale,
            out_dtype=torch.bfloat16,
            input_act="swiglu",
        )
    gate, up = x.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate).mul(up)
    with ck.use_backend("xpu"):
        expected = ck.int8_linear(
            activated,
            qweight,
            scale,
            out_dtype=torch.bfloat16,
        )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


def test_xpu_int8_linear_gelu_input_act_uses_fused_boundary(monkeypatch):
    """The no-ConvRot GELU route must not materialize a floating activation."""
    from omni_xpu_kernel import int8 as omni_int8

    batch, tokens, hidden, output = 2, 17, 256, 96
    x = torch.randn(batch, tokens, hidden, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(output, hidden, device="xpu", dtype=torch.bfloat16)
    with ck.use_backend("xpu"):
        qweight, scale = ck.quantize_int8_tensorwise(weight)
        x_int8, x_scale = omni_int8.fused_gelu_tanh_quantize_rowwise(x)
        expected = omni_int8.int8_linear_prequantized(
            x_int8,
            x_scale,
            qweight,
            scale,
            out_dtype=torch.bfloat16,
        )

        def reject_materialized_activation(*_args, **_kwargs):
            raise AssertionError("gelu_tanh materialized a floating activation")

        monkeypatch.setattr(
            omni_int8, "_apply_input_act", reject_materialized_activation
        )
        actual = ck.int8_linear(
            x,
            qweight,
            scale,
            out_dtype=torch.bfloat16,
            input_act="gelu_tanh",
        )
    assert actual.shape == (batch, tokens, output)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_xpu_int8_linear_gelu_convrot_fallback_matches_explicit_activation():
    """ConvRot still applies GELU exactly once before its existing quantizer."""
    rows, hidden, output = 37, 256, 96
    x = torch.randn(rows, hidden, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(output, hidden, device="xpu", dtype=torch.bfloat16)
    with ck.use_backend("xpu"):
        qweight, scale = ck.quantize_int8_tensorwise(weight)
        actual = ck.int8_linear(
            x,
            qweight,
            scale,
            out_dtype=torch.bfloat16,
            convrot=True,
            convrot_groupsize=256,
            input_act="gelu_tanh",
        )
        expected = ck.int8_linear(
            torch.nn.functional.gelu(x, approximate="tanh"),
            qweight,
            scale,
            out_dtype=torch.bfloat16,
            convrot=True,
            convrot_groupsize=256,
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_xpu_int8_linear_input_act_identity_and_rejects_unknown():
    """None/none are identical and unsupported activation names fail clearly."""
    rows, hidden, output = 17, 256, 64
    x = torch.randn(rows, hidden, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(output, hidden, device="xpu", dtype=torch.bfloat16)
    with ck.use_backend("xpu"):
        qweight, scale = ck.quantize_int8_tensorwise(weight)
        omitted = ck.int8_linear(x, qweight, scale, out_dtype=torch.bfloat16)
        explicit_none = ck.int8_linear(
            x, qweight, scale, out_dtype=torch.bfloat16, input_act=None
        )
        named_none = ck.int8_linear(
            x, qweight, scale, out_dtype=torch.bfloat16, input_act="none"
        )
        with pytest.raises(ValueError, match="unsupported input_act"):
            ck.int8_linear(x, qweight, scale, input_act="silu")
    assert torch.equal(omitted, explicit_none)
    assert torch.equal(omitted, named_none)


def test_xpu_int8_quantized_tensor_lifecycle_and_linear():
    from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

    x = torch.randn(4, 256, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(96, 256, device="xpu", dtype=torch.bfloat16)
    with ck.use_backend("xpu"):
        qdata, params = TensorWiseINT8Layout.quantize(weight)
    qweight = QuantizedTensor(qdata, "TensorWiseINT8Layout", params)
    clone = qweight.clone()
    detached = qweight.detach()
    state = qweight.state_dict()
    with ck.use_backend("xpu"):
        out = torch.nn.functional.linear(x, qweight)
    assert out.shape == (4, 96)
    assert clone._qdata.data_ptr() != qweight._qdata.data_ptr()
    assert detached._qdata.data_ptr() == qweight._qdata.data_ptr()
    assert set(state) == {"", "_scale"}
    assert state[""].dtype == torch.int8
    cpu_weight = qweight.to("cpu")
    roundtrip_weight = cpu_weight.to("xpu")
    assert cpu_weight._qdata.device.type == "cpu"
    assert cpu_weight._params.scale.device.type == "cpu"
    assert roundtrip_weight._qdata.device.type == "xpu"
    assert torch.equal(roundtrip_weight._qdata, qweight._qdata)


def test_xpu_int8_quantized_tensor_mm_addmm_and_transpose():
    from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

    x = torch.randn(4, 256, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(96, 256, device="xpu", dtype=torch.bfloat16)
    bias = torch.randn(96, device="xpu", dtype=torch.bfloat16)
    with ck.use_backend("xpu"):
        qdata, params = TensorWiseINT8Layout.quantize(weight)
    qweight = QuantizedTensor(qdata, "TensorWiseINT8Layout", params)
    transposed = qweight.t()
    with ck.use_backend("xpu"):
        mm_out = torch.mm(x, transposed)
        addmm_out = torch.addmm(bias, x, transposed)
        linear_out = torch.nn.functional.linear(x, qweight, bias)
    assert transposed.shape == (256, 96)
    assert mm_out.shape == (4, 96)
    assert addmm_out.shape == (4, 96)
    torch.testing.assert_close(addmm_out, linear_out)


def test_xpu_int8_compile_smoke():
    x = torch.randn(4, 256, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(96, 256, device="xpu", dtype=torch.bfloat16)
    with ck.use_backend("xpu"):
        qweight, scale = ck.quantize_int8_tensorwise(weight)

    @torch.compile(backend="eager", fullgraph=True)
    def compiled(inp):
        return ck.int8_linear(inp, qweight, scale, out_dtype=torch.bfloat16)

    with ck.use_backend("xpu"):
        actual = compiled(x)
        expected = ck.int8_linear(x, qweight, scale, out_dtype=torch.bfloat16)
        x_dynamic = torch.randn(7, 256, device="xpu", dtype=torch.bfloat16)
        actual_dynamic = compiled(x_dynamic)
        expected_dynamic = ck.int8_linear(x_dynamic, qweight, scale, out_dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_dynamic, expected_dynamic)


def test_xpu_int8_linear_matches_eager():
    x = torch.randn(8, 128, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(64, 128, device="xpu", dtype=torch.bfloat16)
    bias = torch.randn(64, device="xpu", dtype=torch.bfloat16)

    with ck.use_backend("xpu"):
        qweight, weight_scale = ck.quantize_int8_tensorwise(weight)
        actual = ck.int8_linear(x, qweight, weight_scale, bias, torch.bfloat16)

    with ck.use_backend("eager"):
        expected = ck.int8_linear(x, qweight, weight_scale, bias, torch.bfloat16)

    # oneDNN fuses rescaling into the BF16 matmul and can round differently
    # from eager's INT32-then-float sequence. Keep the aggregate error bounded.
    error = (actual.float() - expected.float()).abs()
    assert error.mean().item() < 0.15
    assert error.max().item() < 0.75


def test_xpu_int8_primitive_cache_hits():
    from omni_xpu_kernel import int8

    x = torch.randn(8, 256, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(96, 256, device="xpu", dtype=torch.bfloat16)
    with ck.use_backend("xpu"):
        qweight, scale = ck.quantize_int8_tensorwise(weight)
    int8.int8_cache_clear()
    with ck.use_backend("xpu"):
        ck.int8_linear(x, qweight, scale)
        ck.int8_linear(x, qweight, scale)
    stats = int8.int8_cache_stats()
    assert stats["misses"] >= 1
    assert stats["hits"] >= 1


def test_xpu_convrot_linear_mm_addmm_dispatch():
    from comfy_kitchen.tensor import QuantizedTensor

    x = torch.randn(4, 128, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(64, 128, device="xpu", dtype=torch.bfloat16)
    bias = torch.randn(64, device="xpu", dtype=torch.bfloat16)
    with ck.use_backend("xpu"):
        qweight = QuantizedTensor.from_float(
            weight,
            "TensorWiseINT8Layout",
            per_channel=True,
            convrot=True,
            convrot_groupsize=64,
        )
        linear_out = torch.nn.functional.linear(x, qweight, bias)
        transposed = qweight.t()
        mm_out = torch.mm(x, transposed)
        addmm_out = torch.addmm(bias, x, transposed)
    assert linear_out.shape == (4, 64)
    torch.testing.assert_close(addmm_out, linear_out)
    torch.testing.assert_close(mm_out + bias, linear_out, rtol=0.02, atol=0.01)


def test_xpu_raw_int8_mm_matches_int32_reference():
    a = torch.randint(-8, 8, (8, 128), device="xpu", dtype=torch.int8)
    b = torch.randint(-8, 8, (128, 64), device="xpu", dtype=torch.int8)

    with ck.use_backend("xpu"):
        actual = ck.mm_int8(a, b)

    expected = a.cpu().to(torch.int32) @ b.cpu().to(torch.int32)
    assert actual.dtype == torch.int32
    assert torch.equal(actual.cpu(), expected)


def test_xpu_convrot_roundtrip_uses_dtype_adapter():
    weight = torch.randn(4, 256, device="xpu", dtype=torch.bfloat16)
    kwargs = {"weight": weight, "group_size": 256, "stochastic_rounding": 0}

    with ck.use_backend("xpu"):
        quantize = ck.registry.get_implementation("quantize_int8_convrot_weight", kwargs=kwargs)
        qweight, scale = quantize(**kwargs)
        restored = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
            qweight, scale, 256, 2
        )

    assert restored.dtype == torch.bfloat16
    assert (restored.float() - weight.float()).abs().mean().item() < 0.02


@pytest.mark.parametrize("linear_dtype", ["int4", "int8"])
def test_xpu_convrot_w4a4_layout_linear(linear_dtype, seed):
    from comfy_kitchen.tensor import QuantizedTensor

    x = torch.randn(5, 128, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn(48, 128, device="xpu", dtype=torch.bfloat16)
    bias = torch.randn(48, device="xpu", dtype=torch.bfloat16)
    with ck.use_backend("xpu"):
        qweight = QuantizedTensor.from_float(
            weight,
            "TensorCoreConvRotW4A4Layout",
            convrot_groupsize=64,
            linear_dtype=linear_dtype,
        )
        actual = torch.nn.functional.linear(x, qweight, bias)
        restored = qweight.dequantize()
    expected = torch.nn.functional.linear(x, restored, bias)
    error = (actual.float() - expected.float()).abs()
    assert actual.shape == (5, 48)
    if linear_dtype == "int4":
        # A4 activation quantization is intentionally much coarser than the
        # dequantized-weight W4A16 reference.
        assert error.mean().item() < 1.5
        assert error.max().item() < 5.0
    else:
        assert error.mean().item() < 0.15
        assert error.max().item() < 0.8


def test_xpu_convrot_support_helpers_match_eager_contract():
    from comfy_kitchen.backends.eager.convrot_w4a4 import _build_hadamard

    x = torch.randn(6, 128, device="xpu", dtype=torch.bfloat16)
    h = _build_hadamard(64, x.device, x.dtype)
    kwargs = {"x": x, "H": h, "group_size": 64, "stochastic_rounding": 0}
    with ck.use_backend("xpu"):
        implementation = ck.registry.get_implementation(
            "quantize_and_rotate_rowwise", kwargs=kwargs
        )
        actual_q, actual_scale = implementation(**kwargs)
    with ck.use_backend("eager"):
        reference = ck.registry.get_implementation("quantize_and_rotate_rowwise", kwargs=kwargs)
        expected_q, expected_scale = reference(x, h, 64, 0)
    # Omni's fused rowwise quantizer performs scale multiplication in FP32,
    # while eager casts the scale back to the BF16 input dtype before division.
    # Both implement the same quantization contract but boundary values can be
    # one integer apart. Keep the tolerance aligned with the native Kernel
    # correctness suite and require the row scales themselves to match.
    quant_diff = (actual_q.to(torch.int16) - expected_q.to(torch.int16)).abs()
    assert quant_diff.max().item() <= 1
    torch.testing.assert_close(actual_scale, expected_scale, rtol=1e-6, atol=1e-8)

    packed = torch.randint(-128, 127, (8, 64), device="xpu", dtype=torch.int8)
    prepare = ck.registry.get_implementation(
        "prepare_int4_weight_for_int8_linear", kwargs={"weight": packed}
    )
    unpacked = prepare(packed)
    assert unpacked.shape == (8, 128)
    assert unpacked.min().item() >= -8
    assert unpacked.max().item() <= 7


def test_xpu_svdquant_signed_pipeline_matches_eager():
    from comfy_kitchen.backends.eager.svdquant import _pack_int4_row_major
    from comfy_kitchen.tensor import QuantizedTensor, TensorCoreSVDQuantW4A4Layout

    m, k, n, rank = 7, 128, 64, 8
    x = torch.randn(m, k, device="xpu", dtype=torch.bfloat16)
    smooth = torch.rand(k, device="xpu", dtype=torch.bfloat16) + 0.5
    lora_down = torch.randn(k, rank, device="xpu", dtype=torch.bfloat16) * 0.05
    lora_up = torch.randn(n, rank, device="xpu", dtype=torch.bfloat16) * 0.05
    qweight = _pack_int4_row_major(torch.randint(-7, 8, (n, k), device="xpu", dtype=torch.int8))
    wscales = torch.rand(k // 64, n, device="xpu", dtype=torch.bfloat16) * 0.05
    bias = torch.randn(n, device="xpu", dtype=torch.bfloat16)

    with ck.use_backend("xpu"):
        qact, ascales, lora_act = ck.quantize_svdquant_w4a4(x, smooth, lora_down, pad_size=16)
        actual = ck.scaled_mm_svdquant_w4a4(
            qact, qweight, ascales, wscales, lora_act, lora_up, bias
        )

    with ck.use_backend("eager"):
        expected = ck.scaled_mm_svdquant_w4a4(
            qact, qweight, ascales, wscales, lora_act, lora_up, bias
        )

    assert qact.dtype == torch.int8
    assert qact.shape == (16, k // 2)
    assert lora_act.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    params = TensorCoreSVDQuantW4A4Layout.Params(
        scale=wscales,
        orig_dtype=torch.bfloat16,
        orig_shape=(n, k),
        proj_down=lora_down,
        proj_up=lora_up,
        smooth_factor=smooth,
    )
    quantized_weight = QuantizedTensor(qweight, "TensorCoreSVDQuantW4A4Layout", params)
    with ck.use_backend("xpu"):
        linear_out = torch.nn.functional.linear(x, quantized_weight, bias)
    with ck.use_backend("eager"):
        linear_ref = torch.nn.functional.linear(x, quantized_weight, bias)
    # ESIMD and eager can choose opposite sides of an INT4 rounding tie.
    linear_error = (linear_out.float() - linear_ref.float()).abs()
    assert linear_error.mean().item() < 0.1
    assert linear_error.max().item() < 0.75


def test_xpu_svdquant_unsigned_native_matches_eager():
    from omni_xpu_kernel import svdq

    m, k, n, rank = 5, 128, 32, 4
    x = torch.rand(m, k, device="xpu", dtype=torch.bfloat16) * 3
    smooth = torch.rand(k, device="xpu", dtype=torch.bfloat16) + 0.5
    lora_down = torch.randn(k, rank, device="xpu", dtype=torch.bfloat16) * 0.05
    lora_up = torch.randn(n, rank, device="xpu", dtype=torch.bfloat16) * 0.05
    qweight = torch.randint(-128, 127, (n, k // 2), device="xpu", dtype=torch.int8)
    wscales = torch.rand(k // 64, n, device="xpu", dtype=torch.bfloat16) * 0.05

    with ck.use_backend("xpu"):
        qact, ascales, lora_act = ck.quantize_svdquant_w4a4(
            x, smooth, lora_down, pad_size=8, act_unsigned=True
        )
        actual = ck.scaled_mm_svdquant_w4a4(
            qact, qweight, ascales, wscales, lora_act, lora_up, act_unsigned=True
        )
    with ck.use_backend("eager"):
        expected = ck.scaled_mm_svdquant_w4a4(
            qact, qweight, ascales, wscales, lora_act, lora_up, act_unsigned=True
        )
    unpacked = svdq.unpack_int4(qact.view(torch.uint8), signed=False)
    assert unpacked.min().item() >= 0
    assert unpacked.max().item() <= 15
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_xpu_svdquant_lora_x_uses_separate_source():
    k, rank = 128, 8
    raw = torch.randn(4, k, device="xpu", dtype=torch.bfloat16) * 0.5
    shifted = raw + 0.171875
    smooth = torch.ones(k, device="xpu", dtype=torch.bfloat16)
    lora_down = torch.randn(k, rank, device="xpu", dtype=torch.bfloat16) * 0.1
    with ck.use_backend("xpu"):
        _, _, from_raw = ck.quantize_svdquant_w4a4(
            shifted, smooth, lora_down, pad_size=8, lora_x=raw
        )
        _, _, from_shifted = ck.quantize_svdquant_w4a4(shifted, smooth, lora_down, pad_size=8)
    expected = raw.float() @ lora_down.float()
    assert not torch.allclose(from_raw, from_shifted)
    torch.testing.assert_close(from_raw[:4], expected, rtol=0, atol=0)


@pytest.mark.parametrize("magnitude", [1e-4, 1e3])
def test_xpu_svdquant_extreme_magnitudes_remain_finite(magnitude):
    x = torch.randn(4, 128, device="xpu", dtype=torch.float32) * magnitude
    smooth = torch.ones(128, device="xpu", dtype=torch.float32)
    lora_down = torch.zeros(128, 4, device="xpu", dtype=torch.float32)
    with ck.use_backend("xpu"):
        qact, scales, lora = ck.quantize_svdquant_w4a4(x, smooth, lora_down, pad_size=4)
    assert torch.isfinite(scales).all()
    assert torch.isfinite(lora).all()
    assert qact.view(torch.uint8).shape == (4, 64)


def test_xpu_svdquant_compile_smoke():
    k, n, rank = 128, 32, 4
    smooth = torch.ones(k, device="xpu", dtype=torch.bfloat16)
    lora_down = torch.randn(k, rank, device="xpu", dtype=torch.bfloat16) * 0.01
    lora_up = torch.randn(n, rank, device="xpu", dtype=torch.bfloat16) * 0.01
    qweight = torch.randint(-128, 127, (n, k // 2), device="xpu", dtype=torch.int8)
    wscales = torch.rand(k // 64, n, device="xpu", dtype=torch.bfloat16) * 0.05

    @torch.compile(backend="eager", fullgraph=True)
    def compiled(inp):
        qact, ascales, lora = ck.quantize_svdquant_w4a4(inp, smooth, lora_down, pad_size=8)
        return ck.scaled_mm_svdquant_w4a4(qact, qweight, ascales, wscales, lora, lora_up)

    x = torch.randn(5, k, device="xpu", dtype=torch.bfloat16)
    with ck.use_backend("xpu"):
        actual = compiled(x)
        qact, ascales, lora = ck.quantize_svdquant_w4a4(x, smooth, lora_down, pad_size=8)
        expected = ck.scaled_mm_svdquant_w4a4(qact, qweight, ascales, wscales, lora, lora_up)
    torch.testing.assert_close(actual, expected)


def test_xpu_svdquant_grouped_and_fused_linear_match_individual():
    from comfy_kitchen.tensor import (
        QuantizedTensor,
        TensorCoreSVDQuantW4A4Layout,
        prepare_svdquant_for_xpu,
        svdquant_w4a4_fuse_linear_weights,
        svdquant_w4a4_fused_grouped_linear,
        svdquant_w4a4_grouped_linear,
    )

    m, k, rank = 6, 128, 4
    out_features = (32, 48)
    x = torch.randn(m, k, device="xpu", dtype=torch.bfloat16)
    smooth = torch.rand(k, device="xpu", dtype=torch.bfloat16) + 0.5
    proj_down = torch.randn(k, rank, device="xpu", dtype=torch.bfloat16) * 0.02
    weights = []
    biases = []
    for n in out_features:
        params = TensorCoreSVDQuantW4A4Layout.Params(
            scale=torch.rand(k // 64, n, device="xpu", dtype=torch.bfloat16) * 0.05,
            orig_dtype=torch.bfloat16,
            orig_shape=(n, k),
            proj_down=proj_down,
            proj_up=torch.randn(n, rank, device="xpu", dtype=torch.bfloat16) * 0.02,
            smooth_factor=smooth,
        )
        qdata = torch.randint(-128, 127, (n, k // 2), device="xpu", dtype=torch.int8)
        weights.append(QuantizedTensor(qdata, "TensorCoreSVDQuantW4A4Layout", params))
        biases.append(torch.randn(n, device="xpu", dtype=torch.bfloat16))

    with ck.use_backend("xpu"):
        individual = tuple(
            torch.nn.functional.linear(x, weight, bias)
            for weight, bias in zip(weights, biases, strict=True)
        )
        grouped = svdquant_w4a4_grouped_linear(x, weights, biases)
        fused_weight, fused_bias, splits = svdquant_w4a4_fuse_linear_weights(weights, biases)
        fused = svdquant_w4a4_fused_grouped_linear(x, fused_weight, fused_bias, splits)

    for expected, grouped_out, fused_out in zip(individual, grouped, fused, strict=True):
        torch.testing.assert_close(grouped_out, expected, rtol=0, atol=0)
        torch.testing.assert_close(fused_out, expected, rtol=0, atol=0)

    prepare_svdquant_for_xpu(fused_weight)
    with ck.use_backend("xpu"):
        prepared_fused = svdquant_w4a4_fused_grouped_linear(x, fused_weight, fused_bias, splits)
    for expected, actual in zip(individual, prepared_fused, strict=True):
        error = (actual.float() - expected.float()).abs()
        assert error.mean().item() < 0.1
        assert error.max().item() < 0.75


def test_xpu_non_default_device_int8():
    if torch.xpu.device_count() < 2:
        pytest.skip("multiple XPU devices unavailable")
    device = torch.device("xpu", 1)
    x = torch.randn(2, 128, device=device, dtype=torch.bfloat16)
    with ck.use_backend("xpu"):
        qdata, scale = ck.quantize_int8_rowwise(x)
    assert qdata.device == device
    assert scale.device == device


def test_xpu_svdquant_tile_layout_preparation():
    from comfy_kitchen.backends.xpu.svdquant import prepare_svdquant_weights

    n, k, rank = 128, 128, 8
    natural_weight = torch.randint(-128, 127, (n, k // 2), device="xpu", dtype=torch.int8)
    natural_scales = torch.randn(k // 64, n, device="xpu", dtype=torch.bfloat16)
    natural_lora = torch.randn(n, rank, device="xpu", dtype=torch.bfloat16)

    tiled_weight = (
        natural_weight.view(1, 32, 4, 2, 32).permute(0, 3, 1, 2, 4).contiguous().view(1, 2, 32, 128)
    )
    tiled_scales = natural_scales.view(2, 1, 128).permute(1, 0, 2).contiguous()
    tiled_lora = natural_lora.view(1, 128, rank).permute(0, 2, 1).contiguous()

    weight, scales, lora = prepare_svdquant_weights(tiled_weight, tiled_scales, tiled_lora)
    assert torch.equal(weight, natural_weight)
    assert torch.equal(scales, natural_scales)
    assert torch.equal(lora, natural_lora)


def test_xpu_svdquant_destructive_preconversion_keeps_single_weight_copy():
    from comfy_kitchen.backends.eager.svdquant import _pack_int4_row_major
    from comfy_kitchen.tensor import (
        QuantizedTensor,
        TensorCoreSVDQuantW4A4Layout,
        prepare_svdquant_for_xpu,
        restore_svdquant_standard_format_,
    )

    m, k, n, rank = 8, 128, 64, 8
    x = torch.randn(m, k, device="xpu", dtype=torch.bfloat16)
    standard_qdata = _pack_int4_row_major(
        torch.randint(-7, 8, (n, k), device="xpu", dtype=torch.int8)
    )
    standard_scale = torch.rand(k // 64, n, device="xpu", dtype=torch.bfloat16) * 0.05
    params = TensorCoreSVDQuantW4A4Layout.Params(
        scale=standard_scale,
        orig_dtype=torch.bfloat16,
        orig_shape=(n, k),
        proj_down=torch.randn(k, rank, device="xpu", dtype=torch.bfloat16) * 0.05,
        proj_up=torch.randn(n, rank, device="xpu", dtype=torch.bfloat16) * 0.05,
        smooth_factor=torch.rand(k, device="xpu", dtype=torch.bfloat16) + 0.5,
    )
    weight = QuantizedTensor(standard_qdata, "TensorCoreSVDQuantW4A4Layout", params)
    saved_qdata = standard_qdata.clone()
    saved_scale = standard_scale.clone()

    with ck.use_backend("xpu"):
        expected = torch.nn.functional.linear(x, weight)

    qdata_ptr = weight._qdata.data_ptr()
    qdata_bytes = weight._qdata.untyped_storage().nbytes()
    prepared = prepare_svdquant_for_xpu(weight)

    assert prepared is weight
    assert weight._qdata.data_ptr() == qdata_ptr
    assert weight._qdata.untyped_storage().nbytes() == qdata_bytes
    assert weight._params.xpu_preconverted
    assert weight._params.scale.dtype == torch.float16
    assert (
        weight._qdata.untyped_storage().nbytes() + weight._params.scale.untyped_storage().nbytes()
        == qdata_bytes + saved_scale.untyped_storage().nbytes()
    )

    with ck.use_backend("xpu"):
        actual = torch.nn.functional.linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    state = weight.state_dict()
    assert torch.equal(state[""], saved_qdata)
    torch.testing.assert_close(state["_scale"], saved_scale, rtol=0, atol=2e-4)

    cpu_weight = weight.to("cpu")
    assert not cpu_weight._params.xpu_preconverted
    assert torch.equal(cpu_weight._qdata, saved_qdata.cpu())
    assert cpu_weight._params.scale.dtype == torch.bfloat16

    restored_ptr = weight._qdata.data_ptr()
    restore_svdquant_standard_format_(weight)
    assert weight._qdata.data_ptr() == restored_ptr
    assert not weight._params.xpu_preconverted
    assert torch.equal(weight._qdata, saved_qdata)
