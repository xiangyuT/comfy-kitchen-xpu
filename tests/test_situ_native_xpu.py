"""SiTU public INT8/W4/W6 routes, companion fallback and fullgraph coverage."""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends._activations import apply_input_act


pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available() or not ck.list_backends()["xpu"]["available"],
    reason="native XPU backend required",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,width,group", [(1, 256, 0), (37, 512, 64), (17, 1024, 256), (1, 8192, 256)])
def test_int8_situ_uses_native_gated_quantizer(dtype, rows, width, group, monkeypatch):
    from omni_xpu_kernel import kitchen

    assert kitchen.supports_gated_quantize_int8()
    x = torch.randn(rows, 2 * width, device="xpu", dtype=dtype) * 8
    x[:, :width] *= 8
    weight = torch.randint(-127, 127, (64, width), device="xpu", dtype=torch.int8)
    scale = torch.tensor(0.01, device="xpu")
    calls = []
    quantizer = kitchen.gated_quantize_int8

    def tracked(*args):
        calls.append(args[1:])
        return quantizer(*args)

    monkeypatch.setattr(kitchen, "gated_quantize_int8", tracked)
    kwargs = dict(convrot=bool(group), convrot_groupsize=group or 256, out_dtype=dtype)
    with ck.use_backend("xpu"):
        actual = ck.int8_linear(x, weight, scale, input_act="situ", **kwargs)
        expected = ck.int8_linear(apply_input_act(x, "situ"), weight, scale, **kwargs)
    assert calls == [("situ", group)]
    error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-9)
    assert error.item() < 0.03


def test_situ_old_companion_prepares_before_native_linear(monkeypatch):
    from comfy_kitchen.backends import xpu

    x = torch.randn(3, 512, device="xpu", dtype=torch.bfloat16)
    weight = torch.randint(-127, 127, (64, 256), device="xpu", dtype=torch.int8)
    scale = torch.tensor(0.01, device="xpu")
    calls = []
    linear = xpu._int8.int8_linear

    def tracked(*args):
        calls.append((args[0].shape[-1], args[-1]))
        return linear(*args)

    monkeypatch.setattr(xpu, "_GATED_QUANTIZE_AVAILABLE", False)
    monkeypatch.setattr(xpu._int8, "int8_linear", tracked)
    with ck.use_backend("xpu"):
        actual = ck.int8_linear(x, weight, scale, input_act="situ")
    assert calls == [(256, None)]
    with ck.use_backend("xpu"):
        expected = ck.int8_linear(apply_input_act(x, "situ"), weight, scale)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bits", [4, 6])
def test_w4_w6_situ_native_quantization_and_fullgraph(bits, monkeypatch):
    from comfy_kitchen.tensor import QuantizedTensor
    from omni_xpu_kernel import kitchen

    weight = QuantizedTensor.from_float(
        torch.randn(64, 512, device="xpu", dtype=torch.bfloat16) * 0.02,
        "AsymW4A8Int8Layout", bits=bits, scale_search=False, scale_dtype=torch.float32,
    )
    x = torch.randn(2, 3, 1024, device="xpu", dtype=torch.bfloat16) * 8
    calls = []
    quantizer = kitchen.gated_quantize_int8

    def tracked(*args):
        calls.append(args[1:])
        return quantizer(*args)

    with monkeypatch.context() as context:
        context.setattr(kitchen, "gated_quantize_int8", tracked)
        with ck.use_backend("xpu"):
            eager = weight.linear_input_act(x, input_act="situ")
    assert calls == [("situ", 256)]
    with ck.use_backend("xpu"):
        composed = torch.nn.functional.linear(apply_input_act(x, "situ"), weight)
        compiled = torch.compile(
            lambda value: weight.linear_input_act(value, input_act="situ"),
            fullgraph=True,
        )
        actual = compiled(x)
    torch.testing.assert_close(actual, eager, rtol=0, atol=0)
    error = (eager.float() - composed.float()).norm() / composed.float().norm().clamp_min(1e-9)
    assert error.item() < 0.03


def test_int8_situ_fullgraph():
    x = torch.randn(3, 512, device="xpu", dtype=torch.bfloat16)
    weight = torch.randint(-127, 127, (64, 256), device="xpu", dtype=torch.int8)
    scale = torch.tensor(0.01, device="xpu")
    with ck.use_backend("xpu"):
        def run(value):
            return ck.int8_linear(value, weight, scale, convrot=True, input_act="situ")
        eager = run(x)
        actual = torch.compile(run, fullgraph=True)(x)
    torch.testing.assert_close(actual, eager, rtol=0, atol=0)
