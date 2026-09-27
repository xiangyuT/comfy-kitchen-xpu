"""Kitchen XPU FP16 linear parity for the upstream CUDA unit contracts."""

import pytest
import torch

import comfy_kitchen as ck
from tests.conftest import fp16_accum_tol, rel_err


pytestmark = [
    pytest.mark.xpu,
    pytest.mark.skipif(
        not ck.list_backends()["xpu"]["available"],
        reason="Kitchen XPU backend is unavailable",
    ),
]


@pytest.mark.parametrize("m,n,k", [
    (37, 264, 0),
    (37, 264, 128),
    (512, 512, 512),
    (1797, 6144, 2048),
    (1797, 16384, 2048),
    (1797, 2048, 2048),
    (1797, 2048, 8192),
])
@pytest.mark.parametrize("with_bias", [True, False])
def test_xpu_fp16_linear_matches_cuda_ut(m, n, k, with_bias, seed):
    x = torch.randn(m, k, device="xpu", dtype=torch.float16)
    weight = torch.randn(n, k, device="xpu", dtype=torch.float16) * 0.02
    bias = torch.randn(n, device="xpu", dtype=torch.float16) if with_bias else None
    assert "fp16_linear" in ck.list_backends()["xpu"]["capabilities"]
    with ck.use_backend("xpu"):
        actual = ck.fp16_linear(x, weight, bias)
    expected = torch.nn.functional.linear(x, weight, bias)
    assert actual.shape == (m, n)
    assert rel_err(actual.float(), expected.float()) < fp16_accum_tol(k)


@pytest.mark.parametrize("m,n,k", [
    (37, 264, 128),
    (512, 512, 512),
    (1797, 2048, 2048),
    (1797, 2048, 8192),
])
def test_xpu_fp16_linear_scaled_residual_matches_cuda_ut(m, n, k, seed):
    x = torch.randn(m, k, device="xpu", dtype=torch.float16)
    weight = torch.randn(n, k, device="xpu", dtype=torch.float16) * 0.02
    bias = torch.randn(n, device="xpu", dtype=torch.float16)
    residual = torch.randn(m, n, device="xpu", dtype=torch.float16)
    scale = torch.randn(n, device="xpu", dtype=torch.float16)
    with ck.use_backend("xpu"):
        plain = ck.fp16_linear(x, weight, bias)
        actual = ck.fp16_linear(x, weight, bias, residual, scale)
    expected = torch.addcmul(residual, plain, scale)
    assert rel_err(actual.float(), expected.float()) < 1e-2


def test_xpu_fp16_linear_3d_input(seed):
    x = torch.randn(2, 512, 2048, device="xpu", dtype=torch.float16)
    weight = torch.randn(1024, 2048, device="xpu", dtype=torch.float16) * 0.02
    bias = torch.randn(1024, device="xpu", dtype=torch.float16)
    residual = torch.randn(2, 512, 1024, device="xpu", dtype=torch.float16)
    scale = torch.randn(1024, device="xpu", dtype=torch.float16)
    with ck.use_backend("xpu"):
        actual = ck.fp16_linear(x, weight, bias, residual, scale)
    expected = torch.addcmul(
        residual, torch.nn.functional.linear(x, weight, bias), scale,
    )
    assert actual.shape == expected.shape
    assert rel_err(actual.float(), expected.float()) < fp16_accum_tol(2048)


def test_xpu_fp16_linear_residual_requires_scale():
    x = torch.randn(4, 128, device="xpu", dtype=torch.float16)
    weight = torch.randn(64, 128, device="xpu", dtype=torch.float16)
    residual = torch.randn(4, 64, device="xpu", dtype=torch.float16)
    with ck.use_backend("xpu"), pytest.raises(RuntimeError, match="residual_scale"):
        ck.fp16_linear(x, weight, residual=residual)


@pytest.mark.parametrize("case", ["unaligned_k", "misaligned_input"])
def test_xpu_fp16_linear_unaligned_input_remains_native(case, seed):
    if case == "unaligned_k":
        x = torch.randn(64, 132, device="xpu", dtype=torch.float16)
    else:
        base = torch.randn(64 * 2048 + 4, device="xpu", dtype=torch.float16)
        x = base[4:4 + 64 * 2048].reshape(64, 2048)
        assert x.data_ptr() % 16 == 8
    weight = torch.randn(96, x.shape[-1], device="xpu", dtype=torch.float16) * 0.02
    with ck.use_backend("xpu"):
        actual = ck.fp16_linear(x, weight)
    expected = torch.nn.functional.linear(x, weight)
    assert rel_err(actual.float(), expected.float()) < fp16_accum_tol(x.shape[-1])


def test_xpu_fp16_linear_master_bias_and_misaligned_scale(seed):
    x = torch.randn(64, 2048, device="xpu", dtype=torch.float16)
    weight = torch.randn(512, 2048, device="xpu", dtype=torch.float16) * 0.02
    residual = torch.randn(64, 512, device="xpu", dtype=torch.float16)
    bias = torch.randn(512, device="xpu", dtype=torch.float32)
    scale = torch.randn(512, device="xpu", dtype=torch.float16)
    base = torch.empty(516, device="xpu", dtype=torch.float16)
    offset_scale = base[4:]
    offset_scale.copy_(scale)
    assert offset_scale.is_contiguous() and offset_scale.data_ptr() % 16 == 8
    with ck.use_backend("xpu"):
        expected = ck.fp16_linear(x, weight, bias.half(), residual, scale)
        actual = ck.fp16_linear(x, weight, bias, residual, offset_scale)
    assert torch.equal(actual, expected)


def test_xpu_fp16_linear_public_fullgraph(seed):
    x = torch.randn(8, 128, device="xpu", dtype=torch.float16)
    weight = torch.randn(64, 128, device="xpu", dtype=torch.float16)
    residual = torch.randn(8, 64, device="xpu", dtype=torch.float16)
    scale = torch.randn(64, device="xpu", dtype=torch.float16)

    def run(input, projection, add, add_scale):
        return ck.fp16_linear(input, projection, residual=add,
                              residual_scale=add_scale)

    with ck.use_backend("xpu"):
        expected = run(x, weight, residual, scale)
        actual = torch.compile(run, backend="eager", fullgraph=True)(
            x, weight, residual, scale,
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
