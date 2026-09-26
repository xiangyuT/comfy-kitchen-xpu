"""Kitchen XPU FP16 Conv3D parity for the upstream CUDA unit contracts."""

import math

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from tests.conftest import fp16_accum_tol, rel_err


pytestmark = [
    pytest.mark.xpu,
    pytest.mark.skipif(
        not ck.list_backends()["xpu"]["available"],
        reason="Kitchen XPU backend is unavailable",
    ),
]


@pytest.mark.parametrize(
    "channels,outputs,frames,height,width,kernel_size,stride",
    [
        (128, 128, 5, 130, 130, (3, 3, 3), (1, 1, 1)),
        (256, 256, 5, 130, 130, (3, 3, 3), (1, 1, 1)),
        (128, 128, 9, 129, 129, (3, 3, 3), (1, 2, 2)),
        (128, 256, 5, 128, 128, (1, 1, 1), (1, 1, 1)),
    ],
)
@pytest.mark.parametrize("with_bias,with_residual", [
    (True, False), (False, False), (True, True),
])
def test_xpu_fp16_conv3d_matches_cuda_ut(
    channels, outputs, frames, height, width, kernel_size, stride,
    with_bias, with_residual, seed,
):
    x = torch.randn(1, channels, frames, height, width, device="xpu", dtype=torch.float16)
    x = x.contiguous(memory_format=torch.channels_last_3d)
    weight = torch.randn(outputs, channels, *kernel_size, device="xpu", dtype=torch.float16) * 0.02
    weight = weight.contiguous(memory_format=torch.channels_last_3d)
    bias = torch.randn(outputs, device="xpu", dtype=torch.float16) if with_bias else None
    residual = None
    if with_residual:
        shape = (1, outputs,
                 (frames - kernel_size[0]) // stride[0] + 1,
                 (height - kernel_size[1]) // stride[1] + 1,
                 (width - kernel_size[2]) // stride[2] + 1)
        residual = torch.randn(shape, device="xpu", dtype=torch.float16)
        residual = residual.contiguous(memory_format=torch.channels_last_3d)
    assert "fp16_conv3d" in ck.list_backends()["xpu"]["capabilities"]
    with ck.use_backend("xpu"):
        actual = ck.fp16_conv3d(x, weight, bias, residual, stride)
    expected = functional.conv3d(
        x.float(), weight.float(), None if bias is None else bias.float(),
        stride=stride,
    )
    if residual is not None:
        expected = expected + residual.float()
    torch.xpu.synchronize()
    assert actual.shape == expected.shape
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)
    assert rel_err(actual.float(), expected.float()) < fp16_accum_tol(
        channels * math.prod(kernel_size)
    )


def test_xpu_fp16_conv3d_pixel_channels_and_small_case(seed):
    x = torch.randn(1, 3, 5, 130, 130, device="xpu", dtype=torch.float16)
    weight = torch.randn(128, 3, 3, 3, 3, device="xpu", dtype=torch.float16) * 0.1
    with ck.use_backend("xpu"):
        actual = ck.fp16_conv3d(x, weight)
    expected = functional.conv3d(x.float(), weight.float())
    assert actual.shape == (1, 128, 3, 128, 128)
    assert rel_err(actual.float(), expected.float()) < fp16_accum_tol(3 * 27)


def test_xpu_fp16_conv3d_rejects_bad_stride():
    x = torch.zeros(1, 16, 4, 10, 10, device="xpu", dtype=torch.float16)
    weight = torch.zeros(16, 16, 3, 3, 3, device="xpu", dtype=torch.float16)
    with ck.use_backend("xpu"), pytest.raises(RuntimeError, match="stride"):
        ck.fp16_conv3d(x, weight, stride=(0, 1, 1))


def test_xpu_fp16_conv3d_rejects_bad_residual():
    x = torch.zeros(1, 16, 4, 10, 10, device="xpu", dtype=torch.float16)
    weight = torch.zeros(16, 16, 3, 3, 3, device="xpu", dtype=torch.float16)
    residual = torch.zeros(1, 8, 2, 8, 8, device="xpu", dtype=torch.float16)
    with ck.use_backend("xpu"), pytest.raises(RuntimeError, match="residual"):
        ck.fp16_conv3d(x, weight, residual=residual)


@pytest.mark.parametrize("channels,frames,height,width", [
    (512, 7, 18, 18),
    (64, 3, 6, 6),
])
def test_xpu_fp16_conv3d_deep_and_small_stages(
    channels, frames, height, width, seed,
):
    x = torch.randn(1, channels, frames, height, width,
                    device="xpu", dtype=torch.float16)
    weight = torch.randn(channels, channels, 3, 3, 3,
                         device="xpu", dtype=torch.float16) * 0.02
    residual = torch.randn(1, channels, frames - 2, height - 2, width - 2,
                           device="xpu", dtype=torch.float16)
    with ck.use_backend("xpu"):
        actual = ck.fp16_conv3d(x, weight, residual=residual)
    expected = functional.conv3d(x.float(), weight.float()) + residual.float()
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)
    assert rel_err(actual.float(), expected.float()) < fp16_accum_tol(channels * 27)


def test_xpu_fp16_conv3d_public_fullgraph(seed):
    x = torch.randn(1, 4, 4, 5, 5, device="xpu", dtype=torch.float16)
    weight = torch.randn(8, 4, 2, 2, 2, device="xpu", dtype=torch.float16)
    residual = torch.randn(1, 8, 3, 4, 4, device="xpu", dtype=torch.float16)

    def run(input, kernel, add):
        return ck.fp16_conv3d(input, kernel, residual=add)

    with ck.use_backend("xpu"):
        expected = run(x, weight, residual)
        actual = torch.compile(run, backend="eager", fullgraph=True)(
            x, weight, residual,
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
