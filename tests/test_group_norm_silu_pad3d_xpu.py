"""Kitchen XPU parity for upstream GroupNorm/SiLU/Pad3D CUDA contracts."""

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.eager.group_norm_pad3d import group_norm_silu_pad3d as eager_ref
from tests.conftest import rel_err


pytestmark = [
    pytest.mark.xpu,
    pytest.mark.skipif(
        not ck.list_backends()["xpu"]["available"],
        reason="Kitchen XPU backend is unavailable",
    ),
]


@pytest.mark.parametrize(
    "channels,frames,height,width,pad",
    [
        (128, 3, 40, 56, (1, 1, 1, 1, 2)),
        (256, 2, 33, 17, (1, 1, 1, 1, 2)),
        (512, 2, 9, 9, (0, 1, 0, 1, 2)),
        (1024, 1, 16, 16, (1, 1, 1, 1, 0)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_xpu_group_norm_silu_pad3d_matches_cuda_ut(
    channels, frames, height, width, pad, dtype, seed,
):
    x = torch.randn(1, channels, frames, height, width, device="xpu", dtype=dtype) * 3 + 0.5
    x = x.contiguous(memory_format=torch.channels_last_3d)
    weight = torch.randn(channels, device="xpu", dtype=dtype) * 0.5 + 1
    bias = torch.randn(channels, device="xpu", dtype=dtype) * 0.2
    expected = eager_ref(x, weight, bias, 32, 1e-6, list(pad), True)
    assert "group_norm_silu_pad3d" in ck.list_backends()["xpu"]["capabilities"]
    with ck.use_backend("xpu"):
        actual = ck.group_norm_silu_pad3d(x, weight, bias, 32, 1e-6, pad, True)
    torch.xpu.synchronize()
    assert actual.shape == expected.shape
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)
    assert rel_err(actual.float(), expected.float()) < 5 * torch.finfo(dtype).eps
    if pad[4]:
        assert torch.all(actual[:, :, :pad[4]] == 0)


def test_xpu_group_norm_silu_pad3d_pad_only_is_exact(seed):
    x = torch.randn(1, 128, 3, 31, 31, device="xpu", dtype=torch.float16)
    pad = (0, 1, 0, 1, 2)
    with ck.use_backend("xpu"):
        actual = ck.group_norm_silu_pad3d(x, None, None, 1, 0.0, pad, False)
    expected = eager_ref(x, None, None, 1, 0.0, list(pad), False)
    assert torch.equal(actual, expected)


def test_xpu_group_norm_silu_pad3d_fp32_affine(seed):
    x = torch.randn(1, 128, 2, 12, 12, device="xpu", dtype=torch.float16)
    weight = torch.randn(128, device="xpu")
    bias = torch.randn(128, device="xpu")
    pad = (1, 1, 1, 1, 2)
    with ck.use_backend("xpu"):
        actual = ck.group_norm_silu_pad3d(x, weight, bias, 32, 1e-6, pad, True)
    expected = eager_ref(x, weight, bias, 32, 1e-6, list(pad), True)
    assert rel_err(actual.float(), expected.float()) < 5e-3


def test_xpu_group_norm_silu_pad3d_rejects_negative_pad():
    x = torch.zeros(1, 64, 3, 8, 8, device="xpu", dtype=torch.float16)
    with ck.use_backend("xpu"), pytest.raises(ValueError, match="non-negative"):
        ck.group_norm_silu_pad3d(x, None, None, 1, 0.0, (0, 0, 0, 0, -1), False)
