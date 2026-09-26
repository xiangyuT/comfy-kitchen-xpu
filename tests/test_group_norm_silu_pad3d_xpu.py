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


@pytest.mark.parametrize("channels", [128, 96])
def test_xpu_group_norm_silu_pad3d_fp32_affine(channels, seed):
    x = torch.randn(1, channels, 2, 12, 12, device="xpu", dtype=torch.float16)
    weight = torch.randn(channels, device="xpu")
    bias = torch.randn(channels, device="xpu")
    pad = (1, 1, 1, 1, 2)
    with ck.use_backend("xpu"):
        actual = ck.group_norm_silu_pad3d(x, weight, bias, 32, 1e-6, pad, True)
    expected = eager_ref(x, weight, bias, 32, 1e-6, list(pad), True)
    assert rel_err(actual.float(), expected.float()) < 5e-3


def test_xpu_group_norm_silu_pad3d_rejects_negative_pad():
    x = torch.zeros(1, 64, 3, 8, 8, device="xpu", dtype=torch.float16)
    with ck.use_backend("xpu"), pytest.raises(ValueError, match="non-negative"):
        ck.group_norm_silu_pad3d(x, None, None, 1, 0.0, (0, 0, 0, 0, -1), False)


def test_xpu_group_norm_without_silu(seed):
    x = torch.randn(1, 256, 2, 20, 20, device="xpu", dtype=torch.float16)
    weight = torch.randn(256, device="xpu", dtype=torch.float16)
    bias = torch.randn(256, device="xpu", dtype=torch.float16)
    pad = (1, 1, 1, 1, 0)
    with ck.use_backend("xpu"):
        actual = ck.group_norm_silu_pad3d(x, weight, bias, 32, 1e-6, pad, False)
    expected = eager_ref(x, weight, bias, 32, 1e-6, list(pad), False)
    assert rel_err(actual.float(), expected.float()) < 5e-3


@pytest.mark.parametrize("layout", ["ncdhw", "misaligned_channels_last"])
def test_xpu_group_norm_input_layouts(layout, seed):
    if layout == "ncdhw":
        x = torch.randn(1, 128, 2, 12, 12, device="xpu", dtype=torch.float16)
    else:
        count = 128 * 2 * 12 * 12
        base = torch.randn(count + 4, device="xpu", dtype=torch.float16)
        x = base[4:4 + count].view(1, 2, 12, 12, 128).permute(0, 4, 1, 2, 3)
        assert x.data_ptr() % 16 == 8
    weight = (torch.ones(128, device="xpu", dtype=torch.float16)
              if layout == "ncdhw" else None)
    bias = (torch.zeros(128, device="xpu", dtype=torch.float16)
            if layout == "ncdhw" else None)
    pad = (1, 1, 1, 1, 2)
    with ck.use_backend("xpu"):
        actual = ck.group_norm_silu_pad3d(x, weight, bias, 32, 1e-6, pad, True)
    expected = eager_ref(x, weight, bias, 32, 1e-6, list(pad), True)
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)
    assert rel_err(actual.float(), expected.float()) < 5e-3


def test_xpu_group_norm_three_channel_pad_only(seed):
    x = torch.randn(1, 3, 2, 12, 12, device="xpu", dtype=torch.float16)
    pad = (1, 1, 1, 1, 2)
    with ck.use_backend("xpu"):
        actual = ck.group_norm_silu_pad3d(x, None, None, 1, 0.0, pad, False)
    expected = eager_ref(x, None, None, 1, 0.0, list(pad), False)
    assert torch.equal(actual, expected)


def test_xpu_group_norm_cuda_limit_shapes_remain_supported(seed):
    x = torch.randn(1, 2048, 1, 4, 4, device="xpu", dtype=torch.float16)
    weight = torch.ones(2048, device="xpu", dtype=torch.float16)
    with ck.use_backend("xpu"):
        actual = ck.group_norm_silu_pad3d(
            x, weight, None, 2048, 1e-6, (0, 0, 0, 0, 0), False,
        )
    expected = torch.nn.functional.group_norm(x, 2048, weight, None, 1e-6)
    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)

    many_frames = torch.randn(1, 8, 65536, 1, 1, device="xpu", dtype=torch.float16)
    with ck.use_backend("xpu"):
        padded = ck.group_norm_silu_pad3d(
            many_frames, None, None, 1, 0.0, (0, 0, 0, 0, 1), False,
        )
    assert padded.shape == (1, 8, 65537, 1, 1)
    assert torch.count_nonzero(padded[:, :, :1]) == 0
    assert torch.equal(padded[:, :, 1:], many_frames)


def test_xpu_group_norm_public_fullgraph(seed):
    x = torch.randn(1, 32, 2, 4, 4, device="xpu", dtype=torch.float16)
    weight = torch.randn(32, device="xpu", dtype=torch.float16)

    def run(input, affine):
        return ck.group_norm_silu_pad3d(
            input, affine, None, 8, 1e-6, (1, 1, 1, 1, 1), False,
        )

    with ck.use_backend("xpu"):
        expected = run(x, weight)
        actual = torch.compile(run, backend="eager", fullgraph=True)(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
