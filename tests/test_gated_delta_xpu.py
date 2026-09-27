"""Run the upstream CUDA Gated Delta numerical contracts through Kitchen XPU."""

import pytest
import torch

import comfy_kitchen as ck
from tests.conftest import rel_err
from tests.test_gated_delta_fused import (
    B, C, DK, DV, EPS, HD, HK, HV, KEY_DIM, KS, SCALE,
    _conv_ref, _decode_ref,
)


pytestmark = [
    pytest.mark.xpu,
    pytest.mark.skipif(
        not ck.list_backends()["xpu"]["available"],
        reason="Kitchen XPU backend is unavailable",
    ),
]


@pytest.mark.parametrize("steps", [1, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_xpu_deltanet_conv_step_matches_cuda_ut(steps, dtype, seed):
    assert ck.gated_delta_decode_is_available(torch.device("xpu"), DK, DV)
    proj = torch.randn(B, steps, C, device="xpu", dtype=dtype)
    state = torch.randn(B, C, KS - 1, device="xpu", dtype=dtype)
    weight = torch.randn(C, 1, KS, device="xpu", dtype=dtype) * 0.5
    bias = torch.randn(C, device="xpu", dtype=dtype) * 0.1
    expected, expected_state, expected_snapshots = _conv_ref(
        proj, state, weight, bias, steps,
    )

    actual_state = state.clone()
    snapshots = (
        torch.empty(steps - 1, B, C, KS - 1, device="xpu", dtype=dtype)
        if steps > 1 else None
    )
    actual = ck.deltanet_conv_step(proj, actual_state, weight, bias, snapshots)
    torch.xpu.synchronize()

    assert rel_err(actual, expected) < (1e-5 if dtype == torch.float32 else 1e-2)
    assert torch.equal(actual_state, expected_state)
    if steps > 1:
        assert torch.equal(snapshots, expected_snapshots)


@pytest.mark.parametrize("steps", [1, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_xpu_gated_delta_decode_matches_cuda_ut(steps, dtype, seed):
    assert ck.gated_delta_decode_is_available(torch.device("xpu"), DK, DV)
    mixed_qkv = torch.randn(B, C, steps, device="xpu", dtype=dtype)
    x = torch.randn(B, steps, HD, device="xpu", dtype=dtype)
    w_a = torch.randn(HV, HD, device="xpu", dtype=dtype) * 0.05
    w_b = torch.randn(HV, HD, device="xpu", dtype=dtype) * 0.05
    dt_bias = torch.randn(HV, device="xpu")
    g_decay = -torch.rand(HV, device="xpu") - 0.5
    state = torch.randn(B, HV, DK, DV, device="xpu") * 0.1
    z = torch.randn(B, steps, HV * DV, device="xpu", dtype=dtype)
    norm_weight = torch.rand(DV, device="xpu", dtype=dtype) + 0.5
    expected_state = state.clone()
    expected, expected_snapshots = _decode_ref(
        mixed_qkv, x, w_a, w_b, dt_bias, g_decay,
        expected_state, z, norm_weight, steps,
    )

    actual_state = state.clone()
    snapshots = (
        torch.empty(steps - 1, B, HV, DK, DV, device="xpu")
        if steps > 1 else None
    )
    actual = ck.gated_delta_decode_fused(
        mixed_qkv, x, w_a, w_b, dt_bias, g_decay, actual_state,
        KEY_DIM, HK, SCALE, z, norm_weight, EPS, snapshots,
    )
    torch.xpu.synchronize()

    tolerance = 1e-5 if dtype == torch.float32 else 5e-3
    assert actual.shape == (B, steps, HV, DV)
    assert rel_err(actual.float(), expected.float()) < tolerance
    assert rel_err(actual_state, expected_state) < tolerance
    if steps > 1:
        assert rel_err(snapshots, expected_snapshots) < tolerance


def test_xpu_gated_delta_decode_rejects_long_sequence():
    steps = 9
    assert ck.gated_delta_decode_is_available(torch.device("xpu"), DK, DV)
    mixed_qkv = torch.zeros(B, C, steps, device="xpu", dtype=torch.bfloat16)
    x = torch.zeros(B, steps, HD, device="xpu", dtype=torch.bfloat16)
    weight = torch.zeros(HV, HD, device="xpu", dtype=torch.bfloat16)
    state = torch.zeros(B, HV, DK, DV, device="xpu")
    z = torch.zeros(B, steps, HV * DV, device="xpu", dtype=torch.bfloat16)
    norm_weight = torch.ones(DV, device="xpu", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="1<=S<=8"):
        ck.gated_delta_decode_fused(
            mixed_qkv, x, weight, weight,
            torch.zeros(HV, device="xpu"), -torch.ones(HV, device="xpu"),
            state, KEY_DIM, HK, SCALE, z, norm_weight, EPS,
        )
