"""BMG parity tests for the Nunchaku SVDQuant W4A16 Kitchen boundary."""

from __future__ import annotations

import pytest
import torch

import comfy_kitchen as ck

from .test_svdquant_w4a16 import _make_case


def _xpu_w4a16_available() -> bool:
    try:
        from comfy_kitchen.backends import xpu

        return bool(
            torch.xpu.is_available()
            and ck.list_backends()["xpu"]["available"]
            and xpu._SVDQ_W4A16_AVAILABLE
        )
    except (AttributeError, ImportError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(
    not _xpu_w4a16_available(),
    reason="Kitchen Omni SVDQuant W4A16 XPU backend is unavailable",
)


def _direct_nunchaku_xpu(
    case: dict,
    *,
    with_smooth: bool,
    with_lora: bool,
    with_bias: bool,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    from omni_xpu_kernel import svdq

    x = case["x"].contiguous()
    if with_smooth:
        rcp_smooth = (1.0 / case["smooth"].float()).to(torch.float16)
        x_gemm = svdq.fused_smooth_mul_convert(x, rcp_smooth)
        x_gemm.nan_to_num_(
            nan=0.0,
            posinf=65504.0,
            neginf=-65504.0,
        )
    else:
        x_gemm = x.to(torch.float16)
    packed_u4, scales_f16 = svdq.prepare_onednn_weights(
        case["qweight"].view(torch.uint8),
        case["wscales"],
    )

    if with_lora or with_bias:
        dst = torch.zeros(
            x.shape[0],
            case["weight_int"].shape[0],
            dtype=torch.bfloat16,
            device=x.device,
        )
        if with_lora:
            lora = (
                x.to(torch.bfloat16)
                @ case["lora_down"].to(torch.bfloat16)
            ) @ case["lora_up"].to(torch.bfloat16).t()
            dst.add_(lora)
        if with_bias:
            dst.add_(case["bias"].to(torch.bfloat16))
        svdq.onednn_int4_gemm_add_to_output(
            x_gemm,
            packed_u4,
            scales_f16,
            dst,
        )
        result = dst
    else:
        result = svdq.onednn_int4_gemm_preconverted(
            x_gemm,
            packed_u4,
            scales_f16,
        )
    return result.to(output_dtype)


@pytest.mark.parametrize(
    ("m", "with_smooth", "with_lora", "with_bias", "output_dtype"),
    [
        (1, False, False, False, torch.float16),
        (31, True, False, False, torch.bfloat16),
        (31, True, True, False, torch.bfloat16),
        (126, True, False, True, torch.float16),
        (126, True, True, True, torch.bfloat16),
    ],
)
def test_xpu_is_byte_identical_to_current_nunchaku_route(
    m,
    with_smooth,
    with_lora,
    with_bias,
    output_dtype,
):
    case = _make_case(
        m=m,
        n=64,
        k=128,
        device="xpu",
    )
    direct = _direct_nunchaku_xpu(
        case,
        with_smooth=with_smooth,
        with_lora=with_lora,
        with_bias=with_bias,
        output_dtype=output_dtype,
    )
    prepared = ck.prepare_svdquant_w4a16_for_xpu(
        case["qweight"],
        case["wscales"],
        case["smooth"] if with_smooth else None,
        destructive=False,
    )
    with ck.use_backend("xpu"):
        actual = ck.svdquant_w4a16_linear(
            case["x"],
            prepared,
            lora_down=case["lora_down"] if with_lora else None,
            lora_up=case["lora_up"] if with_lora else None,
            bias=case["bias"] if with_bias else None,
            output_dtype=output_dtype,
        )
    torch.xpu.synchronize()

    assert torch.equal(actual.view(torch.uint8), direct.view(torch.uint8))


def test_destructive_preparation_keeps_one_full_weight_storage_on_xpu():
    case = _make_case(n=64, k=128, device="xpu")
    source_ptr = case["qweight"].data_ptr()
    prepared = ck.prepare_svdquant_w4a16_for_xpu(
        case["qweight"],
        case["wscales"],
        case["smooth"],
    )

    assert prepared.packed_u4.data_ptr() == source_ptr
    assert prepared.source_qweight.data_ptr() == source_ptr
    with ck.use_backend("xpu"):
        output = ck.svdquant_w4a16_linear(case["x"], prepared)
    assert output.shape == (case["x"].shape[0], 64)


def test_prepared_auto_route_uses_cached_xpu_implementation():
    case = _make_case(n=64, k=128, device="xpu")
    prepared = ck.prepare_svdquant_w4a16_for_xpu(
        case["qweight"],
        case["wscales"],
        case["smooth"],
        destructive=False,
    )
    assert prepared.xpu_linear_impl is not None

    ck.get_svdquant_w4a16_route_diagnostics(reset=True)
    actual = ck.svdquant_w4a16_linear(case["x"], prepared)
    with ck.use_backend("xpu"):
        dispatched = ck.svdquant_w4a16_linear(case["x"], prepared)
    assert torch.equal(actual.view(torch.uint8), dispatched.view(torch.uint8))
    assert ck.get_svdquant_w4a16_route_diagnostics(reset=True) == {
        "routes": {"xpu": 2},
        "fallbacks": {},
    }


def test_explicit_eager_override_bypasses_cached_xpu_implementation():
    case = _make_case(n=64, k=128, device="xpu")
    prepared = ck.prepare_svdquant_w4a16_for_xpu(
        case["qweight"],
        case["wscales"],
        case["smooth"],
        destructive=False,
    )

    ck.get_svdquant_w4a16_route_diagnostics(reset=True)
    with ck.use_backend("eager"):
        output = ck.svdquant_w4a16_linear(case["x"], prepared)
    assert output.shape == (case["x"].shape[0], 64)
    assert ck.get_svdquant_w4a16_route_diagnostics(reset=True) == {
        "routes": {"eager": 1},
        "fallbacks": {},
    }


def test_runtime_failure_falls_back_once_then_quarantines(monkeypatch):
    from comfy_kitchen.backends.xpu import svdquant_w4a16 as xpu_w4a16

    case = _make_case(n=64, k=128, device="xpu")
    prepared = ck.prepare_svdquant_w4a16_for_xpu(
        case["qweight"],
        case["wscales"],
        case["smooth"],
        destructive=False,
    )
    xpu_w4a16._reset_svdquant_w4a16_quarantine_for_tests()
    ck.get_svdquant_w4a16_route_diagnostics(reset=True)
    attempts = 0

    def fail_native(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("injected W4A16 failure")

    monkeypatch.setattr(xpu_w4a16, "_call_native", fail_native)
    with (
        ck.use_backend("xpu"),
        pytest.warns(RuntimeWarning, match="quarantined"),
    ):
        first = ck.svdquant_w4a16_linear(case["x"], prepared)
    with ck.use_backend("xpu"):
        second = ck.svdquant_w4a16_linear(case["x"], prepared)
    with ck.use_backend("eager"):
        expected = ck.svdquant_w4a16_linear(case["x"], prepared)

    assert attempts == 1
    assert torch.equal(first.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(second.view(torch.uint8), expected.view(torch.uint8))
    diagnostics = ck.get_svdquant_w4a16_route_diagnostics(reset=True)
    assert diagnostics["routes"] == {"eager": 3}
    assert sum(diagnostics["fallbacks"].values()) == 2
    xpu_w4a16._reset_svdquant_w4a16_quarantine_for_tests()
