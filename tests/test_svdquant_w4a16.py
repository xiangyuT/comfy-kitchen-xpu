"""Backend-neutral contract tests for the Nunchaku SVDQuant W4A16 route."""

from __future__ import annotations

import pytest
import torch

import comfy_kitchen as ck

_GROUP_SIZE = 64


def _pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    low = values[:, 0::2].to(torch.int16) & 0x0F
    high = (values[:, 1::2].to(torch.int16) & 0x0F) << 4
    return (low | high).to(torch.int8).contiguous()


def _make_case(
    *,
    m: int = 5,
    n: int = 32,
    k: int = 128,
    rank: int = 8,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
):
    generator = torch.Generator(device=device).manual_seed(20260728)
    weight_int = torch.randint(
        -8,
        8,
        (n, k),
        dtype=torch.int8,
        device=device,
        generator=generator,
    )
    qweight = _pack_signed_int4(weight_int)
    wscales = (
        torch.rand(
            k // _GROUP_SIZE,
            n,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.05
        + 0.005
    ).to(dtype)
    smooth = (
        torch.rand(k, dtype=torch.float32, device=device, generator=generator)
        * 0.5
        + 0.75
    ).to(dtype)
    x = (
        torch.randn(
            m,
            k,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.25
    ).to(dtype)
    lora_down = (
        torch.randn(
            k,
            rank,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.02
    ).to(dtype)
    lora_up = (
        torch.randn(
            n,
            rank,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.02
    ).to(dtype)
    bias = (
        torch.randn(
            n,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        * 0.01
    ).to(dtype)
    return {
        "weight_int": weight_int,
        "qweight": qweight,
        "wscales": wscales,
        "smooth": smooth,
        "x": x,
        "lora_down": lora_down,
        "lora_up": lora_up,
        "bias": bias,
    }


def _independent_reference(
    case: dict,
    prepared: ck.PreparedSVDQuantW4A16,
    *,
    lora: bool,
    bias: bool,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    x = case["x"]
    rcp = prepared.rcp_smooth_f16
    x_gemm = (
        x.to(torch.float16)
        if rcp is None
        else (x.float() * rcp.float()).to(torch.float16)
    )
    x_gemm = torch.nan_to_num(
        x_gemm,
        nan=0.0,
        posinf=65504.0,
        neginf=-65504.0,
    )
    n, k = case["weight_int"].shape
    groups = k // _GROUP_SIZE
    weight = (
        case["weight_int"].float().reshape(n, groups, _GROUP_SIZE)
        * prepared.scales_f16.float().t().unsqueeze(-1)
    ).reshape(n, k)
    main = x_gemm.float() @ weight.t()

    if lora or bias:
        dst = torch.zeros(
            x.shape[0],
            n,
            dtype=torch.bfloat16,
            device=x.device,
        )
        if lora:
            lora_result = (
                x.to(torch.bfloat16)
                @ case["lora_down"].to(torch.bfloat16)
            ) @ case["lora_up"].to(torch.bfloat16).t()
            dst.add_(lora_result)
        if bias:
            dst.add_(case["bias"].to(torch.bfloat16))
        result = (dst.float() + main).to(torch.bfloat16)
    else:
        result = main.to(torch.float16)
    return result.to(output_dtype)


def test_nondestructive_preparation_preserves_source_and_layout():
    case = _make_case()
    original = case["qweight"].clone()

    prepared = ck.prepare_svdquant_w4a16_for_xpu(
        case["qweight"],
        case["wscales"],
        case["smooth"],
        destructive=False,
    )

    assert torch.equal(case["qweight"], original)
    assert prepared.packed_u4.data_ptr() != case["qweight"].data_ptr()
    assert torch.equal(
        prepared.packed_u4,
        original.view(torch.uint8) ^ 0x88,
    )
    assert prepared.scales_f16.dtype == torch.float16
    assert prepared.rcp_smooth_f16 is not None
    assert prepared.rcp_smooth_f16.dtype == torch.float16
    assert prepared.in_features == case["x"].shape[1]
    assert prepared.out_features == case["weight_int"].shape[0]


def test_destructive_preparation_reuses_full_weight_storage_and_restores():
    case = _make_case()
    qweight = case["qweight"]
    original = qweight.clone()
    original_ptr = qweight.data_ptr()

    prepared = ck.prepare_svdquant_w4a16_for_xpu(
        qweight,
        case["wscales"],
        case["smooth"],
    )

    assert prepared.destructive
    assert prepared.packed_u4.data_ptr() == original_ptr
    assert prepared.source_qweight.data_ptr() == original_ptr
    assert torch.equal(prepared.packed_u4, original.view(torch.uint8) ^ 0x88)
    with pytest.raises(RuntimeError, match="already destructively prepared"):
        ck.prepare_svdquant_w4a16_for_xpu(
            qweight,
            case["wscales"],
            case["smooth"],
        )

    restored = ck.restore_svdquant_w4a16_source_(prepared)
    assert restored.data_ptr() == original_ptr
    assert torch.equal(restored, original)
    with pytest.raises(RuntimeError, match="already restored"):
        ck.restore_svdquant_w4a16_source_(prepared)
    with pytest.raises(RuntimeError, match="was restored"):
        ck.svdquant_w4a16_linear(case["x"], prepared)


@pytest.mark.parametrize(
    ("with_smooth", "with_lora", "with_bias", "output_dtype"),
    [
        (False, False, False, torch.float16),
        (True, False, False, torch.bfloat16),
        (True, True, False, torch.bfloat16),
        (True, False, True, torch.float16),
        (True, True, True, torch.bfloat16),
    ],
)
def test_eager_matches_independent_operator_boundary(
    with_smooth,
    with_lora,
    with_bias,
    output_dtype,
):
    case = _make_case()
    prepared = ck.prepare_svdquant_w4a16_for_xpu(
        case["qweight"],
        case["wscales"],
        case["smooth"] if with_smooth else None,
        destructive=False,
    )
    kwargs = {
        "lora_down": case["lora_down"] if with_lora else None,
        "lora_up": case["lora_up"] if with_lora else None,
        "bias": case["bias"] if with_bias else None,
        "output_dtype": output_dtype,
    }

    with ck.use_backend("eager"):
        actual = ck.svdquant_w4a16_linear(case["x"], prepared, **kwargs)
    expected = _independent_reference(
        case,
        prepared,
        lora=with_lora,
        bias=with_bias,
        output_dtype=output_dtype,
    )

    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        (
            lambda case: case["qweight"].view(torch.uint8).reshape(16, -1),
            ValueError,
            "wscales shape",
        ),
        (
            lambda case: case["qweight"].float(),
            ValueError,
            "natural-layout",
        ),
        (
            lambda case: case["qweight"].t().contiguous().t(),
            ValueError,
            "destructive preparation requires contiguous",
        ),
    ],
)
def test_preparation_rejects_invalid_qweight(mutation, error, message):
    case = _make_case()
    with pytest.raises(error, match=message):
        ck.prepare_svdquant_w4a16_for_xpu(
            mutation(case),
            case["wscales"],
            case["smooth"],
        )


@pytest.mark.parametrize("bad_value", [0.0, float("nan"), float("inf")])
def test_preparation_rejects_invalid_smooth(bad_value):
    case = _make_case()
    case["smooth"][0] = bad_value
    with pytest.raises(ValueError, match="finite non-zero"):
        ck.prepare_svdquant_w4a16_for_xpu(
            case["qweight"],
            case["wscales"],
            case["smooth"],
            destructive=False,
        )


def test_linear_validates_lora_pair_and_shapes():
    case = _make_case()
    prepared = ck.prepare_svdquant_w4a16_for_xpu(
        case["qweight"],
        case["wscales"],
        destructive=False,
    )
    with pytest.raises(ValueError, match="both present"):
        ck.svdquant_w4a16_linear(
            case["x"],
            prepared,
            lora_down=case["lora_down"],
        )
    with pytest.raises(ValueError, match="LoRA shapes"):
        ck.svdquant_w4a16_linear(
            case["x"],
            prepared,
            lora_down=case["lora_down"][:-1],
            lora_up=case["lora_up"],
        )


def test_fake_tensor_preserves_shape_and_dtype():
    case = _make_case()
    prepared = ck.prepare_svdquant_w4a16_for_xpu(
        case["qweight"],
        case["wscales"],
        destructive=False,
    )
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(allow_non_fake_inputs=True):
        output = ck.svdquant_w4a16_linear(
            case["x"],
            prepared,
            output_dtype=torch.bfloat16,
        )
    assert output.shape == (
        case["x"].shape[0],
        case["weight_int"].shape[0],
    )
    assert output.dtype == torch.bfloat16


def test_eager_capability_and_route_diagnostics():
    case = _make_case()
    prepared = ck.prepare_svdquant_w4a16_for_xpu(
        case["qweight"],
        case["wscales"],
        destructive=False,
    )
    assert "svdquant_w4a16_linear" in ck.list_backends()["eager"]["capabilities"]

    ck.get_svdquant_w4a16_route_diagnostics(reset=True)
    with ck.use_backend("eager"):
        ck.svdquant_w4a16_linear(case["x"], prepared)
    assert ck.get_svdquant_w4a16_route_diagnostics(reset=True) == {
        "routes": {"eager": 1},
        "fallbacks": {},
    }
