"""Kitchen INT8 RMSNorm and residual CUDA contracts through native XPU routes."""

import pytest
import torch
from torch.nn import functional

import comfy_kitchen as ck
from tests.conftest import rel_err


pytestmark = [
    pytest.mark.xpu,
    pytest.mark.skipif(
        not ck.list_backends()["xpu"]["available"],
        reason="Kitchen XPU backend is unavailable",
    ),
]


@pytest.mark.parametrize(
    "rows,features,outputs,convrot",
    [
        (1024, 4096, 512, True),
        (1024, 4096, 512, False),
        (64, 18432, 256, True),
        (1, 4096, 256, True),
    ],
)
def test_xpu_int8_rms_norm_matches_cuda_ut(rows, features, outputs, convrot, seed):
    from omni_xpu_kernel import kitchen

    assert kitchen.supports_rms_norm_for_int8()
    x = torch.randn(rows, features, device="xpu", dtype=torch.bfloat16)
    norm_weight = torch.randn(features, device="xpu", dtype=torch.bfloat16)
    weight = torch.randint(-127, 127, (outputs, features), device="xpu", dtype=torch.int8)
    weight_scale = torch.tensor(0.01, device="xpu", dtype=torch.float32)
    kwargs = {"out_dtype": torch.bfloat16, "convrot": convrot,
              "convrot_groupsize": 256}
    with ck.use_backend("xpu"):
        expected = ck.int8_linear(
            functional.rms_norm(x, (features,), norm_weight, 1e-5),
            weight, weight_scale, **kwargs,
        )
        actual = ck.int8_linear(
            x, weight, weight_scale, input_act="rms_norm",
            input_act_weight=norm_weight, input_act_eps=1e-5, **kwargs,
        )
    assert actual.shape == (rows, outputs)
    assert rel_err(actual, expected) < 0.05


@pytest.mark.parametrize("rows,features,outputs", [
    (1797, 2048, 2048), (512, 4096, 512), (37, 2048, 256),
])
@pytest.mark.parametrize("with_bias", [True, False])
def test_xpu_int8_scaled_residual_matches_cuda_ut(
    rows, features, outputs, with_bias, seed,
):
    from omni_xpu_kernel import kitchen

    assert kitchen.supports_scaled_residual()
    x = torch.randn(rows, features, device="xpu", dtype=torch.float16)
    weight = torch.randint(-127, 127, (outputs, features), device="xpu", dtype=torch.int8)
    weight_scale = torch.tensor(0.01, device="xpu", dtype=torch.float32)
    bias = torch.randn(outputs, device="xpu", dtype=torch.float16) if with_bias else None
    residual = torch.randn(rows, outputs, device="xpu", dtype=torch.float16)
    scale = torch.randn(outputs, device="xpu", dtype=torch.float16)
    kwargs = {"out_dtype": torch.float16, "convrot": True,
              "convrot_groupsize": 256}
    with ck.use_backend("xpu"):
        plain = ck.int8_linear(x, weight, weight_scale, bias, **kwargs)
        actual = ck.int8_linear(
            x, weight, weight_scale, bias,
            residual=residual, residual_scale=scale, **kwargs,
        )
    expected = torch.addcmul(residual, plain, scale)
    assert actual.shape == expected.shape
    assert rel_err(actual, expected) < 1e-2


def test_xpu_int8_rms_norm_requires_weight():
    x = torch.randn(4, 256, device="xpu", dtype=torch.bfloat16)
    weight = torch.randint(-127, 127, (64, 256), device="xpu", dtype=torch.int8)
    scale = torch.tensor(0.01, device="xpu")
    with ck.use_backend("xpu"), pytest.raises(ValueError, match="weight"):
        ck.int8_linear(x, weight, scale, input_act="rms_norm")


def test_xpu_int8_residual_requires_scale():
    x = torch.randn(4, 256, device="xpu", dtype=torch.float16)
    weight = torch.randint(-127, 127, (64, 256), device="xpu", dtype=torch.int8)
    scale = torch.tensor(0.01, device="xpu")
    residual = torch.randn(4, 64, device="xpu", dtype=torch.float16)
    with ck.use_backend("xpu"), pytest.raises(ValueError, match="residual_scale"):
        ck.int8_linear(x, weight, scale, residual=residual)
