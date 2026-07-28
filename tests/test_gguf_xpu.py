import struct

import pytest
import torch

import comfy_kitchen as ck


def _xpu_gguf_available() -> bool:
    try:
        from comfy_kitchen.backends import xpu

        return bool(
            torch.xpu.is_available()
            and ck.list_backends()["xpu"]["available"]
            and xpu._GGUF_AVAILABLE
        )
    except (AttributeError, ImportError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(
    not _xpu_gguf_available(), reason="Kitchen Omni GGUF XPU backend is unavailable"
)

BLOCK_BYTES = {
    "q4_0": 18,
    "q4_1": 20,
    "q8_0": 34,
    "q4_k": 144,
    "q6_k": 210,
}


def _half_bytes(value: float) -> torch.Tensor:
    return torch.tensor(list(struct.pack("<e", value)), dtype=torch.uint8)


def _make_blocks(quant_type: str, count: int = 37) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260728)
    blocks = torch.randint(
        0,
        256,
        (count, BLOCK_BYTES[quant_type]),
        dtype=torch.uint8,
        generator=generator,
    )
    for index in range(count):
        if quant_type in {"q4_0", "q8_0"}:
            blocks[index, :2] = _half_bytes(0.125 * (1 + index % 4))
        elif quant_type == "q4_1":
            blocks[index, :2] = _half_bytes(0.125 * (1 + index % 4))
            blocks[index, 2:4] = _half_bytes(-0.25 * (1 + index % 4))
        elif quant_type == "q4_k":
            blocks[index, :2] = _half_bytes(0.03125 * (1 + index % 4))
            blocks[index, 2:4] = _half_bytes(0.015625 * (1 + index % 4))
        else:
            blocks[index, -2:] = _half_bytes(0.03125 * (1 + index % 4))
    return blocks.to("xpu")


def _direct_omni(data, quant_type, dtype, layout):
    from omni_xpu_kernel import gguf

    if quant_type == "q4_0" and layout == "comfyui":
        return gguf.dequantize_q4_0_comfyui(data, dtype)
    return getattr(gguf, f"dequantize_{quant_type}")(data, dtype)


@pytest.mark.parametrize("quant_type", ["q4_0", "q4_1", "q8_0", "q4_k", "q6_k"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_xpu_matches_eager_and_direct_omni(quant_type, dtype):
    data = _make_blocks(quant_type)
    with ck.use_backend("eager"):
        eager = ck.dequantize_gguf(
            data, quant_type, output_dtype=dtype, layout="comfyui"
        )
    with ck.use_backend("xpu"):
        xpu = ck.dequantize_gguf(
            data, quant_type, output_dtype=dtype, layout="comfyui"
        )
    direct = _direct_omni(data, quant_type, dtype, "comfyui")
    torch.xpu.synchronize()

    assert torch.equal(xpu.view(torch.uint8), direct.view(torch.uint8))
    # Omni computes K-quant intermediates at a different precision boundary
    # than the plugin-compatible eager path. Both routes must stay within two
    # output-dtype ULPs while the Kitchen adapter remains byte-identical to
    # direct Omni.
    torch.testing.assert_close(
        xpu,
        eager,
        rtol=2 * torch.finfo(dtype).eps,
        atol=0,
        equal_nan=True,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_xpu_q4_0_interleaved_layout(dtype):
    data = _make_blocks("q4_0")
    with ck.use_backend("xpu"):
        xpu = ck.dequantize_gguf(
            data, "q4_0", output_dtype=dtype, layout="interleaved"
        )
    direct = _direct_omni(data, "q4_0", dtype, "interleaved")
    assert torch.equal(xpu.view(torch.uint8), direct.view(torch.uint8))


def test_runtime_failure_falls_back_once_then_quarantines(monkeypatch):
    from comfy_kitchen.backends.xpu import gguf as xpu_gguf

    data = _make_blocks("q4_0")
    xpu_gguf._reset_gguf_quarantine_for_tests()
    ck.get_gguf_route_diagnostics(reset=True)
    attempts = 0

    def fail_native(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("injected GGUF failure")

    monkeypatch.setattr(xpu_gguf, "_call_native", fail_native)
    with (
        ck.use_backend("xpu"),
        pytest.warns(RuntimeWarning, match="quarantined"),
    ):
        first = ck.dequantize_gguf(data, "q4_0")
    with ck.use_backend("xpu"):
        second = ck.dequantize_gguf(data, "q4_0")
    with ck.use_backend("eager"):
        expected = ck.dequantize_gguf(data, "q4_0")

    assert attempts == 1
    assert torch.equal(first.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(second.view(torch.uint8), expected.view(torch.uint8))
    diagnostics = ck.get_gguf_route_diagnostics(reset=True)
    assert diagnostics["routes"] == {"eager": 3}
    assert sum(diagnostics["fallbacks"].values()) == 2
    xpu_gguf._reset_gguf_quarantine_for_tests()
