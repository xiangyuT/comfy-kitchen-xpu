import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.eager import quantization


def test_clear_nvfp4_lut_cache_selects_one_xpu_device(monkeypatch):
    cache = quantization.E2M1_LUT_CACHE
    original = cache.copy()
    calls = []
    try:
        cache.clear()
        cache[(torch.device("cpu"), torch.bfloat16)] = object()
        cache[(torch.device("xpu:0"), torch.bfloat16)] = object()
        cache[(torch.device("xpu:0"), torch.float16)] = object()
        cache[(torch.device("xpu:1"), torch.bfloat16)] = object()

        def synchronize(index):
            calls.append(index)
            assert (torch.device("xpu:0"), torch.bfloat16) in cache

        monkeypatch.setattr(torch.xpu, "synchronize", synchronize)
        assert ck.clear_nvfp4_lut_cache(0) == 2
        assert calls == [0]
        assert {device.type for device, _ in cache} == {"cpu", "xpu"}
        assert (torch.device("xpu:1"), torch.bfloat16) in cache
        assert ck.clear_nvfp4_lut_cache(0) == 0
        assert calls == [0]
    finally:
        cache.clear()
        cache.update(original)


@pytest.mark.parametrize("index", [-1, True, 0.5, "0"])
def test_clear_nvfp4_lut_cache_rejects_invalid_index(index):
    with pytest.raises(ValueError, match="device_index"):
        ck.clear_nvfp4_lut_cache(index)


@pytest.mark.xpu
def test_clear_nvfp4_lut_cache_rebuilds_on_xpu0():
    if not torch.xpu.is_available():
        pytest.skip("XPU is unavailable")

    device = torch.device("xpu:0")
    ck.clear_nvfp4_lut_cache(0)
    try:
        x = torch.randn((16, 16), device=device, dtype=torch.bfloat16)
        scale = torch.tensor(1.0, device=device, dtype=torch.float32)
        qx, block_scales = quantization.quantize_nvfp4(x, scale)
        before = quantization.dequantize_nvfp4(qx, scale, block_scales)
        assert (device, torch.bfloat16) in quantization.E2M1_LUT_CACHE

        assert ck.clear_nvfp4_lut_cache(0) == 1
        assert (device, torch.bfloat16) not in quantization.E2M1_LUT_CACHE
        after = quantization.dequantize_nvfp4(qx, scale, block_scales)
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    finally:
        ck.clear_nvfp4_lut_cache(0)
