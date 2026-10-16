"""Optional Triton XPU decode contracts, also executable with its CPU interpreter.

TRITON_INTERPRET=1 verifies the arithmetic/indexing without testing the XPU
compiler, stream behavior or performance. Ordinary CPU runs skip these kernels.
"""

import importlib
import os
from types import SimpleNamespace

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.xpu import decode as xpu_decode

int8_decode = importlib.import_module("comfy_kitchen.int8_decode")


@pytest.fixture
def decode_device():
    if os.getenv("TRITON_INTERPRET") == "1":
        return "cpu"
    if not xpu_decode.is_available(torch.device("xpu:0")):
        pytest.skip("XPU Triton decode unavailable")
    return "xpu:0"


def test_optional_decode_imports_without_cuda():
    assert not int8_decode.is_available(torch.device("cpu"))
    assert not ck.flash_attention_decode_gqa_is_available(torch.device("cpu"))
    assert not ck.gated_delta_deferred_is_available(torch.device("cpu"))


def test_xpu_decode_admission(monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 1)
    monkeypatch.setattr(xpu_decode.registry, "is_available", lambda name: name in ("triton", "eager"))
    assert xpu_decode.is_available("xpu:0")
    assert not xpu_decode.is_available("xpu:1")
    assert not xpu_decode.is_available("xpu:0", torch.float32)
    assert not xpu_decode.is_available(None)
    assert not xpu_decode.is_available("cpu")
    with ck.use_backend("eager"):
        assert not xpu_decode.is_available("xpu:0")
    monkeypatch.setattr(xpu_decode.registry, "get_backend_override", lambda: "xpu")
    assert not xpu_decode.is_available("xpu:0")
    monkeypatch.setattr(xpu_decode.registry, "get_backend_override", lambda: None)
    monkeypatch.setattr(xpu_decode.registry, "is_available", lambda _: False)
    assert not xpu_decode.is_available("xpu:0")


def _reference(q, k, v, lengths, *, causal):
    batch, heads, seq, dim = q.shape
    capacity, kv_heads = k.shape[2], k.shape[1]
    k = k.float().repeat_interleave(heads // kv_heads, dim=1)
    v = v.float().repeat_interleave(heads // kv_heads, dim=1)
    live = torch.arange(capacity, device=q.device) < lengths.long().view(-1, 1)
    k = k.masked_fill(~live[:, None, :, None], 0)
    v = v.masked_fill(~live[:, None, :, None], 0)
    scores = q.float() @ k.transpose(-1, -2) * dim ** -0.5
    limit = lengths.long().reshape(-1, 1, 1, 1).clamp(0, capacity)
    if causal:
        limit = limit - seq + torch.arange(seq, device=q.device).view(1, 1, seq, 1) + 1
    scores.masked_fill_(torch.arange(capacity, device=q.device) >= limit, -float("inf"))
    lse = scores.logsumexp(-1)
    probabilities = torch.exp(scores - lse.masked_fill(torch.isneginf(lse), 0).unsqueeze(-1))
    out = (probabilities @ v).transpose(1, 2).reshape(batch, seq, heads * dim).to(q.dtype)
    return out, lse


@pytest.mark.parametrize("dim", [128, 256])
def test_legacy_decode_head_dimensions(decode_device, dim):
    from comfy_kitchen.backends.triton.decode import decode

    torch.manual_seed(89)
    q = torch.randn(2, 1, 4, dim, device=decode_device, dtype=torch.bfloat16)
    k = torch.randn(2, 257, 2, dim, device=decode_device, dtype=q.dtype)
    v = torch.randn_like(k)
    lengths = torch.tensor([0, 257], device=decode_device, dtype=torch.int32)
    expected, _ = _reference(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), lengths, causal=False)
    if decode_device == "cpu":
        actual, _ = decode(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), lengths)
        actual = actual.view_as(q)
    else:
        actual = ck.flash_attention_decode(q, k, v, lengths)
    torch.testing.assert_close(actual, expected.view_as(q), rtol=0.01, atol=0.002)


@pytest.mark.parametrize("seq", [1, 4, 8])
@pytest.mark.parametrize("capacity", [257, 513])
def test_split_decode_staircase_and_empty_rows(decode_device, seq, capacity):
    from comfy_kitchen.backends.triton.decode import decode

    torch.manual_seed(97)
    q = torch.randn(2, 4, seq, 256, device=decode_device, dtype=torch.bfloat16)
    # Non-contiguous BHLD caches are common when views come from BLHD backing.
    k = torch.randn(2, capacity, 2, 256, device=decode_device, dtype=q.dtype).transpose(1, 2)
    v = torch.randn_like(k)
    for lengths in ([0, capacity], [seq - 1, capacity - 3]):
        lengths = torch.tensor(lengths, dtype=torch.int32, device=decode_device)
        out, lse = decode(q, k, v, lengths)
        expected, expected_lse = _reference(q, k, v, lengths, causal=True)
        torch.testing.assert_close(out, expected, rtol=0.01, atol=0.002)
        torch.testing.assert_close(lse, expected_lse, rtol=1e-5, atol=1e-5)
        if decode_device != "cpu":
            torch.testing.assert_close(ck.flash_attention_decode_gqa(q, k, v, lengths), out, rtol=0, atol=0)


@pytest.mark.parametrize("seq", [1, 4, 8])
def test_step_merge_matches_causal_attention_and_aliases_output(decode_device, seq):
    from comfy_kitchen.backends.triton.decode import step_merge

    torch.manual_seed(109)
    q = torch.randn(2, 4, seq, 256, device=decode_device, dtype=torch.bfloat16)
    k = torch.randn(2, 2, seq, 256, device=decode_device, dtype=q.dtype)
    v = torch.randn_like(k)
    prefix_k = torch.randn(2, 2, 257, 256, device=decode_device, dtype=q.dtype)
    prefix_v = torch.randn_like(prefix_k)
    # One batch has no committed prefix; both still include the current rows.
    prefix_lengths = torch.tensor([0, 257], dtype=torch.int32, device=decode_device)
    # Build a reference for prefix-only attention (no verification staircase).
    out, lse = _reference(q, prefix_k, prefix_v, prefix_lengths, causal=False)
    expected = importlib.import_module("comfy_kitchen.flash_attention")._step_merge_torch(
        out, lse, q, k, v, torch.empty_like(out),
    )
    got = step_merge(out, lse, q, k, v, out)
    torch.testing.assert_close(got, expected, rtol=0.01, atol=0.002)
    if decode_device != "cpu":
        prefix_out, prefix_lse = _reference(q, prefix_k, prefix_v, prefix_lengths, causal=False)
        actual = ck.flash_attention_decode_step_merge(prefix_out, prefix_lse, q, k, v, prefix_out)
        torch.testing.assert_close(actual, got, rtol=0, atol=0)


def _empty_cache(device, batch=2, heads=2, capacity=1536):
    if device != "cpu":
        return int8_decode.Int8DecodeCache(batch, heads, capacity, device)
    pages = (capacity + 511) // 512
    return SimpleNamespace(
        _capacity=capacity,
        key=torch.zeros(batch, pages, heads, 512, 256, dtype=torch.int8),
        value=torch.empty(batch, pages, heads, 256, 512, dtype=torch.int8),
        key_scale=torch.empty(batch, pages, heads, 512),
        value_scale=torch.empty(batch, pages, heads, 256),
    )


def _dequant_cache(cache):
    batch, pages, heads, page_size, dim = cache.key.shape
    key = cache.key.float() * cache.key_scale[..., None]
    value = cache.value.float().transpose(-1, -2) * cache.value_scale[..., None, :]
    key = key.permute(0, 2, 1, 3, 4).reshape(batch, heads, pages * page_size, dim)
    value = value.permute(0, 2, 1, 3, 4).reshape_as(key)
    return key, value


@pytest.mark.parametrize("seq", [1, 4])
def test_int8_pages_update_rollback_and_circular_sources(decode_device, seq):
    from comfy_kitchen.backends.triton.decode import decode, update_cache

    torch.manual_seed(113)
    cache = _empty_cache(decode_device)
    k = torch.randn(2, 2, 1536, 256, device=decode_device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q = torch.randn(2, 4, seq, 256, device=decode_device, dtype=k.dtype)
    # Cross two page boundaries, then reject a speculative suffix. A 1024-row
    # circular buffer retains both pages rewritten by incremental updates.
    for n, length in enumerate([0, 511, 513, 1023, 1030, 1025, 1024, 1536]):
        position = torch.tensor([length], dtype=torch.int64, device=decode_device)
        rows = 1536 if n == 0 else 1024
        source_k, source_v = k, v
        if n:
            source_k = torch.empty((2, 2, rows, 256), dtype=k.dtype, device=decode_device)
            source_v = torch.empty_like(source_k)
            begin = max(0, ((max(length - 1, 0) // 512) - 1) * 512)
            for row in range(begin, length):
                source_k[:, :, row % rows].copy_(k[:, :, row])
                source_v[:, :, row % rows].copy_(v[:, :, row])
        if decode_device == "cpu":
            update_cache(cache, source_k, source_v, position, initialize=n == 0)
            out, lse = decode(q, cache.key, cache.value, position,
                              key_scale=cache.key_scale, value_scale=cache.value_scale, capacity=cache._capacity)
        else:
            cache.update(source_k, source_v, position, initialize=n == 0)
            out, lse = cache.attend(q, position)
        key, value = _dequant_cache(cache)
        expected, expected_lse = _reference(q, key, value, position.expand(2), causal=False)
        torch.testing.assert_close(out, expected, rtol=0.01, atol=0.002)
        torch.testing.assert_close(lse, expected_lse, rtol=1e-5, atol=1e-5)
        reference, _ = _reference(q, k, v, position.expand(2), causal=False)
        torch.testing.assert_close(out, reference, rtol=0.03, atol=0.015)


def test_xpu_gqa_nondefault_stream_and_graph_lengths(decode_device):
    if decode_device == "cpu":
        pytest.skip("XPU stream/graph test; interpreter is arithmetic coverage only")
    torch.manual_seed(137)
    q = torch.randn(1, 4, 4, 256, dtype=torch.bfloat16, device=decode_device)
    k = torch.randn(1, 2, 513, 256, dtype=q.dtype, device=decode_device)
    v = torch.randn_like(k)
    lengths = torch.tensor([513], dtype=torch.int32, device=decode_device)
    torch.xpu.synchronize()
    stream = torch.xpu.Stream()
    with torch.xpu.stream(stream):
        actual = ck.flash_attention_decode_gqa(q, k, v, lengths)
    stream.synchronize()
    expected, _ = _reference(q, k, v, lengths, causal=True)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph, stream=stream):
        captured = ck.flash_attention_decode_gqa(q, k, v, lengths)
    for length in (513, 0, 257, 1):
        lengths.fill_(length)
        graph.replay()
        torch.xpu.synchronize()
        expected, _ = _reference(q, k, v, lengths, causal=True)
        torch.testing.assert_close(captured, expected, rtol=0.01, atol=0.002)


def test_xpu_int8_cache_graph_update_and_rollback(decode_device):
    if decode_device == "cpu":
        pytest.skip("XPU graph test; interpreter is arithmetic coverage only")
    torch.manual_seed(139)
    cache = _empty_cache(decode_device, batch=1, heads=1, capacity=1024)
    k = torch.randn(1, 1, 1024, 256, dtype=torch.bfloat16, device=decode_device)
    v = torch.randn_like(k)
    q = torch.randn(1, 2, 1, 256, dtype=k.dtype, device=decode_device)
    length = torch.tensor([513], dtype=torch.int64, device=decode_device)
    cache.update(k, v, length, initialize=True)
    cache.update(k, v, length)
    cache.attend(q, length)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        cache.update(k, v, length)
        out, lse = cache.attend(q, length)
    for value in (0, 511, 513, 1024, 1023):
        length.fill_(value)
        graph.replay()
        torch.xpu.synchronize()
        keys, values = _dequant_cache(cache)
        expected, expected_lse = _reference(q, keys, values, length, causal=False)
        torch.testing.assert_close(out, expected, rtol=0.01, atol=0.002)
        torch.testing.assert_close(lse, expected_lse, rtol=1e-5, atol=1e-5)
