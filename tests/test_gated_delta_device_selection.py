"""Keep Gated Delta's implicit device selection compatible with CUDA/HIP."""

from importlib import import_module
from types import SimpleNamespace

import torch


gated_delta = import_module("comfy_kitchen.gated_delta")


def test_implicit_device_does_not_select_xpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch, "xpu", SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
        raising=False,
    )

    def fail_if_xpu_loaded():
        raise AssertionError("implicit device probed XPU")

    monkeypatch.setattr(gated_delta, "_get_xpu_backend", fail_if_xpu_loaded)

    assert not gated_delta.is_available()
    assert not gated_delta.is_available(0)


def test_explicit_xpu_device_checks_index_and_backend(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch, "xpu", SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
        raising=False,
    )
    calls = []

    def backend_is_available(key_dim, value_dim):
        calls.append((key_dim, value_dim))
        return True

    backend = SimpleNamespace(gated_delta_decode_is_available=backend_is_available)
    monkeypatch.setattr(gated_delta, "_get_xpu_backend", lambda: backend)

    assert gated_delta.is_available(torch.device("xpu:0"), 128, 256)
    assert not gated_delta.is_available(torch.device("xpu:1"), 128, 256)
    assert calls == [(128, 256)]
