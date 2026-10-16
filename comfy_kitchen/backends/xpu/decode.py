"""Optional Triton decode routes; no vLLM extension or serving ABI dependency."""

import torch

from comfy_kitchen.registry import registry


def is_available(device, dtype=torch.bfloat16):
    if device is None or isinstance(device, int):
        return False
    device = torch.device(device)
    if device.type != "xpu" or dtype != torch.bfloat16:
        return False
    if not registry.is_available("triton") or not torch.xpu.is_available():
        return False
    if registry.get_backend_override() not in (None, "triton"):
        return False
    return device.index is None or 0 <= device.index < torch.xpu.device_count()


def gqa(q, k, v, lengths):
    from comfy_kitchen.backends.triton.decode import decode

    _validate(q, k, v, lengths)
    return decode(q, k, v, lengths)[0]


def _validate(q, k, v, lengths=None):
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("decode requires q [B,H,S,D] and matching k/v [B,Hk,L,D]")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("decode batch and head dimensions must match")
    if q.shape[-1] not in (128, 256) or k.shape[1] == 0 or q.shape[1] % k.shape[1] or q.shape[2] == 0 or k.shape[2] == 0:
        raise ValueError("XPU decode requires D=128/256, nonempty cache/query and H divisible by Hk")
    operands = (k, v) if lengths is None else (k, v, lengths)
    if any(t.device != q.device for t in operands):
        raise ValueError("decode inputs must share one device")
    if k.dtype != q.dtype or v.dtype != q.dtype or q.dtype != torch.bfloat16:
        raise ValueError("XPU decode requires BF16 q/k/v")
    if lengths is not None and (lengths.dtype not in (torch.int32, torch.int64) or lengths.shape != (q.shape[0],) or not lengths.is_contiguous()):
        raise ValueError("decode lengths must be a contiguous int32/int64 [B] device tensor")
