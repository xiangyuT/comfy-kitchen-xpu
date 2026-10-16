# SPDX-License-Identifier: Apache-2.0
"""Bounded-workspace decode for XPU, with device-resident cache lengths.

Split KV into independent ranges and combine their normalized outputs with
log-sum-exp weights, as in vLLM's paged decode. Unlike the serving ABI this
adapter reads Kitchen's BHLD cache and its bottom-right verify staircase.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial_decode(
    Q, K, V, Length, KS, VS, Partial, LSE,
    H: tl.constexpr, HK: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    Capacity: tl.constexpr, Splits: tl.constexpr, Span: tl.constexpr,
    QB: tl.constexpr, QH: tl.constexpr, QS: tl.constexpr, QD: tl.constexpr,
    KB: tl.constexpr, KH: tl.constexpr, KL: tl.constexpr, KD: tl.constexpr,
    VB: tl.constexpr, VH: tl.constexpr, VL: tl.constexpr, VD: tl.constexpr,
    Paged: tl.constexpr, Pages: tl.constexpr, PageSize: tl.constexpr,
    Block: tl.constexpr,
):
    row, split = tl.program_id(0), tl.program_id(1)
    token, head, batch = row % S, row // S % H, row // (S * H)
    kv_head = head // (H // HK)
    dim = tl.arange(0, D)
    query = tl.load(Q + batch * QB + head * QH + token * QS + dim * QD).to(tl.float32)
    length = tl.minimum(tl.maximum(tl.load(Length + (0 if Paged else batch)), 0), Capacity)
    if not Paged:
        length = length - S + token + 1
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0, tl.float32)
    acc = tl.full((D,), 0, tl.float32)
    for start in range(split * Span, tl.minimum((split + 1) * Span, Capacity), Block):
        positions = start + tl.arange(0, Block)
        live = positions < length
        if Paged:
            slab = (batch * Pages + positions // PageSize) * HK + kv_head
            keys = tl.load(K + slab[:, None] * PageSize * D
                           + (positions % PageSize)[:, None] * D + dim[None, :],
                           live[:, None], other=0).to(tl.float32)
            values = tl.load(V + slab[:, None] * PageSize * D
                             + dim[None, :] * PageSize + (positions % PageSize)[:, None],
                             live[:, None], other=0).to(tl.float32)
            key_scale = tl.load(KS + slab * PageSize + positions % PageSize, live, other=0)
            value_scale = tl.load(VS + slab[:, None] * D + dim[None, :], live[:, None], other=0)
            keys *= key_scale[:, None]
            values *= value_scale
        else:
            keys = tl.load(K + batch * KB + kv_head * KH
                           + positions[:, None] * KL + dim[None, :] * KD,
                           live[:, None], other=0).to(tl.float32)
            values = tl.load(V + batch * VB + kv_head * VH
                             + positions[:, None] * VL + dim[None, :] * VD,
                             live[:, None], other=0).to(tl.float32)
        scores = tl.sum(keys * query[None, :], 1) * (D ** -0.5)
        scores = tl.where(live, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, 0))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0., next_maximum)
        alpha = tl.exp(maximum - safe_maximum)
        probabilities = tl.exp(scores - safe_maximum)
        acc = acc * alpha + tl.sum(probabilities[:, None] * values, 0)
        denominator = denominator * alpha + tl.sum(probabilities, 0)
        maximum = next_maximum
    safe_denominator = tl.where(denominator > 0., denominator, 1.)
    tl.store(Partial + (row * Splits + split) * D + dim, acc / safe_denominator)
    tl.store(LSE + row * Splits + split,
             tl.where(denominator > 0., maximum + tl.log(safe_denominator), -float("inf")))


@triton.jit
def _combine_decode(Partial, LSE, Out, OutLSE,
                    H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
                    Splits: tl.constexpr, Block: tl.constexpr):
    row = tl.program_id(0)
    splits, dim = tl.arange(0, Block), tl.arange(0, D)
    lse = tl.load(LSE + row * Splits + splits, splits < Splits, other=-float("inf"))
    maximum = tl.max(lse, 0)
    maximum = tl.where(maximum == -float("inf"), 0., maximum)
    weights = tl.exp(lse - maximum)
    total = tl.sum(weights, 0)
    partial = tl.load(Partial + (row * Splits + splits[:, None]) * D + dim[None, :],
                      splits[:, None] < Splits, other=0.)
    result = tl.sum(weights[:, None] * partial, 0) / tl.where(total > 0., total, 1.)
    token, head, batch = row % S, row // S % H, row // (S * H)
    tl.store(Out + ((batch * S + token) * H + head) * D + dim, result)
    tl.store(OutLSE + row, tl.where(total > 0., maximum + tl.log(tl.where(total > 0., total, 1.)), -float("inf")))


def decode(q, k, v, lengths, *, key_scale=None, value_scale=None, capacity=None):
    batch, heads, seq, dim = q.shape
    paged = key_scale is not None
    if paged:
        pages, kv_heads, page_size = k.shape[1:4]
        capacity = pages * page_size if capacity is None else capacity
        strides_k = strides_v = (0, 0, 0, 0)
    else:
        kv_heads, capacity = k.shape[1:3]
        pages, page_size = 0, 0
        strides_k, strides_v = k.stride(), v.stride()
    splits = min(32, triton.cdiv(capacity, 256))
    span = triton.cdiv(capacity, splits * 64) * 64
    partial = torch.empty((batch * heads * seq, splits, dim), dtype=torch.float32, device=q.device)
    lse = torch.empty(partial.shape[:-1], dtype=torch.float32, device=q.device)
    out = torch.empty((batch, seq, heads * dim), dtype=q.dtype, device=q.device)
    out_lse = torch.empty((batch, heads, seq), dtype=torch.float32, device=q.device)
    # Unused pointer parameters remain device tensors on the ordinary BF16 path.
    ks = q if key_scale is None else key_scale
    vs = q if value_scale is None else value_scale
    _partial_decode[(batch * heads * seq, splits)](
        q, k, v, lengths, ks, vs, partial, lse,
        heads, kv_heads, seq, dim, capacity, splits, span,
        *q.stride(), *strides_k, *strides_v, paged, pages, page_size, 64,
        num_warps=4,
    )
    _combine_decode[(batch * heads * seq,)](
        partial, lse, out, out_lse, heads, seq, dim, splits,
        triton.next_power_of_2(splits), num_warps=4,
    )
    return out, out_lse


@triton.jit
def _step_merge(Out, LSE, Q, K, V, Merged,
                H: tl.constexpr, HK: tl.constexpr, S: tl.constexpr,
                D: tl.constexpr, Steps: tl.constexpr):
    row = tl.program_id(0)
    token, head, batch = row % S, row // S % H, row // (S * H)
    kv_head = head // (H // HK)
    dim, steps = tl.arange(0, D), tl.arange(0, Steps)
    query = tl.load(Q + row * D + dim).to(tl.float32)
    offsets = ((batch * HK + kv_head) * S + steps[:, None]) * D + dim[None, :]
    keys = tl.load(K + offsets, steps[:, None] < S, other=0).to(tl.float32)
    values = tl.load(V + offsets, steps[:, None] < S, other=0).to(tl.float32)
    scores = tl.sum(query[None, :] * keys, 1) * D ** -0.5
    scores = tl.where(steps <= token, scores, -float("inf"))
    prefix_lse = tl.load(LSE + row)
    maximum = tl.maximum(prefix_lse, tl.max(scores, 0))
    prefix_weight = tl.exp(prefix_lse - maximum)
    weights = tl.exp(scores - maximum)
    output_offset = ((batch * S + token) * H + head) * D + dim
    prefix_out = tl.load(Out + output_offset).to(tl.float32)
    acc = tl.where(prefix_weight > 0, prefix_out * prefix_weight, 0.)
    acc += tl.sum(weights[:, None] * values, 0)
    tl.store(Merged + output_offset, acc / (prefix_weight + tl.sum(weights, 0)))


def step_merge(out, lse, q, k, v, merged):
    batch, heads, seq, dim = q.shape
    if merged.shape != (batch, seq, heads * dim) or not merged.is_contiguous():
        raise ValueError("merged must be a contiguous [B,S,H*D] tensor")
    if any(t.device != q.device or t.dtype != q.dtype for t in (out, merged)) or lse.device != q.device:
        raise ValueError("merge buffers must be on the query device with matching output dtypes")
    if out.shape != merged.shape or lse.shape != (batch, heads, seq) or lse.dtype != torch.float32:
        raise ValueError("merge requires out [B,S,H*D] and FP32 lse [B,H,S]")
    _step_merge[(batch * heads * seq,)](
        out.contiguous(), lse.contiguous(), q.contiguous(), k.contiguous(), v.contiguous(), merged,
        heads, k.shape[1], seq, dim, triton.next_power_of_2(seq), num_warps=4,
    )
    return merged


@triton.jit
def _round_int8(value):
    # Nearest-even without a backend-specific libdevice call; the same codec
    # can be checked with Triton's CPU interpreter.
    lower = tl.floor(value)
    fraction = value - lower
    up = (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
    return (lower + up.to(tl.float32)).to(tl.int8)


@triton.jit
def _update_key(K, KI, KS, Length,
                H: tl.constexpr, Rows: tl.constexpr, Pages: tl.constexpr, Capacity: tl.constexpr,
                KB: tl.constexpr, KH: tl.constexpr, KL: tl.constexpr, KD: tl.constexpr,
                Initialize: tl.constexpr):
    row, head_batch = tl.program_id(0), tl.program_id(1)
    head, batch = head_batch % H, head_batch // H
    length = tl.minimum(tl.maximum(tl.load(Length), 0), Capacity)
    page = row // 512
    last = (length - 1) // 512
    active = (row < Pages * 512) & (page <= last)
    if not Initialize:
        active &= page >= last - 1
    if active:
        dim = tl.arange(0, 256)
        key = tl.load(K + batch * KB + head * KH + (row % Rows) * KL + dim * KD,
                      row < length, other=0).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(key), 0) / 127., 1.e-30)
        quantized = _round_int8(key / scale)
        slab = (batch * Pages + page) * H + head
        tl.store(KI + (slab * 512 + row % 512) * 256 + dim, quantized)
        tl.store(KS + slab * 512 + row % 512, scale)


@triton.jit
def _update_value(V, VI, VS, Length,
                  H: tl.constexpr, Rows: tl.constexpr, Pages: tl.constexpr, Capacity: tl.constexpr,
                  VB: tl.constexpr, VH: tl.constexpr, VL: tl.constexpr, VD: tl.constexpr,
                  Initialize: tl.constexpr):
    page, head_batch, channel_block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head, batch = head_batch % H, head_batch // H
    length = tl.minimum(tl.maximum(tl.load(Length), 0), Capacity)
    last = (length - 1) // 512
    active = page <= last
    if not Initialize:
        active &= page >= last - 1
    if active:
        rows = page * 512 + tl.arange(0, 512)
        dim = channel_block * 16 + tl.arange(0, 16)
        values = tl.load(V + batch * VB + head * VH + (rows % Rows)[:, None] * VL + dim[None, :] * VD,
                         rows[:, None] < length, other=0).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(values), 0) / 127., 1.e-30)
        quantized = _round_int8(values / scale[None, :])
        slab = (batch * Pages + page) * H + head
        tl.store(VI + slab * 512 * 256 + dim[None, :] * 512 + (rows % 512)[:, None], quantized)
        tl.store(VS + slab * 256 + dim, scale)


def update_cache(cache, key, value, length, initialize):
    batch, heads, rows, _ = key.shape
    pages = cache.key.shape[1]
    capacity = cache._capacity
    _update_key[(pages * 512, batch * heads)](
        key, cache.key, cache.key_scale, length,
        heads, rows, pages, capacity, *key.stride(), initialize, num_warps=4,
    )
    _update_value[(pages, batch * heads, 16)](
        value, cache.value, cache.value_scale, length,
        heads, rows, pages, capacity, *value.stride(), initialize, num_warps=4,
    )
