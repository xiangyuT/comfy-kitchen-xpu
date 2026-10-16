"""Paged CK integer attention over a growing, fixed-capacity decode prefix."""
import torch

if torch.version.hip:
    from .backends import hip as backend
    from .backends.hip import _dl as wrap_for_dlpack
else:
    # The XPU provider wheel excludes CUDA/HIP modules. Resolve CUDA only for
    # an eligible device, so importing this public module remains portable.
    backend = None
    wrap_for_dlpack = None


def _get_backend():
    global backend, wrap_for_dlpack
    if backend is None:
        try:
            from .backends import cuda as cuda_backend
        except ImportError:
            return None
        backend = cuda_backend
        wrap_for_dlpack = backend._wrap_for_dlpack
    return backend


def is_available(device: torch.device | None = None) -> bool:
    """Return whether the native growing INT8 cache supports this device."""
    if device is not None and torch.device(device).type == "xpu":
        from .backends.xpu.decode import int8_is_available as xpu_decode_available
        return xpu_decode_available(device)
    if device is not None and torch.device(device).type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False
    backend = _get_backend()
    return backend is not None and (
        backend._EXT_AVAILABLE
        and hasattr(backend._C, "_int8_decode_update")
        and hasattr(backend._C, "_int8_decode")
        and (backend.has_wmma() if torch.version.hip else torch.cuda.get_device_capability(device) >= (8, 0))
    )


class Int8DecodeCache:
    def __init__(self, batch, heads, capacity, device):
        if not is_available(device):
            raise RuntimeError("Int8DecodeCache requires an available native CUDA/HIP or XPU backend")
        if batch <= 0 or heads <= 0 or capacity <= 0:
            raise ValueError("cache batch, heads and capacity must be positive")
        self._xpu = torch.device(device).type == "xpu"
        self._capacity = capacity
        page_size = 512
        pages = (capacity + page_size - 1) // page_size
        self.key = torch.zeros((batch, pages, heads, page_size, 256), dtype=torch.int8, device=device)
        self.value = torch.empty((batch, pages, heads, 256, page_size), dtype=torch.int8, device=device)
        # Preserve per-thread Q/K groups. XPU stores V without CUDA's MMA
        # permutation; this runtime cache is never serialized across backends.
        scale_rows = page_size // 64 * 4
        self.key_scale = torch.empty((batch, pages, heads, scale_rows), dtype=torch.float32, device=device)
        self.value_scale = torch.empty((batch, pages, heads, 256), dtype=torch.float32, device=device)

    def update(self, key, value, length, initialize=False):
        """Refresh the last two pages, or initialize the prefix. length is one CUDA int64.

        All batches share the length. After initialization the committed length may
        advance by at most one page per call. BF16 sources are [B,H,rows,256].
        Initialization reads the full prefix. Updates can instead read a page-aligned
        circular source: absolute row i lives at i % rows. The caller must retain both
        pages being refreshed, including through speculative writes and rollback.
        """
        if self._xpu:
            self._validate_length(length)
            batch, _, heads, _, _ = self.key.shape
            if key.ndim != 4 or key.shape[:2] != (batch, heads) or key.shape[-1] != 256 or key.shape[2] <= 0:
                raise ValueError("INT8 cache sources must have shape [B,Hk,rows,256]")
            if value.shape != key.shape or any(t.dtype != torch.bfloat16 or t.device != self.key.device for t in (key, value)):
                raise ValueError("INT8 cache sources must be matching BF16 tensors on the cache device")
            rows = key.shape[2]
            if rows > self._capacity or (not initialize and rows != self._capacity and rows % 512):
                raise ValueError("INT8 cache sources must fit capacity; circular updates require page-aligned rows")
            from omni_xpu_kernel import kitchen
            kitchen.int8_decode_update(key, value, self.key, self.value,
                self.key_scale, self.value_scale, length, self._capacity, initialize)
            return
        backend._C._int8_decode_update(
            *map(wrap_for_dlpack, (key, value, self.key, self.value, self.key_scale, self.value_scale, length)),
            initialize, torch.cuda.current_stream(key.device).cuda_stream)

    def attend(self, query, length):
        """BF16 query [B,H,S,256]; return [B,S,H*256] and natural-log LSE [B,H,S].

        Every query attends the same committed prefix. The caller merges the current
        causal rows separately, preserving speculative rollback semantics.
        """
        if self._xpu:
            self._validate_length(length)
            if query.ndim != 4 or query.shape[0] != self.key.shape[0] or query.shape[-1] != 256:
                raise ValueError("INT8 cache query must have shape [B,H,S,256]")
            if query.shape[1] == 0 or query.shape[1] % self.key.shape[2] or query.shape[2] <= 0:
                raise ValueError("INT8 cache requires nonempty query and H divisible by Hk")
            if query.device != self.key.device or query.dtype != torch.bfloat16:
                raise ValueError("INT8 cache query must be BF16 on the cache device")
            from omni_xpu_kernel import kitchen
            return kitchen.int8_decode(query, self.key, self.value,
                self.key_scale, self.value_scale, length, self._capacity)
        batch, heads, seq, dim = query.shape
        kv_heads = self.key.shape[2]
        rows = heads // kv_heads * seq
        q = query.reshape(batch, kv_heads, rows, dim).contiguous()
        qi = torch.empty_like(q, dtype=torch.int8)
        qs = torch.empty((batch, kv_heads, 64), dtype=torch.float32, device=q.device)
        splits = self.key.shape[1] * (8 if torch.version.hip else 1)
        partial = torch.empty((batch, splits, kv_heads, rows, dim), dtype=q.dtype, device=q.device)
        lse = torch.empty(partial.shape[:-1], dtype=torch.float32, device=q.device)
        out = torch.empty((batch, seq, heads * dim), dtype=q.dtype, device=q.device)
        out_lse = torch.empty((batch, heads, seq), dtype=torch.float32, device=q.device)
        backend._C._int8_decode(
            *map(wrap_for_dlpack, (q, qi, qs, self.key, self.value, self.key_scale, self.value_scale,
                                       length, partial, lse, out, out_lse)),
            seq, torch.cuda.current_stream(q.device).cuda_stream)
        return out, out_lse

    def _validate_length(self, length):
        if length.dtype != torch.int64 or length.shape != (1,) or length.device != self.key.device or not length.is_contiguous():
            raise ValueError("INT8 cache length must be a contiguous int64 [1] tensor on the cache device")
