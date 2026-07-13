# Intel XPU Backend Integration Status

Last updated: 2026-07-13

This document records the local implementation status of the experimental
Intel XPU backend powered by
[`omni_xpu_kernel`](https://github.com/intel/llm-scaler/tree/main/omni/omni_xpu_kernel).
It is intended to make the implementation, constraints, and remaining work
recoverable without relying on external discussion history.

## Source Baseline

- Comfy Kitchen base commit: `898017e5c0f23b5b7b9a8473746be6c419baffb3`
- llm-scaler base commit: `cf7b913` (INT8 optimizations from PR #536)
- llm-scaler integration branch: `dev/kitchen_xpu`
- llm-scaler integration commit: `98e7d6e`
- Comfy Kitchen integration branch: `dev/kitchen_xpu`
- Comfy Kitchen version: `0.2.18`
- Tested PyTorch build: `2.10.0+xpu`
- Tested omni_xpu_kernel source version: `0.1.0-b8-dev`
- Test hardware: 4 Intel XPU devices available to the development environment

The Comfy Kitchen integration described here is organized on Kitchen's
`dev/kitchen_xpu` branch. The native Kernel implementation is committed on
llm-scaler's branch of the same name. A release wheel must pin the final full
llm-scaler revision after the milestone commits stop changing.

## Architecture

The backend lives under `comfy_kitchen/backends/xpu/` and treats
`omni_xpu_kernel` as an optional external package. Importing Comfy Kitchen must
remain safe when either the package, its native extension, or XPU hardware is
absent.

Default backend priority is:

```text
cuda -> xpu -> triton -> eager
```

The XPU backend is registered only when:

1. `torch.xpu.is_available()` succeeds;
2. the omni native extension loads;
3. all required native INT8 symbols exist.

SVDQuant, normalization, and FP8 capabilities are detected separately so an
older partial omni build cannot advertise unsupported operations.

### Source Ownership

Comfy Kitchen owns backend selection, `QuantizedTensor` integration, companion
wheel orchestration, and release artifacts. llm-scaler owns the native source,
Python Kernel wrappers, and Kernel-level tests under
`omni/omni_xpu_kernel/`:

| Area | llm-scaler native source | Public `omni_xpu_kernel` surface |
|---|---|---|
| AdaLN | `omni_xpu_kernel/csrc/adaln.cpp` | `norm.fused_adaln` |
| FP8 quantization | `omni_xpu_kernel/csrc/fp8_quant.cpp` | `fp8.quantize_per_tensor`, `fp8.dequantize_per_tensor`, `fp8.stochastic_rounding` |
| Kitchen RoPE | `omni_xpu_kernel/csrc/kitchen_rope.cpp`, `kitchen_rope_sycl.cpp` | `rotary.apply_kitchen_rope*` |
| INT8 ConvRot | `omni_xpu_kernel/csrc/convrot.cpp` | `int8.quantize_int8_convrot_weight`, `int8.dequantize_int8_convrot_weight` |
| Unsigned SVDQuant | `omni_xpu_kernel/csrc/svdq_dequant.cpp` | `svdq.quantize_act_uint4`, `svdq.dequantize_u4` |
| FP8 W8A16 | `omni_xpu_kernel/csrc/onednn_fp8.cpp` | `linear.onednn_w8a16_fp8` |
| CUTE attention | `omni_xpu_kernel/cute/cute_fmha_torch.cpp` | `cute.sdp`, `cute.is_available` |

`omni_xpu_kernel/csrc/bindings.cpp` registers the main extension symbols, and
`native_capabilities()` reports the symbols that actually loaded so Kitchen
does not assume every wheel contains every optional capability. The Linux CUTE
sidecar is required by default when building current llm-scaler source.
Kitchen's companion release flow additionally keeps an explicit
`OMNI_XPU_REQUIRE_CUTE=1` guard; `=0` is reserved for intentional core-only or
Windows builds.

## Implemented Coverage

### INT8 and INT8-ConvRot

The following Kitchen operations are connected to omni:

- `quantize_int8_tensorwise`
- `quantize_int8_rowwise`
- `dequantize_int8_simple`
- `dequantize_int8_simple_dtype`
- `int8_linear`
- `mm_int8`
- `quantize_int8_convrot_weight`
- `dequantize_int8_convrot_weight`
- `dequantize_int8_convrot_weight_dtype`

`mm_int8` was changed from a direct eager call to normal backend-registry
dispatch. Dtype-code adapters preserve the Kitchen custom-op ABI.

The companion omni worktree now exposes native ConvRot rotation,
quantize/dequantize, and online activation support. Rotation uses a cached
regular-Hadamard matrix and the XPU matmul primitive. The largest supported
common matrix (`group_size=256`, BF16) is 128 KiB; no original weight copy is
retained.

### SVDQuant W4A4

The signed activation path uses:

- omni ESIMD INT4 activation quantization;
- omni ESIMD activation dequantization;
- oneDNN INT4 weight GEMM;
- Kitchen-equivalent LoRA down/up and bias operations.

Kitchen natural storage matches omni's row-major packed INT4 layout. Existing
Kitchen tile-packed checkpoints are converted with:

```python
from comfy_kitchen.backends.xpu.svdquant import prepare_svdquant_weights
```

The high-level `TensorCoreSVDQuantW4A4Layout` `F.linear` path is covered.
`act_unsigned=True` uses dedicated omni ESIMD U4 quantize, unpack, and
dequantize kernels with the Kitchen `[0, 15]`, `absmax/15` contract.

### Single-copy SVDQuant Preconversion

Repeated oneDNN signed-to-unsigned nibble and BF16-to-FP16 scale conversion can
be removed with destructive preparation:

```python
from comfy_kitchen.tensor import prepare_svdquant_for_xpu

weight = prepare_svdquant_for_xpu(weight)
weight = weight.to("xpu")
```

For natural row-major storage:

- packed INT4 data is XOR-converted in place;
- the packed allocation and data pointer remain unchanged;
- BF16 scales are replaced by same-sized FP16 scales;
- the original signed weight is not retained;
- inference calls `onednn_int4_gemm_preconverted` directly.

The runtime format is marked with `Params.xpu_preconverted`. It is never
treated as a new checkpoint format:

- `state_dict()` exports standard signed Kitchen tensors;
- moving a prepared tensor away from XPU restores standard storage;
- `restore_svdquant_standard_format_()` restores it in place before
  memory-constrained serialization;
- mixed standard/preconverted weights cannot be fused.

Tile-packed weights should be prepared on CPU before upload. Preparing them
while already resident on XPU is rejected because the permutation would need a
full-size destination allocation and temporarily double packed-weight memory.

Measured for a natural `3840 x 3840` layer:

| Metric | Result |
|---|---:|
| Packed INT4 weight | 7.3728 MB |
| Scale tensor | 0.4608 MB |
| Peak preparation overhead | 0.4608 MB |
| Persistent qdata + scale increase | 0 MB |
| Packed data pointer changed | No |

For `(M,K,N)=(64,3840,3840)`, the preconverted GEMM measured approximately
`0.0540 ms` versus `0.0656 ms` for omni's converting wrapper, or about `1.21x`.
These are local microbenchmarks rather than portable performance guarantees.

### AdaLN

AdaLN uses a single omni ESIMD kernel that fuses LayerNorm, scale modulation,
and shift modulation for global, full-row, and common `(batch, 1, hidden)`
broadcast layouts. The native path supports matching dtypes and hidden
dimensions divisible by 32 and no larger than 8192. Other valid broadcasts or
mixed dtypes explicitly execute the safe composed/eager path.

A local BF16 `(2,256,3072)` microbenchmark measured approximately `0.0293 ms`
for fused XPU versus `0.0453 ms` for eager composition, about `1.55x`.

### FP8 W8A16

When a floating-point XPU activation is multiplied by a
`TensorCoreFP8Layout` weight, the tensor dispatch path uses omni's oneDNN W8A16
kernel. Kitchen's scalar scale is expanded to omni's per-output-channel ABI.
Unsupported oneDNN shapes fall back to dequantized linear execution.

For `(M,K,N)=(64,3840,3840)`, the path measured approximately `12.9x` faster
than dequantizing the FP8 weight on every call before BF16 GEMM. It was slightly
slower than GEMM with an already materialized persistent BF16 weight, so its
main value is preserving FP8 memory capacity and eliminating repeated
dequantization.

FP8 per-tensor Q/DQ and caller-randomness stochastic rounding are also native.
Both E4M3FN and E5M2 are covered for FP32, FP16, and BF16 inputs/outputs, and
the Q/DQ public custom ops pass full-graph compile tests.

### Generic RoPE

omni now provides Kitchen-specific adjacent-pair and split-half RoPE entry
points. They implement the complete broadcastable arbitrary `2 x 2` matrix
contract rather than assuming canonical cosine/sine values. Single-tensor and
query/key pair APIs, BHND/BNHD layouts, differing query/key head counts, and
full-graph compile are covered.

The supported contiguous 4D/6D path is a dedicated single-submission SYCL
kernel. It preserves Kitchen's distinct reduced-precision semantics:
adjacent-pair uses the `addcmul_` order, while split-half preserves two rounded
products followed by addition. Unsupported shapes continue through the safe
ATen composition.

Local BF16-input/FP32-frequency measurements were:

| Workload | Shape | Adjacent speedup | Split-half speedup | Fast extra peak |
|---|---|---:|---:|---:|
| FLUX | `(1,24,4352,128)` | 4.22x | 5.41x | 26 MiB |
| LTX | `(2,32,4996,64)` | 4.04x | 5.69x | 40 MiB |
| Z-Image | `(1,4096,30,128)` | 4.00x | 5.27x | 30 MiB |
| Wan | `(2,12288,16,128)` | 4.02x | 5.33x | 96 MiB |

The eager extra peak ranged from 130 to 480 MiB for adjacent-pair and 208 to
768 MiB for split-half. The fast path therefore uses only the output-sized
allocation and lowers measured extra peak by approximately 5x/8x.

### ComfyUI Attention Routing

`ComfyUI-OmniXPU` keeps `OMNI_ATTN_BACKEND` as an explicit routing control:

- `auto` (default): cute for validated B=1, unmasked, standard-scale d128
  self-attention; ESIMD for supported d64 or cross-attention; PyTorch otherwise;
- `cute`: cute for its validated domain and PyTorch otherwise;
- `esimd`: ESIMD for its supported domain and PyTorch otherwise;
- `torch`: do not patch the original PyTorch attention path.

The current cute scheduler produced inaccurate cross-attention output when
query and key/value sequence lengths differed. ComfyUI now routes that case to
ESIMD, and the public cute Kernel rejects it explicitly so direct callers cannot
receive a silent wrong result. Representative d128 self, d128 cross, d64 self,
and batch=2 paths were compared with ComfyUI's PyTorch attention after installing
the companion wheel. Cute self-attention additionally passed FP16/BF16,
non-tile-aligned length 257, length 1024, length 4352, and `skip_reshape` layout
checks.

Backend changes are numerically close but not bitwise identical. A diffusion
workflow can therefore produce slightly different pixels for the same seed
when its attention route changes, even when correctness tolerances pass.

### ConvRot W4A4

All three public W4A4 capabilities are registered: offline weight quantize,
weight dequantize, and linear for both `linear_dtype="int4"` and `"int8"`.
Persistent weights stay packed W4. The current portable XPU path unpacks only
the transient operands needed by oneDNN INT8 GEMM; it does not materialize or
retain a floating original weight.

For BF16 `(M,K)=(64,3840)` and `group_size=256`, cached native ConvRot measured
approximately `0.0197 ms` versus `0.0221 ms` for the Python cached-matrix
reference. The cached Hadamard constant costs 128 KiB and is shared by calls;
the earlier multi-launch butterfly prototype was rejected after measuring
`0.412 ms`.

## Deliberately Unregistered Operations

### SDP, GGUF, and Standalone Norm APIs

omni provides SDP, GGUF dequantization, RMSNorm, fused add+RMSNorm, and related
operations, but Kitchen currently has no equivalent public APIs. Supporting
these requires API additions rather than backend-only registration.

### NVFP4, MXFP8, and AWQ

These formats were explicitly deferred for this implementation cycle. Seven
eager capabilities remain unregistered: NVFP4 Q/DQ/scaled-mm, MXFP8
Q/DQ/scaled-mm, and AWQ GEMV.

Excluding those seven operations, the final capability comparison is:

```text
Kitchen eager capabilities: 31
Explicitly deferred:          7
XPU target:                  24
XPU registered:              24
Missing target capabilities:  0
```

## Tests and Packaging

The XPU test suite covers:

- backend availability and native-symbol health checks;
- INT8 quantization, dequantization, linear, raw GEMM, and ConvRot;
- deterministic stochastic rounding, single-row linear, FP16/BF16 output,
  QuantizedTensor clone/detach/state lifecycle, and non-default XPU devices;
- signed SVDQuant natural and tile-packed layouts;
- QuantizedTensor `F.linear` dispatch;
- destructive preconversion, single-copy storage, state-dict restoration, and
  CPU migration;
- AdaLN native and fallback shapes;
- FP8 W8A16 dispatch and oneDNN cache use.
- full-graph `torch.compile` smoke tests for INT8 and SVDQuant custom ops;
- dynamic input rows through the compiled INT8 custom-op graph;
- clean subprocess import when `omni_xpu_kernel` is unavailable;
- signed/unsigned SVDQuant, small/large magnitudes, and finite scale checks.
- INT8 QuantizedTensor mm/addmm/transpose and CPU-to-XPU device migration;
- FP8 bias, higher-dimensional activation, and unsupported-shape fallback;
- SVDQuant shared-quant grouped linear, fused linear, and prepared fused weight.
- ConvRot linear/mm/addmm dispatch and oneDNN INT8 primitive cache hits;
- original AdaLN 64/768/3072 hidden-size matrix and SVDQuant `lora_x` routing.
- all 57 portable `test_tensor.py` FP8/INT8/QuantizedTensor lifecycle cases;
- FLUX/LTX/Z-Image/Wan RoPE matrices for adjacent and split-half paths;
- non-default single-stream ordering and a minimal two-stream independence
  check (correctness only, with no overlap/performance guarantee).

The Kernel-side additions are directly exercised by
`test_fused_adaln.py`, `test_kitchen_fp8.py`, `test_kitchen_rope.py`,
`test_native_convrot.py`, `test_streams.py`, and `test_svdq_unsigned.py`.
The existing GGUF, INT8, FP8-linear, SVDQuant, norm, SDP, and packaging suites
provide the broader regression coverage. CUTE execution and its explicit
cross-attention rejection are part of the companion-wheel clean-install smoke
test because that ABI-specific sidecar is not present in a core-only build.

Latest omni kernel acceptance result, excluding `tests/test_packaging.py`
because multi-version wheel execution is deliberately deferred:

```text
434 passed
```

Latest Kitchen XPU result:

```text
50 passed
```

Focused Kitchen RoPE result:

```text
228 passed, 72 skipped
```

Focused Kitchen QuantizedTensor result:

```text
57 passed, 129 skipped
```

Full Kitchen repository result:

```text
467 passed, 328 skipped
```

The earlier core-only wheel matrix result was:

```text
Python 3.10.20: build/install/XPU smoke passed
Python 3.11.15: build/install/XPU smoke passed
Python 3.12.13: build/install/XPU smoke passed
Python 3.13.14: build/install/XPU smoke passed
Python 3.14.6:  build/install/XPU smoke passed
```

Those runs did not require the cute FMHA extension and are no longer sufficient
for Kitchen release acceptance. Current cute-required milestone status is:

```text
Python 3.10.20: pending milestone rebuild
Python 3.11.15: legacy wheel quarantined (cute missing)
Python 3.12.13: build/install/cute XPU execution passed
Python 3.13.14: legacy wheel quarantined (cute missing)
Python 3.14.6:  legacy wheel quarantined (cute missing)
```

These existing artifacts still use the earlier `0.1.0` wheel version. Current
source uses the image-aligned `0.1.0-b8-dev` identifier, normalized to
`0.1.0b8.dev0` by Python packaging. The Python 3.10–3.14 b8 wheel matrix remains
deferred until the milestone rebuild.

Each `linux_x86_64` wheel contains its matching CPython `_C` extension and
LGRF sidecar. Comfy Kitchen now owns these companion artifacts and their build
automation. Detailed sizes, SHA-256 values, and build instructions are in
`packaging/omni_xpu_kernel/README.md`; local artifacts are written to
`wheelhouse/omni_xpu_kernel/` in this repository.

Ruff and `git diff --check` pass. A `py3-none-any` wheel was built and verified
to contain:

```text
comfy_kitchen/backends/xpu/__init__.py
comfy_kitchen/backends/xpu/adaln.py
comfy_kitchen/backends/xpu/fp8.py
comfy_kitchen/backends/xpu/svdquant.py
```

The Kitchen companion build produces a platform wheel containing `_C`, the
LGRF sidecar, and the ABI-specific CUTLASS-SYCL cute FMHA extension. Because
cute is ComfyUI-OmniXPU's default attention backend, the Kitchen build requires
`CUTLASS_SYCL_ROOT` and fails rather than publishing a wheel without it.
`.github/workflows/test-xpu.yml` provides a manual workflow for a self-hosted
runner labeled `self-hosted`, `xpu`, and `bmg`.

Before the PR #536 baseline update, the installed `/llm/ComfyUI` tree passed
both OmniXPU-only and all-custom-node quick startup. That post-install run
covered `414` Omni Kernel tests, `50` Kitchen XPU tests, and `6` ComfyUI
mixed-precision tests. The current llm-scaler source subsequently passed the
`434`-test Kernel acceptance run above; the installed ComfyUI startup check has
not been repeated because the full wheel rebuild is deferred until the
milestone. This remains startup and operator-level coverage; a full
model-generating workflow was not run.

## Known Follow-up Work

1. Integrate CPU-side destructive SVDQuant preparation into the actual model
   loader so tile-packed weights never reach XPU in their source layout.
2. Benchmark full ComfyUI workflows rather than isolated kernels.
3. Add PVC coverage and target-specific wheel testing.
4. Decide whether Kitchen should add public RMSNorm, SDPA, and GGUF APIs.
5. Replace the portable ConvRot W4A4 INT8-unpack GEMM with a packed XPU INT4
   primitive when oneDNN exposes a suitable signed-W4/quantized-activation ABI.
6. Publish the accepted Python 3.10–3.14 artifacts through a tagged release or
   package index, with PyTorch XPU/oneAPI runtime requirements documented.
7. Refactor omni's all-source compile command into incremental object builds;
   a one-file SYCL change currently recompiles the entire native extension.
