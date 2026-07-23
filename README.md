# Comfy Kitchen XPU

Intel XPU integration for
[Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen), backed
by the optional
[`omni_xpu_kernel`](https://github.com/xiangyuT/llm-scaler/tree/main/omni/omni_xpu_kernel)
native package.

This repository keeps Comfy Kitchen's public APIs and backend dispatch model,
then adds an experimental Intel XPU backend, QuantizedTensor integration,
target-aware companion-wheel tooling, and device validation for Intel BMG and
PTL-H GPUs.

> The Kitchen Python source is shared by BMG and PTL-H. The native
> `omni_xpu_kernel` wheel, CUTE/LGRF/core shared objects, Docker image, and
> acceptance results are **not** portable between GPU targets.

## Relationship to upstream

This repository is a fork of
[Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen). We thank
the upstream maintainers and contributors for the library architecture,
operator APIs, QuantizedTensor design, backend registry, eager/CUDA/Triton
implementations, packaging, and tests on which this work is built.

The XPU development line is based on upstream Comfy Kitchen `0.2.18` at
[`898017e`](https://github.com/Comfy-Org/comfy-kitchen/commit/898017e5c0f23b5b7b9a8473746be6c419baffb3).
The Intel-specific work in this fork is intentionally optional: importing
Comfy Kitchen remains safe when PyTorch XPU, `omni_xpu_kernel`, its native
extension, or Intel GPU hardware is absent.

The original upstream CUDA and generic-backend README is retained
[at the bottom of this page](#upstream-readme-reference) for reference.

## What this fork adds

- An `xpu` backend that negotiates the native symbols actually supplied by
  `omni_xpu_kernel` before registering capabilities.
- Intel implementations and adapters for INT8, INT8 ConvRot, FP8
  quantize/dequantize and stochastic rounding, SVDQuant W4A4, arbitrary-2x2
  RoPE, AdaLN, ConvRot W4A4, and FP8 W8A16 dispatch.
- XPU-aware QuantizedTensor lifecycle, device migration, `linear`, `mm`,
  `addmm`, transpose, serialization, and prepared-weight paths.
- Target-checked companion-wheel construction for BMG and PTL-H, including
  Torch ABI, package target, core AOT target, LGRF, CUTE, and clean-install
  validation.
- XPU operator tests, portable tensor tests, self-hosted device workflows, and
  recoverable implementation/acceptance documentation.

Excluding the explicitly deferred NVFP4, MXFP8, and AWQ formats, the XPU
backend registers all 24 capabilities in its current target set. The detailed
operator matrix and implementation notes are in
[`docs/XPU_BACKEND_STATUS.md`](docs/XPU_BACKEND_STATUS.md).

## Platform compatibility

The practical compatibility boundary is:

```text
comfy_kitchen-0.2.18-py3-none-any.whl
├── BMG   + omni_xpu_kernel-...+torch211.bmg-...whl
└── PTL-H + omni_xpu_kernel-...+torch211.ptlh-...whl
```

| Component or result | Shared between BMG and PTL-H? | Rule |
|---|---|---|
| Kitchen source/API at `223eea0` | Yes | Same Python dispatch and adapters |
| `comfy_kitchen-0.2.18-py3-none-any.whl` | Yes | Pure-Python Kitchen wheel |
| `omni_xpu_kernel` wheel | No | Torch-ABI- and GPU-target-specific |
| Core, LGRF, and CUTE shared objects | No | Build AOT for the target GPU |
| Docker image | No | Build with `XPU_TARGET=bmg` or `XPU_TARGET=ptl-h` |
| Correctness/performance acceptance | No | Run and record independently on each target |

Never copy a BMG native wheel into PTL-H, copy a PTL-H wheel into BMG, or
publish either artifact as a generic XPU binary. The build and test workflows
reject mismatches between:

- installed PyTorch major/minor;
- the wheel's `+torch<minor>.<target>` local version;
- `omni_xpu_kernel.__xpu_target__`;
- `omni_xpu_kernel.core_aot_target()`;
- the actual self-hosted runner device.

## Actual validation status

The following table separates the reusable Kitchen Python layer from the
target-specific native tuple. Results from one row do not accept the other.

| Item | BMG | PTL-H |
|---|---|---|
| Kitchen revision | `223eea0`, version `0.2.18` | `223eea0`, version `0.2.18` |
| PyTorch | `2.11.0+xpu` | `2.11.0+xpu` |
| Native wheel identity | `0.1.0b9.dev0+torch211.bmg` | `0.1.0b8.dev0+torch211.ptlh` |
| Package/core AOT target | `bmg` / `bmg` | `ptl-h` / `ptl-h` |
| Device | Intel Graphics `[0xe223]` | Intel Arc B390 |
| Current evidence | Clean BMG image runtime and official LTX 2.3 workflow | Complete Python 3.12 acceptance ladder |

### BMG

The clean BMG ComfyUI image pins this exact Kitchen revision. On the physical
BMG device it reported:

```text
Kitchen version:          0.2.18
Kitchen XPU available:    true
Kitchen XPU capabilities: 24
Kernel wheel:             0.1.0b9.dev0+torch211.bmg
Package target:           bmg
Core AOT target:          bmg
```

The same image completed the official LTX 2.3 template at its user defaults
(1280x720, 5 seconds, 25 fps) with the text encoder on XPU. It passed the
fresh-server run, two functional forced text re-encodes, and 20 measured b9
requests across cached and forced-XPU-text-encode modes without OOM. The
reported b9-versus-b8 endpoint differences are whole-image diagnostics and
must not be attributed to Kitchen alone.

See the
[BMG image and LTX 2.3 report](https://github.com/xiangyuT/omni-xpu-kernel-tuning/blob/main/docs/results/bmg/2026-07-24/comfyui-image-template-default-ltx23.md)
for the image digest, model hashes, workload, raw-data links, and interpretation
limits.

### PTL-H

The Python 3.12 PTL-H tuple passed its complete acceptance ladder on
2026-07-23:

- Kernel installed-wheel runtime: 544 passed, 2 skipped;
- Kernel source packaging: 26 passed;
- Kitchen focused XPU/backend/INT8/version: 78 passed, 28 skipped,
  5 deselected;
- Kitchen portable full selection: 466 passed, 123 skipped, 207 deselected;
- single-process ComfyUI switch:
  Boogu INT8 -> Krea2 INT8 -> Boogu INT8;
- three valid 1024x1024 RGB outputs, healthy service, and no Level Zero OOR or
  OOM.

The accepted artifacts and their SHA-256 values are recorded in
[`docs/PTL_H_MAINTENANCE.md`](docs/PTL_H_MAINTENANCE.md).

## XPU backend behavior

Backend priority is:

```text
cuda -> xpu -> triton -> eager
```

The XPU backend becomes available only when:

1. `torch.xpu.is_available()` succeeds;
2. `omni_xpu_kernel` and its native extension load;
3. the required native INT8 symbols are present.

Optional SVDQuant, normalization, FP8, RoPE, and ConvRot capability groups are
detected separately. A partial or older native package therefore advertises
only the operations it actually implements. If XPU is unavailable, normal
upstream eager/CUDA/Triton behavior remains unchanged.

Check the active runtime explicitly:

```python
import comfy_kitchen as ck
import omni_xpu_kernel
import torch

backend = ck.list_backends()["xpu"]
print(
    {
        "torch": torch.__version__,
        "device": torch.xpu.get_device_name(0),
        "kitchen": ck.__version__,
        "xpu_available": backend["available"],
        "xpu_capabilities": backend["capabilities"],
        "kernel": omni_xpu_kernel.__version__,
        "package_target": omni_xpu_kernel.__xpu_target__,
        "core_aot_target": omni_xpu_kernel.core_aot_target(),
    }
)
```

Do not treat `backend["available"] == True` alone as release acceptance. Also
verify the Torch ABI, target identity, required capabilities, correctness
suite, and representative ComfyUI workflows.

## Build and installation

### Build the shared Kitchen wheel

The XPU integration uses the pure-Python Kitchen wheel; native Intel code stays
in `omni_xpu_kernel`.

```bash
git clone --branch dev/ptl-h-kitchen-xpu \
    https://github.com/xiangyuT/comfy-kitchen-xpu.git
cd comfy-kitchen-xpu
python setup.py bdist_wheel --no-cuda
pip install --force-reinstall --no-deps dist/comfy_kitchen-0.2.18-py3-none-any.whl
```

### Build the target-specific companion wheel

Provide the approved `llm-scaler/omni/omni_xpu_kernel` and CUTLASS-SYCL source
checkouts, then select the real device target:

```bash
export OMNI_XPU_KERNEL_SOURCE=/path/to/llm-scaler/omni/omni_xpu_kernel
export CUTLASS_SYCL_ROOT=/path/to/sycl-tla
export TORCH_SPEC=torch==2.11.0+xpu
export EXPECTED_TORCH_MINOR=2.11

# Choose exactly one target: bmg or ptl-h.
export OMNI_XPU_DEVICE=ptl-h
export EXPECTED_XPU_TARGET=ptl-h

./packaging/omni_xpu_kernel/build_uv_wheel_matrix.sh 3.12
```

The release-oriented Linux build requires CUTE and fails instead of producing
an incomplete companion wheel. Core-only builds are useful for focused Kitchen
operator development but are not equivalent to the accepted ComfyUI stack.
See
[`packaging/omni_xpu_kernel/README.md`](packaging/omni_xpu_kernel/README.md)
for the artifact matrix and clean-install checks.

### Docker image integration

The focused ComfyUI image pins the Kitchen commit and builds the two wheel
branches independently:

```bash
git clone https://github.com/xiangyuT/llm-scaler.git
cd llm-scaler/omni

# On the matching host:
XPU_TARGET=bmg bash build.sh
# or
XPU_TARGET=ptl-h bash build.sh
```

The Docker build produces the same pure-Python Kitchen layer but recompiles
`omni_xpu_kernel` for the selected target. Image labels record the Kitchen
revision/version and XPU target.

## Repository documentation

- [`docs/XPU_BACKEND_STATUS.md`](docs/XPU_BACKEND_STATUS.md): implementation,
  operator coverage, BMG history, tests, and follow-up work.
- [`docs/PTL_H_MAINTENANCE.md`](docs/PTL_H_MAINTENANCE.md): PTL-H ownership,
  pinned tuple, accepted artifacts, and acceptance results.
- [`docs/XPU_TENSOR_TEST_MIGRATION.md`](docs/XPU_TENSOR_TEST_MIGRATION.md):
  portable QuantizedTensor test coverage.
- [`packaging/omni_xpu_kernel/README.md`](packaging/omni_xpu_kernel/README.md):
  target-specific companion-wheel build and artifact rules.

## Current limitations

- Intel XPU support remains experimental and is not an upstream Comfy Kitchen
  release claim.
- NVFP4, MXFP8, and AWQ are explicitly deferred for XPU.
- Native wheels are CPython-, Torch-ABI-, and target-specific.
- The currently accepted local artifact is Python 3.12; other Python versions
  remain build-matrix jobs until their complete target-specific acceptance is
  recorded.
- BMG and PTL-H performance numbers are not portable across devices.
- Full-image measurements include changes outside Kitchen and cannot establish
  an isolated Kitchen speedup.

## Contributing

Please keep upstream-generic changes separate from Intel-specific integration
where possible. XPU changes should include:

- capability and fallback behavior;
- target and Torch ABI identity;
- focused operator correctness;
- portable non-CUDA regression coverage;
- at least one representative workflow when runtime behavior changes;
- separate BMG and PTL-H evidence for claims covering both targets.

Changes that are generally useful to Comfy Kitchen should be proposed to
[the upstream project](https://github.com/Comfy-Org/comfy-kitchen) whenever
practical.

---

## Upstream README reference

The following is the original upstream README content from
[Comfy-Org/comfy-kitchen `0.2.18` at `898017e`](https://github.com/Comfy-Org/comfy-kitchen/blob/898017e5c0f23b5b7b9a8473746be6c419baffb3/README.md).
It is retained here to preserve the CUDA, eager, Triton, installation, and
generic backend documentation supplied by upstream.

<details>
<summary>Expand the upstream Comfy Kitchen README</summary>

# Comfy Kitchen

Fast kernel library for Diffusion inference with multiple compute backends.

## Backend Capabilities Matrix

| Function                    | eager | cuda | triton |
|-----------------------------|-------|------|--------|
| `quantize_per_tensor_fp8`   | ✓     | ✓    | ✓      |
| `dequantize_per_tensor_fp8` | ✓     | ✓    | ✓      |
| `quantize_nvfp4`            | ✓     | ✓    | ✓      |
| `dequantize_nvfp4`          | ✓     | ✓    |        |
| `scaled_mm_nvfp4`           | ✓     | ✓    |        |
| `quantize_mxfp8`            | ✓     | ✓    | ✓      |
| `dequantize_mxfp8`          | ✓     |      |        |
| `scaled_mm_mxfp8`           | ✓     |      |        |
| `apply_rope`                | ✓     | ✓    | ✓      |
| `apply_rope1`               | ✓     | ✓    | ✓      |


## Quantized Tensors

The library provides `QuantizedTensor`, a `torch.Tensor` subclass that transparently intercepts PyTorch operations and dispatches them to optimized quantized kernels when available.

| Layout                 | Format       | HW Requirement  | Description                             |
|------------------------|--------------|-----------------|----------------------------------------|
| `TensorCoreFP8Layout`  | FP8 E4M3     | SM ≥ 8.9 (Ada)  | Per-tensor scaling, 1:1 element mapping |
| `TensorCoreNVFP4Layout`| NVFP4 E2M1   | SM ≥ 10.0 (Blackwell) | Block quantization with 16-element blocks |
| `TensorCoreMXFP8Layout`| MXFP8 E4M3   | SM ≥ 10.0 (Blackwell) | Block quantization with 32-element blocks, E8M0 scales |

```python
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreFP8Layout, TensorCoreNVFP4Layout

# Quantize a tensor
x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
qt = QuantizedTensor.from_float(x, TensorCoreFP8Layout)

# Operations dispatch to optimized kernels automatically
output = torch.nn.functional.linear(qt, weight_qt)

# Dequantize back to float
dq = qt.dequantize()
```


## Installation

### From PyPI

```bash
# Install default (Linux/Windows/MacOS)
pip install comfy-kitchen

# Install with CUBLAS for NVFP4 (+Blackwell)
pip install comfy-kitchen[cublas]
```

### Package Variants

- **CUDA wheels**: Linux x86_64 and Windows x64
- **Pure Python wheel**: Any platform, eager and triton backends only

Wheels are built for Python 3.10, 3.11, and 3.12+ (using Stable ABI for 3.12+).

### From Source

```bash
# Standard installation with CUDA support
pip install .

# Development installation
pip install -e ".[dev]"

# For faster rebuilds during development (skip build isolation)
pip install -e . --no-build-isolation -v
```

#### Build Options

These options require using `setup.py` directly (not `pip install`):

| Option | Command | Description | Default                                                                     |
|--------|---------|-------------|-----------------------------------------------------------------------------|
| `--no-cuda` | `python setup.py bdist_wheel --no-cuda` | Build CPU-only wheel (`py3-none-any`) | Enabled (build with CUDA)                                                   |
| `--cuda-archs=...` | `python setup.py build_ext --cuda-archs="80;89"` | CUDA architectures to build for | `75-virtual;80;89;90a;100f;120f` (Linux), `75-virtual;80;89;120f` (Windows) |
| `--debug-build` | `python setup.py build_ext --debug-build` | Build in debug mode with symbols | Disabled (Release)                                                          |
| `--lineinfo` | `python setup.py build_ext --lineinfo` | Enable NVCC line info for profiling | Disabled                                                                    |

```bash
# Build CPU-only wheel (pure Python, no CUDA required)
python setup.py bdist_wheel --no-cuda

# Build with custom CUDA architectures
python setup.py build_ext --cuda-archs="80;89" bdist_wheel

# Debug build with line info for profiling
python setup.py build_ext --debug-build --lineinfo bdist_wheel
```



### Requirements

- **Python**: ≥3.10
- **PyTorch**: ≥2.5.0
- **CUDA Runtime** (for CUDA wheels): ≥13.0
  - Pre-built wheels require NVIDIA Driver r580+
  - Building from source requires CUDA Toolkit ≥12.8 and `CUDA_HOME` environment variable
- **nanobind**: ≥2.0.0 (for building from source)
- **CMake**: ≥3.18 (for building from source)

## Quick Start

```python
import comfy_kitchen as ck
import torch

# Automatic backend selection (triton -> cuda -> eager)
x = torch.randn(100, 100, device="cuda")
scale = torch.tensor([1.0], device="cuda")
result = ck.quantize_per_tensor_fp8(x, scale)

# Check which backends are available
print(ck.list_backends())

# Force a specific backend
result = ck.quantize_per_tensor_fp8(x, scale, backend="eager")

# Temporarily use a different backend
with ck.use_backend("triton"):
    result = ck.quantize_per_tensor_fp8(x, scale)
```

## Backend System

The library supports multiple backends:
- **eager**: Pure PyTorch implementation
- **cuda**: Custom CUDA C kernels (CUDA only)
- **triton**: Triton JIT-compiled kernels

### Automatic Backend Selection

When you call a function, the registry selects the best backend by checking **constraints** in priority order (`cuda` → `triton` → `eager`):

```python
# Backend is selected automatically based on input constraints
result = ck.quantize_per_tensor_fp8(x, scale)

# On CPU tensors → falls back to eager (only backend supporting CPU)
# On CUDA tensors → uses cuda or triton (higher priority)
```

### Constraint System

Each backend declares constraints for its functions:

| Constraint | Description |
|------------|-------------|
| **Device** | Which device types are supported |
| **Dtype** | Allowed input/output dtypes per parameter |
| **Shape** | Shape requirements (e.g., 2D tensors, dimensions divisible by 16) |
| **Compute Capability** | Minimum GPU architecture (e.g., SM 8.0 for FP8, SM 10.0 for NVFP4) |

The registry validates inputs against these constraints **before** calling the backend—no try/except fallback patterns. If no backend can handle the inputs, a `NoCapableBackendError` is raised with details.

```python
# Debug logging to see backend selection
import logging
logging.getLogger("comfy_kitchen.dispatch").setLevel(logging.DEBUG)
```


## Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_backends.py

# Run with verbose output
pytest -v

# Run specific test
pytest tests/test_backends.py::TestBackendSystem::test_list_backends
```
</details>
