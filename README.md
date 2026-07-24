# Comfy Kitchen XPU

Intel XPU integration for
[Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen), backed
by the optional
[`omni_xpu_kernel`](https://github.com/intel/llm-scaler/tree/main/omni/omni_xpu_kernel)
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

The native XPU implementation comes from Intel's
[`llm-scaler`](https://github.com/intel/llm-scaler) project. We also thank its
maintainers and contributors; this repository provides the Comfy Kitchen
adapter and companion-wheel integration rather than redefining ownership of
that native project.

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
- XPU operator tests, portable tensor tests, and self-hosted device workflows.

Excluding the explicitly deferred NVFP4, MXFP8, and AWQ formats, the XPU
backend registers all 24 capabilities in its current target set.

## XPU backend behavior

Backend priority is:

```text
xpu -> triton -> eager
```

The XPU backend becomes available only when:

1. `torch.xpu.is_available()` succeeds;
2. `omni_xpu_kernel` and its native extension load;
3. the required native INT8 symbols are present.

Optional SVDQuant, normalization, FP8, RoPE, and ConvRot capability groups are
detected separately. A partial or older native package therefore advertises
only the operations it actually implements. Triton remains available on
non-Windows XPU stacks that support it, with eager implementations as the
portable fallback.

On Windows, Triton is registered but disabled by default because its JIT
compiler/runtime is not part of the validated XPU Portable path. Dispatch uses
the XPU backend first and then eager. An advanced environment with a working
Windows Triton toolchain can opt in explicitly:

```python
import comfy_kitchen as ck

ck.enable_backend("triton")
```

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
python -m pip install build
python -m build --wheel
pip install --force-reinstall --no-deps dist/comfy_kitchen-0.2.18-py3-none-any.whl
```

The repository retains upstream CUDA source to keep future upstream updates
reviewable. The XPU wheel does not probe for CUDA, compile the CUDA extension,
or package `comfy_kitchen.backends.cuda`; it contains the XPU, Triton, and eager
Python backends only.

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

The ComfyUI image build uses the Intel upstream repository and compiles the
target-specific native wheel for the selected GPU:

```bash
git clone https://github.com/intel/llm-scaler.git
cd llm-scaler/omni

# On the matching host:
XPU_TARGET=bmg bash build.sh
# or
XPU_TARGET=ptl-h bash build.sh
```

The Docker build produces the same pure-Python Kitchen layer but recompiles
`omni_xpu_kernel` for the selected target. Image labels record the Kitchen
revision/version and XPU target.

## Current limitations

- Intel XPU support remains experimental and is not an upstream Comfy Kitchen
  release claim.
- NVFP4, MXFP8, and AWQ are explicitly deferred for XPU.
- Native wheels are CPython-, Torch-ABI-, and target-specific.
- BMG and PTL-H performance numbers are not portable across devices.
- Full-image measurements include changes outside Kitchen and cannot establish
  an isolated Kitchen speedup.

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
