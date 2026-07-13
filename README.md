# Comfy Kitchen

Fast kernel library for Diffusion inference with multiple compute backends.

Intel XPU support is available experimentally through the optional
[`omni_xpu_kernel`](https://github.com/intel/llm-scaler/tree/main/omni/omni_xpu_kernel)
package. Excluding the explicitly deferred NVFP4, MXFP8, and AWQ formats, the
XPU backend covers all 24 eager API capabilities.
The detailed implementation and validation record is maintained in
[`docs/XPU_BACKEND_STATUS.md`](docs/XPU_BACKEND_STATUS.md).

## Backend Capabilities Matrix

| Function                    | eager | cuda | xpu | triton |
|-----------------------------|-------|------|-----|--------|
| `quantize_per_tensor_fp8`   | ✓     | ✓    | ✓   | ✓      |
| `dequantize_per_tensor_fp8` | ✓     | ✓    | ✓   | ✓      |
| `quantize_nvfp4`            | ✓     | ✓    |     | ✓      |
| `dequantize_nvfp4`          | ✓     | ✓    |     |        |
| `scaled_mm_nvfp4`           | ✓     | ✓    |     |        |
| `quantize_mxfp8`            | ✓     | ✓    |     | ✓      |
| `dequantize_mxfp8`          | ✓     |      |     |        |
| `scaled_mm_mxfp8`           | ✓     |      |     |        |
| `quantize_int8_tensorwise`  | ✓     | ✓    | ✓   |        |
| `quantize_int8_rowwise`     | ✓     | ✓    | ✓   | ✓      |
| `dequantize_int8_simple`    | ✓     | ✓    | ✓   |        |
| `int8_linear`               | ✓     | ✓    | ✓   | ✓      |
| `mm_int8`                   | ✓     |      | ✓   |        |
| `quantize_svdquant_w4a4`    | ✓     | ✓    | ✓   |        |
| `scaled_mm_svdquant_w4a4`   | ✓     | ✓    | ✓   |        |
| `adaln`                     | ✓     | ✓    | ✓   | ✓      |
| `apply_rope`                | ✓     | ✓    | ✓   | ✓      |
| `apply_rope1`               | ✓     | ✓    | ✓   | ✓      |


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

### Intel XPU (experimental)

Install a PyTorch build with XPU support and build/install `omni_xpu_kernel`
for the target GPU first. Comfy Kitchen discovers it at import time:

```bash
git clone https://github.com/intel/llm-scaler.git
OMNI_XPU_REQUIRE_CUTE=0 OMNI_XPU_DEVICE=bmg \
    pip install ./llm-scaler/omni/omni_xpu_kernel --no-build-isolation
pip install comfy-kitchen
```

Use `OMNI_XPU_DEVICE=pvc` for Intel Data Center GPU Max. If the package, native
extension, or XPU device is unavailable, the backend is reported as unavailable
by `list_backends()` and normal eager/CUDA/Triton behavior is unchanged.

The direct install above explicitly selects a core-only build, which is
sufficient for Kitchen's operator backend but omits the CUTE attention
sidecar. CUTE is required by default for normal Linux builds. Kitchen's
companion release wheel supplies `CUTLASS_SYCL_ROOT` and keeps the explicit
`OMNI_XPU_REQUIRE_CUTE=1` release guard; see
[`packaging/omni_xpu_kernel/README.md`](packaging/omni_xpu_kernel/README.md) for
the Python-version matrix, artifact location, and clean-install acceptance
checks.

The backend negotiates optional native symbol groups before registering each
capability. The companion omni implementation includes INT8 ConvRot, complete
arbitrary-2x2 RoPE semantics (adjacent and split-half), FP8 Q/DQ/stochastic
rounding, and ConvRot W4A4.

The signed SVDQuant path uses omni's ESIMD activation quantize/dequantize and
oneDNN INT4 weight GEMM. Kitchen tile-packed checkpoints can be converted once
at load time with `comfy_kitchen.backends.xpu.svdquant.prepare_svdquant_weights`.
The unsigned `act_unsigned=True` path uses dedicated ESIMD U4 quantization and
dequantization.

For repeated SVDQuant inference, weights can be destructively converted to
oneDNN's preconverted representation without retaining a second INT4 copy:

```python
from comfy_kitchen.tensor import prepare_svdquant_for_xpu

weight = prepare_svdquant_for_xpu(weight)  # natural layout: qdata converted in place
weight = weight.to("xpu")                  # prepare tile-packed weights on CPU first
```

The packed weight allocation and total persistent qdata+scale byte count remain
unchanged. For a natural 3840x3840 layer, measured peak conversion overhead is
only the 0.46 MB scale replacement rather than the 7.37 MB INT4 weight. Calling
`state_dict()` returns standard signed Kitchen tensors; in especially
memory-constrained saving code, call `restore_svdquant_standard_format_(weight)`
first to restore qdata in place and avoid the temporary serialized INT4 tensor.

AdaLN fuses LayerNorm and Kitchen modulation in one ESIMD kernel for hidden
dimensions divisible by 32 and no larger than 8192 and common broadcast
layouts. Other valid shapes and broadcasts use the safe composed path.

`TensorCoreFP8Layout` also uses omni's oneDNN W8A16 path when a floating-point
XPU activation is multiplied by an FP8 weight. Kitchen's scalar scale is
expanded to omni's per-output-channel scale ABI. Unsupported oneDNN shapes fall
back to normal dequantized linear execution. For `(M,K,N)=(64,3840,3840)`, this
avoided per-call weight dequantization was about 12.9x faster in a local BMG
microbenchmark; compared with an already materialized BF16 weight, W8A16 itself
was slightly slower, so the benefit is primarily memory capacity and avoiding
repeated dequantization.

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
- **xpu**: Intel XPU kernels supplied by optional `omni_xpu_kernel`
- **triton**: Triton JIT-compiled kernels

### Automatic Backend Selection

When you call a function, the registry selects the best backend by checking **constraints** in priority order (`cuda` → `xpu` → `triton` → `eager`):

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
