# Omni XPU companion wheels

Comfy Kitchen owns the build, validation, and release flow for the
`omni_xpu_kernel` wheels used by its optional XPU backend. The native Kernel
source remains in Intel's
[`llm-scaler`](https://github.com/intel/llm-scaler) repository; it is not
copied into the pure-Python `comfy-kitchen` wheel.

This separation is intentional:

- `comfy-kitchen` selects and calls backend capabilities;
- llm-scaler maintains the Intel XPU Kernel implementations;
- Kitchen's release flow builds CPython- and device-target-specific companion
  wheels from a pinned llm-scaler revision.

## Local build

The default source path expects sibling `comfy-kitchen` and `llm-scaler`
checkouts. `CUTLASS_SYCL_ROOT` is mandatory because cute FMHA is the default
ComfyUI-OmniXPU attention backend. On `dev/ptl-h-kitchen-xpu`, the script
defaults to Torch `2.11.0+xpu` and `OMNI_XPU_DEVICE=ptl-h`:

```bash
CUTLASS_SYCL_ROOT=/path/to/intel-sycl-tla \
    ./packaging/omni_xpu_kernel/build_uv_wheel_matrix.sh \
    3.10 3.11 3.12 3.13 3.14
```

Override it when the repositories are elsewhere:

```bash
OMNI_XPU_KERNEL_SOURCE=/path/to/llm-scaler/omni/omni_xpu_kernel \
CUTLASS_SYCL_ROOT=/path/to/intel-sycl-tla \
OMNI_XPU_DEVICE=ptl-h \
TORCH_SPEC=torch==2.11.0+xpu \
EXPECTED_TORCH_MINOR=2.11 \
EXPECTED_XPU_TARGET=ptl-h \
    ./packaging/omni_xpu_kernel/build_uv_wheel_matrix.sh 3.12
```

Generated wheels are stored under Kitchen's `wheelhouse/omni_xpu_kernel/` and
are ignored by Git. The uv environments, compiler output, egg-info, and local
extensions stay under the llm-scaler Kernel checkout, where they are also
ignored. Release artifacts are produced and validated manually; this repository
does not publish them through GitHub Actions.

The build venv installs `onednn==2025.3.0` plus matching
`onednn-devel==2025.3.0` headers. The clean runtime venv deliberately installs
only `onednn`; the XPU Torch wheel does not declare it, while the omni wheel is
installed with `--no-deps` to prevent a package index from replacing XPU Torch
with a generic build.

Compilation runs from a filtered per-Python source copy under `MATRIX_ROOT`.
Git metadata, old `build`/`dist`/egg-info directories, in-tree shared objects,
and Python bytecode are excluded so a shared llm-scaler checkout cannot leak
stale artifacts into a companion wheel. Set `KEEP_BUILD_TREES=1` only when the
staged source and compiler output are needed for diagnosis.

## Artifact identity and validation

The companion artifacts are CPython-specific `linux_x86_64` wheels, not
`abi3` or manylinux wheels. They are also specific to the PyTorch ABI and GPU
AOT target. A BMG wheel must not be reused on PTL-H, or vice versa.

For each release candidate:

1. Pin the Intel `llm-scaler` and CUTLASS-SYCL source revisions.
2. Set `OMNI_XPU_DEVICE`, `TORCH_SPEC`, `EXPECTED_TORCH_MINOR`, and
   `EXPECTED_XPU_TARGET` explicitly.
3. Build from the filtered source copy and install the result into a clean
   environment with the matching PyTorch XPU runtime.
4. Verify the package target, version tag, loaded core AOT target, required
   native capabilities, LGRF sidecar, and a real BF16 CUTE attention call.
5. Run the Kernel and Kitchen correctness suites plus representative ComfyUI
   workflows on the target device before publishing.

Wheel hashes identify a particular artifact but do not guarantee reproducible
ZIP bytes. Release notes must record the source revisions, wheel hash, Python
and PyTorch versions, oneAPI/oneDNN runtime requirements, and target GPU.
