# Omni XPU companion wheels

Comfy Kitchen owns the build, validation, and release flow for the
`omni_xpu_kernel` wheels used by its optional XPU backend. The native Kernel
source remains in the llm-scaler repository; it is not copied into the pure
Python `comfy-kitchen` wheel.

This separation is intentional:

- `comfy-kitchen` selects and calls backend capabilities;
- llm-scaler maintains the Intel XPU Kernel implementations;
- Kitchen's release flow builds CPython- and device-target-specific companion
  wheels from a pinned llm-scaler revision.

## Local build

The default source path expects sibling `comfy-kitchen` and `llm-scaler`
checkouts. `CUTLASS_SYCL_ROOT` is mandatory because cute FMHA is the default
ComfyUI-OmniXPU attention backend:

```bash
CUTLASS_SYCL_ROOT=/path/to/intel-sycl-tla \
    ./packaging/omni_xpu_kernel/build_uv_wheel_matrix.sh \
    3.10 3.11 3.12 3.13 3.14
```

Override it when the repositories are elsewhere:

```bash
OMNI_XPU_KERNEL_SOURCE=/path/to/llm-scaler/omni/omni_xpu_kernel \
CUTLASS_SYCL_ROOT=/path/to/intel-sycl-tla \
    ./packaging/omni_xpu_kernel/build_uv_wheel_matrix.sh 3.12
```

Generated wheels are stored under Kitchen's `wheelhouse/omni_xpu_kernel/` and
are ignored by Git. The uv environments, compiler output, egg-info, and local
extensions stay under the llm-scaler Kernel checkout, where they are also
ignored. Release artifacts are uploaded by
`.github/workflows/build-omni-xpu-wheels.yml`.

## Current milestone status

Cute became a required Kitchen artifact on 2026-07-13. The full matrix rebuild
is intentionally deferred until the project milestone is declared:

Current llm-scaler source is versioned `0.1.0-b8-dev`; its wheel metadata and
filename use the PEP 440-normalized `0.1.0b8.dev0`. The artifacts below predate
that version alignment and must not be published as b8 wheels.

| Python | Interpreter | Current status | Wheel bytes | SHA-256 |
|---|---|---|---:|---|
| 3.10 | 3.10.20 | pending milestone rebuild | — | — |
| 3.11 | 3.11.15 | legacy, quarantined: missing cute | 908771 | `a4ae12974fa68024036bc3ed38949fd097dce4f18241f103c3015854975b2057` |
| 3.12 | 3.12.13 | accepted with cute | 1015302 | `3fb88eb72532f207d7974ef6a927865de2dfd79c4f8adfd02c95a23eecc7ef7a` |
| 3.13 | 3.13.14 | legacy, quarantined: missing cute | 911267 | `48732d04300a02c93d929ac55b5472dba76d33f356bbc11fc2acd37cdbe2c25d` |
| 3.14 | 3.14.6 | legacy, quarantined: missing cute | 911402 | `b0013da8b99e221425a80ddd5db74169e6ce06c46eb12fa4763d37f36db604ea` |

An accepted wheel must contain and load the ABI-specific
`cute/cute_fmha_torch` extension. Clean-install smoke runs a real BF16 cute
attention call in addition to native extension loading, XPU availability, and
required FP8, RoPE, INT8, ConvRot, AdaLN, and unsigned-SVDQuant symbol checks.
It also verifies that the current cute Kernel rejects cross-attention rather
than returning an unvalidated result.

Only the accepted cp312 wheel is kept at the top level of
`wheelhouse/omni_xpu_kernel/`. Old wheels that predate the cute requirement are
under `legacy-no-cute/` so release globs cannot select them. Wheel hashes are
artifact identifiers, not reproducible-build guarantees; rebuilding can
change ZIP metadata.

These artifacts are CPython-specific `linux_x86_64` wheels, not `abi3` or
manylinux wheels. A release must pin the llm-scaler source revision and state
the required PyTorch XPU, oneAPI runtime, and target GPU.
