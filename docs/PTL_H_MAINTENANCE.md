# PTL-H development maintenance

Last updated: 2026-07-23

## Branch identity

PTL-H Kitchen work is maintained on `dev/ptl-h-kitchen-xpu`. It was forked
from Kitchen `dev/kitchen_xpu` at
`22149d6ee76e00ea933710ed9ff1da5b6c59d7e2`; the existing branch remains the
BMG integration line.

The initial PTL-H companion source profile is:

| Component | Value |
|---|---|
| Kernel repository | `xiangyuT/llm-scaler` |
| Kernel development ref | `dev/roofline-kernel-tuning` |
| Pinned source checkpoint | `dfc364da1f77ea6ea102df13f3177af9b36b4b81` |
| PyTorch XPU | `2.11.0+xpu` |
| AOT target | `ptl-h` |
| Expected wheel local tag | `+torch211.ptlh` |

The branch starts with configuration and source/API compatibility checks. A
previous Kernel result does not by itself accept the Kitchen integration; the
companion wheel, Kitchen tests, and workflow checks below still have to pass
from this branch.

## Ownership boundary

- Kitchen owns backend selection, Python adapters, companion-wheel
  orchestration, and Kitchen-level tests.
- llm-scaler owns native Kernel implementation and Kernel-level tests.
- Kernel tuning records own measurements and rejected experiments; measured
  data is not copied into this public integration branch.
- Model pipeline patches are outside the normal Kitchen iteration path and
  require explicit approval before work begins.

Native wheels are both Torch-ABI- and GPU-target-specific. BMG and PTL-H use
separate AOT artifacts. The build smoke test checks package metadata and the
target embedded in the loaded core extension so a mislabeled or stale binary
fails before Kitchen tests run.

## Build profile

For a local PTL-H build, check out the pinned Kernel source and provide the
CUTLASS-SYCL checkout required by CUTE:

```bash
export OMNI_XPU_KERNEL_SOURCE=/path/to/llm-scaler/omni/omni_xpu_kernel
export CUTLASS_SYCL_ROOT=/path/to/sycl-tla
export OMNI_XPU_DEVICE=ptl-h
export TORCH_SPEC=torch==2.11.0+xpu
export EXPECTED_TORCH_MINOR=2.11
export EXPECTED_XPU_TARGET=ptl-h
./packaging/omni_xpu_kernel/build_uv_wheel_matrix.sh 3.12
```

The script defaults to this PTL-H profile. Its explicit environment variables
remain useful in logs and prevent an inherited shell setting from silently
changing the artifact identity. The manual GitHub workflow uses a runner whose
device label matches `OMNI_XPU_DEVICE`.

## Acceptance ladder

1. Source/API: required FP8, Kitchen RoPE, INT8/ConvRot, AdaLN, SVDQuant, and
   CUTE entry points exist; shell, Python, and workflow files pass static
   validation.
2. Companion wheel: build from the pinned source, clean-install it, verify
   Torch 2.11, `__xpu_target__ == "ptl-h"`, `core_aot_target() == "ptl-h"`,
   required native capabilities, and a real CUTE BF16 attention call.
3. Operator integration: pass the Kernel acceptance suite and Kitchen XPU
   tests, followed by Kitchen's portable full suite.
4. Workflow integration: run the maintained lightweight Boogu and Krea2
   workflows at 1024 x 1024 from the approved Dockerfile-built environment,
   including workflow switching and memory checks.

An image rebuild is a phase boundary, not an inner-loop action. Iterations use
the existing approved container and a rebuild is proposed only after a coherent
fix set is ready for approval.

## Current state

Branch initialization was validated in the existing approved PTL-H container
with Torch `2.11.0+xpu`; the installed wheel reported
`0.1.0b8.dev0+torch211.ptlh`, package target `ptl-h`, and native core AOT target
`ptl-h`. Kitchen results were:

- focused XPU/backend/INT8 selection: 77 passed, 28 skipped, 5 deselected;
- portable full selection (`-k "not cuda"`): 465 passed, 123 skipped, 207
  deselected.

The focused run exposed a deterministic ConvRot row-quantization boundary:
native FP32 inverse-scale math and eager BF16 scale division can differ by one
INT8 unit while producing identical row scales. Kitchen now enforces the same
maximum-one-unit contract as the native Kernel correctness suite.

This is adapter/runtime acceptance against the preinstalled PTL-H artifact,
not acceptance of a wheel newly built by this branch. A clean companion-wheel
matrix and workflow acceptance remain pending and must not be inferred from the
BMG status document or from a different container image.
