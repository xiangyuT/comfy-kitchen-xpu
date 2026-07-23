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
| CUTLASS-SYCL checkpoint | `525faea3f0f43a8aec2d21a70d44111db639a3a9` |
| PyTorch XPU | `2.11.0+xpu` |
| AOT target | `ptl-h` |
| Expected wheel local tag | `+torch211.ptlh` |
| oneDNN build/runtime | `onednn-devel==2025.3.0` / `onednn==2025.3.0` |

Acceptance belongs to the complete source/wheel/test/workflow tuple. A previous
Kernel result or a wheel from another target does not by itself accept the
Kitchen integration.

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

The Python 3.12 PTL-H development tuple passed the complete acceptance ladder
on 2026-07-23. The accepted local artifacts are:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `omni_xpu_kernel-0.1.0b8.dev0+torch211.ptlh-cp312-cp312-linux_x86_64.whl` | 1812687 | `ac614bbcbe86e651a7c1d913be04878c3d09b5318bd95b3e9c4c842ed6b17b09` |
| `comfy_kitchen-0.2.18-py3-none-any.whl` | 129011 | `05362811d68a747d32b2a13eff9f3efb63585fc609ea780f7bd24647f92bea70` |

The native artifact was built from a filtered source copy, contained all three
core/LGRF/CUTE shared objects, contained no Python bytecode, clean-installed
with pip oneDNN runtime only, had no unresolved ELF dependencies, and passed a
real CUTE BF16 call. Package target, version tag, loaded core AOT target, and
Torch minor all matched PTL-H/Torch 2.11.

Acceptance results were:

- Kernel installed-wheel runtime: 544 passed, 2 skipped;
- Kernel source packaging: 26 passed, including a clean-copy LGRF wheel build;
- Kitchen focused XPU/backend/INT8/version: 78 passed, 28 skipped, 5 deselected;
- Kitchen portable full selection (`-k "not cuda"`): 466 passed, 123 skipped,
  207 deselected;
- ComfyUI single-process 1024 x 1024 switch: Boogu INT8 -> Krea2 INT8 -> Boogu
  INT8, three valid RGB outputs, service remained healthy, no Level Zero OOR or
  OOM, and approximately 40.1 GB XPU memory remained available after the final
  workflow.

The focused run exposed a deterministic ConvRot row-quantization boundary:
native FP32 inverse-scale math and eager BF16 scale division can differ by one
INT8 unit while producing identical row scales. Kitchen now enforces the same
maximum-one-unit contract as the native Kernel correctness suite.

The current ComfyUI image declares `comfy-kitchen==0.2.16`; runtime acceptance
explicitly replaced it with the branch's `0.2.18` pure-Python wheel. Updating
that image pin and rebuilding the image is the next release phase and still
requires explicit approval. It is not needed to reproduce the accepted
container-layer integration above.
