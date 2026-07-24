# Comfy Kitchen XPU

Pure-Python Comfy Kitchen integration for Intel XPU. The wheel packages the
XPU, Triton, and eager backends and uses the target-specific
`omni_xpu_kernel` companion wheel for native Intel GPU kernels.

CUDA source is retained in the source repository for synchronization with
upstream Comfy Kitchen, but the CUDA backend and native CUDA artifacts are not
included in this wheel.

See the
[Comfy Kitchen XPU repository](https://github.com/xiangyuT/comfy-kitchen-xpu)
for installation and companion-wheel build instructions. This work is based on
and remains grateful to
[Comfy-Org/comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen).
