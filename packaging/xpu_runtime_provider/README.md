# Comfy Kitchen XPU runtime provider wheel

This builder converts an already-built `comfy-kitchen` XPU wheel into the
co-installable `comfy-kitchen-xpu-runtime` provider distribution. The provider
owns only `comfy_kitchen_xpu_runtime`; its canonical Kitchen implementation is
stored below that package's private `_vendor` directory. It never installs a
top-level `comfy_kitchen` file, so the official distribution can be installed,
reinstalled, or upgraded independently.

Build the normal XPU wheel first, then run:

```bash
python packaging/xpu_runtime_provider/build_wheel.py \
  --source-wheel dist/comfy_kitchen-0.2.35-py3-none-any.whl \
  --output-dir dist/provider \
  --source-revision "$(git rev-parse HEAD)" \
  --torch-version 2.13.0+xpu \
  --xpu-target bmg
```

`--xpu-target` accepts `bmg`, `ptl-h`, or `dg2`. Use the target matching the
installed `omni_xpu_kernel` companion build. This option only records runtime
activation eligibility; the provider does not compile kernels or establish
support for additional operators. Existing operator capability checks and
input constraints still apply. DG2 admission requires a compatible DG2
companion build and matching Torch XPU runtime.

The output wheel contains a lightweight entry point in
`comfyui_omnixpu.runtime_providers` plus a manifest recording the canonical
version, exact source revision, source-wheel hash, supported runtime, and every
vendored file hash. Importing the provider metadata does not import PyTorch or
Kitchen. ComfyUI-OmniXPU validates the manifest during prestartup and routes
the canonical `comfy_kitchen` import only when the official version and XPU
runtime match.

Run the portable contract tests in the target development container:

```bash
python -m pytest -q tests/test_xpu_runtime_provider_wheel.py
```
