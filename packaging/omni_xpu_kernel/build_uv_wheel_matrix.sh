#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${https_proxy:-}" && -n "${http_proxy:-}" ]]; then
    export https_proxy="${http_proxy}"
fi
if [[ -z "${HTTPS_PROXY:-}" && -n "${https_proxy:-}" ]]; then
    export HTTPS_PROXY="${https_proxy}"
fi

KITCHEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OMNI_XPU_KERNEL_SOURCE="${OMNI_XPU_KERNEL_SOURCE:-${KITCHEN_ROOT}/../llm-scaler/omni/omni_xpu_kernel}"
WHEELHOUSE="${WHEELHOUSE:-${KITCHEN_ROOT}/wheelhouse/omni_xpu_kernel}"
MATRIX_ROOT="${MATRIX_ROOT:-${OMNI_XPU_KERNEL_SOURCE}/.uv-wheel-matrix}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.11.0+xpu}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/xpu}"
ONEDNN_SPEC="${ONEDNN_SPEC:-onednn==2025.3.0}"
ONEDNN_DEVEL_SPEC="${ONEDNN_DEVEL_SPEC:-onednn-devel==2025.3.0}"
EXPECTED_ONEDNN_VERSION="${EXPECTED_ONEDNN_VERSION:-2025.3.0}"
OMNI_XPU_DEVICE="${OMNI_XPU_DEVICE:-ptl-h}"
EXPECTED_TORCH_MINOR="${EXPECTED_TORCH_MINOR:-2.11}"
EXPECTED_XPU_TARGET="${EXPECTED_XPU_TARGET:-${OMNI_XPU_DEVICE}}"

case "${OMNI_XPU_DEVICE}" in
    bmg|ptl-h) ;;
    *)
        echo "Unsupported OMNI_XPU_DEVICE: ${OMNI_XPU_DEVICE}; expected bmg or ptl-h" >&2
        exit 2
        ;;
esac
if [[ "${EXPECTED_XPU_TARGET}" != "${OMNI_XPU_DEVICE}" ]]; then
    echo "EXPECTED_XPU_TARGET must match OMNI_XPU_DEVICE" >&2
    exit 2
fi
if [[ ! "${EXPECTED_TORCH_MINOR}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    echo "EXPECTED_TORCH_MINOR must be a major.minor pair" >&2
    exit 2
fi
export OMNI_XPU_DEVICE EXPECTED_TORCH_MINOR EXPECTED_XPU_TARGET EXPECTED_ONEDNN_VERSION

if [[ ! -f "${OMNI_XPU_KERNEL_SOURCE}/setup.py" ]]; then
    echo "omni_xpu_kernel source not found: ${OMNI_XPU_KERNEL_SOURCE}" >&2
    echo "Set OMNI_XPU_KERNEL_SOURCE to the llm-scaler omni/omni_xpu_kernel checkout." >&2
    exit 2
fi

if [[ -z "${CUTLASS_SYCL_ROOT:-}" ]]; then
    echo "CUTLASS_SYCL_ROOT is required for Kitchen's default cute attention backend." >&2
    exit 2
fi
for required_dir in include tools/util/include examples/common applications; do
    if [[ ! -d "${CUTLASS_SYCL_ROOT}/${required_dir}" ]]; then
        echo "Incomplete CUTLASS_SYCL_ROOT: missing ${required_dir} in ${CUTLASS_SYCL_ROOT}" >&2
        exit 2
    fi
done
export OMNI_XPU_REQUIRE_CUTE=1

if [[ "$#" -gt 0 ]]; then
    PYTHON_VERSIONS=("$@")
else
    PYTHON_VERSIONS=(3.10 3.11 3.12 3.13 3.14)
fi

uv_pip_install() {
    local attempt
    for attempt in 1 2 3; do
        if uv pip install "$@"; then
            return 0
        fi
        if [[ "${attempt}" -lt 3 ]]; then
            echo "uv install failed (attempt ${attempt}/3); retrying..." >&2
            sleep $((attempt * 5))
        fi
    done
    return 1
}

mkdir -p "${WHEELHOUSE}" "${MATRIX_ROOT}"

for version in "${PYTHON_VERSIONS[@]}"; do
    tag="${version//./}"
    venv="${MATRIX_ROOT}/py${tag}"
    smoke_venv="${MATRIX_ROOT}/smoke-py${tag}"
    build_source="${MATRIX_ROOT}/source-py${tag}"
    candidate_dir="${WHEELHOUSE}/.candidates/cp${tag}"

    echo "==> Python ${version}: install interpreter"
    uv python install "${version}"
    uv venv --clear --python "${version}" "${venv}"

    echo "==> Python ${version}: install XPU build dependencies"
    uv_pip_install --python "${venv}/bin/python" \
        --index-url "${TORCH_INDEX_URL}" \
        "${TORCH_SPEC}"
    uv_pip_install --python "${venv}/bin/python" \
        build setuptools wheel cmake pytest numpy \
        "${ONEDNN_SPEC}" "${ONEDNN_DEVEL_SPEC}"

    echo "==> Python ${version}: build Kernel in llm-scaler, emit wheel to Kitchen"
    rm -rf "${candidate_dir}"
    mkdir -p "${candidate_dir}"
    # Build from a filtered copy so stale setuptools output, in-tree extension
    # binaries, and bytecode from the shared Kernel checkout cannot leak into
    # the wheel or suppress a required native rebuild.
    rm -rf "${build_source}"
    mkdir -p "${build_source}"
    tar \
        --exclude=.git \
        --exclude=.uv-wheel-matrix \
        --exclude=build \
        --exclude=dist \
        --exclude='*.egg-info' \
        --exclude=__pycache__ \
        --exclude='*.py[co]' \
        --exclude='*.so' \
        -C "${OMNI_XPU_KERNEL_SOURCE}" -cf - . \
        | tar -C "${build_source}" -xf -
    (
        cd "${build_source}"
        "${venv}/bin/python" -m build \
            --wheel --no-isolation --outdir "${candidate_dir}"
    )

    wheel="$(find "${candidate_dir}" -maxdepth 1 -type f -name "omni_xpu_kernel-*-cp${tag}-*.whl" -print -quit)"
    if [[ -z "${wheel}" ]]; then
        echo "Python ${version}: expected cp${tag} wheel was not produced" >&2
        exit 1
    fi
    final_wheel="${WHEELHOUSE}/$(basename "${wheel}")"
    rm -f "${WHEELHOUSE}"/omni_xpu_kernel-*-cp"${tag}"-*.whl
    mv "${wheel}" "${final_wheel}"
    rm -rf "${candidate_dir}"
    wheel="${final_wheel}"
    "${venv}/bin/python" - "${wheel}" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as wheel_zip:
    names = wheel_zip.namelist()
    leaked_bytecode = [
        name
        for name in names
        if "__pycache__" in name or name.endswith((".pyc", ".pyo"))
    ]
    assert not leaked_bytecode, f"wheel contains stale bytecode: {leaked_bytecode}"
PY
    echo "==> Python ${version}: clean install and XPU smoke test"
    rm -rf "${venv}"
    uv venv --clear --python "${version}" "${smoke_venv}"
    uv_pip_install --python "${smoke_venv}/bin/python" \
        --index-url "${TORCH_INDEX_URL}" \
        "${TORCH_SPEC}"
    # The XPU Torch wheel does not declare oneDNN as a dependency. Install the
    # runtime explicitly; development headers belong only in the build venv.
    uv_pip_install --python "${smoke_venv}/bin/python" "${ONEDNN_SPEC}"
    uv_pip_install --python "${smoke_venv}/bin/python" \
        --force-reinstall --no-deps "${wheel}"
    (
        cd "${MATRIX_ROOT}"
        "${smoke_venv}/bin/python" - <<'PY'
import os
from importlib.metadata import version as distribution_version

import torch
import omni_xpu_kernel
from omni_xpu_kernel import cute

expected_torch_minor = os.environ["EXPECTED_TORCH_MINOR"]
expected_xpu_target = os.environ["EXPECTED_XPU_TARGET"]
expected_onednn_version = os.environ["EXPECTED_ONEDNN_VERSION"]
torch_public = torch.__version__.split("+", 1)[0]
torch_minor = ".".join(torch_public.split(".")[:2])
package_torch_minor = ".".join(omni_xpu_kernel.__torch_version__.split(".")[:2])
target_tag = expected_xpu_target.replace("-", "")
torch_tag = expected_torch_minor.replace(".", "")

assert torch_minor == expected_torch_minor, (
    f"expected Torch {expected_torch_minor}, found {torch.__version__}"
)
assert package_torch_minor == expected_torch_minor, (
    "wheel Torch identity mismatch: "
    f"{omni_xpu_kernel.__torch_version__} vs {torch.__version__}"
)
assert omni_xpu_kernel.__xpu_target__ == expected_xpu_target, (
    "wheel target mismatch: "
    f"{omni_xpu_kernel.__xpu_target__} vs {expected_xpu_target}"
)
assert omni_xpu_kernel.__version__.endswith(f"+torch{torch_tag}.{target_tag}"), (
    f"wheel version lacks Torch/target identity: {omni_xpu_kernel.__version__}"
)
assert distribution_version("onednn") == expected_onednn_version, (
    f"unexpected oneDNN runtime: {distribution_version('onednn')}"
)
assert torch.xpu.is_available(), "PyTorch XPU is unavailable"
assert omni_xpu_kernel.is_available(), "omni native extension is unavailable"
assert omni_xpu_kernel.core_aot_target() == expected_xpu_target, (
    "loaded core AOT target mismatch: "
    f"{omni_xpu_kernel.core_aot_target()} vs {expected_xpu_target}"
)
assert cute.is_available(), "cute FMHA extension is unavailable"
required = {
    "fp8": {"quantize_per_tensor", "dequantize_per_tensor", "stochastic_rounding"},
    "rotary": {"apply_kitchen_rope1", "apply_kitchen_rope_split_half1"},
    "int8": {"rotate_convrot", "quantize_int8_convrot_weight"},
    "norm": {"fused_adaln"},
    "svdq": {"quantize_svdq_act_uint4", "dequantize_svdq_u4"},
}
capabilities = omni_xpu_kernel.native_capabilities()
for module, symbols in required.items():
    missing = symbols - set(capabilities.get(module, ()))
    assert not missing, f"{module} missing symbols: {sorted(missing)}"

q = torch.randn(1, 256, 4, 128, device="xpu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
actual = cute.sdp(q, k, v)
expected = torch.nn.functional.scaled_dot_product_attention(
    q.permute(0, 2, 1, 3),
    k.permute(0, 2, 1, 3),
    v.permute(0, 2, 1, 3),
).permute(0, 2, 1, 3)
torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
try:
    cute.sdp(q[:, :128], k, v)
except RuntimeError as exc:
    assert "only self-attention" in str(exc)
else:
    raise AssertionError("cute must reject unvalidated cross-attention")
print(torch.__version__, omni_xpu_kernel.__version__, torch.xpu.get_device_name())
print("cute", actual.shape, actual.dtype)
PY
    )

    if [[ "${KEEP_VENVS:-0}" != "1" ]]; then
        rm -rf "${smoke_venv}"
    fi
    if [[ "${KEEP_BUILD_TREES:-0}" != "1" ]]; then
        rm -rf "${build_source}"
    fi
done

echo "Built Kitchen companion wheels:"
find "${WHEELHOUSE}" -maxdepth 1 -type f -name 'omni_xpu_kernel-*.whl' -print
