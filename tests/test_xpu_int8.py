# SPDX-License-Identifier: Apache-2.0
"""SYCL ConvRot dequantization: numerical, dispatch and stream contracts."""
import json
import os
import subprocess
import sys

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.backends.eager import quantization as eager
from comfy_kitchen.tensor.int8_utils import _build_hadamard


def test_import_without_xpu_runtime():
    code = """
import sys, torch
torch.version.xpu = None
import comfy_kitchen as ck
assert not ck.registry.is_available('xpu')
assert 'comfy_kitchen.backends.xpu._C' not in sys.modules
assert ck.registry.is_available('eager')
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.fixture
def xpu_device():
    if not torch.xpu.is_available() or not ck.registry.is_available("xpu"):
        pytest.skip("compiled SYCL backend and XPU device required")
    return torch.device("xpu", torch.xpu.current_device())


def _compare(q, scale, code):
    kwargs = dict(q=q, scale=scale, group_size=256, output_dtype_code=code)
    assert ck.registry.get_capable_backend("dequantize_int8_convrot_weight_dtype", kwargs) == "xpu"
    out = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(**kwargs)
    ref = eager.dequantize_int8_convrot_weight(q, scale, 256)
    if code == 0:
        # FP32 GEMM reduction order varies with shape/backend. Validate against
        # an independent dense binary64 reference, retaining the FP32 q*scale
        # rounding required by the public API before the rotation.
        products = (q.float() * scale).reshape(-1, 256).cpu()
        precise = torch.empty_like(ref)
        precise_flat = precise.view(-1, 256)
        h = _build_hadamard(256, device="cpu", dtype=torch.float64).T
        unit_roundoff = torch.finfo(torch.float32).eps / 2
        gamma = 256 * unit_roundoff / (1 - 256 * unit_roundoff)
        for start in range(0, products.shape[0], 8192):
            block = products[start:start + 8192].double()
            exact = block @ h
            rounded = exact.float().to(q.device)
            precise_flat[start:start + block.shape[0]].copy_(rounded)
            # Check eager as well; allow its 256-term accumulation error and
            # the final reference rounding, rather than treating it as exact.
            bound = (gamma * block.abs().sum(-1, keepdim=True) / 16
                     + unit_roundoff * exact.abs()).to(q.device)
            eager_block = ref.view(-1, 256)[start:start + block.shape[0]]
            assert ((eager_block - rounded).abs() <= bound).all()
        ref = precise
    # Keep the original 8*eps envelope. For FP32 its reference is now the
    # correctly rounded mathematical result, not another FP32 reduction tree.
    tol = 8 * torch.finfo(torch.float32).eps * ref.abs().amax(dim=1, keepdim=True)
    lo, hi = (ref - tol).to(out.dtype).float(), (ref + tol).to(out.dtype).float()
    assert out.dtype == (torch.float32, torch.float16, torch.bfloat16)[code]
    assert out.shape == q.shape and out.is_contiguous()
    assert torch.isfinite(out).all()
    assert ((out.float() >= lo) & (out.float() <= hi)).all()


@pytest.mark.parametrize("shape", [(1, 256), (17, 256), (257, 768)])
@pytest.mark.parametrize("code", [0, 1, 2])
@pytest.mark.parametrize("scalar", [False, True])
def test_forward(xpu_device, shape, code, scalar):
    q = torch.randint(-128, 128, shape, device=xpu_device, dtype=torch.int8)
    scale = torch.rand((), device=xpu_device) * 1e-2 if scalar else torch.rand(shape[0], 1, device=xpu_device) * 1e-2
    _compare(q, scale, code)


@pytest.mark.parametrize("code", [0, 1, 2])
def test_scale_and_code_boundaries(xpu_device, code):
    q = torch.arange(-128, 128, device=xpu_device, dtype=torch.int8).repeat(8, 1)
    q[1].fill_(127)
    q[2].fill_(-128)
    q[3].zero_()
    q[4, ::2] = 127
    q[4, 1::2] = -127
    scale = torch.tensor([0, 1e-9, 1e-20, 0.5, 0.03125, -0.001, 0.00010001, 1], device=xpu_device).view(-1, 1)
    _compare(q, scale, code)


@pytest.mark.parametrize("code", [0, 1, 2])
def test_unaligned_contiguous_input(xpu_device, code):
    q = torch.randint(-128, 128, (17 * 256 + 1,), device=xpu_device, dtype=torch.int8)[1:].view(17, 256)
    assert q.is_contiguous() and q.storage_offset() == 1
    _compare(q, torch.ones(17, 1, device=xpu_device) * 0.001, code)


@pytest.mark.parametrize("case", ["group64", "strided_q", "strided_scale", "column_scale", "grad_scale"])
def test_unsupported_calls_select_eager(xpu_device, case):
    q = torch.randint(-128, 128, (17, 512), device=xpu_device, dtype=torch.int8)
    scale = torch.rand(17, 1, device=xpu_device)
    group = 256
    if case == "group64":
        group = 64
    elif case == "strided_q":
        q = q[:, ::2]
    elif case == "strided_scale":
        scale = torch.rand(34, 1, device=xpu_device)[::2]
    elif case == "column_scale":
        scale = torch.rand(1, 512, device=xpu_device)
    else:
        scale.requires_grad_(True)
    kwargs = dict(q=q, scale=scale, group_size=group, output_dtype_code=2)
    assert ck.registry.get_capable_backend("dequantize_int8_convrot_weight_dtype", kwargs) == "eager"
    if not scale.requires_grad:
        out = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(**kwargs)
        ref = eager.dequantize_int8_convrot_weight_dtype(**kwargs)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_empty_rows(xpu_device):
    q = torch.empty(0, 256, dtype=torch.int8, device=xpu_device)
    out = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(q, torch.empty(0, 1, device=xpu_device), 256, 2)
    assert out.shape == q.shape and out.dtype == torch.bfloat16


def test_nondefault_stream_producer_and_consumer(xpu_device):
    producer, compute, consumer = (torch.xpu.Stream(device=xpu_device) for _ in range(3))
    default = torch.xpu.current_stream(xpu_device)
    assert len({s.sycl_queue for s in (producer, compute, consumer, default)}) == 4
    host_q = torch.randint(-128, 128, (257, 768), dtype=torch.int8)
    host_scale = torch.rand(257, 1) * 0.001
    for _ in range(5):
        with torch.xpu.stream(producer):
            q = host_q.to(xpu_device)
            scale = host_scale.to(xpu_device)
        with torch.xpu.stream(compute):
            compute.wait_stream(producer)
            out = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(q, scale, 256, 2)
        with torch.xpu.stream(consumer):
            consumer.wait_stream(compute)
            observed = out.clone()
        consumer.synchronize()
        ref = eager.dequantize_int8_convrot_weight_dtype(q, scale, 256, 2)
        torch.testing.assert_close(observed, ref, rtol=0.01, atol=1e-6)


def test_missing_extension_falls_back(xpu_device):
    code = """
import importlib.abc, sys
class MissingExtension(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'comfy_kitchen.backends.xpu._C':
            raise ImportError('extension intentionally absent')
sys.meta_path.insert(0, MissingExtension())
import torch, comfy_kitchen as ck
assert not ck.registry.is_available('xpu')
q = torch.ones(1,256,device='xpu',dtype=torch.int8)
s = torch.ones(1,1,device=q.device)
out = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(q,s,256,2)
assert out.shape == q.shape and torch.isfinite(out).all()
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_native_rejects_cpu_operands(xpu_device):
    from comfy_kitchen.backends import xpu
    q = torch.ones(1, 256, dtype=torch.int8)
    scale = torch.ones(1, 1)
    out = torch.empty(1, 256, device=xpu_device)
    with pytest.raises(TypeError):
        xpu._C.dequantize_convrot(q, scale, out, torch.xpu.current_stream(xpu_device).sycl_queue)


@pytest.mark.parametrize("code", [0, 1, 2])
def test_real_checkpoint_weights(xpu_device, code):
    paths = json.loads(os.getenv("COMFY_KITCHEN_TEST_INT8_CHECKPOINTS", "[]"))
    if not paths:
        pytest.skip("existing checkpoint paths must be supplied explicitly")
    from safetensors import safe_open
    observed = set()
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for key in sorted(keys):
                if not key.endswith(".weight") or key.removesuffix(".weight") + ".weight_scale" not in keys:
                    continue
                view = handle.get_slice(key)
                shape = tuple(view.get_shape())
                if len(shape) != 2 or view.get_dtype() != "I8" or shape in observed:
                    continue
                # Real full-size Qwen DiT and text-encoder matrix families.
                if shape not in {(4096, 4096), (24576, 4096), (4096, 12288), (1024, 4096), (12288, 4096)}:
                    continue
                q = handle.get_tensor(key).to(xpu_device)
                scale = handle.get_tensor(key.removesuffix(".weight") + ".weight_scale").to(xpu_device)
                _compare(q, scale, code)
                observed.add(shape)
    assert len(observed) == 5
