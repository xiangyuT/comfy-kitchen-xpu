import os
import subprocess
import sys

import pytest
import torch

import comfy_kitchen as ck
from comfy_kitchen.exceptions import (
    BackendNotFoundError,
    BackendNotImplementedError,
    NoCapableBackendError,
)


class TestBackendSystem:
    def test_list_backends(self):
        import comfy_kitchen as ck

        backends = ck.list_backends()

        assert isinstance(backends, dict)
        assert "eager" in backends
        assert "xpu" in backends
        assert "triton" in backends

        # Eager backend should always be available
        assert backends["eager"]["available"] is True
        assert "capabilities" in backends["eager"]

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows Triton policy")
    def test_triton_is_unavailable_by_default_on_windows(self):
        env_name = "COMFY_KITCHEN_ENABLE_TRITON_WINDOWS"
        env = os.environ.copy()
        env.pop(env_name, None)

        default_script = f"""
import comfy_kitchen as ck
status = ck.list_backends()["triton"]
assert status["available"] is False, status
assert status["disabled"] is False, status
assert "{env_name}=1" in status["unavailable_reason"], status
"""
        default_result = subprocess.run(
            [sys.executable, "-c", default_script],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert default_result.returncode == 0, default_result.stderr

        env[env_name] = "1"
        opt_in_script = f"""
import comfy_kitchen as ck
status = ck.list_backends()["triton"]
reason = status["unavailable_reason"] or ""
assert "{env_name}=1" not in reason, status
"""
        opt_in_result = subprocess.run(
            [sys.executable, "-c", opt_in_script],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert opt_in_result.returncode == 0, opt_in_result.stderr

    def test_backend_priority(self):
        import comfy_kitchen as ck

        original = list(ck.registry._priority)
        try:
            assert original == ["xpu", "triton", "eager"]
            ck.set_backend_priority(["eager", "xpu", "triton"])
            ck.set_backend_priority(["xpu", "triton", "eager"])
        finally:
            ck.set_backend_priority(original)

    def test_disable_enable_xpu_backend(self):
        status = ck.list_backends()["xpu"]
        if not status["available"]:
            pytest.skip("XPU backend unavailable")
        ck.disable_backend("xpu")
        try:
            assert ck.list_backends()["xpu"]["disabled"] is True
        finally:
            ck.enable_backend("xpu")
        assert ck.list_backends()["xpu"]["disabled"] is False

    def test_disable_enable_backend(self):
        import comfy_kitchen as ck

        # Disable triton
        ck.disable_backend("triton")
        backends = ck.list_backends()
        assert backends["triton"]["disabled"] is True

        # Re-enable
        ck.enable_backend("triton")
        backends = ck.list_backends()
        assert backends["triton"]["disabled"] is False

    def test_int8_capabilities_listed(self):
        """Test that int8 operations are listed in backend capabilities."""
        import comfy_kitchen as ck

        backends = ck.list_backends()

        # Check eager
        eager_caps = backends["eager"]["capabilities"]
        assert "int8_linear" in eager_caps
        assert "quantize_int8_tensorwise" in eager_caps
        assert "quantize_int8_rowwise" in eager_caps
        assert "dequantize_int8_simple" in eager_caps

        # CUDA source remains in the repository for upstream synchronization,
        # but the XPU wheel does not import or package it.
        if backends.get("cuda", {}).get("available", False):
            cuda_caps = backends["cuda"]["capabilities"]
            assert "int8_linear" in cuda_caps

        if backends["xpu"]["available"]:
            from comfy_kitchen.backends import xpu

            xpu_caps = backends["xpu"]["capabilities"]
            assert "int8_linear" in xpu_caps
            assert "mm_int8" in xpu_caps
            assert "quantize_int8_tensorwise" in xpu_caps
            if xpu._SVDQ_AVAILABLE:
                assert "quantize_svdquant_w4a4" in xpu_caps
                assert "scaled_mm_svdquant_w4a4" in xpu_caps
            if xpu._NORM_AVAILABLE:
                assert "adaln" in xpu_caps

    def test_xpu_backend_is_optional(self):
        """Missing XPU hardware or omni extension must not break package import."""
        backends = ck.list_backends()
        status = backends["xpu"]

        assert isinstance(status["available"], bool)
        if not status["available"]:
            assert status["unavailable_reason"]

    def test_missing_omni_does_not_break_clean_import(self):
        script = """
import sys
sys.modules['omni_xpu_kernel'] = None
import comfy_kitchen as ck
status = ck.list_backends()['xpu']
assert status['available'] is False
assert status['unavailable_reason']
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_clean_import_does_not_register_cuda(self):
        script = """
import comfy_kitchen as ck
assert ck.registry._priority == ['xpu', 'triton', 'eager']
assert 'cuda' not in ck.list_backends()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_xpu_constraints_are_device_specific(self):
        from comfy_kitchen.backends.xpu import _build_constraints

        constraints = _build_constraints()
        assert constraints["int8_linear"].default_devices == frozenset({"xpu"})
        assert constraints["mm_int8"].default_devices == frozenset({"xpu"})

    def test_available_xpu_backend_has_native_core(self):
        from comfy_kitchen.backends import xpu

        if not ck.list_backends()["xpu"]["available"]:
            pytest.skip("omni XPU backend is unavailable")

        assert xpu._NATIVE_CAPABILITIES == xpu._REQUIRED_NATIVE_INT8_OPS

    def test_backend_context_manager_override(self, small_tensor):
        """Test that use_backend context manager correctly overrides backend selection."""
        import comfy_kitchen as ck

        scale = torch.tensor([1.0], device=small_tensor.device)

        with ck.use_backend("eager"):
            result = ck.quantize_per_tensor_fp8(small_tensor, scale)

        assert isinstance(result, torch.Tensor)
        assert result.shape == small_tensor.shape


class TestBackendExceptions:
    """Tests for backend exception handling."""

    def test_backend_not_found_error_unregistered(self):
        """Test BackendNotFoundError when requesting unregistered backend."""
        with (
            pytest.raises(BackendNotFoundError, match="not_a_real_backend"),
            ck.use_backend("not_a_real_backend"),
        ):
            pass

    def test_backend_not_found_error_disabled(self):
        """Test BackendNotFoundError when backend is disabled."""
        # Disable eager backend temporarily
        ck.disable_backend("eager")
        try:
            with pytest.raises(BackendNotFoundError, match="disabled"), ck.use_backend("eager"):
                pass
        finally:
            # Re-enable for other tests
            ck.enable_backend("eager")

    def test_backend_not_implemented_error(self):
        """Test BackendNotImplementedError when backend doesn't implement function."""
        # Request a function that doesn't exist from eager backend
        with pytest.raises(BackendNotImplementedError, match="nonexistent_function"):
            ck.registry.get_implementation("nonexistent_function", backend="eager")

    def test_no_capable_backend_error(self):
        """Test NoCapableBackendError when no backend implements function."""
        with pytest.raises(NoCapableBackendError, match="totally_fake_function"):
            ck.registry.get_implementation("totally_fake_function")

    def test_backend_not_found_error_attributes(self):
        """Test BackendNotFoundError has correct attributes."""
        try:
            with ck.use_backend("fake_backend"):
                pass
        except BackendNotFoundError as e:
            assert e.backend_name == "fake_backend"

    def test_backend_not_implemented_error_attributes(self):
        """Test BackendNotImplementedError has correct attributes."""
        try:
            ck.registry.get_implementation("fake_func", backend="eager")
        except BackendNotImplementedError as e:
            assert e.backend_name == "eager"
            assert e.func_name == "fake_func"

    def test_no_capable_backend_error_attributes(self):
        """Test NoCapableBackendError has correct attributes."""
        try:
            ck.registry.get_implementation("fake_function_xyz")
        except NoCapableBackendError as e:
            assert e.func_name == "fake_function_xyz"
            assert isinstance(e.failures, dict)
