# XPU QuantizedTensor Test Migration

Last updated: 2026-07-11

`tests/test_tensor.py` currently contains 117 collected cases under CUDA-only
class markers. They are not 117 missing XPU functions. This audit separates
portable QuantizedTensor behavior from explicitly deferred formats and
CUDA-hardware capability checks.

## Classification

| Category | Cases | Action |
|---|---:|---|
| Portable FP8/INT8 and generic QuantizedTensor behavior | 57 | Parameterize for `cuda` and `xpu` |
| NVFP4/MXFP8 cases deferred by project scope | 52 | Keep CUDA-only |
| CUDA compute-capability/SM checks | 8 | Keep CUDA-only; add separate XPU capability tests if needed |
| Total | 117 | |

### Portable cases (57)

| Test class or subset | Cases | XPU treatment |
|---|---:|---|
| `TestTensorCoreFP8Layout` | 5 | Direct device parameterization |
| `TestQuantizedTensor` generic/FP8 subset | 20 | Replace hard-coded `cuda` with device fixture |
| `TestQuantizedTensorFlatten` FP8/INT8 subset | 2 | Direct device parameterization |
| `TestCopyValidation` | 2 | Use FP8 versus INT8 for mismatched-layout validation |
| `TestBaseLayoutParams` FP8 base inheritance case | 1 | Split from deferred format cases |
| `TestParamsDtypeValidation` | 1 | Direct device parameterization |
| `TestFP8LinearOperations` | 10 | Parameterize; distinguish native W8A16 shapes from safe fallback shapes |
| `TestFP8ViewOperations` | 11 | Direct device parameterization and dequantized-value checks |
| `TestTensorWiseINT8Layout` | 3 | Direct device parameterization |
| `TestINT8LinearOperations` | 2 | Compare XPU dispatch with eager on the same XPU tensors |

### Deferred-format cases (52)

- NVFP4 layout, linear, and shape behavior: 21 cases.
- MXFP8 layout, linear, and shape behavior: 19 cases.
- NVFP4-specific cases inside `TestQuantizedTensor`: 5 cases.
- NVFP4/MXFP8 flatten cases: 2 cases.
- NVFP4/MXFP8 `BaseLayoutParams` cases: 5 cases.

### CUDA-only capability cases (8)

`TestCapabilityChecking` asserts CUDA SM versions and CUDA fast-matmul
requirements. These should not be made device-generic. XPU native-symbol,
device-count, and oneDNN-path health checks live in `tests/test_xpu.py`.

## Acceptance

1. The 57 portable cases run on both CUDA (when present) and XPU.
2. XPU runs do not silently select CPU eager because all source tensors remain
   on XPU; eager references use `ck.use_backend("eager")` on those tensors.
3. Layout lifecycle coverage includes clone, detach, copy, empty-like,
   dtype-only conversion, CPU round-trip, flatten/unflatten, view/reshape,
   transpose, linear, mm, and addmm where the layout supports them.
4. Native FP8 and INT8 fast paths have at least one valid oneDNN shape; small
   shapes explicitly validate the safe fallback.
5. The 52 deferred cases and 8 CUDA capability cases retain explicit reasons
   and are not counted as XPU failures.

## Result

Completed on 2026-07-11. The dual-device fixture collects CUDA and XPU variants
for the portable classes. On the XPU-only validation host:

```text
57 passed, 129 skipped
```

The 57 passes exactly match the portable matrix above. The skip count includes
unavailable CUDA variants plus explicitly deferred NVFP4/MXFP8 and CUDA-SM
cases. Native INT8 versus eager comparisons use the existing oneDNN acceptance
bounds (`mean < 0.15`, `max < 0.75`) because oneDNN fuses BF16 rescaling while
eager uses an INT32-then-float sequence.
