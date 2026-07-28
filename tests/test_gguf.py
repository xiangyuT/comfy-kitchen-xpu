import struct

import pytest
import torch

import comfy_kitchen as ck


BLOCK_BYTES = {
    "q4_0": 18,
    "q8_0": 34,
    "q4_k": 144,
    "q6_k": 210,
}


def _half_bytes(value: float) -> list[int]:
    return list(struct.pack("<e", value))


def _make_blocks(quant_type: str, count: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260728)
    blocks = torch.randint(
        0,
        256,
        (count, BLOCK_BYTES[quant_type]),
        dtype=torch.uint8,
        generator=generator,
    )
    for index in range(count):
        if quant_type in {"q4_0", "q8_0"}:
            blocks[index, :2] = torch.tensor(
                _half_bytes(0.125 * (index + 1)), dtype=torch.uint8
            )
        elif quant_type == "q4_k":
            blocks[index, :2] = torch.tensor(
                _half_bytes(0.03125 * (index + 1)), dtype=torch.uint8
            )
            blocks[index, 2:4] = torch.tensor(
                _half_bytes(0.015625 * (index + 1)), dtype=torch.uint8
            )
        else:
            blocks[index, -2:] = torch.tensor(
                _half_bytes(0.03125 * (index + 1)), dtype=torch.uint8
            )
    return blocks


def _half(data: list[int], dtype: torch.dtype) -> torch.Tensor:
    value = struct.unpack("<e", bytes(data))[0]
    return torch.tensor(value, dtype=dtype)


def _reference_q4_0(
    block: list[int],
    dtype: torch.dtype,
    layout: str,
) -> torch.Tensor:
    scale = _half(block[:2], dtype)
    low = [((value & 0x0F) - 8) for value in block[2:]]
    high = [((value >> 4) - 8) for value in block[2:]]
    if layout == "comfyui":
        values = low + high
    else:
        values = [item for pair in zip(low, high) for item in pair]
    return scale * torch.tensor(values, dtype=dtype)


def _reference_q8_0(block: list[int], dtype: torch.dtype) -> torch.Tensor:
    scale = _half(block[:2], dtype)
    values = [value if value < 128 else value - 256 for value in block[2:]]
    return scale * torch.tensor(values, dtype=dtype)


def _scale_min(scales: list[int]) -> tuple[list[int], list[int]]:
    scale = [value & 0x3F for value in scales[:4]]
    minimum = [value & 0x3F for value in scales[4:8]]
    for index in range(4):
        scale.append((scales[8 + index] & 0x0F) | ((scales[index] >> 2) & 0x30))
        minimum.append(
            (scales[8 + index] >> 4) | ((scales[4 + index] >> 2) & 0x30)
        )
    return scale, minimum


def _reference_q4_k(block: list[int], dtype: torch.dtype) -> torch.Tensor:
    d = _half(block[:2], dtype)
    dmin = _half(block[2:4], dtype)
    scale, minimum = _scale_min(block[4:16])
    packed = block[16:]
    output = []
    for chunk in range(4):
        for shift in (0, 4):
            group = 2 * chunk + shift // 4
            group_scale = d * torch.tensor(scale[group], dtype=dtype)
            group_min = dmin * torch.tensor(minimum[group], dtype=dtype)
            values = [
                (packed[chunk * 32 + index] >> shift) & 0x0F
                for index in range(32)
            ]
            output.append(
                group_scale * torch.tensor(values, dtype=dtype) - group_min
            )
    return torch.cat(output)


def _reference_q6_k(block: list[int], dtype: torch.dtype) -> torch.Tensor:
    ql = block[:128]
    qh = block[128:192]
    scales = [value if value < 128 else value - 256 for value in block[192:208]]
    d = _half(block[208:210], dtype)

    ql_values = []
    for chunk in range(2):
        for shift in (0, 4):
            ql_values.extend(
                (value >> shift) & 0x0F for value in ql[chunk * 64 : (chunk + 1) * 64]
            )

    qh_values = []
    for chunk in range(2):
        for shift in (0, 2, 4, 6):
            qh_values.extend(
                (value >> shift) & 0x03 for value in qh[chunk * 32 : (chunk + 1) * 32]
            )

    output = []
    for index, (low, high) in enumerate(zip(ql_values, qh_values, strict=True)):
        value = (low | (high << 4)) - 32
        group_scale = d * torch.tensor(scales[index // 16], dtype=dtype)
        output.append(group_scale * torch.tensor(value, dtype=dtype))
    return torch.stack(output)


def _reference(
    data: torch.Tensor,
    quant_type: str,
    dtype: torch.dtype,
    layout: str = "comfyui",
) -> torch.Tensor:
    blocks = data.reshape(-1, BLOCK_BYTES[quant_type]).tolist()
    references = []
    for block in blocks:
        if quant_type == "q4_0":
            references.append(_reference_q4_0(block, dtype, layout))
        elif quant_type == "q8_0":
            references.append(_reference_q8_0(block, dtype))
        elif quant_type == "q4_k":
            references.append(_reference_q4_k(block, dtype))
        else:
            references.append(_reference_q6_k(block, dtype))
    return torch.cat(references)


@pytest.mark.parametrize("quant_type", ["q4_0", "q8_0", "q4_k", "q6_k"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_eager_matches_independent_reference(quant_type, dtype):
    data = _make_blocks(quant_type)
    expected = _reference(data, quant_type, dtype)
    with ck.use_backend("eager"):
        actual = ck.dequantize_gguf(data, quant_type, output_dtype=dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_q4_0_layout_is_explicit(dtype):
    data = _make_blocks("q4_0")
    with ck.use_backend("eager"):
        sequential = ck.dequantize_gguf(
            data, "q4_0", output_dtype=dtype, layout="comfyui"
        )
        interleaved = ck.dequantize_gguf(
            data, "q4_0", output_dtype=dtype, layout="interleaved"
        )
    expected = sequential.reshape(-1, 2, 16).transpose(1, 2).reshape(-1)
    assert torch.equal(interleaved.view(torch.uint8), expected.view(torch.uint8))


def test_noncontiguous_storage_is_flattened_in_logical_order():
    data = _make_blocks("q8_0", count=4).transpose(0, 1)
    assert not data.is_contiguous()
    with ck.use_backend("eager"):
        actual = ck.dequantize_gguf(data, "q8_0")
        expected = ck.dequantize_gguf(data.reshape(-1).contiguous(), "q8_0")
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"quant_type": "q5_0"}, ValueError, "unsupported GGUF quant_type"),
        (
            {"quant_type": "q4_0", "output_dtype": torch.float32},
            ValueError,
            "unsupported GGUF output_dtype",
        ),
        (
            {"quant_type": "q8_0", "layout": "interleaved"},
            ValueError,
            "only defined for quant_type='q4_0'",
        ),
        (
            {"quant_type": "q4_0", "layout": "native"},
            ValueError,
            "unsupported GGUF layout",
        ),
    ],
)
def test_public_contract_rejects_unsupported_options(kwargs, error, message):
    with pytest.raises(error, match=message):
        ck.dequantize_gguf(torch.zeros(18, dtype=torch.uint8), **kwargs)


def test_public_contract_rejects_wrong_storage_dtype_and_size():
    with pytest.raises(TypeError, match="torch.uint8"):
        ck.dequantize_gguf(torch.zeros(18, dtype=torch.int8), "q4_0")
    with pytest.raises(ValueError, match="multiple of 18"):
        ck.dequantize_gguf(torch.zeros(17, dtype=torch.uint8), "q4_0")


def test_empty_storage_returns_empty_result():
    with ck.use_backend("eager"):
        output = ck.dequantize_gguf(torch.empty(0, dtype=torch.uint8), "q4_0")
    assert output.shape == (0,)
    assert output.dtype == torch.float16


def test_eager_capability_is_registered():
    backends = ck.list_backends()
    assert "dequantize_gguf" in backends["eager"]["capabilities"]


def test_route_diagnostics_report_completed_eager_route():
    ck.get_gguf_route_diagnostics(reset=True)
    with ck.use_backend("eager"):
        ck.dequantize_gguf(_make_blocks("q4_0"), "q4_0")
    assert ck.get_gguf_route_diagnostics(reset=True) == {
        "routes": {"eager": 1},
        "fallbacks": {},
    }
