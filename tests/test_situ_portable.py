"""SiTU layout, dtype and input-preservation contracts independent of hardware."""

import pytest
import torch

from comfy_kitchen.backends._activations import apply_input_act, input_act_code, input_act_width


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("factor", [0, 1e-6, 1, 1000])
def test_situ_eager_contract(dtype, factor):
    torch.manual_seed(199)
    x = torch.randn(2, 3, 128, dtype=dtype) * factor
    original = x.clone()
    hidden, gate = x.chunk(2, dim=-1)
    expected = (gate / 4).tanh().mul(4).mul(gate.sigmoid()).mul(
        (hidden / 25).tanh().mul(25))
    actual = apply_input_act(x, "situ")
    assert input_act_code("situ") == 4 and input_act_width("situ") == 2
    assert actual.shape == (2, 3, 64) and actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


def test_situ_hidden_gate_order_is_not_swiglu_order():
    x = torch.tensor([[1000.0, -1000.0, 3.0, -3.0]])
    expected = torch.tensor([[25.0, -25.0]]) * (
        4 * torch.tanh(x[:, 2:] / 4) * torch.sigmoid(x[:, 2:]))
    torch.testing.assert_close(apply_input_act(x, "situ"), expected, rtol=0, atol=0)
