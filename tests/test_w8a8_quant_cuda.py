"""ATTENTION_TEST_CUDA=1 ops/test-dev.sh -k w8a8_quant_cuda."""

import os

import pytest
import torch
import torch.nn.functional as F

from comfyui_turing_utils_kernel.turing_sage import core


pytestmark = pytest.mark.skipif(
    os.environ.get("ATTENTION_TEST_CUDA") != "1", reason="opt-in CUDA check"
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [1, 63, 256, 261, 1285, 4096])
@pytest.mark.parametrize("layout", ["packed", "unaligned"])
def test_w8a8_quant_cuda_value_layout(dtype, length, layout):
    # Exercise packed QKV strides and arbitrary storage offsets/odd strides.
    width = 192 if layout == "packed" else 65
    source = torch.randn(2, length, 3, width, device="cuda", dtype=dtype)
    value = source[..., 128:192] if layout == "packed" else source[..., 1:65]
    value = value.transpose(1, 2)
    padded = (length + 63) // 64 * 64
    actual = torch.empty(2, 3, 64, padded, dtype=torch.int8, device="cuda")
    scale = torch.empty(2, 3, 64, device="cuda")
    core._qattn.quantize_v_int8_sm75(value, actual, scale)
    expected_scale = (value.float().abs().amax(2) / 127).clamp_min(1e-12)
    torch.testing.assert_close(scale, expected_scale, rtol=2e-7, atol=0)
    quantized = (value.float() / expected_scale.unsqueeze(2)).round().clamp(-128, 127)
    src = torch.arange(padded, device="cuda")
    dst = (
        (src & ~15)
        | (src & 1)
        | (((src >> 3) & 1) << 1)
        | (((src >> 1) & 1) << 2)
        | (((src >> 2) & 1) << 3)
    )
    # Reciprocal multiplication can differ by one at a rounding midpoint.
    unpacked = actual[..., dst].transpose(2, 3)
    assert (unpacked[:, :, :length].float() - quantized).abs().max() <= 1
    assert not unpacked[:, :, length:].count_nonzero()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_w8a8_quant_cuda_weak_tiles_survive(dtype, head_dim, causal):
    # An early strong tile must not consume the U8 range of later weak tiles.
    # Running-max U8 quantization rounds all their probabilities to zero.
    q = torch.zeros(1, 2, 257, head_dim, dtype=dtype, device="cuda")
    k = torch.zeros_like(q)
    v = torch.ones_like(q)
    q[..., 0] = 1
    k[:, :, :64, 0] = 8
    v[:, :, :64] = 0
    actual = core.w8a8attn(
        q, k, v, sm_scale=1, is_causal=causal, rotate_qk=False, stabilize_k=False
    )
    expected = F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), scale=1, is_causal=causal
    )
    torch.testing.assert_close(actual.float(), expected, rtol=0.015, atol=1e-6)
    assert actual[:, :, -1].min() > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [63, 65, 257])
def test_w8a8_quant_cuda_zero_scores_exclude_padding(dtype, length):
    q = torch.zeros(1, 2, length, 64, device="cuda", dtype=dtype)
    actual = core.w8a8attn(q, q, torch.ones_like(q), rotate_qk=False, stabilize_k=False)
    torch.testing.assert_close(actual, torch.ones_like(actual), rtol=0, atol=0.002)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_w8a8_quant_cuda_sol_summary_scale(dtype):
    # Constant blocks make Sol's skipped-block summary exact. Check that the
    # floating summary and locally quantized exact PV share the same scale.
    q = torch.zeros(1, 2, 256, 128, dtype=dtype, device="cuda")
    k = torch.zeros_like(q)
    v = torch.ones_like(q)
    q[..., 0] = 1
    k[:, :, :64, 0] = 8
    v[:, :, :64] = 0
    actual = core.sol_sparse_sageattn(
        q,
        k,
        v,
        sm_scale=1,
        use_w8a8=True,
        threshold_sigma=100,
        exact_kv_ranges=((0, 64),),
        rotate_qk=False,
        stabilize_k=False,
    )
    expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=1)
    torch.testing.assert_close(actual.float(), expected, rtol=0.025, atol=1e-5)
