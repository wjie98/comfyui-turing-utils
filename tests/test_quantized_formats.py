"""Shared storage identification and the official grouped W6 checkpoint contract."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from comfyui_turing_utils.quantization.convrot import (
    ConvRotSummary,
    _summarize_convrot_modules,
    configure_convrot_activation,
)
from comfyui_turing_utils.quantization.formats import (
    convrot_storage_kind,
    describe_weight_storage,
    grouped_weight_geometry,
)
from comfyui_turing_utils.quantization.fusions import convrot_weight_kind


def packed_codes(codes, bits):
    low = ((codes[:, 0::2] & 15) | ((codes[:, 1::2] & 15) << 4)).to(torch.int8)
    if bits == 4:
        return low
    upper = codes.reshape(codes.shape[0], -1, 4) >> 4
    shifts = torch.arange(4, device=codes.device) * 2
    return torch.cat((low, (upper << shifts).sum(-1).to(torch.int8)), dim=1)


def grouped_weight(
    bits,
    dtype=torch.bfloat16,
    *,
    device="cpu",
    group_size=16,
    scale_dtype=torch.float8_e4m3fn,
):
    from comfy.quant_ops import AsymW4A8Int8Layout, QuantizedTensor

    n, k = 8, 256
    codes = (torch.arange(n * k, device=device) % (1 << bits)).reshape(n, k)
    params = AsymW4A8Int8Layout.Params(
        orig_dtype=dtype,
        orig_shape=(n, k),
        scale=torch.ones((n, k // group_size), dtype=scale_dtype, device=device),
        s_channel=torch.full((n,), 0.01, device=device),
        codebook=torch.linspace(-1, 1, 16, device=device) if bits == 4 else None,
        group_size=group_size,
        convrot_groupsize=256,
    )
    return QuantizedTensor(packed_codes(codes, bits), "AsymW4A8Int8Layout", params)


@pytest.mark.parametrize("bits,kind", [(4, "codebook_w4a8"), (6, "w6a8")])
def test_storage_and_compute_eligibility_are_separate(bits, kind):
    weight = grouped_weight(bits, torch.float16)
    assert describe_weight_storage(weight).kind == kind
    assert convrot_storage_kind(weight) == kind
    assert convrot_weight_kind(weight) is None
    bf16 = weight.to(torch.bfloat16)
    assert convrot_weight_kind(bf16) == kind
    assert convrot_storage_kind(bf16.t()) is None


@pytest.mark.parametrize(
    "packed,scales,group",
    [
        ((8, 128), (7, 16), 16),
        ((8, 129), (8, 16), 16),
        ((8, 144), (8, 24), 8),
        ((8, 18), (8, 3), 8),
        ((8, 128), (8, 16), 5),
    ],
)
def test_reject_invalid_grouped_geometry(packed, scales, group):
    with pytest.raises(ValueError):
        grouped_weight_geometry(packed, scales, group)


def test_w6_rejects_a_codebook():
    weight = grouped_weight(6)
    weight._params = dataclasses.replace(weight._params, codebook=torch.ones(16))
    with pytest.raises(ValueError, match="Only grouped W4A8"):
        describe_weight_storage(weight)


def test_nvfp4_storage_is_not_mistaken_for_already_rotated_weight():
    weight = SimpleNamespace(
        _layout_cls="TensorCoreNVFP4Layout", _params=SimpleNamespace(transposed=False)
    )
    assert describe_weight_storage(weight).kind == "nvfp4"
    assert describe_weight_storage(weight).rotation_group is None
    assert convrot_storage_kind(weight) is None


def test_official_w6_load_and_metadata_roundtrip():
    import comfy.ops

    weight = grouped_weight(6)
    config = {"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256}
    state = weight.state_dict("layer.weight")
    state["layer.comfy_quant"] = torch.tensor(
        list(json.dumps(config).encode()), dtype=torch.uint8
    )
    metadata = {"_quantization_metadata": json.dumps({"layers": {"layer": config}})}
    _, summary = configure_convrot_activation(state, metadata, True)
    assert summary == ConvRotSummary(w6a8=1)
    assert "linear_dtype" not in json.loads(
        state["layer.comfy_quant"].numpy().tobytes()
    )

    root = torch.nn.Module()
    root.layer = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(
        256, 8, bias=False
    )
    root.load_state_dict(state, strict=True)
    assert _summarize_convrot_modules(root) == summary
    assert root.layer.weight._qdata.shape == (8, 192)
    assert root.layer.state_dict()["weight"].shape == (8, 192)


def test_mixed_w4_w6_metadata_and_workspace_scan():
    from comfyui_turing_utils.adapters.memory import scan_quantized_workspaces

    root = torch.nn.Module()
    state = {}
    for bits in (4, 6):
        layer = f"w{bits}"
        weight = grouped_weight(bits)
        state.update(weight.state_dict(layer + ".weight"))
        state[layer + ".comfy_quant"] = torch.tensor(
            list(b'{"format":"asym_w4a8_int8"}'), dtype=torch.uint8
        )
        module = torch.nn.Module()
        module.register_parameter(
            "weight", torch.nn.Parameter(weight, requires_grad=False)
        )
        root.add_module(layer, module)
    _, summary = configure_convrot_activation(state, None, False)
    assert summary == ConvRotSummary(codebook_w4a8=1, w6a8=1)
    profile = scan_quantized_workspaces(root, convrot_storage_kind)
    assert dict(profile.formats) == {"codebook_w4a8": 1, "w6a8": 1}
    assert profile.transient_bytes(10000) > 0


@pytest.mark.parametrize("bits", [4, 6])
def test_grouped_load_rejects_unconsumed_correction(bits):
    weight = grouped_weight(bits)
    state = weight.state_dict("layer.weight")
    state["layer.comfy_quant"] = torch.tensor(
        list(b'{"format":"asym_w4a8_int8"}'), dtype=torch.uint8
    )
    state["layer.weight_correction"] = torch.zeros(8, 16)
    with pytest.raises(ValueError, match="unsupported weight_correction"):
        configure_convrot_activation(state, None, False)


def test_w6_lora_requantization_preserves_six_bit_storage():
    import comfy_kitchen

    weight = grouped_weight(6)
    with comfy_kitchen.use_backend("eager"):
        dense = weight.dequantize()
        delta = torch.randn(8, 2) @ torch.randn(2, 256) * 0.002
        patched = weight.requantize_from_float(
            dense + delta.to(dense), scale="recalculate"
        )
    assert describe_weight_storage(patched).kind == "w6a8"
    assert patched._qdata.shape == weight._qdata.shape
    assert patched._params.codebook is None


def test_w6_custom_op_has_a_fullgraph_fake_contract():
    import comfyui_turing_utils_kernel as kernel

    operation = torch.compile(
        kernel.turing_codebook_w4a8_linear, backend="eager", fullgraph=True
    )
    output = operation(
        torch.empty((8193, 256), device="meta", dtype=torch.int8),
        torch.empty((80, 192), device="meta", dtype=torch.int8),
        torch.empty(8193, device="meta"),
        torch.empty((80, 16), device="meta", dtype=torch.float8_e4m3fn),
        torch.empty(80, device="meta"),
    )
    assert output.shape == (8193, 80)
    assert output.dtype is torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("m,k", [(3, 256), (8193, 96)])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float8_e4m3fn])
def test_w6_cuda_inline_and_staged_match_integer_oracle(m, k, scale_dtype):
    import comfyui_turing_utils_kernel as kernel

    n = 80
    generator = torch.Generator(device="cuda").manual_seed(91)
    activation = torch.randint(
        -127, 128, (m, k), dtype=torch.int8, device="cuda", generator=generator
    )
    codes = torch.randint(1, 64, (n, k), device="cuda", generator=generator)
    weight = packed_codes(codes, 6)
    scales = (torch.rand((n, k // 16), device="cuda", generator=generator) * 4).to(
        scale_dtype
    )
    row_scale = torch.rand(m, device="cuda", generator=generator) * 0.01
    channel_scale = torch.rand(n, device="cuda", generator=generator) * 0.01
    bias = torch.rand(n, device="cuda", generator=generator) * 0.1
    decoded = (
        ((codes - 32) * scales.float().repeat_interleave(16, dim=1))
        .round()
        .clamp(-127, 127)
    )
    reference = (
        (activation.float() @ decoded.float().t()) * row_scale[:, None]
    ) * channel_scale[None, :] + bias
    for chunk in [0, 16, -1] if m > 8192 else [0, 16]:
        output = kernel.turing_codebook_w4a8_linear(
            activation,
            weight,
            row_scale,
            scales,
            channel_scale,
            None,
            bias,
            chunk_rows=chunk,
        )
        torch.testing.assert_close(
            output, reference.to(torch.bfloat16), rtol=0.008, atol=0.001
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("group_size", [16, 32, 64])
@pytest.mark.parametrize("scale_dtype", [torch.float8_e4m3fn, torch.float32])
def test_w6_scoped_dispatch_and_swiglu_use_local_kernel(group_size, scale_dtype):
    from unittest.mock import patch
    from comfyui_turing_utils.quantization import dispatch
    from comfyui_turing_utils.quantization.backend import register_backend
    from comfyui_turing_utils.quantization.fusions import (
        convrot_linear_input_act_from_weight,
    )
    from comfyui_turing_utils.quantization.operator_scope import (
        use_turing_operator_backend,
    )

    weight = grouped_weight(
        6, device="cuda", group_size=group_size, scale_dtype=scale_dtype
    )
    x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    activation, row_scale = dispatch._quantize_turing_int8_activation(x, 256)
    decoded_s8 = (torch.arange(8 * 256, device="cuda") % 64 - 32).reshape(8, 256)
    reference = (
        (activation.float() @ decoded_s8.float().t()) * row_scale.reshape(-1, 1)
    ) * weight._params.s_channel[None, :]
    assert register_backend()
    with patch.object(dispatch, "_kernel_op", wraps=dispatch._kernel_op) as resolve:
        with use_turing_operator_backend():
            output = torch.nn.functional.linear(x, weight)
        assert output.shape == (3, 8)
        torch.testing.assert_close(
            output, reference.to(torch.bfloat16), rtol=0.008, atol=0.001
        )
        assert any(
            call.args == ("turing_codebook_w4a8_linear",)
            for call in resolve.call_args_list
        )
    with patch.object(dispatch, "_kernel_op", wraps=dispatch._kernel_op) as resolve:
        output = convrot_linear_input_act_from_weight(
            weight, None, torch.cat((x, x), dim=-1), "swiglu"
        )
        assert output.shape == (3, 8)
        assert any(
            call.args == ("turing_codebook_w4a8_linear",)
            for call in resolve.call_args_list
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_w6_runtime_contract_and_preflight():
    from comfyui_turing_utils.precision import prepare_turing_runtime
    from comfyui_turing_utils.runtime.capabilities import runtime_capabilities
    from comfyui_turing_utils.runtime.diagnostics import runtime_diagnostics

    device = torch.device("cuda", 0)
    assert runtime_capabilities(device).supports("grouped_w6a8").supported
    assert runtime_diagnostics(device)["support"]["grouped_w6a8"]["supported"]
    prepare_turing_runtime(ConvRotSummary(w6a8=1), device, "sdpa")
