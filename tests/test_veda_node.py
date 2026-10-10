from unittest import mock
import copy
from types import SimpleNamespace

import pytest
import torch

from comfyui_turing_utils.nodes.attention import AttentionStrategy, veda_inputs, _ATTENTION_STRATEGIES
from comfyui_turing_utils.attention.runtime import AttentionRuntimeConfig
from comfyui_turing_utils.adapters.minimax.veda.integration import forward_scope, make_override
from comfyui_turing_utils.adapters.minimax.veda.predictor import PredictorBundle, convert_projection
from comfyui_turing_utils.adapters.minimax.veda.plans import PlanTable, TilePlan
from comfyui_turing_utils.adapters.minimax.veda.tiling import TileShape


def test_veda_node_has_predictor_not_lora():
    with mock.patch("comfyui_turing_utils.nodes.attention.predictor_choices", return_value=["test.safetensors"]):
        inputs = veda_inputs()["required"]
    assert inputs["predictor_precision"][0] == ["w8a8", "bf16", "fp16", "fp32"]
    assert inputs["predictor_precision"][1]["default"] == "w8a8"
    assert "lora" not in str(inputs).lower()
    assert "w4a4" not in str(inputs)
    assert AttentionStrategy.define_schema().outputs[0].io_type == "MODEL"
    assert "execution_mode" not in inputs
    assert "projection_chunk_tiles" not in inputs


def test_veda_configuration_has_only_automatic_execution():
    import inspect
    from comfyui_turing_utils.adapters.minimax.veda.integration import configure
    from comfyui_turing_utils.adapters.minimax.veda.engine import attend
    assert 'execution_mode' not in inspect.signature(configure).parameters
    assert 'projection_chunk_tiles' not in inspect.signature(configure).parameters
    assert 'heterogeneous' not in inspect.signature(attend).parameters


def test_veda_removed_execution_controls_are_rejected():
    with pytest.raises(TypeError, match="execution_mode"):
        AttentionStrategy.execute(None, {"strategy": "veda", "predictor_name": "test.safetensors",
                                           "execution_mode": "heterogeneous"})


def test_veda_strategy_replaces_sol_and_keeps_dense_backend():
    dense, sol, veda = (lambda *args: None for _ in range(3))
    original = AttentionRuntimeConfig("w8a8", "native", dense, "sol", "sol", sol)
    updated = original.with_strategy("veda", "veda:sm75_int8_qk_fp16_pv", veda)
    assert updated.active_override is veda
    assert updated.dense_override is dense
    assert original.active_override is sol
    assert updated.with_strategy("sol", "sol", sol).active_override is sol


def test_veda_configuration_uses_shared_dense_runtime():
    from comfyui_turing_utils.adapters.minimax.veda import integration
    from comfyui_turing_utils.attention.runtime import install_attention_runtime

    dense = mock.Mock()
    runtime = AttentionRuntimeConfig("sdpa", "test", dense)
    model = SimpleNamespace(
        load_device=torch.device("cpu"), model_options={"transformer_options": {}},
        model=SimpleNamespace(diffusion_model=SimpleNamespace(
            blocks=[SimpleNamespace(attn=SimpleNamespace(heads=1, head_dim=64))])),
        remove_wrappers_with_key=mock.Mock(), add_wrapper_with_key=mock.Mock(),
    )
    install_attention_runtime(model.model_options["transformer_options"], runtime)
    bundle = SimpleNamespace(num_layers=1, num_heads=1, head_dim=64, keep_ratio=0.1)
    installed = SimpleNamespace(layout=SimpleNamespace(installed=True), model=model)
    with (
        mock.patch.object(integration, "is_minimax_h3_model", return_value=True),
        mock.patch.object(integration, "kernel_extension_has_symbol", return_value=True),
        mock.patch.object(integration, "register_predictor_folder", return_value=mock.Mock()),
        mock.patch.object(integration, "load_bundle", return_value=bundle),
        mock.patch.object(integration, "install_attention_strategy", return_value=installed) as install,
    ):
        assert integration.configure(model, predictor_name="test.safetensors") is model
    assert install.call_args.kwargs["runtime_config"] is runtime
    assert model.model_options["transformer_options"]["turing_utils_veda"].bundle is bundle
    model.add_wrapper_with_key.assert_called_once()


def test_veda_node_delegates_without_lora_mutation():
    model, result = object(), object()
    configure = mock.Mock(return_value=result)
    with mock.patch.dict(_ATTENTION_STRATEGIES, {"veda": configure}):
        assert AttentionStrategy.execute(model, {"strategy": "veda", "predictor_name": "test.safetensors"}).result == (result,)
    configure.assert_called_once_with(model, predictor_name="test.safetensors")


def test_veda_bundle_is_shared_read_only_between_model_clones():
    projection = convert_projection(torch.ones(1, 12, 4), "bf16")
    table = PlanTable([TilePlan("test", (1, 8, 16), [TileShape(1, 8, 16)], [[0]])])
    bundle = PredictorBundle("bf16", 1, 1, 4, 0.1, table, (projection,), (projection,))
    clone = copy.deepcopy({"bundle": bundle})
    assert clone["bundle"] is bundle
    staged = projection.stage([0], torch.device("cpu"))
    assert staged.weight.data_ptr() != projection.weight.data_ptr()


@pytest.mark.parametrize("fail", [True, False])
def test_veda_forward_scratch_is_cleared_even_on_exception(fail):
    options = {"turing_utils_attention_strategy": "veda"}
    seen = []
    def executor(x, timestep, context, opts):
        scratch = opts["turing_utils_veda_forward_cache"]
        seen.append(scratch)
        scratch["temporary"] = torch.ones(5)
        if fail:
            raise ValueError("test failure")
        return x
    if fail:
        with pytest.raises(ValueError, match="test failure"):
            forward_scope(executor, 1, 2, 3, options)
    else:
        assert forward_scope(executor, 1, 2, 3, options) == 1
    assert seen == [{}]
    assert "turing_utils_veda_forward_cache" not in options


def test_veda_keeps_h3_text_refiner_dense():
    dense = mock.Mock(return_value="dense")
    schedule = mock.Mock()
    schedule.is_dense.return_value = False
    override = make_override(None, dense, schedule)
    q = torch.zeros(1, 2, 3, 128)
    options = {"minimax_h3_layout": SimpleNamespace(seq_len=143)}
    assert override(None, q, q, q, 2, skip_reshape=True, transformer_options=options) == "dense"
    dense.assert_called_once()
    schedule.is_dense.assert_not_called()
