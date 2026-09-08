from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))

import comfy.ops  # noqa: E402
import comfy.sd  # noqa: E402
import nodes as comfy_nodes  # noqa: E402

sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.nodes.loaders import (  # noqa: E402
    ConvRotCLIPLoader,
    ConvRotDiffusionModelLoader,
)
from comfyui_turing_utils.quantization.convrot import (  # noqa: E402
    ConvRotSummary,
    _convrot_skip_reason,
    configure_convrot_activation,
)
from comfyui_turing_utils.loading.convrot import (  # noqa: E402
    convrot_model_names,
    load_convrot_clip,
    load_convrot_model,
    validate_runtime_support,
)


def quant_tensor(config: dict) -> torch.Tensor:
    return torch.tensor(list(json.dumps(config).encode("utf-8")), dtype=torch.uint8)


def quant_config(value: torch.Tensor) -> dict:
    return json.loads(value.numpy().tobytes())


class ConvRotActivationTest(unittest.TestCase):
    def test_grouped_codebook_w4a8_metadata_is_preserved(self):
        key = "blocks.0.mlp.fc1.comfy_quant"
        state_dict = {
            key: quant_tensor(
                {
                    "format": "asym_w4a8_int8",
                    "group_size": 16,
                    "convrot": True,
                    "convrot_groupsize": 256,
                }
            )
        }

        _, summary = configure_convrot_activation(state_dict, None, True)

        self.assertEqual(summary, ConvRotSummary(codebook_w4a8=1))
        self.assertNotIn("linear_dtype", quant_config(state_dict[key]))

    def test_grouped_codebook_w4a8_rejects_non_convrot_metadata(self):
        state_dict = {
            "blocks.0.mlp.fc1.comfy_quant": quant_tensor(
                {"format": "asym_w4a8_int8", "convrot": False}
            )
        }
        with self.assertRaisesRegex(ValueError, "must declare convrot=true"):
            configure_convrot_activation(state_dict, None, False)

    def test_false_uses_w4_format_default(self):
        metadata = {
            "_quantization_metadata": json.dumps(
                {"layers": {"blocks.0.ffn.0": {"format": "convrot_w4a4"}}}
            )
        }

        updated, summary = configure_convrot_activation({}, metadata, False)

        config = json.loads(updated["_quantization_metadata"])["layers"]["blocks.0.ffn.0"]
        self.assertNotIn("linear_dtype", config)
        self.assertEqual(summary, ConvRotSummary(w4a4=1))

    def test_w4_override_updates_header_and_tensor_metadata(self):
        metadata = {
            "_quantization_metadata": json.dumps(
                {"layers": {"blocks.0.ffn.0": {"format": "convrot_w4a4"}}}
            )
        }
        state_dict = {
            "blocks.1.ffn.0.comfy_quant": quant_tensor(
                {"format": "convrot_w4a4", "linear_dtype": "int4"}
            )
        }

        updated, summary = configure_convrot_activation(state_dict, metadata, True)

        header_config = json.loads(updated["_quantization_metadata"])["layers"]["blocks.0.ffn.0"]
        self.assertEqual(header_config["linear_dtype"], "int8")
        self.assertEqual(quant_config(state_dict["blocks.1.ffn.0.comfy_quant"])["linear_dtype"], "int8")
        self.assertEqual(summary, ConvRotSummary(w4a8=2))

    def test_model_prefix_excludes_aio_text_encoder_layers(self):
        state_dict = {
            "model.diffusion_model.blocks.0.ffn.0.comfy_quant": quant_tensor(
                {"format": "convrot_w4a4", "linear_dtype": "int4"}
            ),
            "text_encoders.qwen.layers.0.mlp.comfy_quant": quant_tensor(
                {"format": "convrot_w4a4", "linear_dtype": "int4"}
            ),
        }

        _, summary = configure_convrot_activation(
            state_dict,
            None,
            True,
            model_prefix="model.diffusion_model.",
        )

        self.assertEqual(summary, ConvRotSummary(w4a8=1))
        self.assertEqual(
            quant_config(
                state_dict[
                    "model.diffusion_model.blocks.0.ffn.0.comfy_quant"
                ]
            )["linear_dtype"],
            "int8",
        )
        self.assertEqual(
            quant_config(
                state_dict[
                    "text_encoders.qwen.layers.0.mlp.comfy_quant"
                ]
            )["linear_dtype"],
            "int4",
        )

    def test_false_rejects_w8a4_metadata(self):
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": quant_tensor(
                {"format": "int8_tensorwise", "convrot": True, "linear_dtype": "int4"}
            )
        }

        with self.assertRaisesRegex(ValueError, "W8 ConvRot supports INT8 activations only"):
            configure_convrot_activation(state_dict, None, False)

    def test_force_int8_accepts_mixed_w4_and_w8_convrot(self):
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": quant_tensor({"format": "convrot_w4a4"}),
            "blocks.1.ffn.0.comfy_quant": quant_tensor(
                {"format": "int8_tensorwise", "params": {"convrot": True}}
            ),
        }

        _, summary = configure_convrot_activation(state_dict, None, True)

        self.assertEqual(summary, ConvRotSummary(w4a8=1, w8a8=1))

    def test_legacy_w8_metadata_without_format_is_normalized(self):
        key = "double_blocks.0.img_attn.qkv.comfy_quant"
        state_dict = {
            key: quant_tensor(
                {"convrot": True, "convrot_groupsize": 256, "per_row": True}
            )
        }

        _, summary = configure_convrot_activation(state_dict, None, False)

        self.assertEqual(summary, ConvRotSummary(w8a8=1))
        self.assertEqual(quant_config(state_dict[key])["format"], "int8_tensorwise")

    def test_legacy_int8_rowwise_format_is_normalized(self):
        metadata = {
            "_quantization_metadata": json.dumps(
                {
                    "layers": {
                        "double_blocks.0.img_attn.qkv": {
                            "format": "int8_rowwise",
                            "convrot": True,
                            "convrot_groupsize": 256,
                            "per_row": True,
                        }
                    }
                }
            )
        }

        updated, summary = configure_convrot_activation({}, metadata, False)

        config = json.loads(updated["_quantization_metadata"])["layers"][
            "double_blocks.0.img_attn.qkv"
        ]
        self.assertEqual(summary, ConvRotSummary(w8a8=1))
        self.assertEqual(config["format"], "int8_tensorwise")

    def test_legacy_w8_metadata_requires_per_row_layout(self):
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": quant_tensor(
                {"convrot": True, "convrot_groupsize": 256}
            )
        }

        with self.assertRaisesRegex(ValueError, "unsupported weight format None"):
            configure_convrot_activation(state_dict, None, False)

    def test_normalized_legacy_w8_loads_with_native_comfy_ops(self):
        state_dict = {
            "layer.weight": torch.zeros((8, 256), dtype=torch.int8),
            "layer.weight_scale": torch.ones((8, 1), dtype=torch.float32),
            "layer.comfy_quant": quant_tensor(
                {
                    "convrot": True,
                    "convrot_groupsize": 256,
                    "per_row": True,
                }
            ),
        }
        configure_convrot_activation(state_dict, None, False)
        operations = comfy.ops.mixed_precision_ops(
            {"mixed_ops": True}, compute_dtype=torch.bfloat16
        )
        model = torch.nn.Module()
        model.layer = operations.Linear(256, 8, bias=False, device=torch.device("cpu"))

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        self.assertEqual(model.layer.quant_format, "int8_tensorwise")
        self.assertTrue(model.layer.weight._params.convrot)
        self.assertEqual(model.layer.weight._params.convrot_groupsize, 256)

    def test_false_preserves_explicit_w4a8(self):
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": quant_tensor(
                {"format": "convrot_w4a4", "params": {"linear_dtype": "int8"}}
            )
        }

        _, summary = configure_convrot_activation(state_dict, None, False)

        self.assertEqual(summary, ConvRotSummary(w4a8=1))

    def test_force_int8_overrides_invalid_w8a4_metadata(self):
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": quant_tensor(
                {"format": "int8_tensorwise", "convrot": True, "linear_dtype": "int4"}
            )
        }

        _, summary = configure_convrot_activation(state_dict, None, True)

        self.assertEqual(quant_config(state_dict["blocks.0.ffn.0.comfy_quant"])["linear_dtype"], "int8")
        self.assertEqual(summary, ConvRotSummary(w8a8=1))

    def test_non_convrot_model_is_rejected(self):
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": quant_tensor({"format": "int8_tensorwise"})
        }

        with self.assertRaisesRegex(ValueError, "does not contain supported ConvRot"):
            configure_convrot_activation(state_dict, None, False)

    def test_unsupported_convrot_format_is_rejected(self):
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": quant_tensor(
                {"format": "float8_e4m3fn", "convrot": True}
            )
        }

        with self.assertRaisesRegex(ValueError, "unsupported weight format"):
            configure_convrot_activation(state_dict, None, False)

    def test_w8_metadata_cannot_declare_int4_activations(self):
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": quant_tensor(
                {"format": "int8_tensorwise", "convrot": True, "linear_dtype": "int4"}
            )
        }

        with self.assertRaisesRegex(ValueError, "W8 ConvRot supports INT8 activations only"):
            configure_convrot_activation(state_dict, None, False)

    def test_false_rejects_conflicting_duplicate_metadata(self):
        metadata = {
            "_quantization_metadata": json.dumps(
                {
                    "layers": {
                        "blocks.0.ffn.0": {
                            "format": "convrot_w4a4",
                            "linear_dtype": "int8",
                        }
                    }
                }
            )
        }
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": quant_tensor({"format": "convrot_w4a4"})
        }

        with self.assertRaisesRegex(ValueError, "Conflicting ConvRot metadata"):
            configure_convrot_activation(state_dict, metadata, False)

    def test_invalid_header_json_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contains invalid JSON"):
            configure_convrot_activation({}, {"_quantization_metadata": "{"}, False)

    def test_invalid_tensor_json_is_rejected(self):
        state_dict = {
            "blocks.0.ffn.0.comfy_quant": torch.tensor(list(b"{"), dtype=torch.uint8)
        }

        with self.assertRaisesRegex(ValueError, "contains invalid quantization JSON"):
            configure_convrot_activation(state_dict, None, False)

    def test_non_boolean_force_int8_gemm_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "force_int8_gemm must be boolean"):
            configure_convrot_activation({}, None, "int8")

    def test_w4a8_accepts_installed_cuda_extension_disabled_by_comfyui_policy(self):
        backends = {
            "cuda": {
                "available": True,
                "disabled": True,
                "unavailable_reason": None,
            }
        }
        with (
            mock.patch("comfy_kitchen.list_backends", return_value=backends),
            mock.patch(
                "comfy.model_management.get_torch_device",
                return_value=torch.device("cuda", 0),
            ),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)),
        ):
            validate_runtime_support(ConvRotSummary(w4a8=1))

    def test_w4a8_accepts_enabled_turing_cuda_backend(self):
        backends = {
            "cuda": {
                "available": True,
                "disabled": False,
                "unavailable_reason": None,
            }
        }
        with (
            mock.patch("comfy_kitchen.list_backends", return_value=backends),
            mock.patch("comfy.model_management.get_torch_device", return_value=torch.device("cuda", 0)),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)),
        ):
            validate_runtime_support(ConvRotSummary(w4a8=1))

    def test_w4a8_rejects_explicit_cpu_load_device(self):
        backends = {
            "cuda": {
                "available": True,
                "disabled": False,
                "unavailable_reason": None,
            }
        }
        with (
            mock.patch("comfy_kitchen.list_backends", return_value=backends),
            mock.patch("torch.cuda.is_available", return_value=True),
            self.assertRaisesRegex(RuntimeError, "requires an NVIDIA CUDA load device"),
        ):
            validate_runtime_support(ConvRotSummary(w4a8=1), torch.device("cpu"))


class ConvRotModelFilterTest(unittest.TestCase):
    def _save(self, directory: str, name: str, tensors: dict, metadata=None) -> Path:
        path = Path(directory) / name
        save_file(tensors, path, metadata=metadata)
        return path

    def test_supported_tensor_metadata_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(
                directory,
                "convrot.safetensors",
                {"blocks.0.ffn.0.comfy_quant": quant_tensor({"format": "convrot_w4a4"})},
            )

            self.assertIsNone(_convrot_skip_reason(path))

    def test_supported_header_metadata_is_accepted(self):
        metadata = {
            "_quantization_metadata": json.dumps(
                {"layers": {"blocks.0.ffn.0": {"format": "convrot_w4a4"}}}
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(
                directory,
                "convrot.safetensors",
                {"blocks.0.ffn.0.weight": torch.zeros((1, 1))},
                metadata,
            )

            self.assertIsNone(_convrot_skip_reason(path))

    def test_legacy_w8_tensor_metadata_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(
                directory,
                "legacy-convrot.safetensors",
                {
                    "double_blocks.0.img_attn.qkv.comfy_quant": quant_tensor(
                        {
                            "convrot": True,
                            "convrot_groupsize": 256,
                            "per_row": True,
                        }
                    )
                },
            )

            self.assertIsNone(_convrot_skip_reason(path))

    def test_minimax_h3_int8_convrot_text_encoder_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(
                directory,
                "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                {
                    "model.embed_tokens.comfy_quant": quant_tensor(
                        {"format": "int8_tensorwise"}
                    ),
                    "model.layers.0.mlp.down_proj.comfy_quant": quant_tensor(
                        {
                            "format": "int8_tensorwise",
                            "convrot": True,
                            "convrot_groupsize": 256,
                        }
                    ),
                },
                metadata={
                    "minimax_h3_te": json.dumps(
                        {
                            "num_hidden_layers": 50,
                            "output": "unnormalized_hidden_after_layer_50",
                        }
                    )
                },
            )

            self.assertIsNone(_convrot_skip_reason(path))

    def test_minimax_h3_int4_convrot_text_encoder_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._save(
                directory,
                "qwen3vl_32b_minimax_h3_int4_convrot.safetensors",
                {
                    "model.layers.0.mlp.down_proj.comfy_quant": quant_tensor(
                        {
                            "format": "convrot_w4a4",
                            "convrot_groupsize": 256,
                        }
                    )
                },
            )

            self.assertIsNone(_convrot_skip_reason(path))

    def test_dense_and_non_convrot_quantized_models_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            dense = self._save(
                directory,
                "dense.safetensors",
                {"blocks.0.ffn.0.weight": torch.zeros((1, 1))},
            )
            int8 = self._save(
                directory,
                "int8.safetensors",
                {
                    "blocks.0.ffn.0.comfy_quant": quant_tensor(
                        {"format": "int8_tensorwise"}
                    )
                },
            )

            self.assertIn("does not contain supported ConvRot", _convrot_skip_reason(dense))
            self.assertIn("does not contain supported ConvRot", _convrot_skip_reason(int8))

    def test_model_list_only_contains_supported_convrot_files(self):
        with tempfile.TemporaryDirectory() as directory:
            convrot = self._save(
                directory,
                "convrot.safetensors",
                {"blocks.0.ffn.0.comfy_quant": quant_tensor({"format": "convrot_w4a4"})},
            )
            dense = self._save(
                directory,
                "dense.safetensors",
                {"blocks.0.ffn.0.weight": torch.zeros((1, 1))},
            )
            paths = {
                convrot.name: str(convrot),
                dense.name: str(dense),
            }
            with (
                mock.patch(
                    "folder_paths.get_filename_list",
                    return_value=[convrot.name, dense.name, "model.gguf"],
                ),
                mock.patch(
                    "folder_paths.get_full_path",
                    side_effect=lambda _folder, name: paths.get(name),
                ),
            ):
                names = convrot_model_names("diffusion_models")

            self.assertEqual(names, [convrot.name])

    def test_stale_workflow_selection_is_rejected_before_loading(self):
        with (
            mock.patch(
                "folder_paths.get_full_path_or_raise",
                return_value="/models/dense.safetensors",
            ),
            mock.patch(
                "comfyui_turing_utils.loading.convrot._convrot_skip_reason",
                return_value="does not contain supported ConvRot quantization metadata",
            ),
            mock.patch("comfyui_turing_utils.loading.convrot.load_convrot_model") as load_model,
            self.assertRaisesRegex(ValueError, "is not a supported ConvRot model"),
        ):
            ConvRotDiffusionModelLoader().load_diffusion_model("dense.safetensors")

        load_model.assert_not_called()


class FakeQuantLinear(torch.nn.Module):
    def __init__(self, weight_format: str, activation_dtype: str):
        super().__init__()
        self.quant_format = weight_format
        self.weight = SimpleNamespace(
            _params=SimpleNamespace(
                linear_dtype=activation_dtype,
                convrot=weight_format == "int8_tensorwise",
            )
        )


class FakeCLIP:
    def __init__(self, weight_format: str, activation_dtype: str):
        self.cond_stage_model = torch.nn.Module()
        self.cond_stage_model.layer = FakeQuantLinear(weight_format, activation_dtype)
        self.patcher = SimpleNamespace(cached_patcher_init=None)


class FakeModel:
    def __init__(self, weight_format: str, activation_dtype: str):
        self.model = torch.nn.Module()
        self.model.layer = FakeQuantLinear(weight_format, activation_dtype)
        self.model_options = {}
        self.cached_patcher_init = None
        self.compute_dtype = None

    def set_model_compute_dtype(self, dtype):
        self.compute_dtype = dtype


class ConvRotCLIPLoaderTest(unittest.TestCase):
    def test_load_model_defaults_to_w8a8_and_preserves_it_for_reload(self):
        state_dict = {
            "model.diffusion_model.blocks.0.ffn.0.comfy_quant": quant_tensor(
                {"format": "convrot_w4a4"}
            ),
            "text_encoders.qwen.layers.0.mlp.comfy_quant": quant_tensor(
                {"format": "convrot_w4a4"}
            ),
        }
        fake_model = FakeModel("convrot_w4a4", "int4")

        with (
            mock.patch("comfy.utils.load_torch_file", return_value=(state_dict, {})),
            mock.patch(
                "comfy.model_detection.unet_prefix_from_state_dict",
                return_value="model.diffusion_model.",
            ),
            mock.patch(
                "comfy.model_detection.model_config_from_unet",
                return_value=SimpleNamespace(unet_config={"image_model": "wan"}),
            ),
            mock.patch("comfyui_turing_utils.loading.convrot.validate_runtime_support"),
            mock.patch("comfyui_turing_utils.loading.convrot.prepare_turing_runtime") as prepare_runtime,
            mock.patch("comfyui_turing_utils.loading.convrot.select_compute_dtype", return_value=None),
            mock.patch("comfyui_turing_utils.loading.convrot.normalize_turing_convrot_weight_dtypes") as normalize_dtypes,
            mock.patch("comfy.sd.load_diffusion_model_state_dict", return_value=fake_model) as load_state,
            mock.patch("comfyui_turing_utils.loading.convrot.apply_model_adapters") as apply_adapters,
            mock.patch("comfyui_turing_utils.loading.convrot.apply_attention_backend") as apply_backend,
        ):
            loaded = load_convrot_model("model.safetensors")

        self.assertIs(loaded, fake_model)
        prepare_runtime.assert_called_once_with(
            ConvRotSummary(w4a4=1), torch.device("cuda", 0), "w8a8"
        )
        self.assertEqual(load_state.call_args.kwargs["model_options"], {})
        self.assertIsNone(fake_model.compute_dtype)
        normalize_dtypes.assert_not_called()
        apply_adapters.assert_called_once_with(fake_model, torch.device("cuda", 0))
        apply_backend.assert_called_once_with(
            fake_model,
            "w8a8",
            device=torch.device("cuda", 0),
            native_runtime=True,
        )
        self.assertEqual(
            fake_model.cached_patcher_init,
            (
                load_convrot_model,
                ("model.safetensors", False, "w8a8"),
            ),
        )

    def test_selected_bf16_is_scoped_to_the_model_patcher(self):
        state_dict = {
            "model.diffusion_model.blocks.0.ffn.0.comfy_quant": quant_tensor(
                {"format": "convrot_w4a4"}
            )
        }
        fake_model = FakeModel("convrot_w4a4", "int4")
        model_config = SimpleNamespace(unet_config={"image_model": "minimax_h3"})

        with (
            mock.patch("comfy.utils.load_torch_file", return_value=(state_dict, {})),
            mock.patch("comfy.model_detection.unet_prefix_from_state_dict", return_value="model.diffusion_model."),
            mock.patch("comfy.model_detection.model_config_from_unet", return_value=model_config),
            mock.patch("comfy.model_management.get_torch_device", return_value=torch.device("cuda", 0)),
            mock.patch("comfyui_turing_utils.loading.convrot.validate_runtime_support"),
            mock.patch("comfyui_turing_utils.loading.convrot.prepare_turing_runtime") as prepare_runtime,
            mock.patch("comfyui_turing_utils.loading.convrot.select_compute_dtype", return_value=torch.bfloat16),
            mock.patch("comfyui_turing_utils.loading.convrot.normalize_turing_convrot_weight_dtypes") as normalize_dtypes,
            mock.patch("comfy.sd.load_diffusion_model_state_dict", return_value=fake_model) as load_state,
            mock.patch("comfyui_turing_utils.loading.convrot.apply_model_adapters") as apply_adapters,
            mock.patch("comfyui_turing_utils.loading.convrot.apply_attention_backend") as apply_backend,
        ):
            loaded = load_convrot_model("model.safetensors")

        self.assertIs(loaded, fake_model)
        prepare_runtime.assert_called_once_with(
            ConvRotSummary(w4a4=1), torch.device("cuda", 0), "w8a8"
        )
        self.assertEqual(load_state.call_args.kwargs["model_options"], {"dtype": torch.bfloat16})
        self.assertEqual(fake_model.compute_dtype, torch.bfloat16)
        normalize_dtypes.assert_called_once_with(
            fake_model, torch.device("cuda", 0), torch.bfloat16
        )
        apply_adapters.assert_called_once_with(fake_model, torch.device("cuda", 0))
        apply_backend.assert_called_once_with(
            fake_model,
            "w8a8",
            device=torch.device("cuda", 0),
            native_runtime=True,
        )

    def test_load_clip_forces_mixed_ops_and_preserves_activation_override(self):
        state_dict = {
            "text_model.encoder.layers.0.mlp.fc1.comfy_quant": quant_tensor(
                {"format": "convrot_w4a4"}
            )
        }
        fake_clip = FakeCLIP("convrot_w4a4", "int8")
        captured = {}

        def fake_load_text_encoder(state_dicts, **kwargs):
            captured["state_dicts"] = state_dicts
            captured["kwargs"] = kwargs
            return fake_clip

        with (
            mock.patch("comfy.utils.load_torch_file", return_value=(state_dict, {})),
            mock.patch("comfy.model_management.text_encoder_device", return_value=torch.device("cuda", 0)),
            mock.patch("comfyui_turing_utils.loading.convrot.validate_runtime_support") as validate,
            mock.patch("comfyui_turing_utils.loading.convrot.prepare_turing_runtime") as prepare_runtime,
            mock.patch("comfy.sd.load_text_encoder_state_dicts", side_effect=fake_load_text_encoder),
        ):
            loaded = load_convrot_clip(
                "clip.safetensors",
                clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
                force_int8_gemm=True,
            )

        self.assertIs(loaded, fake_clip)
        validate.assert_called_once_with(ConvRotSummary(w4a8=1), torch.device("cuda", 0))
        prepare_runtime.assert_called_once_with(
            ConvRotSummary(w4a8=1), torch.device("cuda", 0)
        )
        config = quant_config(
            captured["state_dicts"][0][
                "text_model.encoder.layers.0.mlp.fc1.comfy_quant"
            ]
        )
        self.assertEqual(config["linear_dtype"], "int8")
        self.assertEqual(
            captured["kwargs"]["model_options"]["quantization_metadata"],
            {"mixed_ops": True},
        )
        self.assertIsNotNone(fake_clip.patcher.cached_patcher_init[0])

    def test_load_clip_converts_header_metadata_to_tensor_metadata(self):
        metadata = {
            "_quantization_metadata": json.dumps(
                {
                    "layers": {
                        "text_model.encoder.layers.0.mlp.fc1": {
                            "format": "convrot_w4a4"
                        }
                    }
                }
            )
        }
        fake_clip = FakeCLIP("convrot_w4a4", "int4")
        captured = {}

        def fake_load_text_encoder(state_dicts, **kwargs):
            captured["state_dict"] = state_dicts[0]
            return fake_clip

        with (
            mock.patch("comfy.utils.load_torch_file", return_value=({}, metadata)),
            mock.patch("comfy.model_management.text_encoder_device", return_value=torch.device("cpu")),
            mock.patch("comfy.sd.load_text_encoder_state_dicts", side_effect=fake_load_text_encoder),
        ):
            load_convrot_clip("clip.safetensors")

        key = "text_model.encoder.layers.0.mlp.fc1.comfy_quant"
        self.assertIn(key, captured["state_dict"])
        self.assertEqual(quant_config(captured["state_dict"][key])["format"], "convrot_w4a4")

    def test_load_clip_rejects_layers_not_loaded_as_quantized(self):
        state_dict = {
            "text_model.encoder.layers.0.mlp.fc1.comfy_quant": quant_tensor(
                {"format": "convrot_w4a4"}
            )
        }
        fake_clip = SimpleNamespace(
            cond_stage_model=torch.nn.Module(),
            patcher=SimpleNamespace(cached_patcher_init=None),
        )

        with (
            mock.patch("comfy.utils.load_torch_file", return_value=(state_dict, {})),
            mock.patch("comfy.model_management.text_encoder_device", return_value=torch.device("cpu")),
            mock.patch("comfy.sd.load_text_encoder_state_dicts", return_value=fake_clip),
            self.assertRaisesRegex(RuntimeError, "was not applied to every CLIP layer"),
        ):
            load_convrot_clip("clip.safetensors")

    def test_node_matches_single_clip_loader_contract(self):
        inputs = ConvRotCLIPLoader.INPUT_TYPES()
        self.assertIn("clip_name", inputs["required"])
        self.assertIn("type", inputs["required"])
        self.assertIn("mage", inputs["required"]["type"][0])
        self.assertIn("minimax", inputs["required"]["type"][0])
        self.assertEqual(
            inputs["required"]["force_int8_gemm"],
            (
                "BOOLEAN",
                {
                    "default": False,
                    "tooltip": (
                        "False follows each layer's activation format. "
                        "True forces INT8 GEMM activations."
                    ),
                },
            ),
        )
        self.assertNotIn("activation_dtype", inputs["required"])
        self.assertIn("device", inputs["optional"])
        self.assertEqual(ConvRotCLIPLoader.RETURN_TYPES, ("CLIP",))

    def test_node_reads_clip_types_from_official_loader(self):
        official_types = ["stable_diffusion", "mage", "minimax"]
        official_names = [
            "qwen3vl_32b_minimax_h3_int4_convrot.safetensors",
            "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        ]
        official_device = (("default", "cpu"), {"advanced": True})
        with (
            mock.patch.object(
                comfy_nodes.CLIPLoader,
                "INPUT_TYPES",
                return_value={
                    "required": {
                        "clip_name": (official_names,),
                        "type": (official_types,),
                    },
                    "optional": {"device": official_device},
                },
            ),
            mock.patch("comfyui_turing_utils.loading.convrot.convrot_model_names") as filter_names,
        ):
            inputs = ConvRotCLIPLoader.INPUT_TYPES()

        self.assertIs(inputs["required"]["clip_name"][0], official_names)
        self.assertIs(inputs["required"]["type"][0], official_types)
        self.assertIs(inputs["optional"]["device"], official_device)
        filter_names.assert_not_called()

    def test_node_maps_new_official_clip_type_to_enum(self):
        fake_clip = object()
        with (
            mock.patch(
                "comfyui_turing_utils.loading.convrot.official_clip_types",
                return_value=("stable_diffusion", "minimax"),
            ),
            mock.patch(
                "comfyui_turing_utils.loading.convrot.resolve_convrot_model_path",
                return_value="/models/minimax.safetensors",
            ),
            mock.patch(
                "comfyui_turing_utils.loading.convrot.load_convrot_clip", return_value=fake_clip
            ) as load_clip,
        ):
            result = ConvRotCLIPLoader().load_clip(
                "minimax.safetensors", type="minimax"
            )

        self.assertEqual(result, (fake_clip,))
        self.assertEqual(load_clip.call_args.kwargs["clip_type"], comfy.sd.CLIPType.MINIMAX)

    def test_diffusion_node_uses_force_int8_gemm_boolean(self):
        inputs = ConvRotDiffusionModelLoader.INPUT_TYPES()
        self.assertEqual(inputs["required"]["force_int8_gemm"][0], "BOOLEAN")
        self.assertFalse(inputs["required"]["force_int8_gemm"][1]["default"])
        self.assertNotIn("activation_dtype", inputs["required"])
        self.assertEqual(
            inputs["optional"]["patch_attention"][0],
            ("w8a8", "sage", "sdpa"),
        )
        self.assertEqual(inputs["optional"]["patch_attention"][1]["default"], "w8a8")
        self.assertNotIn("turing_bf16_mode", inputs["optional"])

    def test_diffusion_node_forwards_attention_backend(self):
        fake_model = object()
        with (
            mock.patch(
                "comfyui_turing_utils.loading.convrot.resolve_convrot_model_path",
                return_value="/models/convrot.safetensors",
            ),
            mock.patch("comfyui_turing_utils.loading.convrot.load_convrot_model", return_value=fake_model) as load_model,
        ):
            result = ConvRotDiffusionModelLoader().load_diffusion_model(
                "convrot.safetensors",
                force_int8_gemm=True,
                patch_attention="sdpa",
            )

        self.assertEqual(result, (fake_model,))
        load_model.assert_called_once_with(
            "/models/convrot.safetensors",
            True,
            attention_backend="sdpa",
        )

    def test_node_rejects_invalid_clip_type(self):
        with self.assertRaisesRegex(ValueError, "Unsupported ConvRot CLIP type"):
            ConvRotCLIPLoader().load_clip("clip.safetensors", type="unknown")


if __name__ == "__main__":
    unittest.main()
