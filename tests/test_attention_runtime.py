from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.attention import (
    patches as attention_patches,
    runtime as attention_runtime,
)  # noqa: E402


class FakePatcher:
    def __init__(self):
        self.load_device = torch.device("cuda", 0)
        self.model_options = {"transformer_options": {}}

    def clone(self):
        clone = FakePatcher()
        clone.model_options = {
            "transformer_options": self.model_options["transformer_options"].copy()
        }
        return clone


class AttentionRuntimeTest(unittest.TestCase):
    def test_removed_static_image_strategies_are_rejected(self):
        config = attention_runtime.AttentionRuntimeConfig(
            "sdpa", "test", lambda *args: None
        )
        for name in ("h3_virtual_kv", "h3_image_sol"):
            with (
                self.subTest(strategy=name),
                self.assertRaisesRegex(ValueError, "unsupported attention strategy"),
            ):
                config.with_strategy(name, "removed", lambda *args: None)

    def test_dense_backend_installs_native_capability_marker(self):
        model = FakePatcher()
        attention_patches.apply_attention_backend(
            model,
            "sdpa",
            device=torch.device("cuda", 0),
            native_runtime=True,
        )
        options = model.model_options["transformer_options"]
        config = attention_runtime.attention_runtime_config(options)
        self.assertEqual(config.dense_backend, "sdpa")
        self.assertEqual(config.strategy, "dense")
        self.assertTrue(config.native_runtime)
        self.assertTrue(
            attention_runtime.is_attention_runtime_dispatcher(
                options["optimized_attention_override"]
            )
        )

    def test_dispatcher_switches_strategy_without_reinstalling_itself(self):
        def dense(original, *args, **kwargs):
            return "dense"

        def sol(original, *args, **kwargs):
            return "sol"

        def dense_prepared(request):
            return request

        def sol_prepared(request):
            return request

        dense.turing_utils_attention_backend = "sdpa"
        dense.turing_utils_attention_implementation = "test:sdpa"
        dense.prepared_attention_executor = dense_prepared
        sol.prepared_attention_executor = sol_prepared
        dispatcher = attention_runtime.make_attention_runtime_dispatcher(dense)
        options = {}
        base = attention_runtime.AttentionRuntimeConfig(
            dense_backend="sdpa",
            dense_implementation="test:sdpa",
            dense_override=dense,
        )
        attention_runtime.install_attention_runtime(
            options, base, dispatcher=dispatcher
        )
        self.assertEqual(
            dispatcher(None, transformer_options=options),
            "dense",
        )
        self.assertIs(
            options[attention_runtime.ATTENTION_EXECUTOR_KEY],
            dense_prepared,
        )

        sparse = base.with_strategy("sol", "test:sol", sol)
        attention_runtime.install_attention_runtime(
            options, sparse, dispatcher=dispatcher
        )
        self.assertIs(options["optimized_attention_override"], dispatcher)
        self.assertEqual(
            dispatcher(None, transformer_options=options),
            "sol",
        )
        self.assertIs(
            options[attention_runtime.ATTENTION_EXECUTOR_KEY],
            sol_prepared,
        )

        def replacement_dense(original, *args, **kwargs):
            return "replacement"

        replacement = attention_runtime.AttentionRuntimeConfig(
            dense_backend="sage",
            dense_implementation="test:sage",
            dense_override=replacement_dense,
        )
        replacement_dispatcher = attention_runtime.install_attention_runtime(
            options, replacement
        )
        self.assertIsNot(replacement_dispatcher, dispatcher)

    def test_strategy_branch_inherits_sdpa_and_reuses_loader_dispatcher(self):
        model = FakePatcher()

        def dense(original, *args, **kwargs):
            return "dense"

        dense.turing_utils_attention_backend = "sdpa"
        dense.turing_utils_attention_implementation = "test:sdpa"
        base = attention_runtime.AttentionRuntimeConfig(
            dense_backend="sdpa",
            dense_implementation="test:sdpa",
            dense_override=dense,
            native_runtime=True,
        )
        options = model.model_options["transformer_options"]
        dispatcher = attention_runtime.install_attention_runtime(options, base)

        def sol(original, *args, **kwargs):
            return "sol"

        sol.turing_utils_attention_implementation = "bundled_sol_sparse"
        sol.turing_utils_dense_implementation = "test:sdpa"
        sol.turing_utils_sparse_numeric_backend = "fp16"
        with mock.patch(
            "comfyui_turing_utils.attention.patches.make_sparse_attention_override",
            return_value=sol,
        ) as make:
            patched = attention_patches.apply_sparse_attention_patch(model)

        patched_options = patched.model_options["transformer_options"]
        config = attention_runtime.attention_runtime_config(patched_options)
        self.assertIs(patched_options["optimized_attention_override"], dispatcher)
        self.assertEqual(config.strategy, "sol")
        self.assertEqual(config.dense_backend, "sdpa")
        self.assertIs(config.dense_override, dense)
        self.assertEqual(make.call_args.kwargs["dense_backend"], "sdpa")
        self.assertIs(make.call_args.kwargs["dense_override"], dense)
        self.assertNotIn("use_w8a8", make.call_args.kwargs)

    def test_strategy_branches_inherit_loader_sage_runtime(self):
        for apply_name, make_name, strategy in (
            (
                "apply_sparse_attention_patch",
                "make_sparse_attention_override",
                "sol",
            ),
            (
                "apply_sla_attention_patch",
                "make_sla_attention_override",
                "sla",
            ),
        ):
            with self.subTest(strategy=strategy):
                model = FakePatcher()

                def dense(original, *args, **kwargs):
                    return "sage"

                dense.turing_utils_attention_backend = "sage"
                dense.turing_utils_attention_implementation = "comfy:sage"
                base = attention_runtime.AttentionRuntimeConfig(
                    dense_backend="sage",
                    dense_implementation="comfy:sage",
                    dense_override=dense,
                    native_runtime=True,
                )
                options = model.model_options["transformer_options"]
                dispatcher = attention_runtime.install_attention_runtime(options, base)

                def sparse(original, *args, **kwargs):
                    return strategy

                sparse.turing_utils_attention_implementation = (
                    f"bundled_{strategy}_sparse"
                )
                sparse.turing_utils_dense_implementation = "comfy:sage"
                sparse.turing_utils_sparse_numeric_backend = "fp16"
                with mock.patch(
                    f"comfyui_turing_utils.attention.patches.{make_name}",
                    return_value=sparse,
                ) as make:
                    patched = getattr(attention_patches, apply_name)(model)

                patched_options = patched.model_options["transformer_options"]
                config = attention_runtime.attention_runtime_config(patched_options)
                self.assertIs(
                    patched_options["optimized_attention_override"], dispatcher
                )
                self.assertEqual(config.strategy, strategy)
                self.assertEqual(config.dense_backend, "sage")
                self.assertIs(config.dense_override, dense)
                self.assertEqual(make.call_args.kwargs["dense_backend"], "sage")
                self.assertIs(make.call_args.kwargs["dense_override"], dense)
                self.assertNotIn("use_w8a8", make.call_args.kwargs)

    def test_native_loader_branch_does_not_reinstall_model_side_bridges(self):
        model = FakePatcher()

        def dense(original, *args, **kwargs):
            return "dense"

        dense.turing_utils_attention_backend = "w8a8"
        dense.turing_utils_attention_implementation = "test:w8a8"
        base = attention_runtime.AttentionRuntimeConfig(
            dense_backend="w8a8",
            dense_implementation="test:w8a8",
            dense_override=dense,
            native_runtime=True,
        )
        options = model.model_options["transformer_options"]
        options["turing_utils_attention_layout_required"] = "minimax_h3"
        attention_runtime.install_attention_runtime(options, base)

        def sol(original, *args, **kwargs):
            return "sol"

        sol.turing_utils_attention_implementation = "bundled_sol_sparse"
        sol.turing_utils_dense_implementation = "test:w8a8"
        sol.turing_utils_sparse_numeric_backend = "w8a8"
        sol.prepared_attention_executor = lambda request: None
        with (
            mock.patch(
                "comfyui_turing_utils.attention.patches.make_sparse_attention_override",
                return_value=sol,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.orchestration.ensure_attention_layout_provider"
            ) as ensure_layout,
            mock.patch(
                "comfyui_turing_utils.attention.orchestration.ensure_prepared_attention_sites"
            ) as ensure_sites,
        ):
            patched = attention_patches.apply_sparse_attention_patch(model)

        ensure_layout.assert_not_called()
        ensure_sites.assert_not_called()
        self.assertIs(
            patched.model_options["transformer_options"][
                attention_runtime.ATTENTION_EXECUTOR_KEY
            ],
            sol.prepared_attention_executor,
        )

    def test_sdpa_base_selects_fp16_sol_sparse_numeric_path(self):
        q = torch.zeros((1, 2, 256, 128), dtype=torch.bfloat16)
        with (
            mock.patch(
                "comfyui_turing_utils.attention.sol.is_supported_attention_device",
                return_value=True,
            ),
            mock.patch(
                "comfyui_turing_utils.attention.sol.bundled_sparse_available",
                return_value=True,
            ),
            mock.patch("comfyui_turing_utils.attention.sol.preflight_bundled_sparse"),
            mock.patch(
                "comfyui_turing_utils.attention.sol.turing_sol_sparse_attention",
                return_value=q,
            ) as sparse,
        ):
            override = attention_patches.make_sparse_attention_override(
                torch.device("cuda", 0),
                dense_backend="sdpa",
            )
            output = override(mock.Mock(), q, q, q, 2, skip_reshape=True)

        self.assertIs(output, q)
        self.assertEqual(override.turing_utils_dense_backend, "sdpa")
        self.assertEqual(override.turing_utils_sparse_numeric_backend, "fp16")
        self.assertFalse(sparse.call_args.kwargs["use_w8a8"])


if __name__ == "__main__":
    unittest.main()
