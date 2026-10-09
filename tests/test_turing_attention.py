from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT / "kernel"))
sys.path.insert(0, str(PLUGIN_ROOT))

import attention as turing_attention  # noqa: E402


class TuringAttentionContractTest(unittest.TestCase):
    def test_fp32_uses_bf16_kernel_and_restores_fp32_output(self):
        q = torch.zeros((1, 1, 32, 128), dtype=torch.float32)
        kernel_output = torch.ones_like(q, dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sageattn", return_value=kernel_output) as sage,
        ):
            output = turing_attention.turing_sage_attention(
                mock.Mock(), q, q, q, 1, skip_reshape=True, skip_output_reshape=True
            )
        q_arg, k_arg, v_arg = sage.call_args.args[:3]
        self.assertEqual(q_arg.dtype, torch.bfloat16)
        self.assertEqual(k_arg.dtype, torch.bfloat16)
        self.assertEqual(v_arg.dtype, torch.bfloat16)
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, kernel_output.float())

    def test_unsupported_fp32_head_dimension_falls_back_without_casting(self):
        q = torch.zeros((1, 1, 32, 256), dtype=torch.float32)
        original = mock.Mock(return_value=q)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sageattn") as sage,
        ):
            output = turing_attention.turing_sage_attention(
                original, q, q, q, 1, skip_reshape=True
            )
        self.assertIs(output, q)
        original.assert_called_once()
        self.assertIs(original.call_args.args[0], q)
        sage.assert_not_called()

    def test_mask_uses_original_attention(self):
        q = torch.zeros((1, 1, 32, 128), dtype=torch.bfloat16)
        original = mock.Mock(return_value="fallback")
        with mock.patch("attention.is_supported_turing_device", return_value=True):
            output = turing_attention.turing_sage_attention(
                original, q, q, q, 1, mask=torch.ones(1), skip_reshape=True
            )
        self.assertEqual(output, "fallback")
        original.assert_called_once()

    def test_fp16_hnd_input_and_output_are_preserved(self):
        q = torch.zeros((1, 4, 32, 64), dtype=torch.float16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sageattn", return_value=q) as sage,
        ):
            output = turing_attention.turing_sage_attention(
                mock.Mock(),
                q,
                q[:, :2],
                q[:, :2],
                4,
                skip_reshape=True,
                skip_output_reshape=True,
                enable_gqa=True,
            )
        self.assertIs(output, q)
        self.assertEqual(sage.call_args.kwargs["tensor_layout"], "HND")

    def test_causal_and_scale_options_are_forwarded_to_stable_sage(self):
        q = torch.zeros((1, 4, 32, 64), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sageattn", return_value=q) as sage,
        ):
            turing_attention.turing_sage_attention(
                mock.Mock(),
                q,
                q,
                q,
                4,
                skip_reshape=True,
                is_causal=True,
                scale=0.125,
            )
        self.assertTrue(sage.call_args.kwargs["is_causal"])
        self.assertEqual(sage.call_args.kwargs["sm_scale"], 0.125)
        self.assertFalse(sage.call_args.kwargs["smooth_k"])
        self.assertNotIn("variant", sage.call_args.kwargs)

    def test_distinct_sequence_shapes_each_report_bundled_dispatch_once(self):
        q_short = torch.zeros((1, 4, 32, 64), dtype=torch.bfloat16)
        q_long = torch.zeros((1, 4, 96, 64), dtype=torch.bfloat16)
        turing_attention._LOGGED_TURING_KERNELS.clear()
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("attention._sageattn", side_effect=lambda q, *args, **kwargs: q),
            self.assertLogs("comfyui-turing-utils", level="DEBUG") as captured,
        ):
            turing_attention.turing_sage_attention(
                mock.Mock(),
                q_short,
                q_short,
                q_short,
                4,
                skip_reshape=True,
                skip_output_reshape=True,
            )
            turing_attention.turing_sage_attention(
                mock.Mock(),
                q_long,
                q_long,
                q_long,
                4,
                skip_reshape=True,
                skip_output_reshape=True,
            )
            turing_attention.turing_sage_attention(
                mock.Mock(),
                q_long,
                q_long,
                q_long,
                4,
                skip_reshape=True,
                skip_output_reshape=True,
            )

        dispatch_logs = [
            line for line in captured.output if "Bundled Turing Sage active" in line
        ]
        self.assertEqual(len(dispatch_logs), 2)
        self.assertTrue(any("32" in line for line in dispatch_logs))
        self.assertTrue(any("96" in line for line in dispatch_logs))

    def test_bf16_unreshaped_gqa_keeps_compact_kv_heads(self):
        q = torch.zeros((1, 32, 4 * 64), dtype=torch.bfloat16)
        k = torch.zeros((1, 16, 2 * 64), dtype=torch.bfloat16)
        v = torch.zeros_like(k)
        kernel_output = torch.zeros((1, 32, 4, 64), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sageattn", return_value=kernel_output) as sage,
        ):
            output = turing_attention.turing_sage_attention(
                mock.Mock(), q, k, v, 4, enable_gqa=True
            )
        q_arg, k_arg, v_arg = sage.call_args.args[:3]
        self.assertEqual(q_arg.shape, (1, 32, 4, 64))
        self.assertEqual(k_arg.shape, (1, 16, 2, 64))
        self.assertEqual(v_arg.shape, (1, 16, 2, 64))
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(sage.call_args.kwargs["tensor_layout"], "NHD")

    def test_unsupported_head_dimension_uses_original_attention(self):
        q = torch.zeros((1, 1, 32, 256), dtype=torch.bfloat16)
        original = mock.Mock(return_value="fallback")
        with mock.patch("attention.is_supported_turing_device", return_value=True):
            output = turing_attention.turing_sage_attention(
                original, q, q, q, 1, skip_reshape=True
            )
        self.assertEqual(output, "fallback")

    def test_experimental_sparse_uses_kernel_for_generic_long_attention(self):
        q = torch.zeros((1, 4, 4096, 128), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sol_sparse_sageattn", return_value=q) as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                mock.Mock(),
                q,
                q,
                q,
                4,
                skip_reshape=True,
                skip_output_reshape=True,
                min_sequence_tokens=4096,
            )
        self.assertIs(output, q)
        self.assertEqual(sparse.call_args.kwargs["dense_query_ranges"], ())
        self.assertEqual(sparse.call_args.kwargs["exact_kv_ranges"], ())
        self.assertEqual(sparse.call_args.kwargs["threshold_sigma"], 1.0)
        self.assertEqual(sparse.call_args.kwargs["residual_subblocks"], 1)

    def test_h3_sparse_stays_dense_until_complete_runtime_layout_is_available(self):
        q = torch.zeros((1, 4, 4096, 128), dtype=torch.bfloat16)
        baseline = mock.Mock(return_value=q)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sol_sparse_sageattn") as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                baseline,
                q,
                q,
                q,
                4,
                skip_reshape=True,
                transformer_options={
                    "turing_utils_attention_layout_required": "minimax_h3"
                },
            )

        self.assertIs(output, q)
        baseline.assert_called_once()
        sparse.assert_not_called()

    def test_h3_sparse_accepts_complete_loader_independent_runtime_layout(self):
        q = torch.zeros((1, 4, 4096, 128), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sol_sparse_sageattn", return_value=q) as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                mock.Mock(),
                q,
                q,
                q,
                4,
                skip_reshape=True,
                skip_output_reshape=True,
                transformer_options={
                    "turing_utils_attention_layout_required": "minimax_h3",
                    "turing_utils_attention_layout": {
                        "provider": "minimax_h3",
                        "dense_prefix_tokens": 512,
                        "topology_start_tokens": 512,
                        "topology_tokens": 3584,
                        "tokens_per_frame": 256,
                        "spatial_tokens_height": 16,
                        "spatial_tokens_width": 16,
                        "segments": (
                            (0, 256, "text"),
                            (256, 512, "reference_video"),
                            (512, 4096, "target_video"),
                        ),
                        "layer_index": 10,
                        "layer_count": 50,
                    },
                },
            )

        self.assertIs(output, q)
        self.assertEqual(sparse.call_args.kwargs["dense_query_ranges"], ((0, 256),))
        self.assertEqual(sparse.call_args.kwargs["exact_kv_ranges"], ((0, 256),))

    def test_experimental_sparse_keeps_short_calls_dense(self):
        q = torch.zeros((1, 4, 256, 128), dtype=torch.bfloat16)
        baseline = mock.Mock(return_value=q)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sol_sparse_sageattn") as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                baseline,
                q,
                q,
                q,
                4,
                skip_reshape=True,
            )
        self.assertIs(output, q)
        baseline.assert_called_once()
        sparse.assert_not_called()

    def test_experimental_sparse_supports_cross_attention_with_prefix_dense(self):
        q = torch.zeros((1, 4, 4096, 128), dtype=torch.bfloat16)
        k = torch.zeros((1, 2, 4608, 128), dtype=torch.bfloat16)
        v = torch.zeros_like(k)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sol_sparse_sageattn", return_value=q) as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                mock.Mock(),
                q,
                k,
                v,
                4,
                skip_reshape=True,
                skip_output_reshape=True,
                min_sequence_tokens=4096,
                transformer_options={
                    "turing_utils_attention_layout": {"dense_prefix_tokens": 320}
                },
            )
        self.assertIs(output, q)
        self.assertEqual(sparse.call_args.kwargs["dense_query_ranges"], ((0, 320),))
        self.assertEqual(sparse.call_args.kwargs["exact_kv_ranges"], ((0, 320),))

    def test_w8a8_and_sol_accept_head_dims_through_128(self):
        for kernel in ("w8a8", "sol"):
            for head_dim in (1, 32, 63, 64, 65, 96, 127, 128):
                q = torch.zeros((1, 2, 4096, head_dim), dtype=torch.bfloat16)
                with mock.patch(
                    "attention.is_supported_turing_device", return_value=True
                ):
                    call, reason = turing_attention.inspect_turing_attention_call(
                        q,
                        q,
                        q,
                        2,
                        skip_reshape=True,
                        skip_output_reshape=True,
                        kernel=kernel,
                        require_long_sequence=True,
                    )
                self.assertIsNone(reason, (kernel, head_dim, reason))
                self.assertEqual(call.head_dim, head_dim)

    def test_dense_w8a8_accepts_causal_while_sol_rejects_it(self):
        q = torch.zeros((1, 2, 4096, 128), dtype=torch.bfloat16)
        with mock.patch("attention.is_supported_turing_device", return_value=True):
            call, reason = turing_attention.inspect_turing_attention_call(
                q,
                q,
                q,
                2,
                skip_reshape=True,
                skip_output_reshape=True,
                is_causal=True,
                kernel="w8a8",
                require_long_sequence=True,
            )
            sol_call, sol_reason = turing_attention.inspect_turing_attention_call(
                q,
                q,
                q,
                2,
                skip_reshape=True,
                skip_output_reshape=True,
                is_causal=True,
                kernel="sol",
                require_long_sequence=True,
            )
        self.assertIsNone(reason)
        self.assertIsNotNone(call)
        self.assertIsNone(sol_call)
        self.assertIn("causal", sol_reason)

    def test_experimental_sparse_debug_counts_route_only_once_per_shape(self):
        q = torch.zeros((1, 4, 4096, 128), dtype=torch.bfloat16)
        route_keys = set()
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch(
                "attention._sol_sparse_sageattn",
                side_effect=[(q, torch.tensor([1024]), 16384), q],
            ) as sparse,
            self.assertLogs("comfyui-turing-utils", level="INFO") as captured,
        ):
            for _ in range(2):
                output = turing_attention.turing_sol_sparse_attention(
                    mock.Mock(),
                    q,
                    q,
                    q,
                    4,
                    skip_reshape=True,
                    skip_output_reshape=True,
                    min_sequence_tokens=4096,
                    debug_route_density=True,
                    debug_route_keys=route_keys,
                )

        self.assertIs(output, q)
        self.assertEqual(sparse.call_count, 2)
        self.assertTrue(sparse.call_args_list[0].kwargs["return_stats"])
        self.assertFalse(sparse.call_args_list[1].kwargs["return_stats"])
        message = "\n".join(captured.output)
        self.assertIn("density=0.0625", message)
        self.assertIn("step=None/None layer=None/None", message)

    def test_experimental_sparse_debug_aggregates_once_per_step(self):
        q = torch.zeros((1, 4, 4096, 128), dtype=torch.bfloat16)
        route_state = {}
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch(
                "attention._sol_sparse_sageattn",
                side_effect=[
                    (q, torch.tensor([1024]), 16384),
                    (q, torch.tensor([2048]), 16384),
                ],
            ) as sparse,
            self.assertLogs("comfyui-turing-utils", level="INFO") as captured,
        ):
            for layer_index in (2, 3):
                output = turing_attention.turing_sol_sparse_attention(
                    mock.Mock(),
                    q,
                    q,
                    q,
                    4,
                    skip_reshape=True,
                    skip_output_reshape=True,
                    min_sequence_tokens=4096,
                    debug_route_density=True,
                    debug_route_state=route_state,
                    debug_context={
                        "step": 2,
                        "sampling_steps": 8,
                        "layer_index": layer_index,
                        "layer_count": 4,
                    },
                )

        self.assertIs(output, q)
        self.assertEqual(sparse.call_count, 2)
        self.assertTrue(
            all(call.kwargs["return_stats"] for call in sparse.call_args_list)
        )
        self.assertFalse(route_state)
        message = "\n".join(captured.output)
        self.assertIn("step=2/8 layers=2-3 calls=2", message)
        self.assertIn("density[min/mean/max]=0.0625/0.0938/0.1250", message)

    def test_experimental_sparse_unreshaped_gqa_restores_output_layout(self):
        q = torch.zeros((1, 4096, 4 * 128), dtype=torch.float16)
        k = torch.zeros((1, 4096, 2 * 128), dtype=torch.float16)
        v = torch.zeros_like(k)
        kernel_output = torch.zeros((1, 4, 4096, 128), dtype=torch.float16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch(
                "attention._sol_sparse_sageattn", return_value=kernel_output
            ) as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                mock.Mock(), q, k, v, 4, enable_gqa=True, min_sequence_tokens=4096
            )
        self.assertEqual(output.shape, q.shape)
        self.assertEqual(sparse.call_args.args[1].shape, (1, 2, 4096, 128))

    def test_experimental_sparse_fallback_preserves_original_layout(self):
        q = torch.zeros((1, 4096, 4 * 64), dtype=torch.bfloat16)
        baseline = mock.Mock(return_value=q)
        with (
            mock.patch("attention._sol_sparse_sageattn") as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(baseline, q, q, q, 4)
        self.assertIs(output, q)
        baseline.assert_called_once()
        self.assertIs(baseline.call_args.args[0], q)
        sparse.assert_not_called()

    def test_experimental_sparse_forwards_patch_parameters(self):
        q = torch.zeros((1, 4, 8192, 128), dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch("attention._sol_sparse_sageattn", return_value=q) as sparse,
        ):
            turing_attention.turing_sol_sparse_attention(
                mock.Mock(),
                q,
                q,
                q,
                4,
                skip_reshape=True,
                routing_threshold=0.82,
                prefix_policy="auto",
                manual_prefix_tokens=192,
                skipped_residual="1x64",
                sparse_reference_image=True,
                sparse_reference_video=False,
                sparse_reference_audio=True,
                min_sequence_tokens=8000,
                transformer_options={
                    "turing_utils_attention_layout": {
                        "dense_prefix_tokens": 320,
                        "topology_start_tokens": 320,
                        "topology_tokens": 7680,
                        "tokens_per_frame": 960,
                        "segments": (
                            (0, 128, "text"),
                            (128, 256, "reference_image"),
                            (256, 320, "reference_video"),
                            (320, 384, "reference_audio"),
                            (384, 512, "target_audio"),
                            (512, 8192, "target_video"),
                        ),
                    }
                },
            )
        self.assertEqual(
            sparse.call_args.kwargs["dense_query_ranges"],
            ((0, 128), (256, 320), (384, 512)),
        )
        self.assertEqual(
            sparse.call_args.kwargs["exact_kv_ranges"],
            ((0, 128), (256, 320), (384, 512)),
        )
        self.assertEqual(sparse.call_args.kwargs["threshold_sigma"], 0.82)
        self.assertEqual(sparse.call_args.kwargs["residual_subblocks"], 1)

    def test_sparse_prefix_policy_never_guesses_a_generic_layout(self):
        options = {"turing_utils_attention_layout": {"dense_prefix_tokens": 640}}
        self.assertEqual(
            turing_attention._sparse_prefix_tokens("auto", 128, options, 4096),
            640,
        )
        self.assertEqual(
            turing_attention._sparse_prefix_tokens("none", 128, options, 4096),
            0,
        )
        self.assertEqual(
            turing_attention._sparse_prefix_tokens("manual", 128, options, 4096),
            128,
        )
        self.assertEqual(
            turing_attention._sparse_prefix_tokens("auto", 128, {}, 4096),
            0,
        )

    def test_sparse_modality_policy_handles_long_and_mixed_references(self):
        options = {
            "turing_utils_attention_layout": {
                "segments": (
                    (0, 64, "text"),
                    (64, 128, "reference_image"),
                    (128, 1048, "reference_video_anchor"),
                    (1048, 63080, "reference_video"),
                    (63080, 64000, "reference_video_anchor"),
                    (64000, 64128, "reference_audio"),
                    (64128, 64256, "target_audio"),
                    (64256, 100000, "target_video"),
                )
            }
        }
        default_ranges = turing_attention._sparse_protected_ranges(
            "auto",
            0,
            options,
            100000,
            sparse_reference_image=False,
            sparse_reference_video=True,
            sparse_reference_audio=False,
        )
        self.assertEqual(
            default_ranges,
            ((0, 1048), (63080, 64256)),
        )
        image_sparse_video_exact = turing_attention._sparse_protected_ranges(
            "auto",
            0,
            options,
            100000,
            sparse_reference_image=True,
            sparse_reference_video=False,
            sparse_reference_audio=False,
        )
        self.assertEqual(
            image_sparse_video_exact,
            ((0, 64), (1048, 63080), (64000, 64256)),
        )
        all_reference_sparse = turing_attention._sparse_protected_ranges(
            "auto",
            0,
            options,
            100000,
            sparse_reference_image=True,
            sparse_reference_video=True,
            sparse_reference_audio=True,
        )
        self.assertEqual(all_reference_sparse, ((0, 64), (64128, 64256)))
        no_reference_sparse = turing_attention._sparse_protected_ranges(
            "auto",
            0,
            options,
            100000,
            sparse_reference_image=False,
            sparse_reference_video=False,
            sparse_reference_audio=False,
        )
        self.assertEqual(no_reference_sparse, ((0, 64256),))

    def test_sparse_modality_policy_without_references_protects_text_and_audio(self):
        options = {
            "turing_utils_attention_layout": {
                "segments": (
                    (0, 64, "text"),
                    (64, 128, "target_audio"),
                    (128, 4096, "target_video"),
                )
            }
        }
        ranges = turing_attention._sparse_protected_ranges(
            "auto",
            0,
            options,
            4096,
            sparse_reference_image=False,
            sparse_reference_video=True,
            sparse_reference_audio=False,
        )
        self.assertEqual(ranges, ((0, 128),))

    def test_sparse_dense_prefix_steps_use_sampling_progress(self):
        sample_sigmas = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0])
        state = {}
        self.assertTrue(
            turing_attention._sparse_dense_prefix_steps(
                {"sample_sigmas": sample_sigmas, "sigmas": sample_sigmas[0:1]},
                1,
                state,
            )
        )
        self.assertFalse(
            turing_attention._sparse_dense_prefix_steps(
                {"sample_sigmas": sample_sigmas, "sigmas": sample_sigmas[1:2]},
                1,
                state,
            )
        )

    def test_sparse_dense_prefix_steps_accept_inference_tensors(self):
        state = {}
        with torch.inference_mode():
            sample_sigmas = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0])
            current_sigmas = sample_sigmas[0:1]
            self.assertTrue(
                turing_attention._sparse_dense_prefix_steps(
                    {
                        "sample_sigmas": sample_sigmas,
                        "sigmas": current_sigmas,
                    },
                    1,
                    state,
                )
            )
            self.assertTrue(
                turing_attention._sparse_dense_prefix_steps(
                    {
                        "sample_sigmas": sample_sigmas,
                        "sigmas": current_sigmas,
                    },
                    1,
                    state,
                )
            )

    def test_sparse_dense_schedule_supports_a_tail_and_layer_metadata(self):
        sample_sigmas = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0])
        self.assertTrue(
            turing_attention._sparse_dense_schedule(
                {"sample_sigmas": sample_sigmas, "sigmas": sample_sigmas[3:4]},
                0,
                1,
                {},
            )
        )
        self.assertTrue(
            turing_attention._sparse_dense_layer(
                {
                    "turing_utils_attention_layout": {
                        "layer_index": 1,
                        "layer_count": 50,
                    }
                },
                2,
                2,
            )
        )
        self.assertFalse(
            turing_attention._sparse_dense_layer(
                {
                    "turing_utils_attention_layout": {
                        "layer_index": 2,
                        "layer_count": 50,
                    }
                },
                2,
                2,
            )
        )
        self.assertTrue(
            turing_attention._sparse_dense_layer(
                {
                    "turing_utils_attention_layout": {
                        "layer_index": 48,
                        "layer_count": 50,
                    }
                },
                2,
                2,
            )
        )

    def test_sparse_debug_tracks_step_when_dense_schedule_is_disabled(self):
        sample_sigmas = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0])
        state = {}
        self.assertFalse(
            turing_attention._sparse_dense_schedule(
                {"sample_sigmas": sample_sigmas, "sigmas": sample_sigmas[2:3]},
                0,
                0,
                state,
                track_step=True,
            )
        )
        self.assertEqual(state["step"], 2)
        self.assertEqual(state["sampling_steps"], 4)

    def test_sparse_dense_schedule_resets_for_each_sampler_invocation(self):
        full = torch.tensor([1.0, 0.7, 0.3, 0.0])
        partial = torch.tensor([0.2, 0.1, 0.0])
        full_options = {"sample_sigmas": full, "sigmas": full[0:1]}
        partial_options = {"sample_sigmas": partial, "sigmas": partial[0:1]}

        self.assertTrue(
            turing_attention._sparse_dense_schedule(
                full_options,
                1,
                0,
            )
        )
        self.assertTrue(
            turing_attention._sparse_dense_schedule(
                partial_options,
                1,
                0,
            )
        )
        partial_options["sigmas"] = partial[1:2]
        self.assertFalse(
            turing_attention._sparse_dense_schedule(
                partial_options,
                1,
                0,
            )
        )
        state = partial_options["_turing_utils_sparse_schedule_state"]
        self.assertEqual(state["step"], 1)

    def test_experimental_sparse_fp32_uses_bf16_boundary_and_restores_output(self):
        q = torch.zeros((1, 1, 4096, 128), dtype=torch.float32)
        kernel_output = torch.ones_like(q, dtype=torch.bfloat16)
        with (
            mock.patch("attention.is_supported_turing_device", return_value=True),
            mock.patch(
                "attention._sol_sparse_sageattn", return_value=kernel_output
            ) as sparse,
        ):
            output = turing_attention.turing_sol_sparse_attention(
                mock.Mock(),
                q,
                q,
                q,
                1,
                skip_reshape=True,
                skip_output_reshape=True,
                min_sequence_tokens=4096,
            )
        self.assertTrue(
            all(value.dtype == torch.bfloat16 for value in sparse.call_args.args[:3])
        )
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, kernel_output.float())


if __name__ == "__main__":
    unittest.main()
