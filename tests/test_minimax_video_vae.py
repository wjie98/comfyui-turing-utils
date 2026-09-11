from __future__ import annotations

import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.adapters.minimax import video_vae  # noqa: E402
from comfyui_turing_utils.adapters.minimax import video_vae_encode  # noqa: E402
from comfyui_turing_utils.attention.protocol import (  # noqa: E402
    ATTENTION_EXECUTOR_KEY,
    AttentionExecutionOutcome,
)
from comfyui_turing_utils.nodes import minimax_vae as nodes  # noqa: E402


def blend(a, b, extent, dim):
    extent = min(a.shape[dim], b.shape[dim], extent)
    positions = torch.arange(extent, dtype=b.dtype, device=b.device)
    wa = (1 - positions / extent).view(
        [extent if i == dim % b.ndim else 1 for i in range(b.ndim)]
    )
    wb = (positions / extent).view(
        [extent if i == dim % b.ndim else 1 for i in range(b.ndim)]
    )
    a_slice = [slice(None)] * a.ndim
    b_slice = [slice(None)] * b.ndim
    a_slice[dim] = slice(-extent, None)
    b_slice[dim] = slice(0, extent)
    merged = a[tuple(a_slice)] * wa + b[tuple(b_slice)] * wb
    if extent == b.shape[dim]:
        return merged
    b_slice[dim] = slice(extent, None)
    return torch.cat((merged, b[tuple(b_slice)]), dim=dim)


def make_decoder(patch_size_t=1):
    from comfy.ldm.minimax.vae import ViT3DDecoder

    torch.manual_seed(26)
    decoder = ViT3DDecoder(
        patch_size=2, patch_size_t=patch_size_t, in_channels=4, out_channels=3,
        num_layers=3, heads=1, dim_head=64, operations=torch.nn,
    ).eval()
    for name, parameter in decoder.named_parameters():
        if "norm" in name and name.endswith("weight"):
            torch.nn.init.ones_(parameter)
        elif "scale" in name:
            torch.nn.init.constant_(parameter, 0.5)
        else:
            torch.nn.init.uniform_(parameter, -0.1, 0.1)
    return decoder


class MiniMaxVideoVAETest(unittest.TestCase):
    def test_tile_budget_does_not_borrow_resident_models(self):
        patcher = mock.Mock()
        patcher.model_size.return_value = 100
        patcher.loaded_size.return_value = 60
        patcher.get_free_memory.side_effect = AssertionError("includes evictable weights")
        vae = SimpleNamespace(patcher=patcher, device=torch.device("cuda"))
        with (
            mock.patch.object(video_vae.comfy.model_management, "get_free_memory", return_value=200),
            mock.patch.object(video_vae.comfy.model_management, "extra_reserved_memory", return_value=50),
        ):
            self.assertEqual(video_vae._tile_memory_budget(vae), 110)
            patcher.loaded_size.return_value = 0
            patcher.model_size.return_value = 500
            self.assertEqual(video_vae._tile_memory_budget(vae), 0)

    def test_only_one_tile_requests_loading_and_budget_is_measured_afterwards(self):
        vae = SimpleNamespace(patcher=object(), disable_offload=False)
        events = []
        with (
            mock.patch.object(video_vae.comfy.model_management, "load_models_gpu", side_effect=lambda *a, **kw: events.append(kw)) as loader,
            mock.patch.object(video_vae, "_tile_memory_budget", side_effect=lambda v: events.append("budget") or 35),
        ):
            self.assertEqual(video_vae._load_vae_for_tiles(vae, 8, lambda n: 10 * n, 16), (3, 30))
        loader.assert_called_once_with([vae.patcher], memory_required=10, force_full_load=False)
        self.assertEqual(events, [{"memory_required": 10, "force_full_load": False}, "budget"])

    def test_vae_operator_scope_is_optional_and_restored_on_error(self):
        with mock.patch.object(video_vae, "register_backend", return_value=False), mock.patch.object(video_vae, "use_turing_operator_backend") as backend:
            with video_vae._vae_operator_scope():
                pass
            backend.assert_not_called()
        with mock.patch.object(video_vae, "register_backend", return_value=True), mock.patch.object(video_vae, "use_turing_operator_backend") as backend:
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                with video_vae._vae_operator_scope():
                    raise RuntimeError("decode failed")
            backend.return_value.__exit__.assert_called_once()

    def test_custom_encoder_reference_math_is_bitwise_equal(self):
        from comfy.ldm.minimax.vae import EncoderFCN3D

        torch.manual_seed(2)
        encoder = EncoderFCN3D(
            ch=32,
            ch_mult=[1],
            space_down=[1],
            time_down=[1],
            num_res_blocks=1,
            in_channels=3,
            z_channels=4,
        )
        quant_conv = torch.nn.Conv3d(8, 8, 1)
        for parameter in encoder.parameters():
            torch.nn.init.uniform_(parameter, -0.02, 0.02)
        model = SimpleNamespace(encoder=encoder, quant_conv=quant_conv)
        x = torch.randn(1, 3, 2, 8, 8)
        with torch.inference_mode():
            expected = quant_conv(encoder(x.clone()))
            actual = video_vae_encode._encode_moments(
                model,
                x.clone(),
                prefetch_dynamic_vbars=True,
            )
        self.assertTrue(torch.equal(actual, expected))

    def test_encoder_consumes_official_prefetch_queue_in_execution_order(self):
        from comfy.ldm.minimax.vae import EncoderFCN3D

        encoder = EncoderFCN3D(
            ch=32,
            ch_mult=[1],
            space_down=[1],
            time_down=[1],
            num_res_blocks=1,
            in_channels=3,
            z_channels=4,
        )
        model = SimpleNamespace(
            encoder=encoder,
            quant_conv=torch.nn.Conv3d(8, 8, 1),
        )
        value = torch.randn(1, 3, 2, 8, 8)
        queue = object()
        with (
            mock.patch.object(
                video_vae.comfy.model_prefetch,
                "make_prefetch_queue",
                return_value=queue,
            ) as make_queue,
            mock.patch.object(
                video_vae.comfy.model_prefetch,
                "prefetch_queue_pop",
            ) as pop_queue,
            torch.inference_mode(),
        ):
            video_vae_encode._encode_moments(
                model,
                value,
                prefetch_dynamic_vbars=True,
            )

        stages = video_vae_encode._encoder_prefetch_stages(model)
        make_queue.assert_called_once_with(
            stages,
            value.device,
            {"prefetch_dynamic_vbars": True},
        )
        self.assertEqual(
            pop_queue.call_args_list,
            [
                *(mock.call(queue, value.device, stage) for stage in stages),
                mock.call(queue, value.device, None),
            ],
        )

    def test_split_tiles_covers_exact_extent(self):
        for length in (128, 256, 480, 720, 848, 1280):
            for tile in (256, 288, 320):
                starts, lengths, overlaps = video_vae.split_tiles(length, tile, 64, 16)
                self.assertEqual(starts[0], 0)
                self.assertEqual(starts[-1] + lengths[-1], length)
                self.assertTrue(all(value % 16 == 0 for value in starts))
                self.assertTrue(all(value % 16 == 0 for value in overlaps))
                for i, overlap in enumerate(overlaps):
                    self.assertEqual(starts[i + 1], starts[i] + lengths[i] - overlap)

    def test_tiled_encode_matches_full_pointwise_encode(self):
        model = SimpleNamespace(
            vae_ratio=2,
            blend=blend,
            _encode_moments=lambda value: value[..., ::2, ::2],
        )
        pixels = torch.arange(80, dtype=torch.float32).view(1, 1, 1, 8, 10)
        progress = mock.Mock()
        actual = video_vae_encode._tiled_encode(
            model,
            pixels,
            4,
            2,
            tiles_per_batch=3,
            progress=progress,
        )
        expected = model._encode_moments(pixels)
        torch.testing.assert_close(actual, expected)
        expected_tiles = len(video_vae.split_tiles(8, 4, 2, 2)[0]) * len(
            video_vae.split_tiles(10, 4, 2, 2)[0]
        )
        self.assertEqual(
            sum(call.args[0] for call in progress.update.call_args_list),
            expected_tiles,
        )

    def test_small_feed_forward_batches_keep_fused_swiglu(self):
        module = mock.Mock()
        projected = torch.zeros(1, 5, 16)
        module.w1 = mock.Mock(return_value=projected)
        module.w2 = object()
        small = torch.zeros(1, 5, 8)
        expected = torch.ones_like(small)
        with (
            mock.patch.object(video_vae, "_fused_swiglu_eligible", return_value=True),
            mock.patch.object(
                video_vae.comfy.ops,
                "linear_input_act",
                return_value=expected,
            ) as fused,
        ):
            actual = video_vae._feed_forward(module, small)
        self.assertIs(actual, expected)
        module.assert_not_called()
        module.w1.assert_called_once_with(small)
        fused.assert_called_once_with(module.w2, projected, "swiglu")

    def test_large_feed_forward_batches_keep_fused_swiglu(self):
        module = mock.Mock()
        projected = torch.zeros(1, 64, 16)
        module.w1 = mock.Mock(return_value=projected)
        module.w2 = object()
        value = torch.zeros(1, 64, 8)
        expected = torch.ones_like(value)
        with (
            mock.patch.object(video_vae, "_fused_swiglu_eligible", return_value=True),
            mock.patch.object(
                video_vae.comfy.ops,
                "linear_input_act",
                return_value=expected,
            ) as fused,
        ):
            actual = video_vae._feed_forward(module, value)
        self.assertIs(actual, expected)
        module.assert_not_called()
        module.w1.assert_called_once_with(value)
        fused.assert_called_once_with(module.w2, projected, "swiglu")

    def test_feed_forward_rejects_incomplete_output_shape(self):
        module = mock.Mock(return_value=torch.zeros(1, 5, 7))
        module.w2 = object()
        with (
            mock.patch.object(video_vae, "_fused_swiglu_eligible", return_value=False),
            self.assertRaisesRegex(RuntimeError, "feed-forward returned"),
        ):
            video_vae._feed_forward(module, torch.zeros(1, 5, 8))

    def test_latent_fingerprint_is_stable_and_value_sensitive(self):
        latent = torch.arange(96, dtype=torch.float32).reshape(1, 3, 4, 4, 2)
        first = video_vae._latent_fingerprint(latent)
        second = video_vae._latent_fingerprint(latent.clone())
        changed = latent.clone()
        changed.flatten()[47] += 1
        self.assertEqual(first, second)
        self.assertNotEqual(first, video_vae._latent_fingerprint(changed))

    def test_decoder_matches_native_single_window(self):
        decoder = make_decoder()
        value = torch.randn(2, 4, 2, 3, 4)
        original = value.clone()
        with torch.inference_mode():
            expected = decoder(value)
            actual = video_vae._decoder_forward(
                decoder, value, video_vae._attention_options("sdpa", value.device)
            )
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(value, original, rtol=0, atol=0)

    def test_independent_windows_match_native_decoder(self):
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

        model = object.__new__(MiniMaxH3VideoVAE)
        torch.nn.Module.__init__(model)
        model.vae_ratio = 2
        model.tile_size = 8
        model.tile_overlap_min = 4
        model.post_quant_conv = torch.nn.Conv3d(4, 4, 1)
        model.decoder = make_decoder()
        for height, width in ((3, 4), (3, 8), (6, 3), (6, 8), (10, 12)):
            value = torch.randn(2, 4, 2, height, width)
            with torch.inference_mode():
                expected = model.tiled_decode(value)
                for tile_batch in (1, 2, 5):
                    with self.subTest(shape=(height, width), tile_batch=tile_batch):
                        progress = mock.Mock()
                        actual = video_vae._decode_spatial(
                            model, value, video_vae._attention_options("sdpa", value.device),
                            tile_batch, progress,
                        )
                        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                        self.assertEqual(actual.dtype, expected.dtype)
                        tile_count = len(model.split_tiles(height * 2)[0]) * len(
                            model.split_tiles(width * 2)[0]
                        )
                        self.assertEqual(
                            sum(call.args[0] for call in progress.update.call_args_list),
                            tile_count,
                        )

    def test_spatial_stitching_matches_native_with_different_overlap_pixels(self):
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

        model = object.__new__(MiniMaxH3VideoVAE)
        torch.nn.Module.__init__(model)
        model.vae_ratio = 2
        model.tile_size = 8
        model.tile_overlap_min = 4
        model.post_quant_conv = torch.nn.Identity()
        model.decoder = torch.nn.Identity()

        def decode(_decoder, value, _options):
            pixels = value.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
            return pixels + value[..., :1, :1] * 0.125

        model._decode_pixels = lambda value: decode(model.decoder, value, {})
        for dtype in (torch.float32, torch.float16):
            value = torch.randn(2, 3, 2, 7, 9).to(dtype)
            expected = model.tiled_decode(value)
            with mock.patch.object(video_vae, "_decoder_forward", side_effect=decode):
                for tile_batch in (1, 3, 16):
                    actual = video_vae._decode_spatial(model, value, {}, tile_batch, None)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_decoder_matches_native_fp16(self):
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

        model = object.__new__(MiniMaxH3VideoVAE)
        torch.nn.Module.__init__(model)
        model.vae_ratio = 2
        model.tile_size = 8
        model.tile_overlap_min = 4
        model.post_quant_conv = torch.nn.Identity()
        model.decoder = make_decoder().cuda().half()
        value = torch.randn(2, 4, 2, 6, 8, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            expected = model.tiled_decode(value)
            for tile_batch in (1, 3):
                actual = video_vae._decode_spatial(
                    model, value, video_vae._attention_options("sdpa", value.device),
                    tile_batch, None,
                )
                torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_decoder_attention_switching_preserves_state(self):
        decoder = make_decoder().cuda().half()
        value = torch.randn(1, 4, 2, 4, 4, device="cuda", dtype=torch.float16)
        original = value.clone()
        results = []
        with torch.inference_mode():
            for backend in ("sdpa", "sage", "w8a8", "sdpa"):
                result = video_vae._decoder_forward(
                    decoder, value, video_vae._attention_options(backend, value.device)
                )
                results.append(result)
                self.assertEqual(result.shape, (1, 3, 2, 8, 8))
                self.assertEqual(result.dtype, torch.float16)
                self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(results[0], results[-1], rtol=0, atol=0)
        torch.testing.assert_close(value, original, rtol=0, atol=0)
        video_vae._clear_attention_caches(decoder)

    def test_complete_decode_matches_native_spatial_and_temporal_reconstruction(self):
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

        model = object.__new__(MiniMaxH3VideoVAE)
        torch.nn.Module.__init__(model)
        model.vae_ratio = 2
        model.vae_ratio_t = 4
        model.tile_size = 8
        model.tile_overlap_min = 4
        model.tiling = True
        model.clip_length = 17
        model.tokens_chunk_size = 5
        model.token_overlap = 2
        model.token_drop = 3
        model.frame_pre_padding = 3
        model.frame_overlap = 5
        model.post_quant_conv = torch.nn.Conv3d(4, 4, 1)
        model.decoder = make_decoder(patch_size_t=4)
        model.latents_mean = torch.randn(4) * 0.1
        model.latents_std = torch.rand(4) + 0.5
        model.pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        model.pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        vae = SimpleNamespace(
            first_stage_model=model, throw_exception_if_invalid=lambda: None,
            vae_dtype=torch.float32, output_device=torch.device("cpu"),
            device=torch.device("cpu"), disable_offload=False,
            patcher=SimpleNamespace(is_dynamic=lambda: False),
            vae_output_dtype=lambda: torch.float32,
        )
        with (
            mock.patch.object(video_vae, "_load_vae_for_tiles", return_value=(3, 0)),
            mock.patch.object(video_vae, "_TileProgress"),
            mock.patch.object(video_vae.comfy.model_management, "load_models_gpu"),
            torch.inference_mode(),
        ):
            for frames in (1, 2, 7, 12):
                with self.subTest(latent_frames=frames):
                    value = torch.randn(1, 4, frames, 6, 8)
                    original = value.clone()
                    expected = model.decode(value).movedim(1, -1)
                    actual = video_vae.decode_video(vae, value, "sdpa")
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(value, original, rtol=0, atol=0)

    def test_decoder_consumes_official_prefetch_queue_per_block(self):
        decoder = make_decoder()
        value = torch.randn(1, 4, 1, 2, 2)
        queue = object()
        options = video_vae._attention_options(
            "sdpa", value.device, prefetch_dynamic_vbars=True
        )
        with (
            mock.patch.object(
                video_vae.comfy.model_prefetch, "make_prefetch_queue", return_value=queue
            ) as make_queue,
            mock.patch.object(
                video_vae.comfy.model_prefetch, "prefetch_queue_pop"
            ) as pop,
            torch.inference_mode(),
        ):
            video_vae._decoder_forward(decoder, value, options)
        make_queue.assert_called_once_with(
            list(decoder.transformer_blocks), value.device, options
        )
        self.assertEqual(
            pop.call_args_list,
            [mock.call(queue, value.device, block) for block in decoder.transformer_blocks]
            + [mock.call(queue, value.device, None)],
        )

    def test_decode_memory_budget_accounts_for_independent_window_batch(self):
        model = SimpleNamespace(
            vae_ratio=16, vae_ratio_t=4, tile_size=256, tokens_chunk_size=5,
            token_overlap=2, decoder=SimpleNamespace(num_register_tokens=4, out_channels=3),
        )
        vae = SimpleNamespace(
            first_stage_model=model, memory_used_decode=lambda shape, dtype: 0
        )
        one = video_vae._decode_memory_requirement(vae, (1, 24, 7, 30, 54), 1, torch.float16)
        two = video_vae._decode_memory_requirement(vae, (1, 24, 7, 30, 54), 2, torch.float16)
        self.assertGreater(two, one)
        self.assertGreater(one, 0)

    def test_decoder_node_uses_optimized_runtime(self):
        vae = mock.Mock()
        vae.device = torch.device("cpu")
        latent = torch.zeros(1, 24, 2, 1, 1)
        decoded = torch.zeros(1, 2, 4, 4, 3)
        with (
            mock.patch.object(nodes, "require_h3_video_vae"),
            mock.patch.object(nodes, "decode_video", return_value=decoded) as run,
            mock.patch.object(
                nodes.comfy.model_management,
                "cuda_device_context",
                return_value=nullcontext(),
            ),
        ):
            output = nodes.MiniMaxH3VideoVAEDecode().decode(
                {"samples": latent},
                vae,
                "sdpa",
            )[0]
        run.assert_called_once_with(
            vae,
            latent,
            "sdpa",
        )
        self.assertEqual(output.shape, (2, 4, 4, 3))

    def test_encoder_node_uses_optimized_runtime(self):
        vae = mock.Mock()
        vae.device = torch.device("cpu")
        pixels = torch.zeros(5, 16, 16, 3)
        encoded = torch.zeros(1, 24, 2, 1, 1)
        with (
            mock.patch.object(nodes, "require_h3_video_vae"),
            mock.patch.object(nodes, "encode_video", return_value=encoded) as run,
            mock.patch.object(
                nodes.comfy.model_management,
                "cuda_device_context",
                return_value=nullcontext(),
            ),
        ):
            output = nodes.MiniMaxH3VideoVAEEncode().encode(pixels, vae)[0]["samples"]
        run.assert_called_once_with(vae, pixels)
        self.assertEqual(output.shape, (1, 24, 2, 1, 1))

    def test_decode_video_publishes_comfy_vae_output_dtype(self):
        latent = torch.zeros(1, 24, 1, 2, 2)
        decoded = torch.full((1, 3, 1, 2, 2), 0.25, dtype=torch.float32)
        blocks = [SimpleNamespace(attn=SimpleNamespace()) for _ in range(36)]
        model = SimpleNamespace(
            vae_ratio=1,
            latents_mean=torch.zeros(24),
            latents_std=torch.ones(24),
            decoder=SimpleNamespace(transformer_blocks=blocks),
            split_tiles=lambda extent: ([0], [extent], []),
            _finalize_pixels=lambda value: value.float(),
        )
        vae = SimpleNamespace(
            vae_dtype=torch.float32,
            output_device=torch.device("cpu"),
            device=torch.device("cpu"),
            patcher=SimpleNamespace(is_dynamic=lambda: False),
            disable_offload=False,
            vae_output_dtype=lambda: torch.float16,
        )
        progress = mock.Mock()
        with (
            mock.patch.object(video_vae, "require_h3_video_vae", return_value=model),
            mock.patch.object(
                video_vae, "_load_vae_for_tiles", return_value=(1, 0)
            ),
            mock.patch.object(video_vae, "_attention_options", return_value={}),
            mock.patch.object(video_vae, "_decode_spatial", return_value=decoded) as spatial,
            mock.patch.object(video_vae, "_TileProgress", return_value=progress),
            mock.patch.object(video_vae.comfy.model_management, "load_models_gpu"),
        ):
            for compute_dtype in (torch.float16, torch.float32):
                vae.vae_dtype = compute_dtype
                for storage_dtype in (torch.float16, torch.bfloat16, torch.float32):
                    with self.subTest(compute=compute_dtype, storage=storage_dtype):
                        actual = video_vae.decode_video(vae, latent.to(storage_dtype))
                        self.assertEqual(spatial.call_args.args[1].dtype, compute_dtype)
        self.assertEqual(actual.dtype, torch.float16)
        torch.testing.assert_close(actual, decoded.movedim(1, -1).half())
        self.assertEqual(progress.finish.call_count, 6)

    def test_encode_video_publishes_comfy_vae_output_dtype(self):
        pixels = torch.zeros(1, 16, 16, 3)
        moments = torch.cat(
            (
                torch.full((1, 24, 1, 1, 1), 0.5),
                torch.zeros(1, 24, 1, 1, 1),
            ),
            dim=1,
        )
        model = SimpleNamespace(
            clip_length=17,
            latents_mean=torch.zeros(24),
            latents_std=torch.ones(24),
        )
        vae = SimpleNamespace(
            vae_dtype=torch.float32,
            output_device=torch.device("cpu"),
            device=torch.device("cpu"),
            patcher=SimpleNamespace(is_dynamic=lambda: False),
            disable_offload=False,
            vae_output_dtype=lambda: torch.float16,
            vae_encode_crop_pixels=lambda value: value,
            process_input=lambda value: value,
        )
        progress = mock.Mock()
        with (
            mock.patch.object(video_vae_encode, "require_h3_video_vae", return_value=model),
            mock.patch.object(
                video_vae_encode, "_load_vae_for_tiles", return_value=(1, 0)
            ),
            mock.patch.object(video_vae_encode, "_encode_clip", return_value=moments),
            mock.patch.object(video_vae_encode, "_TileProgress", return_value=progress),
            mock.patch.object(video_vae_encode.comfy.model_management, "load_models_gpu"),
        ):
            actual = video_vae_encode.encode_video(vae, pixels)
        self.assertEqual(actual.dtype, torch.float16)
        torch.testing.assert_close(actual, torch.full_like(actual, 0.5))
        progress.finish.assert_called_once_with()

    def test_decoder_attention_is_container_owned(self):
        module = SimpleNamespace(
            heads=1,
            dim_head=4,
            to_qkv=torch.nn.Linear(4, 12),
            to_out=torch.nn.Identity(),
            norm_q=SimpleNamespace(weight=None, eps=1e-5),
            norm_k=SimpleNamespace(weight=None, eps=1e-5),
        )
        seen = []

        def consume(q, k, v, heads, **_kwargs):
            self.assertIsInstance(q, video_vae.AttentionTensorContainer)
            self.assertIsInstance(k, video_vae.AttentionTensorContainer)
            self.assertIsInstance(v, video_vae.AttentionTensorContainer)
            seen.extend((q.take(), k.take(), v.take()))
            return torch.zeros(1, 3, 4)

        override = mock.Mock()
        override.container_function = consume
        output = video_vae._attention_forward(
            module,
            torch.randn(1, 3, 4),
            None,
            {"optimized_attention_override": override},
        )
        self.assertEqual(output.shape, (1, 3, 4))
        self.assertEqual([tuple(tensor.shape) for tensor in seen], [(1, 1, 3, 4)] * 3)
        override.assert_not_called()

    def test_decoder_attention_uses_prepared_qk_transform(self):
        module = SimpleNamespace(
            heads=1,
            dim_head=4,
            to_qkv=torch.nn.Linear(4, 12),
            to_out=torch.nn.Identity(),
            norm_q=SimpleNamespace(weight=None, eps=1e-5),
            norm_k=SimpleNamespace(weight=None, eps=1e-5),
        )
        requests = []

        def execute(request):
            requests.append(request)
            self.assertTrue(
                torch.equal(request.qk_transform.query_norm.weight, torch.ones(4))
            )
            self.assertTrue(
                torch.equal(request.qk_transform.key_norm.weight, torch.ones(4))
            )
            query, _key, _value = request.consume_qkv()
            return AttentionExecutionOutcome(
                torch.zeros(query.shape[0], query.shape[2], 4)
            )

        output = video_vae._attention_forward(
            module,
            torch.randn(1, 3, 4),
            None,
            {ATTENTION_EXECUTOR_KEY: execute},
        )
        self.assertEqual(output.shape, (1, 3, 4))
        self.assertEqual(len(requests), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_pixel_double_buffer_preserves_fp32_values(self):
        output = torch.empty(1, 1, 2, 2, 2, dtype=torch.float32)
        model = SimpleNamespace(_finalize_pixels=lambda value: value.float())
        writer = video_vae._PixelWriter(output, model, torch.device("cuda"))
        first = torch.randn(1, 1, 1, 2, 2, device="cuda", dtype=torch.float32)
        second = torch.randn(1, 1, 1, 2, 2, device="cuda", dtype=torch.float32)
        expected = torch.cat((first, second), dim=2).cpu()
        writer.write(first)
        writer.write(second)
        actual = writer.finish()
        self.assertEqual(writer.staging[0].dtype, torch.float32)
        self.assertEqual(writer.staging[1].dtype, torch.float32)
        self.assertTrue(torch.equal(actual, expected))

    def test_fixed_tile_geometry_and_minimal_node_controls(self):
        self.assertEqual(video_vae.TILE_SIZE, 256)
        self.assertEqual(video_vae.TILE_OVERLAP, 64)
        decoder = nodes.MiniMaxH3VideoVAEDecode.INPUT_TYPES()["required"]
        decoder_optional = nodes.MiniMaxH3VideoVAEDecode.INPUT_TYPES().get("optional", {})
        encoder = nodes.MiniMaxH3VideoVAEEncode.INPUT_TYPES()["required"]
        self.assertEqual(
            set(decoder),
            {
                "samples",
                "vae",
                "attention",
            },
        )
        self.assertEqual(decoder_optional, {})
        self.assertEqual(set(encoder), {"pixels", "vae"})

    def test_prepare_batched_pixels_crops_only_spatial_axes(self):
        vae = SimpleNamespace(
            crop_input=True,
            output_channels=3,
            spacial_compression_encode=lambda: 16,
        )
        pixels = torch.zeros(2, 5, 34, 50, 3)
        actual = video_vae_encode._prepare_encode_pixels(vae, pixels)
        self.assertEqual(actual.shape, (2, 3, 5, 32, 48))

    def test_auto_tile_batch_respects_memory_limit(self):
        vae = SimpleNamespace()
        with mock.patch.object(video_vae, "_tile_memory_budget", return_value=35):
            selected, estimate = video_vae._select_tiles_per_batch(
                vae,
                8,
                lambda count: count * 10,
                4,
            )
        self.assertEqual((selected, estimate), (3, 30))
        self.assertEqual(video_vae._AUTO_DECODE_TILE_BATCH_LIMIT, 16)
        self.assertEqual(video_vae._AUTO_ENCODE_TILE_BATCH_LIMIT, 16)

    def test_tile_progress_uses_comfy_progress_bar(self):
        bar = mock.Mock()
        terminal = mock.Mock()
        with (
            mock.patch.object(
                video_vae.comfy.utils, "ProgressBar", return_value=bar
            ) as factory,
            mock.patch.object(video_vae, "tqdm", return_value=terminal) as tqdm_factory,
        ):
            progress = video_vae._TileProgress(11, description="H3 VAE Test")
            progress.update(4)
            progress.update(3)
            progress.finish()
        factory.assert_called_once_with(11)
        tqdm_factory.assert_called_once_with(
            total=11,
            desc="H3 VAE Test",
            disable=not video_vae.comfy.utils.PROGRESS_BAR_ENABLED,
        )
        self.assertEqual([call.args[0] for call in bar.update.call_args_list], [4, 3])
        self.assertEqual(
            [call.args[0] for call in terminal.update.call_args_list], [4, 3]
        )
        terminal.close.assert_called_once_with()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_tile_progress_waits_for_cuda_events_off_thread(self):
        bar = mock.Mock()
        terminal = mock.Mock()
        with (
            mock.patch.object(video_vae.comfy.utils, "ProgressBar", return_value=bar),
            mock.patch.object(video_vae, "tqdm", return_value=terminal),
        ):
            progress = video_vae._TileProgress(5, torch.device("cuda"))
            progress.update(3)
            progress.update(2)
            progress.finish()
        self.assertEqual([call.args[0] for call in bar.update.call_args_list], [3, 2])
        self.assertEqual(
            [call.args[0] for call in terminal.update.call_args_list], [3, 2]
        )
        terminal.close.assert_called_once_with()
        self.assertIsNone(progress.worker)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_encoder_double_buffer_converts_into_fp16_destination(self):
        model = SimpleNamespace(clip_length=2, token_drop=0)
        pixels = torch.randn(1, 3, 4, 4, 4, dtype=torch.float32)
        seen = []

        def fake_encode(_model, clip, *_args):
            seen.append(clip.cpu())
            return torch.zeros(
                clip.shape[0],
                2,
                1,
                1,
                1,
                dtype=clip.dtype,
                device=clip.device,
            )

        with mock.patch.object(video_vae_encode, "_encode_clip", side_effect=fake_encode):
            output = video_vae_encode._encode_temporal_buffered(
                model,
                pixels,
                lambda value: value,
                torch.float16,
                torch.device("cuda"),
                256,
                64,
                None,
                1,
                None,
            )
        self.assertEqual([value.dtype for value in seen], [torch.float16] * 2)
        torch.testing.assert_close(seen[0], pixels[:, :, :2].half())
        torch.testing.assert_close(seen[1], pixels[:, :, 2:].half())
        self.assertEqual(output.dtype, torch.float16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_encoder_double_buffer_keeps_fp16_roundtrip_normalization_on_device(self):
        model = SimpleNamespace(clip_length=2, token_drop=0)
        pixels = torch.randn(1, 3, 2, 4, 4, dtype=torch.float16)
        normalized = []

        def process_input(value):
            normalized.append((value.device.type, value.dtype))
            return value.mul(2)

        def fake_encode(_model, clip, *_args):
            torch.testing.assert_close(clip.cpu(), pixels.mul(2))
            return torch.zeros(
                clip.shape[0],
                2,
                1,
                1,
                1,
                dtype=clip.dtype,
                device=clip.device,
            )

        with mock.patch.object(video_vae_encode, "_encode_clip", side_effect=fake_encode):
            video_vae_encode._encode_temporal_buffered(
                model,
                pixels,
                process_input,
                torch.float16,
                torch.device("cuda"),
                256,
                64,
                None,
                1,
                None,
            )
        self.assertEqual(normalized, [("cuda", torch.float16)])

    def test_decoder_backend_switching_preserves_cached_latent(self):
        vae = mock.Mock()
        vae.device = torch.device("cpu")
        latent = torch.arange(48, dtype=torch.float32).reshape(1, 3, 2, 4, 2)
        original = latent.clone()
        fingerprints = []

        def decode(_vae, value, attention, **_kwargs):
            fingerprints.append((attention, video_vae._latent_fingerprint(value)))
            return torch.zeros(1, 2, 4, 4, 3)

        with (
            mock.patch.object(nodes, "require_h3_video_vae"),
            mock.patch.object(nodes, "decode_video", side_effect=decode),
            mock.patch.object(
                nodes.comfy.model_management,
                "cuda_device_context",
                return_value=nullcontext(),
            ),
        ):
            node = nodes.MiniMaxH3VideoVAEDecode()
            for backend in ("sdpa", "w8a8", "sdpa"):
                node.decode({"samples": latent}, vae, backend)

        self.assertEqual([item[0] for item in fingerprints], ["sdpa", "w8a8", "sdpa"])
        self.assertEqual(len({item[1] for item in fingerprints}), 1)
        torch.testing.assert_close(latent, original)

    def test_custom_decode_preserves_temporal_length_and_values(self):
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

        model = object.__new__(MiniMaxH3VideoVAE)
        torch.nn.Module.__init__(model)
        model.vae_ratio = 1
        model.vae_ratio_t = 4
        model.tokens_chunk_size = 5
        model.token_overlap = 2
        model.token_drop = 3
        model.clip_length = 17
        model.frame_pre_padding = 3
        model.frame_overlap = 5
        model.decoder = SimpleNamespace(out_channels=1)
        model._finalize_pixels = lambda value: value.float()

        z = torch.arange(12, dtype=torch.float32).view(1, 1, 12, 1, 1)

        def fake_decode(_model, value, *_args):
            return value.repeat_interleave(4, dim=2)

        model._adaptive_decode = lambda value: fake_decode(model, value)
        expected = model.decode_temporal(z.clone())
        with mock.patch.object(video_vae, "_decode_spatial", side_effect=fake_decode):
            actual = video_vae._decode_temporal(
                model,
                z.clone(),
                {},
                1,
                None,
                output_device=torch.device("cpu"),
            )
        self.assertTrue(torch.equal(actual, expected))

    def test_temporal_decode_casts_only_finalized_output_dtype(self):
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

        model = object.__new__(MiniMaxH3VideoVAE)
        torch.nn.Module.__init__(model)
        model.vae_ratio = 1
        model.vae_ratio_t = 4
        model.tokens_chunk_size = 5
        model.token_overlap = 2
        model.token_drop = 3
        model.clip_length = 17
        model.frame_pre_padding = 3
        model.frame_overlap = 5
        model.decoder = SimpleNamespace(out_channels=1)
        finalized_dtypes = []

        def finalize(value):
            finalized_dtypes.append(value.dtype)
            return value.float()

        model._finalize_pixels = finalize
        z = torch.arange(12, dtype=torch.float32).view(1, 1, 12, 1, 1)

        def fake_decode(_model, value, *_args):
            return value.repeat_interleave(4, dim=2)

        model._adaptive_decode = lambda value: fake_decode(model, value)
        with mock.patch.object(video_vae, "_decode_spatial", side_effect=fake_decode):
            actual = video_vae._decode_temporal(
                model,
                z.clone(),
                {},
                1,
                None,
                output_device=torch.device("cpu"),
                output_dtype=torch.float16,
            )
        expected = model.decode_temporal(z.clone()).half()
        self.assertEqual(actual.dtype, torch.float16)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(finalized_dtypes)
        self.assertEqual(set(finalized_dtypes), {torch.float32})


if __name__ == "__main__":
    unittest.main()
