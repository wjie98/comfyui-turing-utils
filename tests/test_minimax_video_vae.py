from __future__ import annotations

import sys
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT.parents[1]))
sys.path.insert(0, str(PLUGIN_ROOT))

import comfy.model_prefetch
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE, ViT3DDecoder
from comfy.sd import VAE
from comfyui_turing_utils.adapters.minimax import video_vae, video_vae_encode
from comfyui_turing_utils.attention.protocol import (
    ATTENTION_EXECUTOR_KEY,
    AttentionExecutionOutcome,
)
from comfyui_turing_utils.nodes import minimax_vae as nodes


def make_decoder(patch_size_t=4):
    torch.manual_seed(26)
    decoder = ViT3DDecoder(
        patch_size=2,
        patch_size_t=patch_size_t,
        in_channels=4,
        out_channels=3,
        num_layers=2,
        heads=1,
        dim_head=64,
        operations=torch.nn,
    ).eval()
    for name, parameter in decoder.named_parameters():
        if "norm" in name and name.endswith("weight"):
            torch.nn.init.ones_(parameter)
        elif "scale" in name:
            torch.nn.init.constant_(parameter, 0.5)
        else:
            torch.nn.init.uniform_(parameter, -0.1, 0.1)
    return decoder


class _TinyEncoder(torch.nn.Module):
    def forward(self, value):
        # Cheap stand-in for the CNN; exercise the actual upstream spatial and
        # temporal tiling, not a second implementation of those algorithms.
        value = value[:, :, ::4, ::2, ::2]
        return torch.cat((value, value, value), dim=1)[:, :8]


def make_vae():
    # Avoid allocating the full 2.6B-parameter VAE in lifecycle regressions.
    model = object.__new__(MiniMaxH3VideoVAE)
    torch.nn.Module.__init__(model)
    model.vae_ratio = 2
    model.vae_ratio_t = 4
    model.clip_length = 17
    model.token_drop = 3
    model.frame_pre_padding = 3
    model.tokens_chunk_size = 5
    model.token_overlap = 2
    model.frame_overlap = 5
    model.tiling = True
    model.tile_size = 4
    model.tile_overlap_min = 2
    model.encoder = _TinyEncoder()
    model.quant_conv = torch.nn.Conv3d(8, 8, 1)
    model.post_quant_conv = torch.nn.Identity()
    model.decoder = make_decoder()
    model.register_buffer("latents_mean", torch.zeros(4))
    model.register_buffer("latents_std", torch.ones(4))
    model.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1, 1))
    model.register_buffer("pixel_std", torch.ones(1, 3, 1, 1, 1))

    vae = object.__new__(VAE)
    vae.first_stage_model = model
    vae.device = torch.device("cpu")
    vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.float32
    vae.disable_offload = False
    vae.crop_input = True
    vae.output_channels = 3
    vae.pad_channel_value = None
    vae.latent_dim = 3
    vae.latent_channels = 4
    vae.not_video = False
    vae.extra_1d_channel = None
    vae.handles_tiling = True
    vae.downscale_ratio = (4, 2, 2)
    vae.upscale_ratio = (4, 2, 2)
    vae.format_encoded = None
    vae.process_input = lambda pixels: pixels * 2 - 1
    vae.process_output = lambda pixels: pixels
    vae.memory_used_encode = lambda shape, dtype: 1024
    vae.memory_used_decode = lambda shape, dtype: 1024
    vae.patcher = mock.Mock()
    vae.patcher.get_free_memory.return_value = 1024 * 1024
    return vae


class MiniMaxVideoVAETest(unittest.TestCase):
    def test_requires_h3_and_propagates_loader_validation(self):
        vae = SimpleNamespace(
            first_stage_model=torch.nn.Identity(),
            throw_exception_if_invalid=mock.Mock(),
        )
        with self.assertRaisesRegex(ValueError, "MiniMax H3 video VAE"):
            video_vae.require_h3_video_vae(vae)
        vae.throw_exception_if_invalid.side_effect = RuntimeError("invalid VAE")
        with self.assertRaisesRegex(RuntimeError, "invalid VAE"):
            video_vae.require_h3_video_vae(vae)

    def test_vae_operator_scope_is_optional_and_restored_on_error(self):
        with (
            mock.patch.object(video_vae, "register_backend", return_value=False),
            mock.patch.object(video_vae, "use_turing_operator_backend") as backend,
        ):
            with video_vae._vae_operator_scope():
                pass
            backend.assert_not_called()
        with (
            mock.patch.object(video_vae, "register_backend", return_value=True),
            mock.patch.object(video_vae, "use_turing_operator_backend") as backend,
        ):
            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                with video_vae._vae_operator_scope():
                    raise RuntimeError("decode failed")
            backend.return_value.__exit__.assert_called_once()

    def test_entry_points_delegate_unchanged_tensors_without_memory_management(self):
        vae = make_vae()
        pixels = torch.rand(5, 7, 9, 4)
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(video_vae, "register_backend", return_value=False)
            )
            # Official methods are mocked in this test: only plugin-owned calls
            # would trip these guards. Real native entry points are tested below.
            for owner, name in (
                (video_vae.comfy.model_management, "load_models_gpu"),
                (video_vae.comfy.model_management, "get_free_memory"),
                (video_vae.comfy.model_management, "soft_empty_cache"),
                (comfy.model_prefetch, "make_prefetch_queue"),
                (comfy.model_prefetch, "cleanup_prefetch_queues"),
                (torch.cuda, "Stream"),
                (torch.cuda, "Event"),
                (torch.cuda, "synchronize"),
            ):
                stack.enter_context(
                    mock.patch.object(
                        owner,
                        name,
                        side_effect=AssertionError("plugin must not own " + name),
                    )
                )
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                with self.subTest(dtype=dtype):
                    latent = torch.zeros(1, 4, 2, 3, 4, dtype=dtype)
                    encoded = object()
                    decoded = object()
                    with (
                        mock.patch.object(
                            vae, "encode", return_value=encoded
                        ) as encode,
                        mock.patch.object(
                            vae, "decode", return_value=decoded
                        ) as decode,
                    ):
                        self.assertIs(
                            video_vae_encode.encode_video(vae, pixels), encoded
                        )
                        self.assertIs(video_vae.decode_video(vae, latent), decoded)
                        encode.assert_called_once_with(pixels)
                        decode.assert_called_once_with(latent)
                        self.assertIs(encode.call_args.args[0], pixels)
                        self.assertIs(decode.call_args.args[0], latent)

    def test_native_encoder_output_and_input_cropping_are_unchanged(self):
        vae = make_vae()
        for frames in (1, 5, 18):
            pixels = torch.rand(frames, 7, 9, 4)
            original = pixels.clone()
            batches = []
            handle = vae.first_stage_model.encoder.register_forward_pre_hook(
                lambda _module, args: batches.append(args[0].shape[0])
            )
            try:
                with (
                    mock.patch.object(
                        video_vae, "register_backend", return_value=False
                    ),
                    mock.patch.object(
                        video_vae.comfy.model_management, "load_models_gpu"
                    ),
                    torch.inference_mode(),
                ):
                    expected = vae.encode(pixels)
                    expected_batches = batches[:]
                    batches.clear()
                    actual = video_vae_encode.encode_video(vae, pixels)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(batches, expected_batches)
                self.assertTrue(all(batch == 1 for batch in batches))
                torch.testing.assert_close(pixels, original, rtol=0, atol=0)
            finally:
                handle.remove()

    def test_native_decoder_spatial_temporal_batch_and_dtype_parity(self):
        vae = make_vae()
        for batch, frames, output_dtype in (
            (1, 1, torch.float32),
            (1, 2, torch.float16),
            (2, 7, torch.float32),
            (1, 12, torch.bfloat16),
        ):
            with self.subTest(batch=batch, frames=frames, output_dtype=output_dtype):
                latent = torch.randn(batch, 4, frames, 3, 4)
                original = latent.clone()
                batches = []
                handle = vae.first_stage_model.decoder.register_forward_pre_hook(
                    lambda _module, args: batches.append(args[0].shape[0])
                )
                try:
                    with (
                        mock.patch.object(
                            video_vae, "register_backend", return_value=False
                        ),
                        mock.patch.object(
                            video_vae.comfy.model_management, "load_models_gpu"
                        ),
                        mock.patch.object(
                            vae, "vae_output_dtype", return_value=output_dtype
                        ),
                        torch.inference_mode(),
                    ):
                        expected = vae.decode(latent)
                        expected_batches = batches[:]
                        batches.clear()
                        actual = video_vae.decode_video(vae, latent, "sdpa")
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(actual.dtype, output_dtype)
                    self.assertEqual(batches, expected_batches)
                    self.assertTrue(all(size % batch == 0 for size in batches))
                    self.assertTrue(any(size > batch for size in batches))
                    torch.testing.assert_close(latent, original, rtol=0, atol=0)
                finally:
                    handle.remove()

    def test_official_oom_fallback_is_not_replaced_or_retried_by_plugin(self):
        vae = make_vae()
        for operation in ("encode", "decode"):
            with self.subTest(operation=operation):
                if operation == "encode":
                    value = torch.rand(5, 6, 8, 3)
                    result = torch.zeros(1, 4, 2, 3, 4)
                    run = video_vae_encode.encode_video
                else:
                    value = torch.zeros(1, 4, 2, 3, 4)
                    result = torch.zeros(1, 3, 5, 6, 8)
                    run = video_vae.decode_video
                with (
                    mock.patch.object(
                        video_vae, "register_backend", return_value=False
                    ),
                    mock.patch.object(
                        video_vae.comfy.model_management, "load_models_gpu"
                    ),
                    mock.patch.object(
                        video_vae.comfy.model_management, "soft_empty_cache"
                    ),
                    mock.patch.object(
                        vae.first_stage_model,
                        operation,
                        side_effect=[torch.cuda.OutOfMemoryError("native OOM"), result],
                    ) as native,
                    torch.inference_mode(),
                ):
                    actual = run(vae, value)
                self.assertEqual(native.call_count, 2)
                expected = result if operation == "encode" else result.movedim(1, -1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_errors_and_cancellation_restore_overrides_and_hooks_without_retry(self):
        vae = make_vae()
        model = vae.first_stage_model
        for operation in ("encode", "decode"):
            for error in (
                RuntimeError("VRAM grow failed"),
                torch.cuda.OutOfMemoryError("OOM"),
                KeyboardInterrupt(),
            ):
                with self.subTest(operation=operation, error=type(error)):
                    terminal = mock.Mock()
                    factory = mock.Mock()
                    factory.return_value.__enter__ = mock.Mock(return_value=terminal)
                    factory.return_value.__exit__ = mock.Mock(return_value=False)
                    existing = lambda value, rotary_pos_emb=None: value
                    model.decoder.transformer_blocks[0].attn.forward = existing
                    with (
                        mock.patch.object(
                            video_vae, "register_backend", return_value=False
                        ),
                        mock.patch.object(video_vae, "tqdm", factory),
                        mock.patch.object(vae, operation, side_effect=error) as native,
                    ):
                        run = (
                            video_vae_encode.encode_video
                            if operation == "encode"
                            else video_vae.decode_video
                        )
                        with self.assertRaises(type(error)):
                            run(vae, torch.zeros(1))
                    native.assert_called_once()
                    self.assertIs(
                        model.decoder.transformer_blocks[0].attn.forward, existing
                    )
                    self.assertNotIn(
                        "forward", model.decoder.transformer_blocks[1].attn.__dict__
                    )
                    for block in model.decoder.transformer_blocks:
                        self.assertNotIn("forward", block.ff.__dict__)
                    self.assertFalse(model.decoder._forward_hooks)
                    self.assertFalse(model.quant_conv._forward_hooks)
                    factory.return_value.__exit__.assert_called_once()
                    del model.decoder.transformer_blocks[0].attn.forward

    def test_partial_override_installation_rolls_back(self):
        decoder = make_decoder()
        original = video_vae._temporary_forward
        count = 0

        @contextmanager
        def fail_second(module, forward):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("installation failed")
            with original(module, forward):
                yield

        with mock.patch.object(video_vae, "_temporary_forward", fail_second):
            with self.assertRaisesRegex(RuntimeError, "installation failed"):
                with video_vae._decoder_overrides(decoder, "sdpa", torch.device("cpu")):
                    self.fail("must not execute")
        for block in decoder.transformer_blocks:
            self.assertNotIn("forward", block.attn.__dict__)
            self.assertNotIn("forward", block.ff.__dict__)

    def test_backend_selection_is_scoped_to_one_decoder(self):
        decoder, other = make_decoder(), make_decoder()
        original = other.transformer_blocks[0].attn.forward
        for backend in ("sdpa", "w8a8", "sage", "sdpa"):
            with mock.patch.object(
                video_vae, "_attention_options", return_value={}
            ) as options:
                with video_vae._decoder_overrides(
                    decoder, backend, torch.device("cpu")
                ):
                    self.assertIn(
                        "forward", decoder.transformer_blocks[0].attn.__dict__
                    )
                    self.assertEqual(other.transformer_blocks[0].attn.forward, original)
                options.assert_called_once_with(backend, torch.device("cpu"))
            for block in decoder.transformer_blocks:
                self.assertNotIn("forward", block.attn.__dict__)

    def test_native_attention_never_replaces_upstream_forward(self):
        decoder = make_decoder()
        original = [block.attn.forward for block in decoder.transformer_blocks]
        with mock.patch.object(
            video_vae, "_attention_options", side_effect=AssertionError("no override")
        ):
            with video_vae._decoder_overrides(decoder, "native", torch.device("cpu")):
                for block, forward in zip(decoder.transformer_blocks, original):
                    self.assertEqual(block.attn.forward, forward)
                    self.assertNotIn("forward", block.attn.__dict__)

    def test_pre_norm_cast_lives_through_fused_linear_and_releases_on_error(self):
        x = torch.rand(1, 3, 4)
        norm = SimpleNamespace(weight=torch.ones(4), eps=1e-5, comfy_cast_weights=True)
        cast_weight = torch.full((4,), 2.0)
        active = []

        @contextmanager
        def cast(owner, reference, *, offloadable):
            self.assertIs(owner, norm)
            self.assertIs(reference, x)
            self.assertTrue(offloadable)
            active.append(True)
            try:
                yield cast_weight, None
            finally:
                active.pop()

        def linear(layer, value, act, weight, eps):
            self.assertEqual(active, [True])
            self.assertIs(weight, cast_weight)
            self.assertEqual(eps, norm.eps)
            raise RuntimeError("linear failed")

        with (
            mock.patch.object(video_vae.comfy.ops, "CastBiasWeightContext", cast),
            mock.patch.object(video_vae.comfy.ops, "linear_input_act", linear),
        ):
            with self.assertRaisesRegex(RuntimeError, "linear failed"):
                video_vae._attention_forward(
                    SimpleNamespace(to_qkv=object()),
                    x,
                    None,
                    norm,
                    x,
                    torch.ones(4),
                    options={},
                )
        self.assertEqual(active, [])

    def test_modern_norm_owns_its_cast_context(self):
        x = torch.rand(1, 3, 4)
        cast_weight = torch.ones(4)
        owner = mock.MagicMock()
        owner.__enter__.return_value = (cast_weight, None)
        norm = SimpleNamespace(cast_weight=mock.Mock(return_value=owner))
        with video_vae._norm_weight_context(norm, x) as weight:
            self.assertIs(weight, cast_weight)
        norm.cast_weight.assert_called_once_with(x)
        owner.__exit__.assert_called_once()

    def test_plain_and_nonaffine_norm_preserve_input_dtype(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            x = torch.rand(1, 3, 4).to(dtype)
            for affine in (True, False):
                norm = torch.nn.RMSNorm(4, elementwise_affine=affine)
                with video_vae._norm_weight_context(norm, x) as weight:
                    if affine:
                        self.assertEqual(weight.dtype, dtype)
                        self.assertEqual(weight.device, x.device)
                    else:
                        self.assertIsNone(weight)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_comfy_cpu_norm_weight_is_cast_without_moving_owned_parameter(self):
        norm = video_vae.comfy.ops.disable_weight_init.RMSNorm(4, eps=1e-5)
        norm.weight.data.fill_(2.0)
        original = norm.weight
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            x = torch.rand(1, 3, 4, device="cuda", dtype=dtype)
            with video_vae._norm_weight_context(norm, x) as weight:
                self.assertEqual(weight.device, x.device)
                self.assertEqual(weight.dtype, x.dtype)
                torch.testing.assert_close(weight, torch.full_like(weight, 2.0))
            self.assertIs(norm.weight, original)
            self.assertEqual(norm.weight.device.type, "cpu")

    def test_progress_only_observes_forwards_and_removes_hook(self):
        module = torch.nn.Identity()
        first = torch.rand(4, 3)
        second = torch.rand(2, 3)
        terminal = mock.Mock()
        terminal.total = 3
        with mock.patch.object(video_vae, "tqdm") as factory:
            factory.return_value.__enter__.return_value = terminal
            with video_vae._tile_progress(module, "test", total=3, base_batch_size=2):
                self.assertIs(module(first), first)
                self.assertIs(module(second), second)
            self.assertFalse(module._forward_hooks)
            module(second)
        self.assertEqual(terminal.update.call_args_list, [mock.call(2), mock.call(1)])
        factory.assert_called_once_with(
            total=3,
            desc="test",
            unit="tile",
            disable=not video_vae.comfy.utils.PROGRESS_BAR_ENABLED,
        )

    def test_progress_becomes_open_ended_if_a_retry_exceeds_the_plan(self):
        module = torch.nn.Identity()
        terminal = mock.Mock()
        terminal.total = 2
        with mock.patch.object(video_vae, "tqdm") as factory:
            factory.return_value.__enter__.return_value = terminal
            with video_vae._tile_progress(module, "test", total=2):
                module(torch.zeros(1))
                module(torch.zeros(1))
                module(torch.zeros(1))
        self.assertIsNone(terminal.total)
        terminal.refresh.assert_called_once_with()

    def test_h3_tile_totals_match_native_spatial_and_temporal_plans(self):
        vae = make_vae()
        model = vae.first_stage_model

        pixels = torch.zeros(18, 6, 10, 3)
        self.assertEqual(video_vae._encode_tile_total(vae, model, pixels), 16)

        latent = torch.zeros(1, 4, 37, 3, 5)
        self.assertEqual(video_vae._decode_tile_total(model, latent), 56)

    def test_decode_tile_total_is_independent_of_video_batch_size(self):
        model = make_vae().first_stage_model
        latent = torch.zeros(2, 4, 37, 3, 5)
        self.assertEqual(video_vae._decode_tile_total(model, latent), 56)

    def test_decoder_override_leaves_official_ffn_untouched(self):
        decoder = make_decoder()
        module = decoder.transformer_blocks[0].ff
        with video_vae._decoder_overrides(decoder, "sdpa", torch.device("cpu")):
            self.assertNotIn("forward", module.__dict__)

    def test_decoder_attention_uses_containers_and_prepared_qk_transform(self):
        module = SimpleNamespace(
            heads=1,
            dim_head=4,
            to_qkv=torch.nn.Linear(4, 12),
            to_out=torch.nn.Linear(4, 4),
            norm_q=SimpleNamespace(weight=None, eps=1e-5),
            norm_k=SimpleNamespace(weight=None, eps=1e-5),
        )
        value = torch.randn(1, 3, 4)
        pre_norm = torch.nn.RMSNorm(4)
        residual = value.clone()
        residual_scale = torch.ones(4)
        seen = []

        def consume(q, k, v, heads, **kwargs):
            for tensor in (q, k, v):
                self.assertIsInstance(tensor, video_vae.AttentionTensorContainer)
                seen.append(tensor.take())
            return torch.zeros(1, 3, 4)

        override = mock.Mock()
        override.container_function = consume
        output = video_vae._attention_forward(
            module,
            value,
            None,
            pre_norm,
            residual,
            residual_scale,
            options={"optimized_attention_override": override},
        )
        self.assertEqual(output.shape, value.shape)
        self.assertEqual([tuple(t.shape) for t in seen], [(1, 1, 3, 4)] * 3)
        override.assert_not_called()

        def execute(request):
            self.assertTrue(
                torch.equal(request.qk_transform.query_norm.weight, torch.ones(4))
            )
            self.assertTrue(
                torch.equal(request.qk_transform.key_norm.weight, torch.ones(4))
            )
            request.consume_qkv()
            return AttentionExecutionOutcome(torch.zeros(1, 3, 4))

        video_vae._attention_forward(
            module,
            value,
            None,
            pre_norm,
            residual,
            residual_scale,
            options={ATTENTION_EXECUTOR_KEY: execute},
        )
        self.assertFalse(any(key.startswith("_turing_utils") for key in vars(module)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_native_cuda_fp16_decoder_and_nondefault_stream(self):
        decoder = make_decoder().to(device="cuda", dtype=torch.float16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(stream):
            value = torch.randn(1, 4, 2, 3, 4, device="cuda", dtype=torch.float16)
            original = value.clone()
            expected = decoder(value)
            with video_vae._decoder_overrides(decoder, "sdpa", value.device):
                actual = decoder(value)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(value, original, rtol=0, atol=0)
        stream.synchronize()

    def test_node_interfaces_and_av_unwrap_remain_compatible(self):
        decode_inputs = nodes.MiniMaxH3VideoVAEDecode.INPUT_TYPES()["required"]
        encode_inputs = nodes.MiniMaxH3VideoVAEEncode.INPUT_TYPES()["required"]
        self.assertEqual(set(decode_inputs), {"samples", "vae", "attention"})
        self.assertEqual(set(encode_inputs), {"pixels", "vae"})
        self.assertEqual(decode_inputs["attention"][1]["default"], "w8a8")
        latent = torch.zeros(1, 4, 2, 3, 4)
        nested = mock.Mock(is_nested=True)
        nested.unbind.return_value = (latent, torch.zeros(1))
        vae = SimpleNamespace(device=torch.device("cpu"))
        decoded = torch.zeros(2, 5, 6, 8, 3)
        with (
            mock.patch.object(nodes, "require_h3_video_vae"),
            mock.patch.object(nodes, "decode_video", return_value=decoded) as decode,
            mock.patch.object(nodes, "encode_video", return_value=latent) as encode,
            mock.patch.object(
                nodes.comfy.model_management,
                "cuda_device_context",
                return_value=nullcontext(),
            ),
        ):
            output = nodes.MiniMaxH3VideoVAEDecode().decode(
                {"samples": nested}, vae, "sdpa"
            )[0]
            self.assertEqual(output.shape, (10, 6, 8, 3))
            decode.assert_called_once_with(vae, latent, "sdpa")
            pixels = torch.rand(5, 6, 8, 3)
            self.assertIs(
                nodes.MiniMaxH3VideoVAEEncode().encode(pixels, vae)[0]["samples"],
                latent,
            )
            encode.assert_called_once_with(vae, pixels)


if __name__ == "__main__":
    unittest.main()
