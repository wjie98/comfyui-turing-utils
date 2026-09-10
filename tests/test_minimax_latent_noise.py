from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

import comfy.latent_formats  # noqa: E402
import comfy.model_base  # noqa: E402
import comfy.model_sampling  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.utils  # noqa: E402
from comfy.nested_tensor import NestedTensor  # noqa: E402
from comfy_extras.nodes_custom_sampler import DisableNoise, Noise_RandomNoise  # noqa: E402
from comfyui_turing_utils.nodes.minimax import H3AddNoise, H3ConcatAVLatent, H3SeparateAVLatent  # noqa: E402


class H3Sampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
    pass


class FixedNoise:
    def __init__(self, samples):
        self.samples = samples
        self.calls = []

    def generate_noise(self, latent):
        self.calls.append(latent)
        return self.samples


class TestModel:
    __test__ = False

    def __init__(self, shift=12.0, audio_shift=3.0, noise_scale=1.0):
        # Exercise actual H3 format/sampler methods without constructing a DiT.
        self.model = comfy.model_base.MiniMaxH3.__new__(comfy.model_base.MiniMaxH3)
        torch.nn.Module.__init__(self.model)
        self.model.model_sampling = H3Sampling()
        self.model.model_sampling.set_parameters(shift=shift, audio_shift=audio_shift)
        self.model.model_sampling.noise_scale = noise_scale
        self.model.latent_format = comfy.latent_formats.MiniMaxH3AV()
        self.model.latent_shapes = None
        self.object_patches = {}

    def get_model_object(self, name):
        return self.object_patches[name] if name in self.object_patches else getattr(self.model, name)


class H3AddNoiseTest(unittest.TestCase):
    def setUp(self):
        self.model = TestModel()
        self.video = torch.randn(2, 24, 2, 3, 4)
        self.audio = torch.randn(2, 32, 2, 9)
        self.video_noise = torch.randn_like(self.video)
        self.audio_noise = torch.randn_like(self.audio)
        self.sigmas = torch.tensor([0.6, 0.3, 0.0])

    def prepare(self, samples, noise, sigmas=None, model=None, **metadata):
        return H3AddNoise.execute(
            model or self.model,
            FixedNoise(noise),
            self.sigmas if sigmas is None else sigmas,
            {"samples": samples, **metadata},
        ).result[0]

    def test_schema(self):
        schema = H3AddNoise.define_schema()
        self.assertEqual(schema.node_id, "TuringUtilsH3AddNoise")
        self.assertEqual(schema.display_name, "H3 Add Noise")
        self.assertEqual([item.id for item in schema.inputs], ["model", "noise", "sigmas", "latent_image"])
        self.assertEqual(len(schema.outputs), 1)

    def test_standalone_video_round_trip_to_sampler_sigma(self):
        prepared = self.prepare(self.video, self.video_noise)["samples"]
        start = self.model.model.model_sampling.noise_scaling(
            self.sigmas[0], torch.zeros_like(prepared), prepared
        )
        torch.testing.assert_close(start, 0.4 * self.video + 0.6 * self.video_noise)

    def test_standalone_audio_uses_video_schedule_and_audio_carry(self):
        prepared = self.prepare(self.audio, self.audio_noise)["samples"]
        torch.testing.assert_close(prepared, self.audio + (0.6 / 0.4 / 4.0) * self.audio_noise)

    def test_av_changes_both_streams_without_mutating_input(self):
        samples = NestedTensor([self.video, self.audio])
        noise = FixedNoise(NestedTensor([self.video_noise, self.audio_noise]))
        masks = NestedTensor([torch.zeros_like(self.video), torch.ones_like(self.audio)])
        metadata = {"purpose": "continuation"}
        latent = {"samples": samples, "noise_mask": masks, "batch_index": [2, 2], "metadata": metadata}
        video_copy, audio_copy = self.video.clone(), self.audio.clone()
        output = H3AddNoise.execute(self.model, noise, self.sigmas, latent).result[0]
        self.assertEqual(noise.calls, [latent])
        self.assertIsNot(output, latent)
        self.assertIs(output["noise_mask"], masks)
        self.assertIs(output["metadata"], metadata)
        self.assertIs(output["batch_index"], latent["batch_index"])
        for original, modified in zip(samples.unbind(), output["samples"].unbind()):
            self.assertFalse(torch.equal(original, modified))
        torch.testing.assert_close(self.video, video_copy)
        torch.testing.assert_close(self.audio, audio_copy)

    def test_combined_and_separate_streams_have_identical_math(self):
        combined = self.prepare(
            NestedTensor([self.video, self.audio]), NestedTensor([self.video_noise, self.audio_noise])
        )["samples"]
        video = self.prepare(self.video, self.video_noise)["samples"]
        audio = self.prepare(self.audio, self.audio_noise)["samples"]
        torch.testing.assert_close(combined.unbind()[0], video)
        torch.testing.assert_close(combined.unbind()[1], audio)

    def test_random_noise_respects_batch_index_and_seed(self):
        latent = {"samples": NestedTensor([self.video, self.audio]), "batch_index": [3, 3]}
        noise = Noise_RandomNoise(17)
        generated = noise.generate_noise(latent)
        output = H3AddNoise.execute(self.model, noise, self.sigmas, latent).result[0]["samples"]
        for clean, epsilon, prepared, scale in zip(
            latent["samples"].unbind(), generated.unbind(), output.unbind(), [1.0, 4.0]
        ):
            torch.testing.assert_close(epsilon[0], epsilon[1])
            torch.testing.assert_close(prepared, clean + 1.5 / scale * epsilon)

    def test_first_sigma_not_difference_and_single_sigma_supported(self):
        expected = self.prepare(self.video, self.video_noise)["samples"]
        for sigmas in [torch.tensor([0.6]), torch.tensor([0.6, 0.4, 0.2])]:
            with self.subTest(sigmas=sigmas):
                actual = self.prepare(self.video, self.video_noise, sigmas)["samples"]
                torch.testing.assert_close(actual, expected)

    def test_empty_and_zero_sigmas_skip_noise_generation(self):
        latent = {"samples": NestedTensor([self.video, self.audio])}
        noise = FixedNoise(None)
        for sigmas in [torch.tensor([]), torch.tensor([0.0])]:
            self.assertIs(H3AddNoise.execute(self.model, noise, sigmas, latent).result[0], latent)
        self.assertEqual(noise.calls, [])

    def test_invalid_sigma_fails_before_generating_noise(self):
        for value in [1.0, 1.5, -0.2, float("inf"), float("nan")]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "0 <= sigmas"):
                self.prepare(self.video, self.video_noise, torch.tensor([value]))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            self.prepare(self.video, self.video_noise, torch.tensor([[0.5]]))

    def test_model_sampling_patches_are_used_without_mutating_live_state(self):
        patched = TestModel(shift=8, audio_shift=4, noise_scale=1.7).model.model_sampling
        self.model.object_patches["model_sampling"] = patched
        stale_shapes = [(1, 24, 7, 99, 99), (1, 32, 2, 38)]
        self.model.model.latent_shapes = stale_shapes
        actual = self.prepare(self.audio, self.audio_noise)["samples"]
        torch.testing.assert_close(actual, self.audio + 1.5 * 1.7 / 2.0 * self.audio_noise)
        self.assertIs(self.model.model.latent_shapes, stale_shapes)
        self.assertEqual(self.model.model.model_sampling.audio_scale, 4.0)

    def test_absent_audio_shift_means_equal_stream_scales(self):
        actual = self.prepare(self.audio, self.audio_noise, model=TestModel(audio_shift=None))["samples"]
        torch.testing.assert_close(actual, self.audio + 1.5 * self.audio_noise)

    def test_half_inputs_promote_for_safe_near_one_sigma(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                clean = self.video.to(dtype)
                epsilon = self.video_noise.to(dtype)
                actual = self.prepare(clean, epsilon, torch.tensor([0.99999]))["samples"]
                self.assertEqual(actual.dtype, torch.promote_types(dtype, torch.float32))
                self.assertEqual(actual.device, clean.device)
                self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_non_h3_model_is_rejected(self):
        self.model.model.latent_format = comfy.latent_formats.SD15()
        with self.assertRaisesRegex(ValueError, "MiniMax H3"):
            self.prepare(self.video, self.video_noise)

    def test_double_sigma_is_not_rounded_to_one(self):
        sigmas = torch.tensor([1.0 - 1e-10], dtype=torch.float64)
        actual = self.prepare(self.video, self.video_noise, sigmas)["samples"]
        self.assertEqual(actual.dtype, torch.float64)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        start = self.model.model.model_sampling.noise_scaling(sigmas[0], torch.zeros_like(actual), actual)
        expected = (1 - sigmas[0]) * self.video.double() + sigmas[0] * self.video_noise.double()
        torch.testing.assert_close(start, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cpu_random_noise_with_cuda_av_latents(self):
        latent = {"samples": NestedTensor([self.video.cuda(), self.audio.cuda()])}
        noise = Noise_RandomNoise(19)
        generated = noise.generate_noise(latent)
        actual = H3AddNoise.execute(self.model, noise, self.sigmas, latent).result[0]["samples"]
        for clean, epsilon, prepared, carry in zip(
            latent["samples"].unbind(), generated.unbind(), actual.unbind(), [1.0, 4.0]
        ):
            self.assertEqual(prepared.device, clean.device)
            self.assertEqual(epsilon.device.type, "cpu")
            torch.testing.assert_close(prepared, clean + 1.5 / carry * epsilon.to(clean.device))

    def test_wrong_stream_order_and_broadcasting_noise_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected H3"):
            self.prepare(NestedTensor([self.audio, self.video]), NestedTensor([self.audio_noise, self.video_noise]))
        with self.assertRaisesRegex(ValueError, "same shape"):
            self.prepare(self.video, self.video_noise[:1])
        with self.assertRaisesRegex(ValueError, "same video/audio structure"):
            self.prepare(NestedTensor([self.video, self.audio]), self.video_noise)
        with self.assertRaisesRegex(ValueError, "video and audio"):
            self.prepare(NestedTensor([self.video]), NestedTensor([self.video_noise]))

    def sampler_start(self, latent):
        """Capture actual KSAMPLER's input with native H3 packed AV scaling."""
        empty_noise = DisableNoise.execute().result[0].generate_noise(latent)
        packed, shapes = comfy.utils.pack_latents(latent["samples"].unbind())
        noise, _ = comfy.utils.pack_latents(empty_noise.unbind())
        self.model.model.latent_shapes = shapes
        internal = self.model.model.process_latent_in(packed)
        captured = []

        def capture(_model, x, _sigmas, **kwargs):
            captured.append(x)
            return x

        comfy.samplers.KSAMPLER(capture).sample(
            SimpleNamespace(inner_model=self.model.model),
            self.sigmas, {}, None, noise, latent_image=internal, disable_pbar=True,
        )
        return comfy.utils.unpack_latents(captured[0], shapes)

    def test_native_disable_noise_sampler_receives_correct_av_state(self):
        latent = self.prepare(
            NestedTensor([self.video, self.audio]), NestedTensor([self.video_noise, self.audio_noise])
        )
        video, audio = self.sampler_start(latent)
        torch.testing.assert_close(video, 0.4 * self.video + 0.6 * self.video_noise)
        torch.testing.assert_close(audio, 0.4 * 4.0 * self.audio + 0.6 * self.audio_noise)

    def test_upscaled_video_can_rejoin_unchanged_partial_output_audio(self):
        # First sampler exports intermediate audio via its native inverse and
        # process_latent_out. The learned upscaler supplies a new clean video.
        self.model.model.latent_shapes = [self.video.shape, self.audio.shape]
        partial_audio = torch.randn_like(self.audio)
        first_output = self.model.model.process_latent_out(NestedTensor([
            self.model.model.model_sampling.inverse_noise_scaling(self.sigmas[0], self.video),
            self.model.model.model_sampling.inverse_noise_scaling(self.sigmas[0], partial_audio),
        ]))
        _, original_audio = H3SeparateAVLatent.execute({"samples": first_output}).result
        larger_video = torch.randn(2, 24, 2, 6, 8)
        epsilon = torch.randn_like(larger_video)
        video = self.prepare(larger_video, epsilon)
        joined = H3ConcatAVLatent.execute(video, original_audio).result[0]
        self.assertIs(joined["samples"].unbind()[1], original_audio["samples"])
        actual_video, actual_audio = self.sampler_start(joined)
        torch.testing.assert_close(actual_video, 0.4 * larger_video + 0.6 * epsilon)
        torch.testing.assert_close(actual_audio, partial_audio)


if __name__ == "__main__":
    unittest.main()
