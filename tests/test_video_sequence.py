from __future__ import annotations

import math
import sys
import tempfile
import unittest
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))


from comfyui_turing_utils.media import segments, prefix_noise
from comfyui_turing_utils.nodes import video_sequence as nodes  # noqa: E402


class VideoSequenceTest(unittest.TestCase):
    def test_concat_broadcasts_only_single_mask_and_preserves_per_frame_masks(self):
        frames = torch.zeros(3, 8, 12, 3)
        masks = torch.zeros(3, 8, 12)
        for index in range(3):
            masks[index, 2:5, 1 + index * 3 : 3 + index * 3] = 1
        for mode in ("concat", "replace"):
            repeated = nodes.VideoContinuationConcat.execute(
                body_images=frames,
                body_mask=masks[:1],
                mode=mode,
            ).result[1]
            torch.testing.assert_close(repeated, masks[:1].expand_as(masks))
            tracked = nodes.VideoContinuationConcat.execute(
                body_images=frames,
                body_mask=masks,
                mode=mode,
            ).result[1]
            torch.testing.assert_close(tracked, masks)

    def test_schemas_use_stable_ids_and_no_preview_outputs(self):
        expected = {
            nodes.LoadIndexedVideoSegment: "TuringUtilsLoadIndexedVideoSegment",
            nodes.SaveIndexedVideoSegment: "TuringUtilsSaveIndexedVideoSegment",
            nodes.MergeIndexedVideoSegments: "TuringUtilsMergeIndexedVideoSegments",
            nodes.VideoPrefixContextNoise: "TuringUtilsVideoPrefixContextNoise",
            nodes.VideoContinuationConcat: "TuringUtilsVideoContinuationConcat",
            nodes.TrimVideoContinuationPrefix: "TuringUtilsTrimVideoContinuationPrefix",
            nodes.H3SetAudioPrefixNoiseMask: "TuringUtilsH3SetAudioPrefixNoiseMask",
        }
        for node, node_id in expected.items():
            with self.subTest(node=node.__name__):
                schema = node.define_schema()
                self.assertEqual(schema.node_id, node_id)
                self.assertNotIn("preview", [output.id for output in schema.outputs])

        loader_inputs = {
            input_.id: input_
            for input_ in nodes.LoadIndexedVideoSegment.define_schema().inputs
        }
        self.assertEqual(loader_inputs["segment_index"].min, -1)
        self.assertEqual(loader_inputs["tail_frames"].default, 22)

        merger_inputs = {
            input_.id: input_
            for input_ in nodes.MergeIndexedVideoSegments.define_schema().inputs
        }
        self.assertEqual(merger_inputs["max_index"].default, -1)
        self.assertEqual(merger_inputs["max_index"].min, -1)
        self.assertEqual(merger_inputs["max_index"].max, 999_999)

        concat_schema = nodes.VideoContinuationConcat.define_schema()
        self.assertEqual(
            [input_.id for input_ in concat_schema.inputs[:6]],
            [
                "prefix_images",
                "prefix_mask",
                "prefix_audio",
                "body_images",
                "body_mask",
                "body_audio",
            ],
        )
        concat_inputs = {input_.id: input_ for input_ in concat_schema.inputs}
        self.assertEqual(
            set(concat_inputs),
            {
                "prefix_images",
                "prefix_mask",
                "prefix_audio",
                "body_images",
                "body_mask",
                "body_audio",
                "frame_rate",
                "mode",
            },
        )
        self.assertEqual(concat_inputs["mode"].default, "concat")

        noise_inputs = {
            input_.id: input_
            for input_ in nodes.VideoPrefixContextNoise.define_schema().inputs
        }
        self.assertEqual(noise_inputs["strength"].default, 0.45)
        self.assertEqual(noise_inputs["end_strength"].default, 0.10)
        self.assertEqual(noise_inputs["tail_protection_frames"].default, 5)
        self.assertEqual(noise_inputs["transition_frames"].default, 4)
        self.assertTrue(noise_inputs["images"].optional)
        self.assertNotIn("noise_frames", noise_inputs)
        self.assertFalse(noise_inputs["end_strength"].advanced)
        self.assertFalse(noise_inputs["transition_frames"].advanced)
        self.assertEqual(concat_schema.outputs[-1].display_name, "trim_info")
        self.assertEqual(
            nodes.TrimVideoContinuationPrefix.define_schema().inputs[-1].id, "trim_info"
        )
        self.assertEqual(
            nodes.H3SetAudioPrefixNoiseMask.define_schema().inputs[-1].id, "trim_info"
        )

        concat_schema.finalize()
        concat_v1 = concat_schema.get_v1_info(nodes.VideoContinuationConcat)
        self.assertEqual(
            concat_v1.input_order["optional"],
            [
                "prefix_images",
                "prefix_mask",
                "prefix_audio",
                "body_images",
                "body_mask",
                "body_audio",
            ],
        )
        trim_schema = nodes.TrimVideoContinuationPrefix.define_schema()
        trim_schema.finalize()
        self.assertEqual(
            trim_schema.get_v1_info(nodes.TrimVideoContinuationPrefix).input_order[
                "optional"
            ],
            ["audio", "trim_info"],
        )
        audio_mask_schema = nodes.H3SetAudioPrefixNoiseMask.define_schema()
        audio_mask_schema.finalize()
        self.assertEqual(
            audio_mask_schema.get_v1_info(nodes.H3SetAudioPrefixNoiseMask).input_order[
                "optional"
            ],
            ["trim_info"],
        )

    def test_segment_paths_are_six_digit_and_relative_paths_are_confined_to_output(
        self,
    ):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                segments.folder_paths, "get_output_directory", return_value=directory
            ),
        ):
            path = segments._segment_path("series/a", 324, create=True)
            self.assertEqual(path, Path(directory) / "series" / "a" / "000324.mp4")
            self.assertTrue(path.parent.is_dir())
            with self.assertRaisesRegex(ValueError, "inside"):
                segments._segment_path("../outside", 0)

    def test_segment_paths_accept_absolute_root_directory(self):
        with (
            tempfile.TemporaryDirectory() as output_directory,
            tempfile.TemporaryDirectory() as root_directory,
            mock.patch.object(
                segments.folder_paths,
                "get_output_directory",
                return_value=output_directory,
            ),
        ):
            path = segments._segment_path(root_directory, 324, create=True)
            self.assertEqual(path, Path(root_directory) / "000324.mp4")
            self.assertTrue(path.parent.is_dir())

    def test_missing_segment_returns_empty_without_preview(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                segments.folder_paths, "get_output_directory", return_value=directory
            ),
        ):
            output = nodes.LoadIndexedVideoSegment.execute("segments", 8, 21)
            self.assertEqual(output.result, (None, None, 0.0))
            self.assertIsNone(output.ui)

    def test_loader_zero_returns_empty_without_reading_or_scanning(self):
        with (
            mock.patch.object(nodes.InputImpl, "VideoFromFile") as video_from_file,
            mock.patch.object(
                segments.folder_paths, "get_output_directory"
            ) as get_output_directory,
        ):
            output = nodes.LoadIndexedVideoSegment.execute("ignored", 0, 21)
            fingerprint = nodes.LoadIndexedVideoSegment.fingerprint_inputs(
                "ignored", 0, 21
            )
        self.assertEqual(output.result, (None, None, 0.0))
        self.assertEqual(fingerprint, (0, None, 21))
        video_from_file.assert_not_called()
        get_output_directory.assert_not_called()

    def test_loader_fingerprint_conservatively_reloads_linked_inputs(self):
        self.assertTrue(
            math.isnan(
                nodes.LoadIndexedVideoSegment.fingerprint_inputs(None, None, None, True)
            )
        )

    def test_loader_can_ignore_file_changes_while_debugging(self):
        with mock.patch.object(nodes, "_segment_path_for_load") as segment_path:
            fingerprint = nodes.LoadIndexedVideoSegment.fingerprint_inputs(
                "segments", 7, 22, False
            )
        self.assertEqual(fingerprint, ("reuse_cached",))
        segment_path.assert_not_called()

    def test_loader_positive_index_reads_previous_saved_segment(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                segments.folder_paths, "get_output_directory", return_value=directory
            ),
        ):
            target = Path(directory) / "segments" / "000006.mp4"
            target.parent.mkdir()
            target.write_bytes(b"video")
            self.assertEqual(segments._segment_path_for_load("segments", 7), target)
            self.assertEqual(
                segments._segment_path_for_load("segments", 1).name, "000000.mp4"
            )
            self.assertEqual(
                segments._segment_path_for_load("segments", 1_000_000).name,
                "999999.mp4",
            )

    def test_loader_minus_one_selects_highest_six_digit_segment(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                segments.folder_paths, "get_output_directory", return_value=directory
            ),
        ):
            segment_directory = Path(directory) / "segments"
            segment_directory.mkdir()
            for name in (
                "000002.mp4",
                "000117.mp4",
                "999999.mp4",
                "1000000.mp4",
                "999998.MP4",
                "notes.mp4",
            ):
                (segment_directory / name).write_bytes(b"video")
            self.assertEqual(
                segments._segment_path_for_load("segments", -1),
                segment_directory / "999999.mp4",
            )

    def test_loader_minus_one_returns_empty_when_no_segment_exists(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                segments.folder_paths, "get_output_directory", return_value=directory
            ),
        ):
            output = nodes.LoadIndexedVideoSegment.execute("segments", -1, 21)
            self.assertEqual(output.result, (None, None, 0.0))

    def test_segment_merger_rejects_numbering_gaps(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                segments.folder_paths, "get_output_directory", return_value=directory
            ),
        ):
            segment_directory = Path(directory) / "segments"
            segment_directory.mkdir()
            (segment_directory / "000000.mp4").write_bytes(b"video")
            (segment_directory / "000002.mp4").write_bytes(b"video")
            with self.assertRaisesRegex(ValueError, "000001.mp4"):
                segments._indexed_segment_paths("segments")

    def test_segment_merger_max_index_is_inclusive_and_ignores_later_segments(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                segments.folder_paths, "get_output_directory", return_value=directory
            ),
        ):
            segment_directory = Path(directory) / "segments"
            segment_directory.mkdir()
            for index in range(3):
                (segment_directory / f"{index:06d}.mp4").write_bytes(b"video")

            self.assertEqual(
                [path.name for path in segments._indexed_segment_paths("segments", 0)],
                ["000000.mp4"],
            )
            self.assertEqual(
                [path.name for path in segments._indexed_segment_paths("segments", 1)],
                ["000000.mp4", "000001.mp4"],
            )
            self.assertEqual(
                [path.name for path in segments._indexed_segment_paths("segments", -1)],
                ["000000.mp4", "000001.mp4", "000002.mp4"],
            )
            with self.assertRaisesRegex(ValueError, "000003.mp4"):
                segments._indexed_segment_paths("segments", 3)

    def test_merged_filename_stays_inside_root_and_avoids_segment_namespace(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                segments.folder_paths, "get_output_directory", return_value=directory
            ),
        ):
            self.assertEqual(
                segments._merged_segment_path("segments", "merged.mp4"),
                Path(directory) / "segments" / "merged.mp4",
            )
            with self.assertRaisesRegex(ValueError, "inside root_directory"):
                segments._merged_segment_path("segments", "../merged.mp4")
            with self.assertRaisesRegex(ValueError, "reserved"):
                segments._merged_segment_path("segments", "000123.mp4")

    def test_merger_pads_a_small_legacy_aac_shortfall(self):
        class FakeAudioFrame:
            def to_ndarray(self):
                return np.ones((2, 530), dtype=np.float32)

        class FakeResampler:
            def resample(self, frame):
                return [] if frame is None else [FakeAudioFrame()]

        class FakeContainer:
            streams = SimpleNamespace(audio=[SimpleNamespace(codec_context=object())])

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def decode(self, stream):
                return [object()]

        with (
            mock.patch.object(segments.av, "open", return_value=FakeContainer()),
            mock.patch.object(
                segments.av, "AudioResampler", return_value=FakeResampler()
            ),
            self.assertLogs(level="WARNING"),
        ):
            waveform = segments._decode_segment_audio(
                Path("000000.mp4"),
                sample_rate=48_000,
                layout="stereo",
                channels=2,
                expected_samples=1_000,
            )

        self.assertEqual(waveform.shape, (2, 1_000))
        np.testing.assert_array_equal(
            waveform[:, :530], np.ones((2, 530), dtype=np.float32)
        )
        np.testing.assert_array_equal(
            waveform[:, 530:], np.zeros((2, 470), dtype=np.float32)
        )

    def test_merger_rejects_audio_shortfall_larger_than_one_aac_frame(self):
        class FakeContainer:
            streams = SimpleNamespace(audio=[SimpleNamespace(codec_context=object())])

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def decode(self, stream):
                return []

        class FakeResampler:
            def resample(self, frame):
                return []

        with (
            mock.patch.object(segments.av, "open", return_value=FakeContainer()),
            mock.patch.object(
                segments.av, "AudioResampler", return_value=FakeResampler()
            ),
            self.assertRaisesRegex(ValueError, "1025.*1024"),
        ):
            segments._decode_segment_audio(
                Path("000000.mp4"),
                sample_rate=48_000,
                layout="stereo",
                channels=2,
                expected_samples=1_025,
            )

    def test_loader_keeps_tail_frames_and_matching_audio(self):
        images = torch.arange(10, dtype=torch.float32)[:, None, None, None].expand(
            10, 2, 2, 3
        )
        audio = {
            "waveform": torch.arange(20, dtype=torch.float32).reshape(1, 1, 20),
            "sample_rate": 48,
        }

        class FakeVideo:
            def get_frame_rate(self):
                return Fraction(24, 1)

            def get_frame_count(self):
                return 10

            def as_trimmed(self, start_time, duration, strict_duration):
                self.trim = (start_time, duration, strict_duration)
                return self

            def get_components(self):
                return SimpleNamespace(images=images, audio=audio)

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                segments.folder_paths, "get_output_directory", return_value=directory
            ),
            mock.patch.object(
                nodes.InputImpl, "VideoFromFile", return_value=FakeVideo()
            ),
        ):
            target = Path(directory) / "segments" / "000001.mp4"
            target.parent.mkdir()
            target.write_bytes(b"video")
            output_images, output_audio, frame_rate = (
                nodes.LoadIndexedVideoSegment.execute("segments", 2, 5).result
            )
        torch.testing.assert_close(output_images, images[-5:])
        torch.testing.assert_close(
            output_audio["waveform"], audio["waveform"][..., -10:]
        )
        self.assertEqual(output_audio["sample_rate"], 48)
        self.assertEqual(frame_rate, 24.0)

    def test_saver_is_atomic_honors_overwrite_and_has_no_preview(self):
        class FakeVideo:
            def save_to(self, path, **kwargs):
                Path(path).write_bytes(b"encoded")

        images = torch.zeros(2, 4, 6, 3)
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                segments.folder_paths, "get_output_directory", return_value=directory
            ),
            mock.patch.object(
                nodes.InputImpl, "VideoFromComponents", return_value=FakeVideo()
            ),
        ):
            result = nodes.SaveIndexedVideoSegment.execute(
                images, "segments", 12, 24.0, False
            )
            target = Path(result.result[0])
            self.assertEqual(target.name, "000012.mp4")
            self.assertEqual(target.read_bytes(), b"encoded")
            self.assertFalse(any(target.parent.glob(".*.tmp.mp4")))
            self.assertIsNone(result.ui)
            with self.assertRaises(FileExistsError):
                nodes.SaveIndexedVideoSegment.execute(
                    images, "segments", 12, 24.0, False
                )
            nodes.SaveIndexedVideoSegment.execute(images, "segments", 12, 24.0, True)

    def test_saver_fits_audio_to_the_exact_video_frame_duration(self):
        captured = {}

        class FakeVideo:
            def save_to(self, path, **kwargs):
                Path(path).write_bytes(b"encoded")

        def video_from_components(components, **kwargs):
            captured["components"] = components
            return FakeVideo()

        images = torch.zeros(3, 4, 6, 3)
        audio = {
            "waveform": torch.ones(1, 2, 1_030),
            "sample_rate": 8_000,
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                segments.folder_paths, "get_output_directory", return_value=directory
            ),
            mock.patch.object(
                nodes.InputImpl,
                "VideoFromComponents",
                side_effect=video_from_components,
            ),
        ):
            nodes.SaveIndexedVideoSegment.execute(
                images,
                "segments",
                0,
                16.0,
                False,
                audio=audio,
            )

        waveform = captured["components"].audio["waveform"]
        self.assertEqual(tuple(waveform.shape), (1, 2, 1_500))
        torch.testing.assert_close(waveform[..., :1_030], audio["waveform"])
        self.assertEqual(waveform[..., 1_030:].count_nonzero().item(), 0)

    def test_merger_stream_copies_video_and_encodes_one_continuous_audio_track(self):
        frame_rate = 16.0
        sample_rate = 8000
        frames_per_segment = 3
        samples_per_segment = 1500
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(
                segments.folder_paths, "get_output_directory", return_value=directory
            ),
        ):
            for index, value in enumerate((0.1, 0.2)):
                images = torch.zeros(frames_per_segment, 32, 32, 3)
                images[..., index] = 1.0
                audio = {
                    "waveform": torch.full((1, 2, samples_per_segment), value),
                    "sample_rate": sample_rate,
                }
                nodes.SaveIndexedVideoSegment.execute(
                    images,
                    "segments",
                    index,
                    frame_rate,
                    False,
                    audio=audio,
                )

            segment_directory = Path(directory) / "segments"
            source_audio_samples = 0
            for path in sorted(
                segment_directory.glob("[0-9][0-9][0-9][0-9][0-9][0-9].mp4")
            ):
                source_audio_samples += int(
                    nodes.InputImpl.VideoFromFile(str(path))
                    .get_components()
                    .audio["waveform"]
                    .shape[-1]
                )

            filename, segment_count = nodes.MergeIndexedVideoSegments.execute(
                "segments",
                -1,
                "merged.mp4",
                False,
            ).result
            merged_path = Path(filename)
            merged = nodes.InputImpl.VideoFromFile(str(merged_path))
            components = merged.get_components()
            self.assertEqual(segment_count, 2)
            self.assertEqual(merged_path, segment_directory / "merged.mp4")
            self.assertEqual(merged.get_frame_count(), 6)
            self.assertEqual(merged.get_frame_rate(), Fraction(16, 1))
            self.assertEqual(tuple(components.images.shape), (6, 32, 32, 3))
            self.assertEqual(components.audio["sample_rate"], sample_rate)
            self.assertLess(
                int(components.audio["waveform"].shape[-1]), source_audio_samples
            )
            self.assertFalse(any(segment_directory.glob(".*.tmp.mp4")))
            with self.assertRaises(FileExistsError):
                nodes.MergeIndexedVideoSegments.execute(
                    "segments",
                    -1,
                    "merged.mp4",
                    False,
                )

    def test_concat_builds_default_masks_and_aligned_audio(self):
        prefix_images = torch.full((9, 8, 8, 3), 0.5)
        body_images = torch.full((4, 8, 8, 3), 0.25)
        prefix_audio = {
            "waveform": torch.arange(9, dtype=torch.float32).reshape(1, 1, 9),
            "sample_rate": 24,
        }
        images, mask, audio, info = nodes.VideoContinuationConcat.execute(
            prefix_images=prefix_images,
            prefix_audio=prefix_audio,
            body_images=body_images,
            frame_rate=24.0,
        ).result
        torch.testing.assert_close(images[:9], prefix_images)
        torch.testing.assert_close(images[9:], body_images)
        self.assertEqual(mask.shape, (13, 8, 8))
        self.assertEqual(mask[:9].count_nonzero().item(), 0)
        self.assertTrue(torch.all(mask[9:] == 1))
        self.assertEqual(audio["sample_rate"], 24)
        self.assertEqual(tuple(audio["waveform"].shape), (1, 1, 13))
        torch.testing.assert_close(audio["waveform"][..., :9], prefix_audio["waveform"])
        self.assertEqual(audio["waveform"][..., 9:].count_nonzero().item(), 0)
        self.assertEqual(info["prefix_audio_samples"], 9)
        self.assertEqual(info["total_audio_samples"], 13)
        self.assertTrue(info["prefix_audio_present"])
        self.assertEqual(info["composition_mode"], "concat")
        self.assertEqual(info["trim_frames"], 9)

    def test_replace_mode_overwrites_prefix_without_extending_timeline(self):
        prefix_images = torch.full((2, 4, 4, 3), 0.75)
        body_images = (
            torch.arange(5, dtype=torch.float32)[:, None, None, None].expand(5, 4, 4, 3)
            / 10
        )
        prefix_audio = {
            "waveform": torch.tensor([[[0.7, 0.8]]]),
            "sample_rate": 1,
        }
        body_audio = {
            "waveform": torch.tensor([[[0.0, 0.1, 0.2, 0.3, 0.4]]]),
            "sample_rate": 1,
        }
        images, mask, audio, info = nodes.VideoContinuationConcat.execute(
            prefix_images=prefix_images,
            prefix_audio=prefix_audio,
            body_images=body_images,
            body_audio=body_audio,
            frame_rate=1.0,
            mode="replace",
        ).result
        self.assertEqual(images.shape[0], 5)
        torch.testing.assert_close(images[:2], prefix_images)
        torch.testing.assert_close(images[2:], body_images[2:])
        self.assertEqual(mask[:2].count_nonzero().item(), 0)
        self.assertTrue(torch.all(mask[2:] == 1))
        torch.testing.assert_close(
            audio["waveform"],
            torch.tensor([[[0.7, 0.8, 0.2, 0.3, 0.4]]]),
        )
        self.assertEqual(info["composition_mode"], "replace")
        self.assertEqual(info["prefix_frames"], 2)
        self.assertEqual(info["trim_frames"], 2)
        self.assertEqual(info["total_audio_samples"], 5)

        audio_latent = {"samples": torch.zeros(1, 32, 2, 5)}
        masked_audio_latent = nodes.H3SetAudioPrefixNoiseMask.execute(
            audio_latent, info, "protect_prefix_generate_body"
        ).result[0]
        self.assertEqual(
            masked_audio_latent["noise_mask"][..., :2].count_nonzero().item(), 0
        )
        self.assertTrue(torch.all(masked_audio_latent["noise_mask"][..., 2:] == 1))

        trimmed_images, trimmed_audio = nodes.TrimVideoContinuationPrefix.execute(
            images, info, audio
        ).result
        torch.testing.assert_close(trimmed_images, images[2:])
        torch.testing.assert_close(
            trimmed_audio["waveform"], audio["waveform"][..., 2:]
        )

    def test_replace_mode_rejects_prefix_longer_than_body(self):
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            nodes.VideoContinuationConcat.execute(
                prefix_images=torch.zeros(5, 4, 4, 3),
                body_images=torch.zeros(4, 4, 4, 3),
                mode="replace",
            )

    def test_concat_accepts_lazy_audio_mappings(self):
        class LazyAudio(Mapping):
            def __init__(self, value):
                self.value = value

            def __getitem__(self, key):
                return self.value[key]

            def __iter__(self):
                return iter(self.value)

            def __len__(self):
                return len(self.value)

        body = torch.zeros(4, 8, 8, 3)
        lazy = LazyAudio({"waveform": torch.ones(1, 2, 4), "sample_rate": 24})
        audio = nodes.VideoContinuationConcat.execute(
            body_images=body, frame_rate=24.0, body_audio=lazy
        ).result[2]
        torch.testing.assert_close(audio["waveform"], lazy["waveform"])

    def test_prefix_noise_rejects_invalid_cross_field_values(self):
        images = torch.zeros(22, 8, 8, 3)
        with self.assertRaisesRegex(ValueError, "between 0 and strength"):
            nodes.VideoPrefixContextNoise.execute(
                images, strength=0.2, end_strength=0.3
            )

    def test_prefix_noise_passes_through_none(self):
        self.assertIsNone(nodes.VideoPrefixContextNoise.execute(None).result[0])

    def test_prefix_noise_short_batches_prioritize_the_clean_tail(self):
        images = torch.ones(3, 8, 8, 3)
        zero_grid = lambda pattern, width, height, palette_rng, generator: torch.zeros(
            height, width, 3
        )
        with mock.patch.object(
            prefix_noise, "_coarse_noise_frame", side_effect=zero_grid
        ):
            output = nodes.VideoPrefixContextNoise.execute(
                images,
                tail_protection_frames=5,
                strength=0.45,
                end_strength=0.10,
                transition_frames=4,
            ).result[0]
        self.assertEqual(output.shape, images.shape)
        torch.testing.assert_close(output, images, rtol=0, atol=0)

    def test_prefix_noise_short_batches_shrink_transition_before_protected_tail(self):
        images = torch.ones(7, 8, 8, 3)
        zero_grid = lambda pattern, width, height, palette_rng, generator: torch.zeros(
            height, width, 3
        )
        with mock.patch.object(
            prefix_noise, "_coarse_noise_frame", side_effect=zero_grid
        ):
            output = nodes.VideoPrefixContextNoise.execute(
                images,
                tail_protection_frames=5,
                strength=0.45,
                end_strength=0.10,
                transition_frames=4,
            ).result[0]
        torch.testing.assert_close(output[0], torch.full_like(output[0], 0.725))
        torch.testing.assert_close(output[1], torch.full_like(output[1], 0.90))
        torch.testing.assert_close(output[2:], images[2:], rtol=0, atol=0)

    def test_validated_noise_schedule_is_flat_then_tapers_to_point_one(self):
        schedule = prefix_noise._noise_alpha_schedule(17, 0.45, 0.10, 4)
        self.assertEqual(schedule[:13], [0.45] * 13)
        self.assertAlmostEqual(schedule[13], 0.3625, places=12)
        self.assertAlmostEqual(schedule[14], 0.275, places=12)
        self.assertAlmostEqual(schedule[15], 0.1875, places=12)
        self.assertAlmostEqual(schedule[16], 0.10, places=12)

    def test_prefix_noise_uses_blend_alpha_and_only_changes_initial_frames(self):
        images = torch.ones(22, 8, 8, 3)
        zero_grid = lambda pattern, width, height, palette_rng, generator: torch.zeros(
            height, width, 3
        )
        with mock.patch.object(
            prefix_noise, "_coarse_noise_frame", side_effect=zero_grid
        ):
            output = nodes.VideoPrefixContextNoise.execute(
                images,
                tail_protection_frames=5,
                strength=0.45,
                seed=7,
                end_strength=0.10,
                transition_frames=4,
            ).result[0]
        torch.testing.assert_close(output[:13], torch.full_like(output[:13], 0.55))
        torch.testing.assert_close(output[13], torch.full_like(output[13], 0.6375))
        torch.testing.assert_close(output[14], torch.full_like(output[14], 0.725))
        torch.testing.assert_close(output[15], torch.full_like(output[15], 0.8125))
        torch.testing.assert_close(output[16], torch.full_like(output[16], 0.90))
        torch.testing.assert_close(output[17:], images[17:], rtol=0, atol=0)

    def test_prefix_noise_is_deterministic_and_bounded(self):
        images = torch.full((24, 20, 12, 3), 0.5)
        first = nodes.VideoPrefixContextNoise.execute(images, seed=11).result[0]
        second = nodes.VideoPrefixContextNoise.execute(images, seed=11).result[0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertGreaterEqual(first.amin().item(), 0.0)
        self.assertLessEqual(first.amax().item(), 1.0)
        torch.testing.assert_close(first[19:], images[19:], rtol=0, atol=0)

    def test_prefix_noise_protecting_the_complete_prefix_is_an_exact_no_op(self):
        images = torch.rand(22, 8, 8, 3)
        output = nodes.VideoPrefixContextNoise.execute(
            images, tail_protection_frames=22
        ).result[0]
        torch.testing.assert_close(output, images, rtol=0, atol=0)
        self.assertIsNot(output, images)

    def test_concat_requires_prefix_images_for_prefix_side_data(self):
        body = torch.zeros(3, 8, 8, 3)
        with self.assertRaisesRegex(ValueError, "require prefix_images"):
            nodes.VideoContinuationConcat.execute(
                body_images=body,
                frame_rate=24.0,
                prefix_audio={"waveform": torch.zeros(1, 2, 4), "sample_rate": 24},
            )

    def test_trim_uses_exact_frame_rate_for_images_and_audio(self):
        info = {
            "version": 1,
            "prefix_frames": 5,
            "body_frames": 4,
            "frame_rate_numerator": 24,
            "frame_rate_denominator": 1,
            "prefix_audio_samples": 10,
            "total_audio_samples": 18,
            "audio_sample_rate": 48,
        }
        images = torch.arange(9, dtype=torch.float32)[:, None, None, None].expand(
            9, 2, 2, 3
        )
        waveform = torch.arange(18, dtype=torch.float32).reshape(1, 1, 18)
        output_images, output_audio = nodes.TrimVideoContinuationPrefix.execute(
            images, info, {"waveform": waveform, "sample_rate": 48}
        ).result
        torch.testing.assert_close(output_images, images[5:])
        torch.testing.assert_close(output_audio["waveform"], waveform[..., 10:])

    def test_h3_audio_mask_maps_waveform_boundary_to_latent_time(self):
        info = {
            "version": 1,
            "prefix_frames": 9,
            "body_frames": 4,
            "frame_rate_numerator": 24,
            "frame_rate_denominator": 1,
            "prefix_audio_samples": 9,
            "total_audio_samples": 13,
            "audio_sample_rate": 24,
        }
        samples = torch.randn(1, 32, 2, 13, dtype=torch.float16)
        metadata = object()
        latent = {"samples": samples, "metadata": metadata}
        output = nodes.H3SetAudioPrefixNoiseMask.execute(
            latent, info, "protect_prefix_generate_body"
        ).result[0]
        self.assertIs(output["samples"], samples)
        self.assertIs(output["metadata"], metadata)
        self.assertEqual(output["noise_mask"].dtype, torch.float32)
        self.assertEqual(output["noise_mask"][..., :9].count_nonzero().item(), 0)
        self.assertTrue(torch.all(output["noise_mask"][..., 9:] == 1))
        self.assertNotIn("noise_mask", latent)

        protected = nodes.H3SetAudioPrefixNoiseMask.execute(
            latent, info, "protect_all"
        ).result[0]
        generated = nodes.H3SetAudioPrefixNoiseMask.execute(
            latent, info, "generate_all"
        ).result[0]
        self.assertEqual(protected["noise_mask"].count_nonzero().item(), 0)
        self.assertTrue(torch.all(generated["noise_mask"] == 1))

    def test_h3_audio_mask_regenerates_silent_prefix_when_no_audio_was_supplied(self):
        prefix = torch.zeros(9, 8, 8, 3)
        body = torch.zeros(4, 8, 8, 3)
        _, _, _, info = nodes.VideoContinuationConcat.execute(
            prefix_images=prefix, body_images=body, frame_rate=24.0
        ).result
        latent = {"samples": torch.zeros(1, 32, 2, 13)}
        output = nodes.H3SetAudioPrefixNoiseMask.execute(
            latent, info, "protect_prefix_generate_body"
        ).result[0]
        self.assertFalse(info["prefix_audio_present"])
        self.assertTrue(torch.all(output["noise_mask"] == 1))


if __name__ == "__main__":
    unittest.main()
