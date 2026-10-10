"""Indexed video segments and continuation-timeline helpers."""

from __future__ import annotations

import os
import uuid
from fractions import Fraction

from comfy_api.latest import InputImpl, Types, io

VideoTrimInfo = io.Custom("TURING_UTILS_VIDEO_TRIM_INFO")
from ..media.segments import (
    _segment_path,
    _segment_path_for_load,
    _merge_indexed_segments,
)

from ..media.timeline import (
    _frame_rate,
    _ceil_fraction,
    _validate_images,
    _validate_audio,
    _fit_waveform,
    concat_video_continuation,
    _trim_info_values,
)

from ..adapters.minimax.latent_noise import set_audio_prefix_noise_mask

from ..media.prefix_noise import (
    _add_prefix_chroma_blocks,
)


class LoadIndexedVideoSegment(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsLoadIndexedVideoSegment",
            display_name="Load Indexed Video Segment",
            category="Turing Utils/Video",
            description=(
                "Load the segment before the requested continuation index from root_directory. Relative roots "
                "are resolved below the current ComfyUI output directory; absolute roots are used directly. "
                "Index 0 returns empty, i loads i-1, and -1 loads the highest six-digit MP4. "
                "Optionally decode only its final frames and matching audio. A missing file returns empty outputs."
            ),
            inputs=[
                io.String.Input("root_directory", default="video/segments"),
                io.Int.Input("segment_index", default=0, min=-1, max=1_000_000, step=1),
                io.Int.Input(
                    "tail_frames",
                    default=22,
                    min=0,
                    max=16384,
                    step=1,
                    tooltip="Final frames to load; 0 loads the complete segment.",
                ),
                io.Boolean.Input(
                    "reload_on_change",
                    default=True,
                    tooltip=(
                        "Reload when the selected file's timestamp or size changes. Disable while debugging "
                        "to reuse the cached decode until another graph input changes."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
                io.Float.Output(display_name="frame_rate"),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        root_directory=None,
        segment_index=None,
        tail_frames=None,
        reload_on_change=True,
    ):
        # ComfyUI intentionally does not resolve linked inputs while evaluating a
        # node fingerprint. In automatic mode we must conservatively reload because
        # the selected file cannot be stat'ed here. Cached mode explicitly opts out
        # of that external-file check and lets the normal graph signature decide.
        if reload_on_change is False:
            return ("reuse_cached",)
        if root_directory is None or segment_index is None or tail_frames is None:
            return float("NaN")

        path = _segment_path_for_load(root_directory, segment_index)
        if path is None:
            return (int(segment_index), None, int(tail_frames))
        if not path.is_file():
            return (int(segment_index), str(path), None, int(tail_frames))
        stat = path.stat()
        return (
            int(segment_index),
            str(path),
            stat.st_mtime_ns,
            stat.st_size,
            int(tail_frames),
        )

    @classmethod
    def execute(
        cls,
        root_directory: str,
        segment_index: int,
        tail_frames: int,
        reload_on_change: bool = True,
    ) -> io.NodeOutput:
        path = _segment_path_for_load(root_directory, segment_index)
        if path is None or not path.is_file():
            return io.NodeOutput(None, None, 0.0)

        source = InputImpl.VideoFromFile(str(path))
        frame_rate = Fraction(source.get_frame_rate())
        requested = int(tail_frames)
        if requested > 0:
            frame_count = int(source.get_frame_count())
            if frame_count > requested:
                start_time = Fraction(frame_count - requested, 1) / frame_rate
                duration = Fraction(requested, 1) / frame_rate
                source = source.as_trimmed(
                    float(start_time), float(duration), strict_duration=False
                )
        components = source.get_components()
        images = components.images
        if int(images.shape[0]) < 1:
            raise ValueError(f"Indexed video contains no decodable frames: {path}")

        dropped_leading_frames = requested > 0 and int(images.shape[0]) > requested
        if dropped_leading_frames:
            images = images[-requested:]
        audio = components.audio
        if audio is not None:
            waveform, sample_rate = _validate_audio(audio, "decoded audio")
            expected = _ceil_fraction(
                Fraction(int(images.shape[0]) * sample_rate, 1) / frame_rate
            )
            if dropped_leading_frames:
                waveform = waveform[..., -expected:]
            waveform = _fit_waveform(waveform, expected)
            audio = {"waveform": waveform, "sample_rate": sample_rate}
        return io.NodeOutput(images, audio, float(frame_rate))


class SaveIndexedVideoSegment(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsSaveIndexedVideoSegment",
            display_name="Save Indexed Video Segment",
            category="Turing Utils/Video",
            description=(
                "Atomically save IMAGE frames and optional AUDIO as NNNNNN.mp4 in root_directory. Relative "
                "roots are resolved below the current ComfyUI output directory; absolute roots are used "
                "directly. Audio is trimmed or zero-padded to the exact video-frame duration before encoding. "
                "This node deliberately creates no preview."
            ),
            is_output_node=True,
            inputs=[
                io.Image.Input("images"),
                io.String.Input("root_directory", default="video/segments"),
                io.Int.Input("segment_index", default=0, min=0, max=999999, step=1),
                io.Float.Input(
                    "frame_rate", default=24.0, min=0.01, max=1000.0, step=0.01
                ),
                io.Boolean.Input("overwrite", default=False),
                io.Float.Input("crf", default=19.0, min=0.0, max=51.0, step=1.0),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[io.String.Output(display_name="filename")],
        )

    @classmethod
    def execute(
        cls,
        images,
        root_directory,
        segment_index,
        frame_rate,
        overwrite,
        crf=19.0,
        audio=None,
    ) -> io.NodeOutput:
        images = _validate_images(images, "images")
        rate = _frame_rate(frame_rate)
        target = _segment_path(root_directory, segment_index, create=True)
        if target.exists() and not overwrite:
            raise FileExistsError(f"Indexed video already exists: {target}")
        if audio is not None:
            waveform, sample_rate = _validate_audio(audio, "audio")
            expected_samples = _ceil_fraction(
                Fraction(int(images.shape[0]) * sample_rate, 1) / rate
            )
            waveform = _fit_waveform(waveform, expected_samples)
            audio = {"waveform": waveform, "sample_rate": sample_rate}

        temporary = target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp.mp4")
        video = InputImpl.VideoFromComponents(
            Types.VideoComponents(images=images, audio=audio, frame_rate=rate),
            bit_depth=8,
            color_space="sRGB",
        )
        try:
            video.save_to(
                str(temporary),
                format=Types.VideoContainer.MP4,
                codec=Types.VideoCodec.H264,
                crf=float(crf),
            )
            if target.exists() and not overwrite:
                raise FileExistsError(f"Indexed video already exists: {target}")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return io.NodeOutput(str(target))


class MergeIndexedVideoSegments(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsMergeIndexedVideoSegments",
            display_name="Merge Indexed Video Segments",
            category="Turing Utils/Video",
            description=(
                "Merge contiguous NNNNNN.mp4 segments from index 0 through max_index into one MP4 without "
                "decoding or re-encoding the video stream. A max_index of -1 selects every segment through "
                "the highest existing index. Audio is decoded per segment, fitted to exact cumulative "
                "video-frame boundaries, and encoded once as a continuous track so per-segment AAC padding "
                "and rounding errors cannot accumulate. Segments are joined exactly as stored, so continuation "
                "context must be trimmed before each segment is saved. This output node creates no preview."
            ),
            is_output_node=True,
            inputs=[
                io.String.Input("root_directory", default="video/segments"),
                io.Int.Input(
                    "max_index",
                    default=-1,
                    min=-1,
                    max=999_999,
                    step=1,
                    tooltip=(
                        "Last segment index to include, inclusive. -1 merges every segment; "
                        "0 merges only 000000.mp4."
                    ),
                ),
                io.String.Input("output_filename", default="merged.mp4"),
                io.Boolean.Input("overwrite", default=False),
                io.Int.Input(
                    "audio_bitrate_kbps",
                    default=192,
                    min=32,
                    max=512,
                    step=1,
                ),
            ],
            outputs=[
                io.String.Output(display_name="filename"),
                io.Int.Output(display_name="segment_count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        root_directory: str,
        max_index: int,
        output_filename: str,
        overwrite: bool,
        audio_bitrate_kbps: int = 192,
    ) -> io.NodeOutput:
        target, segment_count = _merge_indexed_segments(
            root_directory,
            int(max_index),
            output_filename,
            bool(overwrite),
            int(audio_bitrate_kbps),
        )
        return io.NodeOutput(str(target), segment_count)


class VideoPrefixContextNoise(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsVideoPrefixContextNoise",
            display_name="Video Prefix Context Noise",
            category="Turing Utils/Video",
            description=(
                "Treat the complete IMAGE batch as a video prefix. Preserve a clean tail first, place a "
                "noise transition immediately before it, and apply full coarse colour-block noise to every "
                "earlier frame. Short batches prioritize the protected tail, then the transition, and only "
                "use full noise when frames remain. A missing IMAGE passes through as absent."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    optional=True,
                    tooltip="A missing IMAGE is passed through as None.",
                ),
                io.Int.Input(
                    "tail_protection_frames",
                    default=5,
                    min=0,
                    max=16384,
                    step=1,
                    tooltip="Clean frames reserved at the end of the complete prefix before any noise is allocated.",
                ),
                io.Float.Input(
                    "strength",
                    default=0.45,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Flat blend alpha for the validated H3 context-noise recipe; this is not Gaussian sigma.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Float.Input(
                    "end_strength", default=0.10, min=0.0, max=1.0, step=0.01
                ),
                io.Int.Input(
                    "transition_frames",
                    default=4,
                    min=0,
                    max=4096,
                    step=1,
                    tooltip="Frames immediately before the clean tail that taper from full to end strength.",
                ),
                io.Combo.Input(
                    "pattern",
                    options=["poc_chroma_blocks", "gaussian_rgb", "uniform_rgb"],
                    default="poc_chroma_blocks",
                ),
                io.DynamicCombo.Input(
                    "grid_mode",
                    options=[
                        io.DynamicCombo.Option("poc_36x64", []),
                        io.DynamicCombo.Option(
                            "block_size",
                            [
                                io.Int.Input(
                                    "block_size",
                                    default=16,
                                    min=1,
                                    max=256,
                                    step=1,
                                    optional=True,
                                ),
                            ],
                        ),
                    ],
                    extra_dict={"advanced": False},
                ),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(
        cls,
        images=None,
        tail_protection_frames=5,
        strength=0.45,
        seed=0,
        end_strength=0.10,
        transition_frames=4,
        pattern="poc_chroma_blocks",
        grid_mode="poc_36x64",
    ) -> io.NodeOutput:
        block_size = 16
        if isinstance(grid_mode, dict):
            block_size = grid_mode.get("block_size", 16)
            grid_mode = grid_mode["grid_mode"]
        if block_size is None:
            block_size = 16
        if images is None:
            return io.NodeOutput(None)
        return io.NodeOutput(
            _add_prefix_chroma_blocks(
                _validate_images(images, "images"),
                float(strength),
                int(seed),
                end_strength=float(end_strength),
                transition_frames=int(transition_frames),
                tail_protection_frames=int(tail_protection_frames),
                pattern=str(pattern),
                grid_mode=str(grid_mode),
                block_size=int(block_size),
            )
        )


class VideoContinuationConcat(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsVideoContinuationConcat",
            display_name="Video Continuation Concat",
            category="Turing Utils/Video",
            description=(
                "Compose optional prefix frames/audio with a body timeline. Concat mode prepends the prefix; "
                "replace mode fills leading context slots already included in the body timeline without "
                "increasing its duration. In both modes the prefix remains temporary context and is marked "
                "for trimming after generation. Missing prefix masks preserve the prefix, and missing body "
                "masks redraw the body."
            ),
            inputs=[
                io.Image.Input("prefix_images", optional=True),
                io.Mask.Input(
                    "prefix_mask",
                    optional=True,
                    tooltip="Prefix redraw mask, not an audio mask. One mask repeats over every prefix frame; a batch must match prefix_images frame count. This node does not track motion.",
                ),
                io.Audio.Input(
                    "prefix_audio",
                    optional=True,
                    tooltip="Optional prefix waveform content. Audio preservation is controlled later by the H3 audio latent noise mask.",
                ),
                io.Image.Input(
                    "body_images",
                    optional=True,
                    tooltip="Required generated or source body frames.",
                ),
                io.Mask.Input(
                    "body_mask",
                    optional=True,
                    tooltip="Body redraw mask, not an audio mask. One mask repeats over every body frame; a batch must match body_images frame count. Use per-frame tracked masks for moving objects.",
                ),
                io.Audio.Input(
                    "body_audio",
                    optional=True,
                    tooltip="Optional body waveform content. Leave empty when H3 should generate the body audio.",
                ),
                io.Float.Input(
                    "frame_rate", default=24.0, min=0.01, max=1000.0, step=0.01
                ),
                io.Combo.Input(
                    "mode",
                    options=["concat", "replace"],
                    default="concat",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Mask.Output(display_name="mask"),
                io.Audio.Output(display_name="audio"),
                VideoTrimInfo.Output(display_name="trim_info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        prefix_images=None,
        prefix_mask=None,
        prefix_audio=None,
        body_images=None,
        body_mask=None,
        body_audio=None,
        frame_rate=24.0,
        mode="concat",
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *concat_video_continuation(
                prefix_images,
                prefix_mask,
                prefix_audio,
                body_images,
                body_mask,
                body_audio,
                frame_rate,
                mode,
            )
        )


class TrimVideoContinuationPrefix(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsTrimVideoContinuationPrefix",
            display_name="Trim Video Continuation Prefix",
            category="Turing Utils/Video",
            description=(
                "Remove temporary continuation context from generated IMAGE frames and matching AUDIO. This "
                "trims prefixes prepended in concat mode as well as leading context slots filled in replace "
                "mode. No preview is created."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Audio.Input("audio", optional=True),
                VideoTrimInfo.Input("trim_info", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
            ],
        )

    @classmethod
    def execute(cls, images, trim_info=None, audio=None) -> io.NodeOutput:
        images = _validate_images(images, "images")
        _, trim_frames, rate = _trim_info_values(trim_info)
        if trim_frames > int(images.shape[0]):
            raise ValueError(
                f"trim_info requests {trim_frames} trim frames but images contains {int(images.shape[0])}"
            )
        images = images[trim_frames:]
        if audio is not None:
            waveform, sample_rate = _validate_audio(audio, "audio")
            trim_samples = _ceil_fraction(Fraction(trim_frames * sample_rate, 1) / rate)
            waveform = waveform[..., min(trim_samples, int(waveform.shape[-1])) :]
            audio = {"waveform": waveform, "sample_rate": sample_rate}
        return io.NodeOutput(images, audio)


class H3SetAudioPrefixNoiseMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3SetAudioPrefixNoiseMask",
            display_name="H3 Set Audio Prefix Noise Mask",
            category="Turing Utils/Video",
            description=(
                "Set a standalone H3 audio latent noise mask from Video Continuation Concat metadata. "
                "Only prefix protection requires trim_info. Whole-audio protection/generation needs no metadata. "
                "Zero preserves audio and one generates it; concatenate with the video latent afterward."
            ),
            inputs=[
                io.Latent.Input("audio_latent"),
                io.Combo.Input(
                    "mode",
                    options=[
                        "protect_prefix_generate_body",
                        "protect_all",
                        "generate_all",
                    ],
                    default="protect_prefix_generate_body",
                ),
                VideoTrimInfo.Input("trim_info", optional=True),
            ],
            outputs=[io.Latent.Output(display_name="audio_latent")],
        )

    @classmethod
    def execute(cls, audio_latent, trim_info=None, mode="protect_prefix_generate_body"):
        return io.NodeOutput(set_audio_prefix_noise_mask(audio_latent, trim_info, mode))
