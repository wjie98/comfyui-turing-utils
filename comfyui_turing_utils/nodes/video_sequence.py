"""Indexed video segments and continuation-timeline helpers."""

from __future__ import annotations

import logging
import math
import os
import random
import uuid
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

import comfy.utils
import folder_paths
from comfy_api.latest import InputImpl, Types, io


VideoTrimInfo = io.Custom("TURING_UTILS_VIDEO_TRIM_INFO")

_DEFAULT_AUDIO_SAMPLE_RATE = 32000
_AAC_FRAME_SAMPLES = 1024
_CHROMA_BLOCK_SIZE = 16
_CHROMA_POC_GRID = (36, 64)
# Empirical MIT-licensed recipe documented by MacroSony and packaged for
# ComfyUI by beijinren/ComfyUI-H3-Context-Noise.
_CHROMA_PALETTE = (
    (185, 115, 215),
    (115, 195, 140),
    (150, 148, 162),
    (205, 150, 192),
    (138, 182, 148),
    (160, 120, 175),
)


def _frame_rate(value: float) -> Fraction:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("frame_rate must be finite and positive")
    return Fraction(str(value)).limit_denominator(1_000_000)


def _ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def _segment_directory(root_directory: str, *, create: bool = False) -> Path:
    output_root = Path(folder_paths.get_output_directory()).resolve()
    configured = Path(str(root_directory).strip().replace("\\", "/"))
    if configured.is_absolute():
        directory = configured.resolve()
    else:
        directory = (output_root / configured).resolve()
        try:
            directory.relative_to(output_root)
        except ValueError as error:
            raise ValueError(
                "relative root_directory must stay inside the ComfyUI output directory"
            ) from error
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def _segment_path(root_directory: str, segment_index: int, *, create: bool = False) -> Path:
    segment_index = int(segment_index)
    if not 0 <= segment_index <= 999999:
        raise ValueError("segment_index must be between 0 and 999999")

    directory = _segment_directory(root_directory, create=create)
    return directory / f"{segment_index:06d}.mp4"


def _segment_path_for_read(root_directory: str, segment_index: int) -> Path:
    path = _segment_path(root_directory, segment_index)
    directory = path.parent
    resolved = path.resolve()
    try:
        resolved.relative_to(directory)
    except ValueError as error:
        raise ValueError("Indexed video path must stay inside root_directory") from error
    return resolved


def _latest_segment_path(root_directory: str) -> Path | None:
    directory = _segment_directory(root_directory)
    if not directory.is_dir():
        return None

    latest_index = None
    for candidate in directory.iterdir():
        name = candidate.name
        if (
            len(name) == 10
            and name[:6].isascii()
            and name[:6].isdigit()
            and name[6:] == ".mp4"
            and not candidate.is_symlink()
            and candidate.is_file()
        ):
            index = int(name[:6])
            latest_index = index if latest_index is None else max(latest_index, index)
    if latest_index is None:
        return None
    return _segment_path_for_read(root_directory, latest_index)


def _segment_path_for_load(root_directory: str, segment_index: int) -> Path | None:
    segment_index = int(segment_index)
    if segment_index == 0:
        return None
    if segment_index == -1:
        return _latest_segment_path(root_directory)
    if not 1 <= segment_index <= 1_000_000:
        raise ValueError("load segment_index must be -1 or between 0 and 1000000")
    return _segment_path_for_read(root_directory, segment_index - 1)


def _indexed_segment_paths(root_directory: str, max_index: int = -1) -> list[Path]:
    max_index = int(max_index)
    if not -1 <= max_index <= 999_999:
        raise ValueError("max_index must be -1 or between 0 and 999999")

    directory = _segment_directory(root_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Indexed segment directory does not exist: {directory}")

    indexed = {}
    for candidate in directory.iterdir():
        name = candidate.name
        if (
            len(name) == 10
            and name[:6].isascii()
            and name[:6].isdigit()
            and name[6:] == ".mp4"
            and not candidate.is_symlink()
            and candidate.is_file()
        ):
            indexed[int(name[:6])] = candidate.resolve()
    if not indexed:
        raise FileNotFoundError(f"No six-digit MP4 segments found in: {directory}")

    final_index = max(indexed) if max_index == -1 else max_index
    missing = [index for index in range(final_index + 1) if index not in indexed]
    if missing:
        preview = ", ".join(f"{index:06d}.mp4" for index in missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(f"Indexed segments must be contiguous from 000000.mp4; missing {preview}{suffix}")
    return [indexed[index] for index in range(final_index + 1)]


def _merged_segment_path(root_directory: str, output_filename: str) -> Path:
    directory = _segment_directory(root_directory, create=True)
    normalized = str(output_filename).strip().replace("\\", "/")
    relative = Path(normalized)
    if not normalized or relative.is_absolute() or len(relative.parts) != 1:
        raise ValueError("output_filename must be a filename inside root_directory")
    if relative.suffix.lower() != ".mp4":
        raise ValueError("output_filename must end with .mp4")
    if len(relative.name) == 10 and relative.name[:6].isascii() and relative.name[:6].isdigit():
        raise ValueError("output_filename must not use the reserved NNNNNN.mp4 segment format")
    return directory / relative.name


def _probe_indexed_segment(path: Path) -> dict:
    source = InputImpl.VideoFromFile(str(path))
    frame_count = int(source.get_frame_count())
    frame_rate = Fraction(source.get_frame_rate())
    if frame_count < 1 or frame_rate <= 0:
        raise ValueError(f"Indexed segment has invalid video timing: {path}")

    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"Indexed segment has no video stream: {path}")
        video = container.streams.video[0]
        audio = next(
            (stream for stream in container.streams.audio if stream.codec_context is not None),
            None,
        )
        video_format = video.format.name if video.format is not None else None
        color_metadata = tuple(
            None if value is None else int(value)
            for value in (
                video.color_primaries,
                video.color_trc,
                video.colorspace,
                video.color_range,
            )
        )
        video_signature = (
            video.codec.canonical_name,
            int(video.width),
            int(video.height),
            video_format,
            bytes(video.codec_context.extradata or b""),
            *color_metadata,
        )
        if audio is None:
            audio_info = None
        else:
            sample_rate = int(audio.codec_context.sample_rate or 0)
            layout = audio.codec_context.layout
            if sample_rate <= 0 or layout is None or layout.nb_channels < 1:
                raise ValueError(f"Indexed segment has invalid audio metadata: {path}")
            audio_info = {
                "sample_rate": sample_rate,
                "layout": layout.name,
                "channels": int(layout.nb_channels),
            }
    return {
        "path": path,
        "frame_count": frame_count,
        "frame_rate": frame_rate,
        "video_signature": video_signature,
        "audio": audio_info,
    }


def _decode_segment_audio(
    path: Path,
    sample_rate: int,
    layout: str,
    channels: int,
    expected_samples: int,
) -> np.ndarray:
    chunks = []
    with av.open(str(path), mode="r") as container:
        stream = next(
            (candidate for candidate in container.streams.audio if candidate.codec_context is not None),
            None,
        )
        if stream is None:
            raise ValueError(f"Indexed segment has no decodable audio stream: {path}")
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
        for frame in container.decode(stream):
            for output in resampler.resample(frame):
                chunks.append(output.to_ndarray())
        for output in resampler.resample(None):
            chunks.append(output.to_ndarray())

    waveform = (
        np.concatenate(chunks, axis=1)
        if chunks
        else np.zeros((channels, 0), dtype=np.float32)
    )
    if waveform.shape[0] != channels:
        raise ValueError(
            f"Decoded audio from {path} has {waveform.shape[0]} channels after resampling; expected {channels}"
        )
    waveform = waveform.astype(np.float32, copy=False)
    current_samples = int(waveform.shape[1])
    if current_samples < expected_samples:
        shortfall = expected_samples - current_samples
        if shortfall > _AAC_FRAME_SAMPLES:
            raise ValueError(
                f"Decoded audio from {path} is {shortfall} samples shorter than its exact video-frame "
                f"duration, exceeding the automatic AAC padding limit of {_AAC_FRAME_SAMPLES} samples"
            )
        logging.warning(
            "Decoded audio from %s is %d samples shorter than its exact video-frame duration; "
            "padding the legacy segment with silence",
            path,
            shortfall,
        )
        waveform = np.pad(waveform, ((0, 0), (0, shortfall)), mode="constant")
    return np.ascontiguousarray(waveform[:, :expected_samples])


def _encode_audio_chunk(output, stream, waveform: np.ndarray, sample_rate: int, start_sample: int) -> None:
    chunk_samples = 65536
    layout = stream.codec_context.layout.name
    for offset in range(0, int(waveform.shape[1]), chunk_samples):
        chunk = np.ascontiguousarray(waveform[:, offset:offset + chunk_samples])
        frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout=layout)
        frame.sample_rate = sample_rate
        frame.pts = start_sample + offset
        frame.time_base = Fraction(1, sample_rate)
        for packet in stream.encode(frame):
            output.mux(packet)


def _merge_indexed_segments(
    root_directory: str,
    max_index: int,
    output_filename: str,
    overwrite: bool,
    audio_bitrate_kbps: int,
) -> tuple[Path, int]:
    paths = _indexed_segment_paths(root_directory, max_index)
    target = _merged_segment_path(root_directory, output_filename)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Merged video already exists: {target}")

    segments = [_probe_indexed_segment(path) for path in paths]
    frame_rate = segments[0]["frame_rate"]
    video_signature = segments[0]["video_signature"]
    for segment in segments[1:]:
        if segment["frame_rate"] != frame_rate:
            raise ValueError(
                f"Indexed segment frame rates differ: {frame_rate} and {segment['frame_rate']} "
                f"in {segment['path']}"
            )
        if segment["video_signature"] != video_signature:
            raise ValueError(
                f"Indexed segment video codecs, dimensions, pixel formats, or colour metadata differ: "
                f"{segment['path']}"
            )

    audio_presence = [segment["audio"] is not None for segment in segments]
    if any(audio_presence) and not all(audio_presence):
        raise ValueError("Either every indexed segment must contain audio or none of them may contain audio")
    audio_info = segments[0]["audio"] if all(audio_presence) else None

    temporary = target.with_name(f".{target.stem}.{uuid.uuid4().hex}.tmp.mp4")
    progress = comfy.utils.ProgressBar(len(segments))
    first_container = None
    try:
        first_container = av.open(str(paths[0]), mode="r")
        first_video = first_container.streams.video[0]
        with av.open(
            str(temporary),
            mode="w",
            format="mp4",
            options={"movflags": "use_metadata_tags+faststart"},
        ) as output:
            output_video = output.add_stream_from_template(first_video)
            output_audio = None
            if audio_info is not None:
                output_audio = output.add_stream(
                    "aac",
                    rate=audio_info["sample_rate"],
                    layout=audio_info["layout"],
                )
                output_audio.bit_rate = int(audio_bitrate_kbps) * 1000

            cumulative_frames = 0
            last_video_dts = None
            for segment in segments:
                start_frame = cumulative_frames
                end_frame = start_frame + segment["frame_count"]
                with av.open(str(segment["path"]), mode="r") as source:
                    source_video = source.streams.video[0]
                    for packet in source.demux(source_video):
                        if packet.dts is None:
                            continue
                        time_base = Fraction(packet.time_base)
                        offset_ticks = Fraction(start_frame, 1) / frame_rate / time_base
                        if offset_ticks.denominator != 1:
                            raise ValueError(
                                f"Video time base in {segment['path']} cannot represent its exact frame boundary"
                            )
                        offset_ticks = offset_ticks.numerator
                        if packet.pts is not None:
                            packet.pts += offset_ticks
                        packet.dts += offset_ticks
                        absolute_dts = Fraction(packet.dts) * time_base
                        if last_video_dts is not None and absolute_dts <= last_video_dts:
                            raise ValueError(
                                f"Video timestamps are not strictly increasing at {segment['path']}"
                            )
                        last_video_dts = absolute_dts
                        packet.stream = output_video
                        packet.time_base = time_base
                        output.mux(packet)

                if output_audio is not None:
                    sample_rate = audio_info["sample_rate"]
                    start_sample = _ceil_fraction(
                        Fraction(start_frame * sample_rate, 1) / frame_rate
                    )
                    end_sample = _ceil_fraction(
                        Fraction(end_frame * sample_rate, 1) / frame_rate
                    )
                    waveform = _decode_segment_audio(
                        segment["path"],
                        sample_rate,
                        audio_info["layout"],
                        audio_info["channels"],
                        end_sample - start_sample,
                    )
                    _encode_audio_chunk(
                        output,
                        output_audio,
                        waveform,
                        sample_rate,
                        start_sample,
                    )

                cumulative_frames = end_frame
                progress.update(1)

            if output_audio is not None:
                for packet in output_audio.encode(None):
                    output.mux(packet)
        if target.exists() and not overwrite:
            raise FileExistsError(f"Merged video already exists: {target}")
        os.replace(temporary, target)
    finally:
        if first_container is not None:
            first_container.close()
        if temporary.exists():
            temporary.unlink()
    return target, len(segments)


def _validate_images(images, name: str) -> torch.Tensor:
    if not torch.is_tensor(images) or images.ndim != 4 or int(images.shape[-1]) < 3:
        shape = tuple(images.shape) if hasattr(images, "shape") else type(images).__name__
        raise ValueError(f"{name} must be IMAGE frames [frames,height,width,channels], got {shape}")
    if any(int(size) < 1 for size in images.shape[:3]):
        raise ValueError(f"{name} must contain at least one non-empty frame")
    return images[..., :3]


def _validate_audio(audio, name: str):
    if not isinstance(audio, Mapping) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError(f"{name} must be an AUDIO dictionary")
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not torch.is_tensor(waveform) or waveform.ndim != 3:
        shape = tuple(waveform.shape) if hasattr(waveform, "shape") else type(waveform).__name__
        raise ValueError(f"{name} waveform must be [batch,channels,samples], got {shape}")
    if int(waveform.shape[0]) != 1 or int(waveform.shape[1]) < 1:
        raise ValueError(f"{name} must contain one batch and at least one channel")
    if sample_rate <= 0:
        raise ValueError(f"{name} sample_rate must be positive")
    if not waveform.dtype.is_floating_point:
        waveform = waveform.float()
    return waveform, sample_rate


def _fit_waveform(waveform: torch.Tensor, length: int) -> torch.Tensor:
    current = int(waveform.shape[-1])
    if current >= length:
        return waveform[..., :length]
    padding = torch.zeros(
        (*waveform.shape[:-1], length - current),
        device=waveform.device,
        dtype=waveform.dtype,
    )
    return torch.cat((waveform, padding), dim=-1)


def _resample_waveform(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    return torchaudio.functional.resample(waveform, source_rate, target_rate)


def _match_audio_channels(waveform: torch.Tensor, channels: int, name: str) -> torch.Tensor:
    current = int(waveform.shape[1])
    if current == channels:
        return waveform
    if current == 1:
        return waveform.expand(-1, channels, -1)
    raise ValueError(f"{name} has {current} channels and cannot be matched to {channels}")


def _audio_timeline(
    prefix_audio,
    body_audio,
    prefix_frames: int,
    body_frames: int,
    frame_rate: Fraction,
    mode: str,
):
    prefix = _validate_audio(prefix_audio, "prefix_audio") if prefix_audio is not None else None
    body = _validate_audio(body_audio, "body_audio") if body_audio is not None else None
    sample_rate = prefix[1] if prefix is not None else body[1] if body is not None else _DEFAULT_AUDIO_SAMPLE_RATE
    prefix_length = _ceil_fraction(Fraction(prefix_frames * sample_rate, 1) / frame_rate)
    body_length = _ceil_fraction(Fraction(body_frames * sample_rate, 1) / frame_rate)

    present = [item[0] for item in (prefix, body) if item is not None]
    if present:
        reference = present[0]
        channels = max(int(waveform.shape[1]) for waveform in present)
        batch = int(reference.shape[0])
        device = reference.device
        dtype = reference.dtype
    else:
        channels = 2
        batch = 1
        device = torch.device("cpu")
        dtype = torch.float32

    def prepare(item, length: int, name: str):
        if item is None:
            return torch.zeros((batch, channels, length), device=device, dtype=dtype)
        waveform, source_rate = item
        waveform = _resample_waveform(waveform, source_rate, sample_rate)
        waveform = _match_audio_channels(waveform, channels, name).to(device=device, dtype=dtype)
        return _fit_waveform(waveform, length)

    body_waveform = prepare(body, body_length, "body_audio")
    if mode == "concat":
        prefix_waveform = prepare(prefix, prefix_length, "prefix_audio")
        waveform = torch.cat((prefix_waveform, body_waveform), dim=-1)
    elif mode == "replace":
        waveform = body_waveform.clone()
        if prefix is not None and prefix_length > 0:
            prefix_waveform = prepare(prefix, prefix_length, "prefix_audio")
            waveform[..., :prefix_length] = prefix_waveform
    else:
        raise ValueError(f"Unknown continuation mode: {mode!r}")
    return {"waveform": waveform, "sample_rate": sample_rate}, prefix_length


def _prepare_mask(mask, frames: int, height: int, width: int, default: float, device) -> torch.Tensor:
    if mask is None:
        return torch.full((frames, height, width), default, device=device, dtype=torch.float32)
    if not torch.is_tensor(mask) or mask.ndim not in (2, 3):
        shape = tuple(mask.shape) if hasattr(mask, "shape") else type(mask).__name__
        raise ValueError(f"MASK must be [height,width] or [frames,height,width], got {shape}")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if int(mask.shape[0]) == 1 and frames != 1:
        mask = mask.expand(frames, -1, -1)
    elif int(mask.shape[0]) != frames:
        raise ValueError(f"MASK contains {int(mask.shape[0])} frames, expected 1 or {frames}")
    if not mask.dtype.is_floating_point and mask.dtype != torch.bool:
        mask = mask.float()
    if not torch.isfinite(mask).all() or mask.amin() < 0 or mask.amax() > 1:
        raise ValueError("MASK values must be finite and within [0,1]")
    mask = mask.to(device=device, dtype=torch.float32)
    if tuple(mask.shape[-2:]) != (height, width):
        mask = F.interpolate(mask.unsqueeze(1), size=(height, width), mode="nearest").squeeze(1)
    return mask.clone()


def _noise_alpha_schedule(frame_count: int, strength: float, end_strength: float, transition_frames: int) -> list[float]:
    frame_count = int(frame_count)
    strength = float(strength)
    end_strength = float(end_strength)
    transition_frames = int(transition_frames)
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if not 0.0 <= end_strength <= strength <= 1.0:
        raise ValueError("end_strength must be between 0 and strength")
    if transition_frames < 0:
        raise ValueError("transition_frames must not be negative")
    transition_frames = min(transition_frames, frame_count)
    if transition_frames == 0:
        return [strength] * frame_count
    result = []
    for position in range(frame_count):
        from_end = frame_count - 1 - position
        if from_end >= transition_frames:
            result.append(strength)
        else:
            result.append(
                strength
                + (end_strength - strength)
                * (transition_frames - from_end)
                / transition_frames
            )
    return result


def _nearest_indices(output_size: int, input_size: int, device) -> torch.Tensor:
    scale = input_size / output_size
    values = [min(input_size - 1, int((position + 0.5) * scale)) for position in range(output_size)]
    return torch.tensor(values, dtype=torch.long, device=device)


def _coarse_noise_frame(pattern: str, grid_width: int, grid_height: int, palette_rng, torch_generator) -> torch.Tensor:
    if pattern == "poc_chroma_blocks":
        rows = [
            [palette_rng.choice(_CHROMA_PALETTE) for _ in range(grid_width)]
            for _ in range(grid_height)
        ]
        return torch.tensor(rows, dtype=torch.float32).div_(255.0)
    if pattern == "gaussian_rgb":
        return torch.randn(
            (grid_height, grid_width, 3), generator=torch_generator, dtype=torch.float32
        ).mul_(0.25).add_(0.5).clamp_(0.0, 1.0)
    if pattern == "uniform_rgb":
        return torch.rand(
            (grid_height, grid_width, 3), generator=torch_generator, dtype=torch.float32
        )
    raise ValueError(f"Unknown pattern: {pattern!r}")


def _add_prefix_chroma_blocks(
    images: torch.Tensor,
    strength: float,
    seed: int,
    *,
    end_strength: float = 0.10,
    transition_frames: int = 4,
    tail_protection_frames: int = 5,
    pattern: str = "poc_chroma_blocks",
    grid_mode: str = "poc_36x64",
    block_size: int = _CHROMA_BLOCK_SIZE,
) -> torch.Tensor:
    frame_count, height, width, _ = images.shape
    tail_protection_frames = int(tail_protection_frames)
    if tail_protection_frames < 0:
        raise ValueError("tail_protection_frames must not be negative")
    affected_frames = frame_count - min(tail_protection_frames, frame_count)
    if affected_frames == 0:
        return images.clone()
    alphas = _noise_alpha_schedule(
        affected_frames,
        strength,
        end_strength,
        transition_frames,
    )
    if grid_mode == "poc_36x64":
        grid_width, grid_height = _CHROMA_POC_GRID
    elif grid_mode == "block_size":
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError("block_size must be positive")
        grid_width = max(1, round(width / block_size))
        grid_height = max(1, round(height / block_size))
    else:
        raise ValueError(f"Unknown grid_mode: {grid_mode!r}")

    palette_rng = random.Random(int(seed))
    torch_generator = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    row_indices = _nearest_indices(height, grid_height, images.device)
    column_indices = _nearest_indices(width, grid_width, images.device)
    output = images.clone()
    for position, alpha in enumerate(alphas):
        noise = _coarse_noise_frame(
            pattern, grid_width, grid_height, palette_rng, torch_generator
        ).to(device=images.device, dtype=images.dtype)
        noise = noise.index_select(0, row_indices).index_select(1, column_indices)
        output[position] = output[position] * (1.0 - alpha) + noise * alpha
    return output


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
                io.Int.Input("tail_frames", default=22, min=0, max=16384, step=1, tooltip="Final frames to load; 0 loads the complete segment."),
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
        return (int(segment_index), str(path), stat.st_mtime_ns, stat.st_size, int(tail_frames))

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
                source = source.as_trimmed(float(start_time), float(duration), strict_duration=False)
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
            expected = _ceil_fraction(Fraction(int(images.shape[0]) * sample_rate, 1) / frame_rate)
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
                io.Float.Input("frame_rate", default=24.0, min=0.01, max=1000.0, step=0.01),
                io.Boolean.Input("overwrite", default=False),
                io.Float.Input("crf", default=19.0, min=0.0, max=51.0, step=1.0),
                io.Audio.Input("audio", optional=True),
            ],
            outputs=[io.String.Output(display_name="filename")],
        )

    @classmethod
    def execute(cls, images, root_directory, segment_index, frame_rate, overwrite, crf=19.0, audio=None) -> io.NodeOutput:
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
            accept_all_inputs=True,
            category="Turing Utils/Video",
            description=(
                "Treat the complete IMAGE batch as a video prefix. Preserve a clean tail first, place a "
                "noise transition immediately before it, and apply full coarse colour-block noise to every "
                "earlier frame. Short batches prioritize the protected tail, then the transition, and only "
                "use full noise when frames remain. A missing IMAGE passes through as absent."
            ),
            inputs=[
                io.Image.Input("images", optional=True, tooltip="A missing IMAGE is passed through as None."),
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
                    max=0xffffffffffffffff,
                    control_after_generate=True,
                ),
                io.Float.Input("end_strength", default=0.10, min=0.0, max=1.0, step=0.01),
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
                        io.DynamicCombo.Option("block_size", [
                            io.Int.Input("block_size", default=16, min=1, max=256, step=1, optional=True),
                        ]),
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
        block_size=16,
    ) -> io.NodeOutput:
        # Old API prompts use flat block_size; the UI migrates it to the branch.
        if isinstance(grid_mode, dict):
            block_size = grid_mode.get("block_size") if grid_mode.get("block_size") is not None else block_size
            grid_mode = grid_mode["grid_mode"]
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
                io.Mask.Input("prefix_mask", optional=True, tooltip="Prefix redraw mask, not an audio mask. One mask repeats over every prefix frame; a batch must match prefix_images frame count. This node does not track motion."),
                io.Audio.Input("prefix_audio", optional=True, tooltip="Optional prefix waveform content. Audio preservation is controlled later by the H3 audio latent noise mask."),
                io.Image.Input("body_images", optional=True, tooltip="Required generated or source body frames."),
                io.Mask.Input("body_mask", optional=True, tooltip="Body redraw mask, not an audio mask. One mask repeats over every body frame; a batch must match body_images frame count. Use per-frame tracked masks for moving objects."),
                io.Audio.Input("body_audio", optional=True, tooltip="Optional body waveform content. Leave empty when H3 should generate the body audio."),
                io.Float.Input("frame_rate", default=24.0, min=0.01, max=1000.0, step=0.01),
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
        if body_images is None:
            raise ValueError("body_images is required")
        if mode not in ("concat", "replace"):
            raise ValueError(f"Unknown continuation mode: {mode!r}")
        body_images = _validate_images(body_images, "body_images")
        rate = _frame_rate(frame_rate)
        body_frames, height, width, _ = body_images.shape
        prefix_frames = 0
        if prefix_images is not None:
            prefix_images = _validate_images(prefix_images, "prefix_images")
            if tuple(prefix_images.shape[1:3]) != (height, width):
                raise ValueError(
                    f"prefix_images and body_images must have the same height/width; got "
                    f"{tuple(prefix_images.shape[1:3])} and {(height, width)}"
                )
            prefix_frames = int(prefix_images.shape[0])
            prefix_images = prefix_images.to(device=body_images.device, dtype=body_images.dtype)
        elif prefix_mask is not None or prefix_audio is not None:
            raise ValueError("prefix_mask and prefix_audio require prefix_images")

        if mode == "replace" and prefix_frames > int(body_frames):
            raise ValueError(
                f"replace mode cannot fit {prefix_frames} prefix frames into a {int(body_frames)}-frame body"
            )

        prefix_mask = _prepare_mask(prefix_mask, prefix_frames, height, width, 0.0, body_images.device)
        body_mask = _prepare_mask(body_mask, int(body_frames), height, width, 1.0, body_images.device)
        if mode == "concat":
            images = body_images if prefix_images is None else torch.cat((prefix_images, body_images), dim=0)
            mask = torch.cat((prefix_mask, body_mask), dim=0)
            trim_frames = prefix_frames
        else:
            images = (
                body_images
                if prefix_images is None
                else torch.cat((prefix_images, body_images[prefix_frames:]), dim=0)
            )
            mask = body_mask.clone()
            if prefix_frames > 0:
                mask[:prefix_frames] = prefix_mask
            trim_frames = prefix_frames
        audio, prefix_audio_samples = _audio_timeline(
            prefix_audio,
            body_audio,
            prefix_frames,
            int(body_frames),
            rate,
            mode,
        )
        trim_info = {
            "version": 1,
            "composition_mode": mode,
            "prefix_frames": prefix_frames,
            "trim_frames": trim_frames,
            "body_frames": int(body_frames),
            "frame_rate_numerator": rate.numerator,
            "frame_rate_denominator": rate.denominator,
            "prefix_audio_samples": prefix_audio_samples,
            "total_audio_samples": int(audio["waveform"].shape[-1]),
            "audio_sample_rate": int(audio["sample_rate"]),
            "prefix_audio_present": prefix_audio is not None,
        }
        return io.NodeOutput(images, mask, audio, trim_info)


def _trim_info_values(trim_info):
    if not isinstance(trim_info, dict) or int(trim_info.get("version", 0)) != 1:
        raise ValueError("trim_info must come from Video Continuation Concat")
    prefix_frames = int(trim_info["prefix_frames"])
    trim_frames = int(trim_info.get("trim_frames", prefix_frames))
    rate = Fraction(
        int(trim_info["frame_rate_numerator"]),
        int(trim_info["frame_rate_denominator"]),
    )
    if prefix_frames < 0 or not 0 <= trim_frames <= prefix_frames or rate <= 0:
        raise ValueError("trim_info contains an invalid prefix boundary")
    return prefix_frames, trim_frames, rate


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
            raise ValueError(f"trim_info requests {trim_frames} trim frames but images contains {int(images.shape[0])}")
        images = images[trim_frames:]
        if audio is not None:
            waveform, sample_rate = _validate_audio(audio, "audio")
            trim_samples = _ceil_fraction(Fraction(trim_frames * sample_rate, 1) / rate)
            waveform = waveform[..., min(trim_samples, int(waveform.shape[-1])):]
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
                    options=["protect_prefix_generate_body", "protect_all", "generate_all"],
                    default="protect_prefix_generate_body",
                ),
                VideoTrimInfo.Input("trim_info", optional=True),
            ],
            outputs=[io.Latent.Output(display_name="audio_latent")],
        )

    @classmethod
    def execute(cls, audio_latent, trim_info=None, mode="protect_prefix_generate_body"):
        if not isinstance(audio_latent, dict) or "samples" not in audio_latent:
            raise ValueError("audio_latent must be a LATENT dictionary")
        samples = audio_latent["samples"]
        if getattr(samples, "is_nested", False) or not torch.is_tensor(samples) or samples.ndim != 4 or tuple(samples.shape[1:3]) != (32, 2):
            shape = tuple(samples.shape) if hasattr(samples, "shape") else type(samples).__name__
            raise ValueError(f"Expected standalone H3 audio latent [B,32,2,T], got {shape}")
        if mode not in ("protect_prefix_generate_body", "protect_all", "generate_all"):
            raise ValueError(f"Unknown audio mask mode: {mode!r}")
        if mode == "protect_all":
            mask = torch.zeros_like(samples, dtype=torch.float32)
        elif mode == "generate_all":
            mask = torch.ones_like(samples, dtype=torch.float32)
        else:
            _trim_info_values(trim_info)
            total_samples = int(trim_info.get("total_audio_samples", 0))
            prefix_samples = int(trim_info.get("prefix_audio_samples", 0))
            if total_samples < 1 or not 0 <= prefix_samples <= total_samples:
                raise ValueError("trim_info contains an invalid audio boundary")
            latent_length = int(samples.shape[-1])
            if trim_info.get("prefix_audio_present", True):
                boundary = (2 * prefix_samples * latent_length + total_samples) // (2 * total_samples)
            else:
                boundary = 0
            boundary = min(max(boundary, 0), latent_length)
            mask = torch.ones_like(samples, dtype=torch.float32)
            mask[..., :boundary] = 0.0

        output = audio_latent.copy()
        output["noise_mask"] = mask
        return io.NodeOutput(output)
