"""Indexed video segment storage, probing and merging."""

from __future__ import annotations

import logging
import os
import uuid
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import comfy.utils
import folder_paths
from comfy_api.latest import InputImpl

from .timeline import _AAC_FRAME_SAMPLES, _ceil_fraction


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


def _segment_path(
    root_directory: str, segment_index: int, *, create: bool = False
) -> Path:
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
        raise ValueError(
            "Indexed video path must stay inside root_directory"
        ) from error
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
        raise FileNotFoundError(
            f"Indexed segment directory does not exist: {directory}"
        )

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
        raise ValueError(
            f"Indexed segments must be contiguous from 000000.mp4; missing {preview}{suffix}"
        )
    return [indexed[index] for index in range(final_index + 1)]


def _merged_segment_path(root_directory: str, output_filename: str) -> Path:
    directory = _segment_directory(root_directory, create=True)
    normalized = str(output_filename).strip().replace("\\", "/")
    relative = Path(normalized)
    if not normalized or relative.is_absolute() or len(relative.parts) != 1:
        raise ValueError("output_filename must be a filename inside root_directory")
    if relative.suffix.lower() != ".mp4":
        raise ValueError("output_filename must end with .mp4")
    if (
        len(relative.name) == 10
        and relative.name[:6].isascii()
        and relative.name[:6].isdigit()
    ):
        raise ValueError(
            "output_filename must not use the reserved NNNNNN.mp4 segment format"
        )
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
            (
                stream
                for stream in container.streams.audio
                if stream.codec_context is not None
            ),
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
            (
                candidate
                for candidate in container.streams.audio
                if candidate.codec_context is not None
            ),
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


def _encode_audio_chunk(
    output, stream, waveform: np.ndarray, sample_rate: int, start_sample: int
) -> None:
    chunk_samples = 65536
    layout = stream.codec_context.layout.name
    for offset in range(0, int(waveform.shape[1]), chunk_samples):
        chunk = np.ascontiguousarray(waveform[:, offset : offset + chunk_samples])
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
        raise ValueError(
            "Either every indexed segment must contain audio or none of them may contain audio"
        )
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
                        if (
                            last_video_dts is not None
                            and absolute_dts <= last_video_dts
                        ):
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
