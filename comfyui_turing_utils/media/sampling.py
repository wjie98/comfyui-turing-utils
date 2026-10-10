"""Uniform and motion-aware sampling of tensors and streamed videos."""

from __future__ import annotations

import bisect

import av
import torch
from comfy_api.latest import InputImpl


def uniform_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count < 1:
        raise ValueError("A motion contact sheet requires at least one frame")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if sample_count == 1:
        return [frame_count // 2]
    return [
        round(i * (frame_count - 1) / (sample_count - 1)) for i in range(sample_count)
    ]


def motion_weighted_frame_indices(motion: list[float], sample_count: int) -> list[int]:
    frame_count = len(motion) + 1
    if sample_count == 1:
        return [frame_count // 2]
    if frame_count == 1 or not any(value > 0.0 for value in motion):
        return uniform_frame_indices(frame_count, sample_count)

    mean_motion = sum(motion) / len(motion)
    baseline = max(mean_motion * 0.15, 1e-6)
    cumulative = [0.0]
    for value in motion:
        cumulative.append(cumulative[-1] + max(value, 0.0) + baseline)

    indices = []
    total = cumulative[-1]
    for i in range(sample_count):
        target = total * i / max(sample_count - 1, 1)
        index = bisect.bisect_left(cumulative, target)
        if 0 < index < frame_count and abs(cumulative[index - 1] - target) <= abs(
            cumulative[index] - target
        ):
            index -= 1
        indices.append(min(index, frame_count - 1))
    indices[0] = 0
    indices[-1] = frame_count - 1
    return indices


def _motion_scores(frames: torch.Tensor) -> list[float]:
    frame_count, height, width, _ = frames.shape
    if frame_count < 2:
        return []
    stride_y = max(1, height // 64)
    stride_x = max(1, width // 64)
    previous = None
    scores = []
    for start in range(0, frame_count, 128):
        chunk = frames[start : start + 128, ::stride_y, ::stride_x, :3].float()
        gray = chunk[..., 0] * 0.299 + chunk[..., 1] * 0.587 + chunk[..., 2] * 0.114
        if previous is not None:
            scores.append(float((gray[0] - previous).abs().mean().item()))
        if gray.shape[0] > 1:
            scores.extend(
                (gray[1:] - gray[:-1]).abs().mean(dim=(1, 2)).detach().cpu().tolist()
            )
        previous = gray[-1]
    return scores


def _iter_video_frames(video):
    source = video.get_stream_source()
    start_time, duration = video.get_active_trim_window()
    end_time = start_time + duration if duration > 0.0 else None
    with av.open(source, mode="r") as container:
        if not container.streams.video:
            raise ValueError("The VIDEO input contains no video stream")
        stream = container.streams.video[0]
        if start_time > 0.0:
            container.seek(max(0, int(start_time / stream.time_base)), stream=stream)
        fallback_index = 0
        for frame in container.decode(stream):
            timestamp = (
                float(frame.pts * stream.time_base) if frame.pts is not None else None
            )
            if timestamp is not None and timestamp + 1e-9 < start_time:
                continue
            if timestamp is not None and end_time is not None and timestamp >= end_time:
                break
            relative_time = (
                max(0.0, timestamp - start_time)
                if timestamp is not None
                else fallback_index / float(video.get_frame_rate())
            )
            yield frame, relative_time
            fallback_index += 1


def _frame_to_tensor(frame) -> torch.Tensor:
    array = frame.to_ndarray(format="rgb24")
    return torch.from_numpy(array.copy()).float().div_(255.0)


def _decode_video_indices(
    video, indices: list[int]
) -> tuple[torch.Tensor, list[float]]:
    wanted = set(indices)
    decoded = {}
    last_frame = None
    last_time = 0.0
    max_index = max(indices)
    for index, (frame, timestamp) in enumerate(_iter_video_frames(video)):
        last_frame = frame
        last_time = timestamp
        if index in wanted:
            decoded[index] = (_frame_to_tensor(frame), timestamp)
        if index >= max_index:
            break
    if last_frame is None:
        raise ValueError("The VIDEO input contains no decodable frames")

    tail = (_frame_to_tensor(last_frame), last_time)
    available = sorted(decoded)
    output_frames = []
    output_times = []
    for index in indices:
        if index in decoded:
            image, timestamp = decoded[index]
        elif not available or index > available[-1]:
            image, timestamp = tail
        else:
            nearest = min(available, key=lambda candidate: abs(candidate - index))
            image, timestamp = decoded[nearest]
        output_frames.append(image)
        output_times.append(timestamp)
    return torch.stack(output_frames), output_times


def _video_motion_scores(video) -> list[float]:
    previous = None
    scores = []
    for frame, _ in _iter_video_frames(video):
        thumb_width = min(64, frame.width)
        thumb_height = max(1, round(frame.height * thumb_width / frame.width))
        thumb = frame.reformat(width=thumb_width, height=thumb_height, format="gray")
        gray = torch.from_numpy(thumb.to_ndarray().copy()).float()
        if previous is not None:
            scores.append(float((gray - previous).abs().mean().item()))
        previous = gray
    if previous is None:
        raise ValueError("The VIDEO input contains no decodable frames")
    return scores


def sample_video(
    video, sample_count: int, sampling: str
) -> tuple[torch.Tensor, list[float]]:
    if isinstance(video, InputImpl.VideoFromComponents):
        components = video.get_components()
        return sample_images(
            components.images, sample_count, sampling, float(components.frame_rate)
        )
    if sampling == "motion_weighted":
        indices = motion_weighted_frame_indices(
            _video_motion_scores(video), sample_count
        )
    else:
        indices = uniform_frame_indices(video.get_frame_count(), sample_count)
    return _decode_video_indices(video, indices)


def sample_images(
    frames: torch.Tensor, sample_count: int, sampling: str, frame_rate: float
) -> tuple[torch.Tensor, list[float]]:
    if not torch.is_tensor(frames) or frames.ndim != 4 or frames.shape[-1] < 3:
        shape = (
            tuple(frames.shape) if hasattr(frames, "shape") else type(frames).__name__
        )
        raise ValueError(
            f"Expected IMAGE frames shaped [frames,height,width,channels], got {shape}"
        )
    if frames.shape[0] < 1:
        raise ValueError("The IMAGE input contains no frames")
    if frame_rate <= 0.0:
        raise ValueError("image_frame_rate must be positive")
    if sampling == "motion_weighted":
        indices = motion_weighted_frame_indices(_motion_scores(frames), sample_count)
    else:
        indices = uniform_frame_indices(int(frames.shape[0]), sample_count)
    return frames[indices, ..., :3], [index / frame_rate for index in indices]
