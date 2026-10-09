"""General media preparation nodes."""

from __future__ import annotations

import bisect

import av
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

import comfy.utils
from comfy_api.latest import InputImpl, io


def uniform_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count < 1:
        raise ValueError("A motion contact sheet requires at least one frame")
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if sample_count == 1:
        return [frame_count // 2]
    return [round(i * (frame_count - 1) / (sample_count - 1)) for i in range(sample_count)]


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
        if 0 < index < frame_count and abs(cumulative[index - 1] - target) <= abs(cumulative[index] - target):
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
            scores.extend((gray[1:] - gray[:-1]).abs().mean(dim=(1, 2)).detach().cpu().tolist())
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
            timestamp = float(frame.pts * stream.time_base) if frame.pts is not None else None
            if timestamp is not None and timestamp + 1e-9 < start_time:
                continue
            if timestamp is not None and end_time is not None and timestamp >= end_time:
                break
            relative_time = max(0.0, timestamp - start_time) if timestamp is not None else fallback_index / float(video.get_frame_rate())
            yield frame, relative_time
            fallback_index += 1


def _frame_to_tensor(frame) -> torch.Tensor:
    array = frame.to_ndarray(format="rgb24")
    return torch.from_numpy(array.copy()).float().div_(255.0)


def _decode_video_indices(video, indices: list[int]) -> tuple[torch.Tensor, list[float]]:
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


def _sample_video(video, sample_count: int, sampling: str) -> tuple[torch.Tensor, list[float]]:
    if isinstance(video, InputImpl.VideoFromComponents):
        components = video.get_components()
        return _sample_images(components.images, sample_count, sampling, float(components.frame_rate))
    if sampling == "motion_weighted":
        indices = motion_weighted_frame_indices(_video_motion_scores(video), sample_count)
    else:
        indices = uniform_frame_indices(video.get_frame_count(), sample_count)
    return _decode_video_indices(video, indices)


def _sample_images(frames: torch.Tensor, sample_count: int, sampling: str, frame_rate: float) -> tuple[torch.Tensor, list[float]]:
    if not torch.is_tensor(frames) or frames.ndim != 4 or frames.shape[-1] < 3:
        shape = tuple(frames.shape) if hasattr(frames, "shape") else type(frames).__name__
        raise ValueError(f"Expected IMAGE frames shaped [frames,height,width,channels], got {shape}")
    if frames.shape[0] < 1:
        raise ValueError("The IMAGE input contains no frames")
    if frame_rate <= 0.0:
        raise ValueError("image_frame_rate must be positive")
    if sampling == "motion_weighted":
        indices = motion_weighted_frame_indices(_motion_scores(frames), sample_count)
    else:
        indices = uniform_frame_indices(int(frames.shape[0]), sample_count)
    return frames[indices, ..., :3], [index / frame_rate for index in indices]


def _resolved_size(source_width: int, source_height: int, width: int, height: int) -> tuple[int, int]:
    if width == 0 and height == 0:
        return source_width, source_height
    if width == 0:
        return max(1, round(height * source_width / source_height)), height
    if height == 0:
        return width, max(1, round(width * source_height / source_width))
    return width, height


def _split_extent(total: int, count: int, gap: int) -> list[tuple[int, int]]:
    usable = total - gap * (count - 1)
    if usable < count:
        raise ValueError(f"Output extent {total} is too small for {count} cells and gap={gap}")
    base, remainder = divmod(usable, count)
    segments = []
    position = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        segments.append((position, size))
        position += size + gap
    return segments


def _fit_image(image: Image.Image, width: int, height: int, resize_mode: str) -> Image.Image:
    if resize_mode == "stretch":
        return image.resize((width, height), Image.Resampling.LANCZOS)
    if resize_mode == "crop":
        return ImageOps.fit(image, (width, height), Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    resized = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
    output = Image.new("RGB", (width, height), (8, 8, 8))
    output.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return output


def _align_dimension(value: int, multiple: int) -> int:
    if multiple <= 1:
        return max(1, value)
    return max(multiple, value - value % multiple)


def _resize_tensor_image(image: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
    return comfy.utils.common_upscale(image.movedim(-1, 1), width, height, method, "disabled").movedim(1, -1)


def _resize_tensor_mask(mask: torch.Tensor | None, width: int, height: int) -> torch.Tensor | None:
    if mask is None:
        return None
    return F.interpolate(mask.unsqueeze(1).float(), size=(height, width), mode="nearest-exact").squeeze(1).to(mask.dtype)


def _position_offset(extra_width: int, extra_height: int, position: str) -> tuple[int, int]:
    if position == "top":
        return extra_width // 2, 0
    if position == "bottom":
        return extra_width // 2, extra_height
    if position == "left":
        return 0, extra_height // 2
    if position == "right":
        return extra_width, extra_height // 2
    return extra_width // 2, extra_height // 2


def _pad_color(value: str, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    try:
        values = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError:
        values = [float(component) for component in ImageColor.getrgb(value.strip())]
    if not values:
        values = [0.0]
    if max(abs(component) for component in values) > 1.0:
        values = [component / 255.0 for component in values]
    if len(values) == 1:
        values *= channels
    elif len(values) < channels:
        values.extend([1.0 if len(values) == 3 else values[-1]] * (channels - len(values)))
    values = values[:channels]
    return torch.tensor(values, device=device, dtype=dtype)


def resize_image_if_present(
    image: torch.Tensor | None,
    mask: torch.Tensor | None,
    width: int,
    height: int,
    resize_mode: str,
    upscale_method: str,
    crop_position: str,
    divisible_by: int,
    pad_color: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int, int]:
    if image is None:
        return None, None, 0, 0
    if image.ndim != 4 or image.shape[1] < 1 or image.shape[2] < 1 or image.shape[3] < 1:
        raise ValueError(f"image must be shaped [batch,height,width,channels], got {tuple(image.shape)}")
    if mask is not None:
        if mask.ndim != 3:
            raise ValueError(f"mask must be shaped [batch,height,width], got {tuple(mask.shape)}")
        if mask.shape[-2:] != image.shape[1:3]:
            mask = _resize_tensor_mask(mask, int(image.shape[2]), int(image.shape[1]))

    source_height, source_width = int(image.shape[1]), int(image.shape[2])
    width, height = int(width), int(height)
    if width < 0 or height < 0:
        raise ValueError("width and height must be non-negative")
    if width == 0 and height == 0:
        return image, mask, source_width, source_height

    target_width, target_height = _resolved_size(source_width, source_height, width, height)
    multiple = max(1, int(divisible_by))
    target_width = _align_dimension(target_width, multiple)
    target_height = _align_dimension(target_height, multiple)

    if resize_mode == "stretch" or width == 0 or height == 0:
        output = _resize_tensor_image(image, target_width, target_height, upscale_method)
        output_mask = _resize_tensor_mask(mask, target_width, target_height)
        return output, output_mask, target_width, target_height

    if resize_mode == "crop":
        source_aspect = source_width / source_height
        target_aspect = target_width / target_height
        if source_aspect > target_aspect:
            crop_width = max(1, round(source_height * target_aspect))
            crop_height = source_height
        else:
            crop_width = source_width
            crop_height = max(1, round(source_width / target_aspect))
        x, y = _position_offset(source_width - crop_width, source_height - crop_height, crop_position)
        image = image[:, y : y + crop_height, x : x + crop_width]
        if mask is not None:
            mask = mask[:, y : y + crop_height, x : x + crop_width]
        output = _resize_tensor_image(image, target_width, target_height, upscale_method)
        output_mask = _resize_tensor_mask(mask, target_width, target_height)
        return output, output_mask, target_width, target_height

    ratio = min(target_width / source_width, target_height / source_height)
    content_width = min(target_width, max(1, round(source_width * ratio)))
    content_height = min(target_height, max(1, round(source_height * ratio)))
    content_width = min(target_width, _align_dimension(content_width, multiple))
    content_height = min(target_height, _align_dimension(content_height, multiple))
    output = _resize_tensor_image(image, content_width, content_height, upscale_method)
    output_mask = _resize_tensor_mask(mask, content_width, content_height)
    if resize_mode == "fit":
        return output, output_mask, content_width, content_height

    extra_width = target_width - content_width
    extra_height = target_height - content_height
    x, y = _position_offset(extra_width, extra_height, crop_position)
    left, right = x, extra_width - x
    top, bottom = y, extra_height - y
    if resize_mode == "pad_edge":
        output = F.pad(output.movedim(-1, 1), (left, right, top, bottom), mode="replicate").movedim(1, -1)
        if output_mask is not None:
            output_mask = F.pad(output_mask.unsqueeze(1), (left, right, top, bottom), mode="replicate").squeeze(1)
    else:
        fill = _pad_color(pad_color, int(output.shape[-1]), output.device, output.dtype)
        canvas = fill.reshape(1, 1, 1, -1).expand(output.shape[0], target_height, target_width, -1).clone()
        canvas[:, top : top + content_height, left : left + content_width] = output
        output = canvas
        if output_mask is not None:
            mask_canvas = output_mask.new_zeros((output_mask.shape[0], target_height, target_width))
            mask_canvas[:, top : top + content_height, left : left + content_width] = output_mask
            output_mask = mask_canvas
    return output, output_mask, target_width, target_height


def _label(annotation: str, index: int, timestamp: float) -> str:
    if annotation == "index":
        return f"{index + 1:02d}"
    if annotation == "timestamp":
        return f"{timestamp:.2f}s"
    if annotation == "index_timestamp":
        return f"{index + 1:02d} | {timestamp:.2f}s"
    return ""


def _draw_centered_label(draw: ImageDraw.ImageDraw, label: str, y0: int, width: int, height: int):
    if not label or height < 6:
        return None
    font_size = max(8, round(height * 0.62))
    font = ImageFont.load_default(size=font_size)
    bounds = draw.textbbox((0, 0), label, font=font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    x = max(1, (width - text_width) // 2)
    y = y0 + max(0, (height - text_height) // 2 - bounds[1])
    draw.text((x, y), label, fill=(235, 235, 220), font=font)
    return x, x + text_width


def _draw_perforations(draw: ImageDraw.ImageDraw, width: int, tile_height: int, rail_height: int, label_span):
    hole_height = max(2, rail_height // 3)
    hole_width = max(4, round(hole_height * 1.7))
    step = max(hole_width + 3, round(hole_width * 1.8))
    top_y = max(1, (rail_height - hole_height) // 2)
    bottom_y = tile_height - rail_height + top_y
    for x in range(3, width - hole_width, step):
        draw.rounded_rectangle((x, top_y, x + hole_width, top_y + hole_height), radius=max(1, hole_height // 4), fill=(224, 216, 188))
        if label_span is None or x + hole_width < label_span[0] - 4 or x > label_span[1] + 4:
            draw.rounded_rectangle((x, bottom_y, x + hole_width, bottom_y + hole_height), radius=max(1, hole_height // 4), fill=(224, 216, 188))


def _render_tile(frame: torch.Tensor, width: int, height: int, resize_mode: str, film_border: bool, label: str) -> Image.Image:
    if width < 8 or height < 8:
        raise ValueError(f"Contact-sheet cells must be at least 8x8 pixels, got {width}x{height}")
    array = frame.detach().float().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
    source = Image.fromarray(array[..., :3])
    tile = Image.new("RGB", (width, height), (9, 9, 8))
    draw = ImageDraw.Draw(tile)

    if film_border:
        rail_height = min(max(10, round(height * 0.09)), max(10, height // 4))
        viewport_height = height - rail_height * 2
        if viewport_height < 8:
            raise ValueError(f"Cell height {height} is too small for the film border")
        tile.paste(_fit_image(source, width, viewport_height, resize_mode), (0, rail_height))
        draw.rectangle((0, rail_height, width - 1, height - rail_height - 1), outline=(65, 62, 52))
        label_span = _draw_centered_label(draw, label, height - rail_height, width, rail_height)
        _draw_perforations(draw, width, height, rail_height, label_span)
    elif label:
        caption_height = min(max(10, round(height * 0.08)), max(10, height // 4))
        tile.paste(_fit_image(source, width, height - caption_height, resize_mode), (0, 0))
        _draw_centered_label(draw, label, height - caption_height, width, caption_height)
    else:
        tile.paste(_fit_image(source, width, height, resize_mode), (0, 0))
    return tile


def render_contact_sheet(
    frames: torch.Tensor,
    timestamps: list[float],
    grid_size: int,
    width: int,
    height: int,
    resize_mode: str,
    gap: int,
    film_border: bool,
    annotation: str,
) -> torch.Tensor:
    expected_frames = int(grid_size) ** 2
    if int(frames.shape[0]) != expected_frames or len(timestamps) != expected_frames:
        raise ValueError(
            f"A {grid_size}x{grid_size} contact sheet requires {expected_frames} frames and timestamps; "
            f"got {int(frames.shape[0])} frames and {len(timestamps)} timestamps"
        )
    source_height, source_width = int(frames.shape[1]), int(frames.shape[2])
    width, height = _resolved_size(source_width, source_height, width, height)
    columns = _split_extent(width, grid_size, gap)
    rows = _split_extent(height, grid_size, gap)
    sheet = Image.new("RGB", (width, height), (5, 5, 5))
    for index, frame in enumerate(frames):
        row, column = divmod(index, grid_size)
        x, tile_width = columns[column]
        y, tile_height = rows[row]
        tile = _render_tile(frame, tile_width, tile_height, resize_mode, film_border, _label(annotation, index, timestamps[index]))
        sheet.paste(tile, (x, y))
    array = np.array(sheet, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


class VideoMotionContactSheet(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsVideoMotionContactSheet",
            display_name="Video Motion Contact Sheet",
            category="Turing Utils/Video",
            description="Sample N x N frames from a VIDEO or IMAGE batch and arrange them as an optional annotated filmstrip-style motion storyboard.",
            is_experimental=True,
            inputs=[
                io.Int.Input("grid_size", default=3, min=2, max=8, step=1, tooltip="Sample exactly grid_size squared frames in chronological row-major order."),
                io.Combo.Input("sampling", options=["uniform", "motion_weighted"], default="uniform", tooltip="Uniform preserves timing. Motion weighted allocates more panels to intervals with stronger visual change."),
                io.Int.Input("width", default=0, min=0, max=8192, step=32, tooltip="Final sheet width. 0 derives it from the source aspect and the height setting."),
                io.Int.Input("height", default=0, min=0, max=8192, step=32, tooltip="Final sheet height. 0 derives it from the source aspect and the width setting."),
                io.Combo.Input("resize_mode", options=["fit", "crop", "stretch"], default="fit", tooltip="How each sampled frame is fitted into its panel."),
                io.Int.Input("gap", default=0, min=0, max=64, step=1, tooltip="Black pixels between adjacent panels."),
                io.Boolean.Input("film_border", default=True, tooltip="Wrap every panel in film rails and perforations. Labels are printed on the lower rail."),
                io.Combo.Input("annotation", options=["none", "index", "timestamp", "index_timestamp"], default="index", tooltip="Optional chronological label. Without the film border, labels use a narrow caption rail outside the frame."),
                io.Float.Input("image_frame_rate", default=24.0, min=0.01, max=1000.0, step=0.01, tooltip="Used only to derive timestamps when the source is an IMAGE batch."),
                io.Video.Input("video", optional=True, tooltip="A loaded ComfyUI VIDEO. Connect either video or frames, not both."),
                io.Image.Input("frames", optional=True, tooltip="An already decoded video frame batch. Connect either frames or video, not both."),
            ],
            outputs=[
                io.Image.Output(display_name="contact_sheet"),
                io.Image.Output(display_name="sampled_frames"),
                io.String.Output(display_name="prompt_hint"),
            ],
        )

    @classmethod
    def execute(
        cls,
        grid_size: int,
        sampling: str,
        width: int,
        height: int,
        resize_mode: str,
        gap: int,
        film_border: bool,
        annotation: str,
        image_frame_rate: float,
        video=None,
        frames=None,
    ) -> io.NodeOutput:
        if (video is None) == (frames is None):
            raise ValueError("Connect exactly one source: video or frames")
        sample_count = int(grid_size) ** 2
        if video is not None:
            sampled, timestamps = _sample_video(video, sample_count, sampling)
        else:
            sampled, timestamps = _sample_images(frames, sample_count, sampling, image_frame_rate)
        sheet = render_contact_sheet(sampled, timestamps, int(grid_size), int(width), int(height), resize_mode, int(gap), bool(film_border), annotation)
        hint = (
            "The reference image is a chronological motion storyboard. Read its frames from left to right, "
            "then top to bottom. Follow the depicted motion and camera progression, but do not reproduce "
            "frame numbers, timestamps, film borders, perforations, or other storyboard markings."
        )
        return io.NodeOutput(sheet, sampled, hint)


class ResizeImageIfPresent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsResizeImageIfPresent",
            display_name="Resize Image If Present",
            category="Turing Utils/Video",
            description="Resize and optionally crop or pad an image. An unconnected image produces an absent image instead of a placeholder frame.",
            inputs=[
                io.Image.Input("image", optional=True, tooltip="Leave unconnected to produce no image."),
                io.Mask.Input("mask", optional=True, tooltip="Optional mask transformed with the same geometry as the image."),
                io.Int.Input("width", default=0, min=0, max=16384, step=1, tooltip="Target width. 0 derives it from height; width=height=0 passes the input through."),
                io.Int.Input("height", default=0, min=0, max=16384, step=1, tooltip="Target height. 0 derives it from width; width=height=0 passes the input through."),
                io.Combo.Input("resize_mode", options=["stretch", "fit", "crop", "pad", "pad_edge"], default="crop"),
                io.Combo.Input("upscale_method", options=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], default="lanczos"),
                io.Combo.Input("crop_position", options=["center", "top", "bottom", "left", "right"], default="center", tooltip="Crop anchor or padded-content alignment."),
                io.Int.Input("divisible_by", default=1, min=1, max=512, step=1),
                io.String.Input("pad_color", default="0, 0, 0", tooltip="RGB values in 0-255 or 0-1 form, a hex color, or a CSS color name."),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, width, height, resize_mode, upscale_method, crop_position, divisible_by, pad_color, image=None, mask=None):
        return io.NodeOutput(*resize_image_if_present(
            image,
            mask,
            width,
            height,
            resize_mode,
            upscale_method,
            crop_position,
            divisible_by,
            pad_color,
        ))
