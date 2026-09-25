"""Video region-of-interest, outpaint, and inverse-composite nodes."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from comfy_api.latest import io


VideoCropInfoType = io.Custom("TURING_UTILS_VIDEO_CROP_INFO")


def _validate_images(images: torch.Tensor, name: str) -> tuple[int, int, int, int]:
    if not torch.is_tensor(images) or images.ndim != 4:
        shape = tuple(images.shape) if hasattr(images, "shape") else type(images).__name__
        raise ValueError(f"{name} must be an IMAGE batch shaped [frames,height,width,channels], got {shape}")
    frames, height, width, channels = (int(value) for value in images.shape)
    if frames < 1 or height < 1 or width < 1 or channels < 1:
        raise ValueError(f"{name} must be non-empty, got {tuple(images.shape)}")
    if not images.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype, got {images.dtype}")
    return frames, height, width, channels


def _prepare_masks(
    masks: torch.Tensor,
    frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    if not torch.is_tensor(masks):
        raise ValueError(f"masks must be a MASK tensor, got {type(masks).__name__}")
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if masks.ndim != 3:
        raise ValueError(f"masks must be shaped [frames,height,width], got {tuple(masks.shape)}")
    if int(masks.shape[0]) != frames:
        raise ValueError(
            f"masks and images must have the same frame count; got {int(masks.shape[0])} and {frames}"
        )
    masks = masks.detach().float()
    if tuple(masks.shape[-2:]) != (height, width):
        masks = F.interpolate(
            masks.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    return masks.clamp_(0.0, 1.0)


def _mask_observations(
    masks: torch.Tensor,
    threshold: float,
    context_scale: float,
    output_ratio: float,
    max_crop_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = int(masks.shape[0])
    centers_x = np.full(frames, np.nan, dtype=np.float64)
    centers_y = np.full(frames, np.nan, dtype=np.float64)
    log_heights = np.full(frames, np.nan, dtype=np.float64)
    valid = np.zeros(frames, dtype=bool)

    masks_cpu = masks.detach().cpu()
    for index in range(frames):
        ys, xs = torch.where(masks_cpu[index] >= threshold)
        if int(xs.numel()) < 4:
            continue
        x0 = float(xs.min().item())
        x1 = float(xs.max().item()) + 1.0
        y0 = float(ys.min().item())
        y1 = float(ys.max().item()) + 1.0
        mask_width = max(1.0, x1 - x0)
        mask_height = max(1.0, y1 - y0)
        base_height = max(mask_height, mask_width / output_ratio)
        crop_height = min(max_crop_height, max(1.0, base_height * context_scale))
        centers_x[index] = (x0 + x1) * 0.5
        centers_y[index] = (y0 + y1) * 0.5
        log_heights[index] = math.log(crop_height)
        valid[index] = True
    return centers_x, centers_y, log_heights, valid


def _fill_missing(values: np.ndarray, valid: np.ndarray, mode: str) -> np.ndarray:
    indices = np.flatnonzero(valid)
    if not len(indices):
        raise ValueError("No frame contains a usable mask region")
    if mode == "interpolate":
        return np.interp(np.arange(len(values)), indices, values[indices])
    if mode != "hold":
        raise ValueError(f"Unsupported missing_mode: {mode}")

    output = np.asarray(values, dtype=np.float64).copy()
    first = int(indices[0])
    output[: first + 1] = output[first]
    previous = output[first]
    for index in range(first + 1, len(output)):
        if valid[index]:
            previous = output[index]
        else:
            output[index] = previous
    return output


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1 or len(values) < 3:
        return values
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window <= 1:
        return values
    if window % 2 == 0:
        window -= 1
    radius = window // 2
    offsets = np.arange(window, dtype=np.float64) - radius
    sigma = max(window / 6.0, 0.5)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def mask_guided_crop_boxes(
    masks: torch.Tensor,
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    context_scale: float,
    missing_mode: str,
    smooth_window: int,
    mask_threshold: float,
) -> tuple[list[tuple[float, float, float, float]], list[bool]]:
    if output_width < 1 or output_height < 1:
        raise ValueError("width and height must be positive")
    if context_scale < 1.0:
        raise ValueError("context_scale must be at least 1.0")
    if not 0.0 < mask_threshold <= 1.0:
        raise ValueError("mask_threshold must be in (0, 1]")

    ratio = float(output_width) / float(output_height)
    max_crop_height = min(float(source_height), float(source_width) / ratio)
    if max_crop_height <= 0.0:
        raise ValueError("The requested crop aspect ratio does not fit the source frame")

    centers_x, centers_y, log_heights, valid = _mask_observations(
        masks,
        mask_threshold,
        context_scale,
        ratio,
        max_crop_height,
    )
    centers_x = _smooth_series(_fill_missing(centers_x, valid, missing_mode), smooth_window)
    centers_y = _smooth_series(_fill_missing(centers_y, valid, missing_mode), smooth_window)
    log_heights = _smooth_series(_fill_missing(log_heights, valid, missing_mode), smooth_window)

    boxes: list[tuple[float, float, float, float]] = []
    for center_x, center_y, log_height in zip(centers_x, centers_y, log_heights):
        crop_height = min(max_crop_height, max(1.0, math.exp(float(log_height))))
        crop_width = crop_height * ratio
        center_x = min(max(float(center_x), crop_width * 0.5), source_width - crop_width * 0.5)
        center_y = min(max(float(center_y), crop_height * 0.5), source_height - crop_height * 0.5)
        boxes.append(
            (
                center_x - crop_width * 0.5,
                center_y - crop_height * 0.5,
                crop_width,
                crop_height,
            )
        )
    return boxes, valid.tolist()


def _sampling_dtype(tensor: torch.Tensor) -> torch.dtype:
    if tensor.dtype in (torch.float32, torch.float64):
        return tensor.dtype
    if tensor.device.type == "cuda" and tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.dtype
    return torch.float32


def _crop_theta(
    boxes: list[tuple[float, float, float, float]],
    source_width: int,
    source_height: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    theta = torch.zeros((len(boxes), 2, 3), device=device, dtype=dtype)
    for index, (x, y, width, height) in enumerate(boxes):
        theta[index, 0, 0] = width / source_width
        theta[index, 1, 1] = height / source_height
        theta[index, 0, 2] = 2.0 * (x + width * 0.5) / source_width - 1.0
        theta[index, 1, 2] = 2.0 * (y + height * 0.5) / source_height - 1.0
    return theta


def _stitch_theta(
    boxes: list[tuple[float, float, float, float]],
    source_width: int,
    source_height: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    theta = torch.zeros((len(boxes), 2, 3), device=device, dtype=dtype)
    for index, (x, y, width, height) in enumerate(boxes):
        theta[index, 0, 0] = source_width / width
        theta[index, 1, 1] = source_height / height
        theta[index, 0, 2] = (source_width - 2.0 * x - width) / width
        theta[index, 1, 2] = (source_height - 2.0 * y - height) / height
    return theta


def crop_video_by_mask(
    images: torch.Tensor,
    masks: torch.Tensor,
    width: int,
    height: int,
    context_scale: float,
    missing_mode: str,
    smooth_window: int,
    mask_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    frames, source_height, source_width, channels = _validate_images(images, "images")
    masks = _prepare_masks(masks, frames, source_height, source_width)
    boxes, valid = mask_guided_crop_boxes(
        masks,
        source_width,
        source_height,
        int(width),
        int(height),
        float(context_scale),
        missing_mode,
        int(smooth_window),
        float(mask_threshold),
    )

    sample_dtype = _sampling_dtype(images)
    image_nchw = images.movedim(-1, 1).to(dtype=sample_dtype)
    theta = _crop_theta(
        boxes,
        source_width,
        source_height,
        device=images.device,
        dtype=sample_dtype,
    )
    grid = F.affine_grid(
        theta,
        size=(frames, channels, int(height), int(width)),
        align_corners=False,
    )
    cropped_images = F.grid_sample(
        image_nchw,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).movedim(1, -1).to(dtype=images.dtype)
    mask_nchw = masks.to(device=images.device, dtype=sample_dtype).unsqueeze(1)
    cropped_masks = F.grid_sample(
        mask_nchw,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(1).float().clamp_(0.0, 1.0)

    crop_info = {
        "version": 1,
        "boxes": boxes,
        "source_size": (source_width, source_height),
        "output_size": (int(width), int(height)),
        "frames": frames,
        "mask_valid": valid,
    }
    return cropped_images, cropped_masks, crop_info


def _blur_masks(masks: torch.Tensor, radius: int) -> torch.Tensor:
    radius = max(0, int(radius))
    if radius == 0:
        return masks
    sigma = max(radius / 3.0, 0.5)
    offsets = torch.arange(-radius, radius + 1, device=masks.device, dtype=masks.dtype)
    kernel = torch.exp(-(offsets**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    values = masks.unsqueeze(1)
    horizontal = kernel.reshape(1, 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1)
    values = F.pad(values, (radius, radius, 0, 0), mode="replicate")
    values = F.conv2d(values, horizontal)
    values = F.pad(values, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(values, vertical).squeeze(1)


def stitch_video_crops(
    base_images: torch.Tensor,
    cropped_images: torch.Tensor,
    cropped_masks: torch.Tensor,
    crop_info: dict,
    feather: int,
) -> torch.Tensor:
    frames, source_height, source_width, channels = _validate_images(base_images, "base_images")
    crop_frames, _, _, crop_channels = _validate_images(cropped_images, "cropped_images")
    if crop_channels != channels:
        raise ValueError(
            f"base_images and cropped_images must have the same channel count; got {channels} and {crop_channels}"
        )
    if not isinstance(crop_info, dict) or int(crop_info.get("version", -1)) != 1:
        raise ValueError("crop_info is not a supported Video Mask Guided Crop result")
    if tuple(crop_info.get("source_size", ())) != (source_width, source_height):
        raise ValueError(
            f"crop_info source size {crop_info.get('source_size')} does not match base_images "
            f"{source_width}x{source_height}"
        )
    if int(crop_info.get("frames", -1)) != frames:
        raise ValueError(
            f"crop_info frame count {crop_info.get('frames')} does not match base_images frame count {frames}"
        )
    if crop_frames < frames:
        raise ValueError(f"cropped_images has {crop_frames} frames but {frames} are required")

    if cropped_masks.ndim == 2:
        cropped_masks = cropped_masks.unsqueeze(0)
    if cropped_masks.ndim != 3 or int(cropped_masks.shape[0]) < frames:
        shape = tuple(cropped_masks.shape) if hasattr(cropped_masks, "shape") else type(cropped_masks).__name__
        raise ValueError(f"cropped_masks must provide at least {frames} frames, got {shape}")

    boxes = crop_info.get("boxes")
    if not isinstance(boxes, (list, tuple)) or len(boxes) != frames:
        raise ValueError("crop_info boxes do not match its frame count")
    boxes = [tuple(float(value) for value in box) for box in boxes]

    cropped_images = cropped_images[:frames].to(device=base_images.device)
    cropped_masks = cropped_masks[:frames].detach().float()
    if tuple(cropped_masks.shape[-2:]) != tuple(cropped_images.shape[1:3]):
        cropped_masks = F.interpolate(
            cropped_masks.unsqueeze(1),
            size=tuple(cropped_images.shape[1:3]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    sample_dtype = _sampling_dtype(base_images)
    crop_nchw = cropped_images.movedim(-1, 1).to(dtype=sample_dtype)
    theta = _stitch_theta(
        boxes,
        source_width,
        source_height,
        device=base_images.device,
        dtype=sample_dtype,
    )
    grid = F.affine_grid(
        theta,
        size=(frames, channels, source_height, source_width),
        align_corners=False,
    )
    warped_images = F.grid_sample(
        crop_nchw,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).movedim(1, -1)
    mask_nchw = cropped_masks.to(device=base_images.device, dtype=sample_dtype).unsqueeze(1)
    warped_masks = F.grid_sample(
        mask_nchw,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(1).clamp_(0.0, 1.0)
    warped_masks = _blur_masks(warped_masks, int(feather)).clamp_(0.0, 1.0)
    alpha = warped_masks.unsqueeze(-1)
    base = base_images.to(dtype=sample_dtype)
    return (base * (1.0 - alpha) + warped_images * alpha).to(dtype=base_images.dtype)


def _aligned_target_size(
    canvas_width: float,
    canvas_height: float,
    megapixels: float,
    multiple: int,
) -> tuple[int, int]:
    if not math.isfinite(canvas_width) or not math.isfinite(canvas_height):
        raise ValueError("Outpaint canvas dimensions must be finite")
    if canvas_width <= 0.0 or canvas_height <= 0.0:
        raise ValueError("Outpaint canvas dimensions must be positive")
    if not math.isfinite(megapixels) or megapixels <= 0.0:
        raise ValueError("megapixels must be a positive finite number")
    multiple = int(multiple)
    if multiple < 1:
        raise ValueError("multiple must be positive")

    target_pixels = float(megapixels) * 1024.0 * 1024.0
    ratio = canvas_width / canvas_height
    ideal_width = math.sqrt(target_pixels * ratio)
    ideal_height = math.sqrt(target_pixels / ratio)
    width = max(multiple, int(round(ideal_width / multiple)) * multiple)
    height = max(multiple, int(round(ideal_height / multiple)) * multiple)
    return width, height


def _outpaint_canvas_geometry(
    source_width: int,
    source_height: int,
    layout: dict,
) -> tuple[float, float, float, float]:
    if not isinstance(layout, dict):
        raise ValueError("layout must be a Video Pad For Outpaint mode value")
    mode = layout.get("layout")
    if mode == "pixels":
        margins = {}
        for name in ("left", "top", "right", "bottom"):
            value = int(layout.get(name, 0))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            margins[name] = value
        return (
            float(source_width + margins["left"] + margins["right"]),
            float(source_height + margins["top"] + margins["bottom"]),
            float(margins["left"]),
            float(margins["top"]),
        )
    if mode == "relative_frame":
        expand_ratio = float(layout.get("expand_ratio", 0.0))
        offset_x = float(layout.get("offset_x", 0.0))
        offset_y = float(layout.get("offset_y", 0.0))
        if not math.isfinite(expand_ratio) or expand_ratio < 0.0:
            raise ValueError("expand_ratio must be a non-negative finite number")
        if not math.isfinite(offset_x) or not math.isfinite(offset_y):
            raise ValueError("offset_x and offset_y must be finite")

        extra_width = float(source_width) * expand_ratio
        extra_height = float(source_height) * expand_ratio
        shift_x = offset_x * float(source_width)
        shift_y = offset_y * float(source_height)
        if not bool(layout.get("allow_image_outside_frame", False)):
            shift_x = min(max(shift_x, -extra_width * 0.5), extra_width * 0.5)
            shift_y = min(max(shift_y, -extra_height * 0.5), extra_height * 0.5)
        return (
            float(source_width) + extra_width,
            float(source_height) + extra_height,
            extra_width * 0.5 + shift_x,
            extra_height * 0.5 + shift_y,
        )
    raise ValueError(f"Unsupported outpaint layout mode: {mode!r}")


def _aligned_inner_bounds(start: float, end: float, limit: int, multiple: int) -> tuple[int, int]:
    visible_start = min(float(limit), max(0.0, start))
    visible_end = min(float(limit), max(0.0, end))
    epsilon = 1e-6
    aligned_start = int(math.ceil((visible_start - epsilon) / multiple) * multiple)
    aligned_end = int(math.floor((visible_end + epsilon) / multiple) * multiple)
    return max(0, min(limit, aligned_start)), max(0, min(limit, aligned_end))


def pad_video_for_outpaint(
    images: torch.Tensor,
    layout: dict,
    megapixels: float,
    multiple: int,
    padding_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    frames, source_height, source_width, channels = _validate_images(images, "images")
    multiple = int(multiple)
    canvas_width, canvas_height, source_canvas_x, source_canvas_y = _outpaint_canvas_geometry(
        source_width,
        source_height,
        layout,
    )
    output_width, output_height = _aligned_target_size(
        canvas_width,
        canvas_height,
        float(megapixels),
        multiple,
    )

    # Use one isotropic scale and centre the small remainder introduced by
    # output-size alignment. This preserves the source and virtual-canvas aspect
    # ratios instead of independently stretching width and height.
    scale = min(output_width / canvas_width, output_height / canvas_height)
    canvas_x = (output_width - canvas_width * scale) * 0.5
    canvas_y = (output_height - canvas_height * scale) * 0.5
    source_x0 = canvas_x + source_canvas_x * scale
    source_y0 = canvas_y + source_canvas_y * scale
    source_scaled_width = float(source_width) * scale
    source_scaled_height = float(source_height) * scale
    source_x1 = source_x0 + source_scaled_width
    source_y1 = source_y0 + source_scaled_height

    sample_dtype = _sampling_dtype(images)
    x = torch.arange(output_width, device=images.device, dtype=sample_dtype) + 0.5
    y = torch.arange(output_height, device=images.device, dtype=sample_dtype) + 0.5
    grid_x = 2.0 * (x - source_x0) / source_scaled_width - 1.0
    grid_y = 2.0 * (y - source_y0) / source_scaled_height - 1.0
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    frame_grid = grid.expand(frames, -1, -1, -1)
    source = images.movedim(-1, 1).to(dtype=sample_dtype)

    if padding_mode == "edge":
        padded = F.grid_sample(
            source,
            frame_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
    elif padding_mode in ("neutral_gray", "black"):
        padded = F.grid_sample(
            source,
            frame_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        if padding_mode == "neutral_gray":
            validity = F.grid_sample(
                torch.ones(
                    (1, 1, source_height, source_width),
                    device=images.device,
                    dtype=sample_dtype,
                ),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            padded = padded + (1.0 - validity) * 0.5
    else:
        raise ValueError(f"Unsupported padding_mode: {padding_mode!r}")

    output_images = padded.movedim(1, -1).to(dtype=images.dtype)
    mask = torch.ones(
        (output_height, output_width),
        device=images.device,
        dtype=torch.float32,
    )
    protect_left, protect_right = _aligned_inner_bounds(
        source_x0,
        source_x1,
        output_width,
        multiple,
    )
    protect_top, protect_bottom = _aligned_inner_bounds(
        source_y0,
        source_y1,
        output_height,
        multiple,
    )
    if protect_right > protect_left and protect_bottom > protect_top:
        mask[protect_top:protect_bottom, protect_left:protect_right] = 0.0
    output_masks = mask.unsqueeze(0).expand(frames, -1, -1).clone()
    return output_images, output_masks


class VideoMaskGuidedCrop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsVideoMaskGuidedCrop",
            display_name="Video Mask Guided Crop",
            category="Turing Utils/video",
            description=(
                "Derive a stable, fixed-aspect in-frame crop from one mask per video frame. "
                "Missing observations are interpolated or held without introducing padded pixels."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Mask.Input("masks"),
                io.Int.Input("width", default=768, min=1, max=16384, step=8, tooltip="Output crop width and crop-box aspect numerator."),
                io.Int.Input("height", default=768, min=1, max=16384, step=8, tooltip="Output crop height and crop-box aspect denominator."),
                io.Float.Input("context_scale", default=2.5, min=1.0, max=10.0, step=0.05, tooltip="Expand the smallest target-aspect box containing the mask by this factor."),
                io.Combo.Input("missing_mode", options=["interpolate", "hold"], default="interpolate", tooltip="Interpolate bounded gaps and hold edge gaps, or always hold the last valid crop."),
                io.Int.Input("smooth_window", default=5, min=1, max=101, step=2, advanced=True, tooltip="Gaussian temporal smoothing window for crop centre and logarithmic size. 1 disables smoothing."),
                io.Float.Input("mask_threshold", default=0.5, min=0.001, max=1.0, step=0.01, advanced=True, tooltip="Mask values at or above this level define each frame's observed region."),
            ],
            outputs=[
                io.Image.Output("images"),
                io.Mask.Output("masks"),
                VideoCropInfoType.Output("crop_info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        masks,
        width,
        height,
        context_scale,
        missing_mode,
        smooth_window,
        mask_threshold,
    ) -> io.NodeOutput:
        return io.NodeOutput(*crop_video_by_mask(
            images,
            masks,
            width,
            height,
            context_scale,
            missing_mode,
            smooth_window,
            mask_threshold,
        ))


class VideoMaskGuidedStitch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsVideoMaskGuidedStitch",
            display_name="Video Mask Guided Stitch",
            category="Turing Utils/video",
            description=(
                "Map regenerated crops and their masks back through Video Mask Guided Crop's "
                "per-frame float transforms and composite them over the original video."
            ),
            inputs=[
                io.Image.Input("base_images"),
                io.Image.Input("cropped_images"),
                io.Mask.Input("cropped_masks"),
                VideoCropInfoType.Input("crop_info"),
                io.Int.Input("feather", default=8, min=0, max=256, step=1, tooltip="Gaussian blend radius in source-video pixels. 0 keeps the supplied mask unchanged."),
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(cls, base_images, cropped_images, cropped_masks, crop_info, feather) -> io.NodeOutput:
        return io.NodeOutput(stitch_video_crops(
            base_images,
            cropped_images,
            cropped_masks,
            crop_info,
            feather,
        ))


class VideoPadForOutpaint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsVideoPadForOutpaint",
            display_name="Video Pad For Outpaint",
            category="Turing Utils/video",
            description=(
                "Place a video in a larger target-resolution canvas and create a hard outpaint mask. "
                "Pixel margins and aspect-preserving relative placement use separate dynamic controls."
            ),
            inputs=[
                io.Image.Input("images"),
                io.DynamicCombo.Input(
                    "layout",
                    options=[
                        io.DynamicCombo.Option("pixels", [
                            io.Int.Input("left", default=0, min=0, max=16384, step=1),
                            io.Int.Input("top", default=0, min=0, max=16384, step=1),
                            io.Int.Input("right", default=0, min=0, max=16384, step=1),
                            io.Int.Input("bottom", default=0, min=0, max=16384, step=1),
                        ]),
                        io.DynamicCombo.Option("relative_frame", [
                            io.Float.Input(
                                "expand_ratio",
                                default=0.25,
                                min=0.0,
                                max=8.0,
                                step=0.01,
                                tooltip=(
                                    "Increase both canvas dimensions by this fraction of the original. "
                                    "0.25 creates a 1.25x canvas while preserving its aspect ratio."
                                ),
                            ),
                            io.Float.Input(
                                "offset_x",
                                default=0.0,
                                min=-8.0,
                                max=8.0,
                                step=0.01,
                                tooltip="Horizontal source offset as a fraction of the original width; positive moves right.",
                            ),
                            io.Float.Input(
                                "offset_y",
                                default=0.0,
                                min=-8.0,
                                max=8.0,
                                step=0.01,
                                tooltip="Vertical source offset as a fraction of the original height; positive moves down.",
                            ),
                            io.Boolean.Input(
                                "allow_image_outside_frame",
                                default=False,
                                tooltip=(
                                    "Allow placement to crop the original at the output boundary. "
                                    "When disabled, offsets are clamped so the whole original remains visible."
                                ),
                            ),
                        ]),
                    ],
                    tooltip="Use explicit source-pixel margins or expand the current aspect ratio and reposition the source.",
                ),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.01,
                    max=64.0,
                    step=0.01,
                    tooltip="Approximate pixel count of the final canvas in 1024x1024 megapixels.",
                ),
                io.Int.Input(
                    "multiple",
                    default=32,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip=(
                        "Align final width, height, and the protected source rectangle inward to this pixel multiple. "
                        "32 matches MiniMax H3's effective spatial token grid."
                    ),
                ),
                io.Combo.Input(
                    "padding_mode",
                    options=["edge", "neutral_gray", "black"],
                    default="edge",
                    tooltip="Pixels outside the original are edge-extended, neutral gray, or black before repainting.",
                    advanced=True,
                ),
            ],
            outputs=[
                io.Image.Output("images"),
                io.Mask.Output("masks"),
            ],
        )

    @classmethod
    def execute(cls, images, layout, megapixels, multiple, padding_mode) -> io.NodeOutput:
        return io.NodeOutput(*pad_video_for_outpaint(
            images,
            layout,
            megapixels,
            multiple,
            padding_mode,
        ))
