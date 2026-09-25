"""Derive reusable point, box, and mask prompts from one image/mask frame."""

from __future__ import annotations

import json
import math

import numpy as np
import torch
from PIL import Image as PILImage, ImageDraw
from scipy import ndimage

from comfy_api.latest import io


def _select_first_image_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(image) or image.ndim != 4 or int(image.shape[-1]) < 1:
        shape = tuple(image.shape) if hasattr(image, "shape") else type(image).__name__
        raise ValueError(f"image must be [frames,height,width,channels], got {shape}")
    if not torch.is_tensor(mask) or mask.ndim != 3:
        shape = tuple(mask.shape) if hasattr(mask, "shape") else type(mask).__name__
        raise ValueError(f"mask must be [frames,height,width], got {shape}")

    if int(image.shape[0]) < 1 or int(mask.shape[0]) < 1:
        raise ValueError("image and mask must contain at least one frame")
    if tuple(image.shape[1:3]) != tuple(mask.shape[1:3]):
        raise ValueError(
            "image and mask spatial dimensions must match, got "
            f"{tuple(image.shape[1:3])} and {tuple(mask.shape[1:3])}"
        )

    selected_image = image[:1]
    selected_mask = mask[:1]
    if not selected_mask.is_floating_point():
        selected_mask = selected_mask.float()
    return selected_image, selected_mask


def _spread_points(
    score: np.ndarray,
    count: int,
    initial_points: tuple[tuple[int, int], ...] | list[tuple[int, int]] = (),
) -> list[tuple[int, int]]:
    """Select high-score points while progressively favouring spatial coverage."""

    count = max(0, int(count))
    ys, xs = np.nonzero(score > 0.0)
    if count == 0 or not len(xs):
        return []

    candidates = np.column_stack((ys, xs)).astype(np.float64, copy=False)
    weights = score[ys, xs].astype(np.float64, copy=False)
    blocked = np.zeros(len(candidates), dtype=bool)
    min_distance_sq: np.ndarray | None = None
    diagonal_sq = max(float(score.shape[0] ** 2 + score.shape[1] ** 2), 1.0)

    for x, y in initial_points:
        delta = candidates - np.asarray((y, x), dtype=np.float64)
        distance_sq = np.sum(delta * delta, axis=1)
        min_distance_sq = (
            distance_sq
            if min_distance_sq is None
            else np.minimum(min_distance_sq, distance_sq)
        )
        blocked |= (xs == int(x)) & (ys == int(y))

    result: list[tuple[int, int]] = []
    for _ in range(min(count, len(candidates))):
        if min_distance_sq is None:
            ranking = weights.copy()
        else:
            # Interior depth keeps points away from uncertain boundaries; the
            # distance factor prevents all points collapsing into one thick area.
            coverage = np.sqrt(min_distance_sq / diagonal_sq)
            ranking = weights * (0.25 + coverage)
        ranking[blocked] = -1.0
        choice = int(np.argmax(ranking))
        if ranking[choice] < 0.0:
            break
        blocked[choice] = True
        result.append((int(xs[choice]), int(ys[choice])))
        delta = candidates - candidates[choice]
        distance_sq = np.sum(delta * delta, axis=1)
        min_distance_sq = (
            distance_sq
            if min_distance_sq is None
            else np.minimum(min_distance_sq, distance_sq)
        )
    return result


def _bounded_inside_distance(binary: np.ndarray) -> np.ndarray:
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    return ndimage.distance_transform_edt(padded)[1:-1, 1:-1]


def _perceptual_image(image: torch.Tensor, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim == 4:
        if int(image.shape[0]) != 1:
            raise ValueError(f"expected one selected image, got {int(image.shape[0])}")
        image = image[0]
    if image.ndim != 3 or tuple(image.shape[:2]) != (height, width):
        raise ValueError(
            "selected image must be [height,width,channels] and match the mask, got "
            f"{tuple(image.shape)}"
        )
    values = image.detach().float().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("image contains non-finite values")
    if values.shape[-1] == 1:
        rgb = np.repeat(values, 3, axis=-1)
    else:
        rgb = values[..., :3]
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32, copy=False)

    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.asarray(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    ).T
    xyz /= np.asarray((0.95047, 1.0, 1.08883), dtype=np.float32)
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3.0 * delta * delta) + 4.0 / 29.0,
    )
    lab = np.empty_like(transformed)
    lab[..., 0] = (116.0 * transformed[..., 1] - 16.0) / 100.0
    lab[..., 1] = (500.0 * (transformed[..., 0] - transformed[..., 1])) / 128.0
    lab[..., 2] = (200.0 * (transformed[..., 1] - transformed[..., 2])) / 128.0
    return rgb, lab


def _color_region_points(
    rgb: np.ndarray,
    lab: np.ndarray,
    binary: np.ndarray,
    interior_depth: np.ndarray,
    count: int,
    initial_points: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Choose stable representatives from perceptually distinct colour regions."""

    count = min(max(0, int(count)), 12)
    ys, xs = np.nonzero(binary)
    if count == 0 or not len(xs):
        return []
    pixels = lab[ys, xs]
    quantized = np.rint(rgb[ys, xs] * 15.0).astype(np.int16)
    keys = (
        quantized[:, 0].astype(np.int32) * 256
        + quantized[:, 1].astype(np.int32) * 16
        + quantized[:, 2].astype(np.int32)
    )
    _, inverse, populations = np.unique(keys, return_inverse=True, return_counts=True)
    bins = len(populations)
    bin_features = np.column_stack(
        [np.bincount(inverse, weights=pixels[:, channel], minlength=bins) for channel in range(3)]
    ) / populations[:, None]

    seed_features = [lab[y, x] for x, y in initial_points]
    if seed_features:
        delta = bin_features[:, None, :] - np.asarray(seed_features)[None, :, :]
        min_distance_sq = np.min(np.sum(delta * delta, axis=2), axis=1)
    else:
        mean = np.average(bin_features, axis=0, weights=populations)
        min_distance_sq = np.sum((bin_features - mean) ** 2, axis=1)
    popularity = np.power(populations / max(int(populations.max()), 1), 0.25)
    chosen_bins: list[int] = []
    for _ in range(min(count, bins)):
        ranking = min_distance_sq * (0.35 + 0.65 * popularity)
        if chosen_bins:
            ranking[np.asarray(chosen_bins, dtype=np.int64)] = -1.0
        choice = int(np.argmax(ranking))
        if ranking[choice] <= 1e-12:
            break
        chosen_bins.append(choice)
        distance_sq = np.sum((bin_features - bin_features[choice]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
    if not chosen_bins:
        return []

    centres = bin_features[np.asarray(chosen_bins, dtype=np.int64)]
    bin_distances = np.sum((bin_features[:, None, :] - centres[None, :, :]) ** 2, axis=2)
    assignments = np.argmin(bin_distances, axis=1)[inverse]
    result: list[tuple[int, int]] = []
    existing = {(int(x), int(y)) for x, y in initial_points}
    for cluster in range(len(chosen_bins)):
        members = np.nonzero(assignments == cluster)[0]
        if not len(members):
            continue
        region_y = ys[members]
        region_x = xs[members]
        y0, y1 = int(region_y.min()), int(region_y.max()) + 1
        x0, x1 = int(region_x.min()), int(region_x.max()) + 1
        region = np.zeros((y1 - y0 + 2, x1 - x0 + 2), dtype=bool)
        region[region_y - y0 + 1, region_x - x0 + 1] = True
        region_depth = ndimage.distance_transform_edt(region)[1:-1, 1:-1]
        global_depth = interior_depth[y0:y1, x0:x1]
        global_scale = max(float(global_depth.max()), 1.0)
        score = np.where(
            region[1:-1, 1:-1],
            region_depth + 0.05 * global_depth / global_scale,
            -1.0,
        )
        local_y, local_x = np.unravel_index(int(np.argmax(score)), score.shape)
        point = (x0 + int(local_x), y0 + int(local_y))
        if point not in existing:
            existing.add(point)
            result.append(point)
    return result


def visual_prompts_from_mask(
    mask: torch.Tensor,
    mask_threshold: float,
    positive_point_count: int,
    negative_point_count: int,
    bbox_padding: float,
    image: torch.Tensor | None = None,
) -> tuple[str, str, list[dict], list[list[dict]]]:
    if mask.ndim == 3:
        if int(mask.shape[0]) != 1:
            raise ValueError(f"expected one selected mask, got {int(mask.shape[0])}")
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"selected mask must resolve to [height,width], got {tuple(mask.shape)}")
    if not 0.0 < mask_threshold <= 1.0:
        raise ValueError("mask_threshold must be in (0, 1]")
    if positive_point_count < 1:
        raise ValueError("positive_point_count must be at least 1")
    if negative_point_count < 0:
        raise ValueError("negative_point_count must be non-negative")
    if bbox_padding < 0.0:
        raise ValueError("bbox_padding must be non-negative")

    values = mask.detach().float().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("mask contains non-finite values")
    binary = values >= float(mask_threshold)
    ys, xs = np.nonzero(binary)
    if not len(xs):
        raise ValueError("the selected mask contains no foreground at mask_threshold")

    height, width = binary.shape
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    pad_x = int(math.ceil((x1 - x0) * float(bbox_padding)))
    pad_y = int(math.ceil((y1 - y0) * float(bbox_padding)))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)

    interior_depth = _bounded_inside_distance(binary)
    anchor_y, anchor_x = np.unravel_index(int(np.argmax(interior_depth)), interior_depth.shape)
    positive = [(int(anchor_x), int(anchor_y))]
    remaining = max(0, int(positive_point_count) - 1)
    thin_budget = max(1, int(math.floor(remaining * 0.4))) if remaining >= 2 else 0
    color_budget = remaining - thin_budget

    color_candidates: list[tuple[int, int]] = []
    if image is not None and color_budget > 0:
        rgb, lab = _perceptual_image(image, height, width)
        color_candidates = _color_region_points(
            rgb,
            lab,
            binary,
            interior_depth,
            max(color_budget * 2, color_budget),
            positive,
        )
        for point in color_candidates:
            if len(positive) >= 1 + color_budget:
                break
            if point not in positive:
                positive.append(point)

    if thin_budget > 0:
        local_max = ndimage.maximum_filter(interior_depth, size=3, mode="constant")
        ridge = binary & (interior_depth >= local_max - 1e-6)
        thin_score = np.where(ridge, 1.0 / (interior_depth + 0.25), 0.0)
        positive.extend(_spread_points(thin_score, thin_budget, positive))

    for point in color_candidates:
        if len(positive) >= int(positive_point_count):
            break
        if point not in positive:
            positive.append(point)
    if len(positive) < int(positive_point_count):
        positive.extend(
            _spread_points(
                interior_depth,
                int(positive_point_count) - len(positive),
                positive,
            )
        )

    negative: list[tuple[int, int]] = []
    if negative_point_count > 0 and not bool(binary.all()):
        outside_depth = ndimage.distance_transform_edt(~binary)
        object_scale = max(x1 - x0, y1 - y0)
        ring_radius = max(2.0, min(50.0, object_scale * 0.15))
        ring_score = np.where(
            (outside_depth > 0.0) & (outside_depth <= ring_radius),
            outside_depth,
            0.0,
        )
        negative = _spread_points(ring_score, negative_point_count)

    def encode(points: list[tuple[int, int]]) -> str:
        return json.dumps(
            [{"x": x, "y": y} for x, y in points],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    # Preserve the legacy KJNodes BBOX contract for older consumers.
    legacy_bbox = [{"startX": x0, "startY": y0, "endX": x1, "endY": y1}]
    # ComfyUI's built-in SAM3 uses the canonical BOUNDING_BOX contract, nested
    # once because this node always outputs exactly one selected frame.
    canonical_bbox = [[{"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}]]
    return encode(positive), encode(negative), legacy_bbox, canonical_bbox


def _render_prompt_preview(
    image: torch.Tensor,
    mask: torch.Tensor,
    mask_threshold: float,
    positive_coords: str,
    negative_coords: str,
    bbox: list[dict],
) -> torch.Tensor:
    """Render the exact point and box prompts over the first input frame."""

    height, width = int(mask.shape[-2]), int(mask.shape[-1])
    rgb, _ = _perceptual_image(image, height, width)
    pixels = np.rint(rgb * 255.0).astype(np.uint8)
    mask_values = mask[0].detach().float().cpu().numpy()
    binary = mask_values >= float(mask_threshold)

    # A restrained teal tint and a brighter contour make the source region
    # legible without hiding the image content that drives colour-aware points.
    tint = np.asarray((36, 211, 196), dtype=np.float32)
    blended = pixels.astype(np.float32)
    blended[binary] = blended[binary] * 0.86 + tint * 0.14
    boundary = binary & ~ndimage.binary_erosion(binary)
    boundary = ndimage.binary_dilation(boundary, iterations=1)
    blended[boundary] = blended[boundary] * 0.30 + tint * 0.70
    canvas = PILImage.fromarray(np.rint(blended).clip(0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")

    short_side = max(1, min(height, width))
    line_width = max(2, min(7, int(round(short_side * 0.005))))
    point_radius = max(5, min(16, int(round(short_side * 0.012))))

    box = bbox[0]
    x0, y0 = int(box["startX"]), int(box["startY"])
    x1, y1 = int(box["endX"]) - 1, int(box["endY"]) - 1
    radius = max(2, min(point_radius, (x1 - x0) // 5, (y1 - y0) // 5))
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=radius,
        outline=(0, 0, 0, 185),
        width=line_width + 3,
    )
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=radius,
        outline=(255, 201, 71, 255),
        width=line_width,
    )

    positive = json.loads(positive_coords)
    negative = json.loads(negative_coords)

    def marker(point: dict, *, positive_point: bool) -> None:
        x, y = int(round(point["x"])), int(round(point["y"]))
        outer = point_radius + 3
        draw.ellipse(
            (x - outer, y - outer, x + outer, y + outer),
            fill=(0, 0, 0, 175),
        )
        draw.ellipse(
            (
                x - point_radius - 1,
                y - point_radius - 1,
                x + point_radius + 1,
                y + point_radius + 1,
            ),
            fill=(255, 255, 255, 245),
        )
        fill = (28, 214, 126, 255) if positive_point else (255, 72, 101, 255)
        draw.ellipse(
            (
                x - point_radius,
                y - point_radius,
                x + point_radius,
                y + point_radius,
            ),
            fill=fill,
        )
        symbol_extent = max(2, int(round(point_radius * 0.48)))
        symbol_width = max(2, line_width // 2 + 1)
        if positive_point:
            draw.line(
                (x - symbol_extent, y, x + symbol_extent, y),
                fill=(255, 255, 255, 255),
                width=symbol_width,
            )
            draw.line(
                (x, y - symbol_extent, x, y + symbol_extent),
                fill=(255, 255, 255, 255),
                width=symbol_width,
            )
        else:
            draw.line(
                (x - symbol_extent, y - symbol_extent, x + symbol_extent, y + symbol_extent),
                fill=(255, 255, 255, 255),
                width=symbol_width,
            )
            draw.line(
                (x - symbol_extent, y + symbol_extent, x + symbol_extent, y - symbol_extent),
                fill=(255, 255, 255, 255),
                width=symbol_width,
            )

    for point in positive:
        marker(point, positive_point=True)
    for point in negative:
        marker(point, positive_point=False)

    result = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(result.copy()).unsqueeze(0)


class MaskToVisualPrompts(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsMaskToVisualPrompts",
            display_name="Mask to Visual Prompts",
            category="Turing Utils/mask",
            description=(
                "Use the first IMAGE/MASK frame to derive reusable positive/negative point "
                "JSON, legacy KJ BBOX, and canonical SeC/SAM3 BOUNDING_BOX prompts. Positive points "
                "cover perceptually distinct colour regions and prioritise thin mask structures. "
                "The preview renders the thresholded mask, prompts, and box over the source image."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Float.Input("mask_threshold", default=0.5, min=0.001, max=1.0, step=0.01, advanced=True),
                io.Int.Input("positive_point_count", default=3, min=1, max=32, step=1, advanced=True),
                io.Int.Input("negative_point_count", default=4, min=0, max=32, step=1, advanced=True),
                io.Float.Input("bbox_padding", default=0.05, min=0.0, max=1.0, step=0.01, advanced=True, tooltip="Padding on each side as a fraction of the tight mask bounding-box size."),
            ],
            outputs=[
                io.Image.Output("preview"),
                io.String.Output("positive_coords"),
                io.String.Output("negative_coords"),
                io.BBOX.Output("bbox"),
                io.BoundingBox.Output("bounding_box"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        mask,
        mask_threshold,
        positive_point_count,
        negative_point_count,
        bbox_padding,
    ) -> io.NodeOutput:
        selected_image, selected_mask = _select_first_image_mask(
            image,
            mask,
        )
        prompts = visual_prompts_from_mask(
            selected_mask,
            float(mask_threshold),
            int(positive_point_count),
            int(negative_point_count),
            float(bbox_padding),
            selected_image,
        )
        preview = _render_prompt_preview(
            selected_image,
            selected_mask,
            float(mask_threshold),
            prompts[0],
            prompts[1],
            prompts[2],
        )
        return io.NodeOutput(preview, *prompts)
