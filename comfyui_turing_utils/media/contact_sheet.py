"""Contact-sheet rendering independent of node schemas."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .images import _resolved_size


def _split_extent(total: int, count: int, gap: int) -> list[tuple[int, int]]:
    usable = total - gap * (count - 1)
    if usable < count:
        raise ValueError(
            f"Output extent {total} is too small for {count} cells and gap={gap}"
        )
    base, remainder = divmod(usable, count)
    segments = []
    position = 0
    for index in range(count):
        size = base + (1 if index < remainder else 0)
        segments.append((position, size))
        position += size + gap
    return segments


def _fit_image(
    image: Image.Image, width: int, height: int, resize_mode: str
) -> Image.Image:
    if resize_mode == "stretch":
        return image.resize((width, height), Image.Resampling.LANCZOS)
    if resize_mode == "crop":
        return ImageOps.fit(
            image, (width, height), Image.Resampling.LANCZOS, centering=(0.5, 0.5)
        )
    resized = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
    output = Image.new("RGB", (width, height), (8, 8, 8))
    output.paste(
        resized, ((width - resized.width) // 2, (height - resized.height) // 2)
    )
    return output


def _label(annotation: str, index: int, timestamp: float) -> str:
    if annotation == "index":
        return f"{index + 1:02d}"
    if annotation == "timestamp":
        return f"{timestamp:.2f}s"
    if annotation == "index_timestamp":
        return f"{index + 1:02d} | {timestamp:.2f}s"
    return ""


def _draw_centered_label(
    draw: ImageDraw.ImageDraw, label: str, y0: int, width: int, height: int
):
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


def _draw_perforations(
    draw: ImageDraw.ImageDraw,
    width: int,
    tile_height: int,
    rail_height: int,
    label_span,
):
    hole_height = max(2, rail_height // 3)
    hole_width = max(4, round(hole_height * 1.7))
    step = max(hole_width + 3, round(hole_width * 1.8))
    top_y = max(1, (rail_height - hole_height) // 2)
    bottom_y = tile_height - rail_height + top_y
    for x in range(3, width - hole_width, step):
        draw.rounded_rectangle(
            (x, top_y, x + hole_width, top_y + hole_height),
            radius=max(1, hole_height // 4),
            fill=(224, 216, 188),
        )
        if (
            label_span is None
            or x + hole_width < label_span[0] - 4
            or x > label_span[1] + 4
        ):
            draw.rounded_rectangle(
                (x, bottom_y, x + hole_width, bottom_y + hole_height),
                radius=max(1, hole_height // 4),
                fill=(224, 216, 188),
            )


def _render_tile(
    frame: torch.Tensor,
    width: int,
    height: int,
    resize_mode: str,
    film_border: bool,
    label: str,
) -> Image.Image:
    if width < 8 or height < 8:
        raise ValueError(
            f"Contact-sheet cells must be at least 8x8 pixels, got {width}x{height}"
        )
    array = frame.detach().float().clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
    source = Image.fromarray(array[..., :3])
    tile = Image.new("RGB", (width, height), (9, 9, 8))
    draw = ImageDraw.Draw(tile)

    if film_border:
        rail_height = min(max(10, round(height * 0.09)), max(10, height // 4))
        viewport_height = height - rail_height * 2
        if viewport_height < 8:
            raise ValueError(f"Cell height {height} is too small for the film border")
        tile.paste(
            _fit_image(source, width, viewport_height, resize_mode), (0, rail_height)
        )
        draw.rectangle(
            (0, rail_height, width - 1, height - rail_height - 1), outline=(65, 62, 52)
        )
        label_span = _draw_centered_label(
            draw, label, height - rail_height, width, rail_height
        )
        _draw_perforations(draw, width, height, rail_height, label_span)
    elif label:
        caption_height = min(max(10, round(height * 0.08)), max(10, height // 4))
        tile.paste(
            _fit_image(source, width, height - caption_height, resize_mode), (0, 0)
        )
        _draw_centered_label(
            draw, label, height - caption_height, width, caption_height
        )
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
        tile = _render_tile(
            frame,
            tile_width,
            tile_height,
            resize_mode,
            film_border,
            _label(annotation, index, timestamps[index]),
        )
        sheet.paste(tile, (x, y))
    array = np.array(sheet, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)
