"""Frame-grid padding shared by video preprocessing and mask-only branches."""

from __future__ import annotations

from comfy_api.latest import io
from ..media.temporal import VIDEO_MASK_SPECS
from ..media.temporal import (
    repeat_last_frame,
    validate_frame_mask,
    padded_frame_count,
)


class VideoFramesPadding(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsVideoFramesPadding",
            display_name="Video Frames Padding",
            category="Turing Utils/Video",
            description="Repeat the final frame to the selected VAE frame grid. Accepts images, masks, or both. Use separate image and mask branches to keep VAE encoding cached when masks change. No inputs passes through as absent.",
            inputs=[
                io.Image.Input("image", optional=True),
                io.Combo.Input(
                    "type", options=list(VIDEO_MASK_SPECS), default="minimax"
                ),
                io.Int.Input(
                    "target_frame_count",
                    default=0,
                    min=0,
                    max=16385,
                    tooltip="0 rounds up. Wan/Hunyuan: 4*n+1; LTX: 8*n+1; Mochi: 6*n+1 (minimum 7); H3: 17*n+5.",
                ),
                io.Mask.Input("mask", optional=True),
            ],
            outputs=[
                io.Image.Output("image"),
                io.Mask.Output("mask"),
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.Int.Output("length"),
                io.Int.Output("input_length"),
            ],
        )

    @classmethod
    def execute(cls, type="minimax", target_frame_count=0, image=None, mask=None):
        if type not in VIDEO_MASK_SPECS:
            raise ValueError(f"Unknown video type: {type!r}")
        if image is None and mask is None:
            return io.NodeOutput(None, None, 0, 0, 0, 0)
        if image is not None:
            if image.ndim != 4 or image.shape[-1] < 3:
                raise ValueError("IMAGE must have shape [frames,height,width,channels]")
            count, height, width = image.shape[:3]
        else:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim != 3:
                raise ValueError("MASK must have shape [frames,height,width]")
            count, height, width = mask.shape
        if min(count, height, width) < 1:
            raise ValueError("Video frames must be non-empty")
        mask = validate_frame_mask(mask, count, height, width)
        length = padded_frame_count(count, type, int(target_frame_count))
        return io.NodeOutput(
            repeat_last_frame(image, length - count) if image is not None else None,
            repeat_last_frame(mask, length - count) if mask is not None else None,
            width,
            height,
            length,
            count,
        )
