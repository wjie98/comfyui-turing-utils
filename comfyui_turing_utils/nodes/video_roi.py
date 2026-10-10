"""Video region-of-interest, outpaint, and inverse-composite nodes."""

from __future__ import annotations

from comfy_api.latest import io

VideoCropInfoType = io.Custom("TURING_UTILS_VIDEO_CROP_INFO")
from ..media.geometry import (
    crop_video_by_mask,
    stitch_video_crops,
    pad_video_for_outpaint,
)


class VideoMaskGuidedCrop(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsVideoMaskGuidedCrop",
            display_name="Video Mask Guided Crop",
            category="Turing Utils/Video",
            description=(
                "Derive a stable, fixed-aspect in-frame crop from one mask per video frame. "
                "Missing observations are interpolated or held without introducing padded pixels."
            ),
            inputs=[
                io.Image.Input("images"),
                io.Mask.Input("masks"),
                io.Int.Input(
                    "width",
                    default=768,
                    min=1,
                    max=16384,
                    step=8,
                    tooltip="Output crop width and crop-box aspect numerator.",
                ),
                io.Int.Input(
                    "height",
                    default=768,
                    min=1,
                    max=16384,
                    step=8,
                    tooltip="Output crop height and crop-box aspect denominator.",
                ),
                io.Float.Input(
                    "context_scale",
                    default=2.5,
                    min=1.0,
                    max=10.0,
                    step=0.05,
                    tooltip="Expand the smallest target-aspect box containing the mask by this factor.",
                ),
                io.Combo.Input(
                    "missing_mode",
                    options=["interpolate", "hold"],
                    default="interpolate",
                    tooltip="Interpolate bounded gaps and hold edge gaps, or always hold the last valid crop.",
                ),
                io.Int.Input(
                    "smooth_window",
                    default=5,
                    min=1,
                    max=101,
                    step=2,
                    tooltip="Gaussian temporal smoothing window for crop centre and logarithmic size. 1 disables smoothing.",
                ),
                io.Float.Input(
                    "mask_threshold",
                    default=0.5,
                    min=0.001,
                    max=1.0,
                    step=0.01,
                    tooltip="Mask values at or above this level define each frame's observed region.",
                ),
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
        return io.NodeOutput(
            *crop_video_by_mask(
                images,
                masks,
                width,
                height,
                context_scale,
                missing_mode,
                smooth_window,
                mask_threshold,
            )
        )


class VideoMaskGuidedStitch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsVideoMaskGuidedStitch",
            display_name="Video Mask Guided Stitch",
            category="Turing Utils/Video",
            description=(
                "Map regenerated crops and their masks back through Video Mask Guided Crop's "
                "per-frame float transforms and composite them over the original video."
            ),
            inputs=[
                io.Image.Input("base_images"),
                io.Image.Input("cropped_images"),
                io.Mask.Input("cropped_masks"),
                VideoCropInfoType.Input("crop_info"),
                io.Int.Input(
                    "feather",
                    default=8,
                    min=0,
                    max=256,
                    step=1,
                    tooltip="Gaussian blend radius in source-video pixels. 0 keeps the supplied mask unchanged.",
                ),
            ],
            outputs=[io.Image.Output("images")],
        )

    @classmethod
    def execute(
        cls, base_images, cropped_images, cropped_masks, crop_info, feather
    ) -> io.NodeOutput:
        return io.NodeOutput(
            stitch_video_crops(
                base_images,
                cropped_images,
                cropped_masks,
                crop_info,
                feather,
            )
        )


class VideoPadForOutpaint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsVideoPadForOutpaint",
            display_name="Video Pad For Outpaint",
            category="Turing Utils/Video",
            description=(
                "Place a video in a larger target-resolution canvas and create a hard outpaint mask. "
                "Pixel margins and aspect-preserving relative placement use separate dynamic controls."
            ),
            inputs=[
                io.Image.Input("images"),
                io.DynamicCombo.Input(
                    "layout",
                    options=[
                        io.DynamicCombo.Option(
                            "pixels",
                            [
                                io.Int.Input(
                                    "left", default=0, min=0, max=16384, step=1
                                ),
                                io.Int.Input(
                                    "top", default=0, min=0, max=16384, step=1
                                ),
                                io.Int.Input(
                                    "right", default=0, min=0, max=16384, step=1
                                ),
                                io.Int.Input(
                                    "bottom", default=0, min=0, max=16384, step=1
                                ),
                            ],
                        ),
                        io.DynamicCombo.Option(
                            "relative_frame",
                            [
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
                            ],
                        ),
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
                ),
            ],
            outputs=[
                io.Image.Output("images"),
                io.Mask.Output("masks"),
            ],
        )

    @classmethod
    def execute(
        cls, images, layout, megapixels, multiple, padding_mode
    ) -> io.NodeOutput:
        return io.NodeOutput(
            *pad_video_for_outpaint(
                images,
                layout,
                megapixels,
                multiple,
                padding_mode,
            )
        )
