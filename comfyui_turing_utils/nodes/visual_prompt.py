"""Derive reusable point, box, and mask prompts from one image/mask frame."""

from __future__ import annotations

from comfy_api.latest import io
from ..media.visual_prompts import (
    _select_first_image_mask,
    visual_prompts_from_mask,
    _render_prompt_preview,
)


class MaskToVisualPrompts(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsMaskToVisualPrompts",
            display_name="Mask to Visual Prompts",
            category="Turing Utils/Mask",
            description=(
                "Use the first IMAGE/MASK frame to derive reusable positive/negative point "
                "JSON, legacy KJ BBOX, and canonical SeC/SAM3 BOUNDING_BOX prompts. Positive points "
                "cover perceptually distinct colour regions and prioritise thin mask structures. "
                "The preview renders the thresholded mask, prompts, and box over the source image."
            ),
            inputs=[
                io.Image.Input("image"),
                io.Mask.Input("mask"),
                io.Float.Input(
                    "mask_threshold", default=0.5, min=0.001, max=1.0, step=0.01
                ),
                io.Int.Input("positive_point_count", default=3, min=1, max=32, step=1),
                io.Int.Input("negative_point_count", default=4, min=0, max=32, step=1),
                io.Float.Input(
                    "bbox_padding",
                    default=0.05,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Padding on each side as a fraction of the tight mask bounding-box size.",
                ),
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
