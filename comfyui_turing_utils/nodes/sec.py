"""SeC visual-concept video tracking nodes."""

from __future__ import annotations

from comfy_api.latest import io

from ..adapters.sec import load_sec_model, sec_model_choices, track_visual_concept


SeCModelType = io.Custom("TURING_UTILS_SEC_MODEL")


class SeCModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        choices = sec_model_choices()
        return io.Schema(
            node_id="TuringUtilsSeCModelLoader",
            display_name="Load SeC Model",
            category="Turing Utils/SeC",
            description=(
                "Load a SeC visual-concept tracking model through ComfyUI's model "
                "lifecycle. Device placement, residency, and unloading are managed by ComfyUI."
            ),
            inputs=[
                io.Combo.Input("model_name", options=choices, default=choices[0]),
                io.Combo.Input(
                    "attention",
                    options=["auto", "sdpa"],
                    default="auto",
                    tooltip=(
                        "Auto selects a compatible Flash Attention implementation independently for "
                        "the vision and language encoders, then falls back to SDPA. SDPA forces the "
                        "portable PyTorch backend throughout SeC."
                    ),
                ),
            ],
            outputs=[SeCModelType.Output("model")],
        )

    @classmethod
    def execute(
        cls,
        model_name: str,
        attention: str = "auto",
    ) -> io.NodeOutput:
        return io.NodeOutput(load_sec_model(model_name, attention))


class SeCTrackVisualConcept(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsSeCTrackVisualConcept",
            display_name="SeC Track Visual Concept",
            category="Turing Utils/SeC",
            description=(
                "Track one visual concept through a video. With mask connected, the mask is "
                "authoritative and points must agree with it; the bounding box limits its region. "
                "Without a mask, the box and positive/negative coordinates form one SAM2 prompt."
            ),
            inputs=[
                SeCModelType.Input("model"),
                io.Image.Input("frames"),
                io.String.Input(
                    "positive_coords",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip='JSON point list such as [{"x": 120, "y": 240}].',
                ),
                io.String.Input(
                    "negative_coords",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip='JSON exclusion-point list such as [{"x": 80, "y": 200}].',
                ),
                io.BoundingBox.Input(
                    "bounding_box",
                    optional=True,
                    force_input=True,
                    tooltip="Canonical ComfyUI BOUNDING_BOX prompt.",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "One mask or one mask per video frame. If batched, the annotation-frame mask "
                        "is selected. Mask prompt state takes precedence over points because SAM2 "
                        "cannot retain mask and click state simultaneously."
                    ),
                ),
                io.Combo.Input(
                    "tracking_direction",
                    options=["forward", "backward", "bidirectional"],
                    default="forward",
                ),
                io.Int.Input("annotation_frame_idx", default=0, min=0, max=1_000_000, step=1),
                io.Int.Input(
                    "max_frames_to_track",
                    default=-1,
                    min=-1,
                    max=1_000_000,
                    step=1,
                    advanced=True,
                    tooltip="-1 tracks every reachable frame in the selected direction.",
                ),
                io.Int.Input(
                    "semantic_keyframes",
                    default=7,
                    min=1,
                    max=20,
                    step=1,
                    advanced=True,
                    tooltip=(
                        "Maximum semantic keyframes retained for scene-change recovery. Larger values "
                        "can improve concept recovery but also increase MLLM activation memory."
                    ),
                ),
            ],
            outputs=[io.Mask.Output("masks")],
        )

    @classmethod
    def execute(
        cls,
        model,
        frames,
        positive_coords="",
        negative_coords="",
        bounding_box=None,
        mask=None,
        tracking_direction="forward",
        annotation_frame_idx=0,
        max_frames_to_track=-1,
        semantic_keyframes=7,
    ) -> io.NodeOutput:
        masks = track_visual_concept(
            model,
            frames,
            positive_coords=positive_coords,
            negative_coords=negative_coords,
            bounding_box=bounding_box,
            mask=mask,
            tracking_direction=tracking_direction,
            annotation_frame_idx=int(annotation_frame_idx),
            max_frames_to_track=int(max_frames_to_track),
            semantic_keyframes=int(semantic_keyframes),
        )
        return io.NodeOutput(masks)


__all__ = ["SeCModelLoader", "SeCTrackVisualConcept"]
