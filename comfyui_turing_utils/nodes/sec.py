"""SeC visual-concept video tracking nodes."""

from __future__ import annotations

from comfy_api.latest import io

from ..adapters.sec import load_sec_model, sec_model_choices, track_visual_concept


SeCModelType = io.Custom("TURING_UTILS_SEC_MODEL")


class _SeCLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        choices = sec_model_choices()
        return io.Schema(
            node_id="_TuringUtilsSeCLoader",
            display_name="SeC Loader (Internal)",
            is_dev_only=True,
            category="",
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


class _SeCApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="_TuringUtilsSeCApply",
            is_dev_only=True,
            display_name="SeC Track Visual Concept (Internal Apply)",
            category="",
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
                    optional=True,
                    tooltip='JSON point list such as [{"x": 120, "y": 240}].',
                ),
                io.String.Input(
                    "negative_coords",
                    default="",
                    optional=True,
                    tooltip='JSON exclusion-point list such as [{"x": 80, "y": 200}].',
                ),
                io.BoundingBox.Input(
                    "bounding_box",
                    optional=True,
                    socketless=False,
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
                io.Int.Input(
                    "annotation_frame_idx",
                    default=0,
                    min=-1_000_000,
                    max=1_000_000,
                    step=1,
                    tooltip=(
                        "Non-negative values are absolute frame indexes. Negative values use "
                        "standard Python indexing: -1 selects the final frame, -2 the "
                        "penultimate frame."
                    ),
                ),
                io.Int.Input(
                    "max_frames_to_track",
                    default=-1,
                    min=-1,
                    max=1_000_000,
                    step=1,
                    tooltip="-1 tracks every reachable frame in the selected direction.",
                ),
                io.Int.Input(
                    "semantic_keyframes",
                    default=7,
                    min=1,
                    max=20,
                    step=1,
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


class SeCTrackVisualConcept(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = _SeCApply.define_schema()
        schema.node_id = "TuringUtilsSeCTrackVisualConcept"
        schema.display_name = "SeC Track Visual Concept"
        schema.is_dev_only = False
        schema.category = "Turing Utils/Mask"
        schema.inputs = _SeCLoader.define_schema().inputs + schema.inputs[1:]
        return schema

    @classmethod
    def execute(cls, model_name, frames, attention="auto", **kwargs):
        return _SeCApply.execute(
            load_sec_model(model_name, attention), frames, **kwargs)


__all__ = ["_SeCLoader", "SeCTrackVisualConcept", "_SeCApply"]
