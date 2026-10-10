"""General media preparation nodes."""

from __future__ import annotations

from comfy_api.latest import io
from ..media.sampling import sample_images, sample_video

from ..media.images import (
    resize_image_if_present,
)

from ..media.contact_sheet import (
    render_contact_sheet,
)


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
                io.Int.Input(
                    "grid_size",
                    default=3,
                    min=2,
                    max=8,
                    step=1,
                    tooltip="Sample exactly grid_size squared frames in chronological row-major order.",
                ),
                io.Combo.Input(
                    "sampling",
                    options=["uniform", "motion_weighted"],
                    default="uniform",
                    tooltip="Uniform preserves timing. Motion weighted allocates more panels to intervals with stronger visual change.",
                ),
                io.Int.Input(
                    "width",
                    default=0,
                    min=0,
                    max=8192,
                    step=32,
                    tooltip="Final sheet width. 0 derives it from the source aspect and the height setting.",
                ),
                io.Int.Input(
                    "height",
                    default=0,
                    min=0,
                    max=8192,
                    step=32,
                    tooltip="Final sheet height. 0 derives it from the source aspect and the width setting.",
                ),
                io.Combo.Input(
                    "resize_mode",
                    options=["fit", "crop", "stretch"],
                    default="fit",
                    tooltip="How each sampled frame is fitted into its panel.",
                ),
                io.Int.Input(
                    "gap",
                    default=0,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="Black pixels between adjacent panels.",
                ),
                io.Boolean.Input(
                    "film_border",
                    default=True,
                    tooltip="Wrap every panel in film rails and perforations. Labels are printed on the lower rail.",
                ),
                io.Combo.Input(
                    "annotation",
                    options=["none", "index", "timestamp", "index_timestamp"],
                    default="index",
                    tooltip="Optional chronological label. Without the film border, labels use a narrow caption rail outside the frame.",
                ),
                io.Float.Input(
                    "image_frame_rate",
                    default=24.0,
                    min=0.01,
                    max=1000.0,
                    step=0.01,
                    tooltip="Used only to derive timestamps when the source is an IMAGE batch.",
                ),
                io.Video.Input(
                    "video",
                    optional=True,
                    tooltip="A loaded ComfyUI VIDEO. Connect either video or frames, not both.",
                ),
                io.Image.Input(
                    "frames",
                    optional=True,
                    tooltip="An already decoded video frame batch. Connect either frames or video, not both.",
                ),
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
            sampled, timestamps = sample_video(video, sample_count, sampling)
        else:
            sampled, timestamps = sample_images(
                frames, sample_count, sampling, image_frame_rate
            )
        sheet = render_contact_sheet(
            sampled,
            timestamps,
            int(grid_size),
            int(width),
            int(height),
            resize_mode,
            int(gap),
            bool(film_border),
            annotation,
        )
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
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Leave unconnected to produce no image.",
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="Optional mask transformed with the same geometry as the image.",
                ),
                io.Int.Input(
                    "width",
                    default=0,
                    min=0,
                    max=16384,
                    step=1,
                    tooltip="Target width. 0 derives it from height; width=height=0 passes the input through.",
                ),
                io.Int.Input(
                    "height",
                    default=0,
                    min=0,
                    max=16384,
                    step=1,
                    tooltip="Target height. 0 derives it from width; width=height=0 passes the input through.",
                ),
                io.Combo.Input(
                    "resize_mode",
                    options=["stretch", "fit", "crop", "pad", "pad_edge"],
                    default="crop",
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                    default="lanczos",
                ),
                io.Combo.Input(
                    "crop_position",
                    options=["center", "top", "bottom", "left", "right"],
                    default="center",
                    tooltip="Crop anchor or padded-content alignment.",
                ),
                io.Int.Input("divisible_by", default=1, min=1, max=512, step=1),
                io.String.Input(
                    "pad_color",
                    default="0, 0, 0",
                    tooltip="RGB values in 0-255 or 0-1 form, a hex color, or a CSS color name.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(
        cls,
        width,
        height,
        resize_mode,
        upscale_method,
        crop_position,
        divisible_by,
        pad_color,
        image=None,
        mask=None,
    ):
        return io.NodeOutput(
            *resize_image_if_present(
                image,
                mask,
                width,
                height,
                resize_mode,
                upscale_method,
                crop_position,
                divisible_by,
                pad_color,
            )
        )
