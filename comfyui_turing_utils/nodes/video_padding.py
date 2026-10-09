"""Frame-grid padding shared by video preprocessing and mask-only branches."""

from comfy_api.latest import io

from .latent import VIDEO_MASK_SPECS
import torch


def repeat_last_frame(tensor: torch.Tensor, count: int) -> torch.Tensor:
    """Append copies of the final frame along the first dimension."""
    count = int(count)
    if count <= 0:
        return tensor
    if tensor.shape[0] < 1:
        raise ValueError("Cannot repeat the final item of an empty tensor")
    tail = tensor[-1:].repeat(count, *([1] * (tensor.ndim - 1)))
    return torch.cat((tensor, tail), dim=0)


def validate_frame_mask(mask, frame_count: int, height: int, width: int):
    if mask is None:
        return None
    if mask.ndim == 2 and frame_count == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim < 3:
        raise ValueError(f"Expected MASK tensor shaped [frames, height, width], got {tuple(mask.shape)}.")
    if int(mask.shape[0]) != frame_count:
        raise ValueError(
            f"MASK frame count must match IMAGE frame count before padding; "
            f"got mask={int(mask.shape[0])}, image={frame_count}."
        )
    if int(mask.shape[-2]) != height or int(mask.shape[-1]) != width:
        raise ValueError(
            f"MASK spatial size must match IMAGE before padding; got mask="
            f"{int(mask.shape[-1])}x{int(mask.shape[-2])}, image={width}x{height}."
        )
    return mask


def padded_frame_count(count, model_type, target=0):
    stride = VIDEO_MASK_SPECS[model_type]
    minimum, step = (5, 17) if stride is None else (7 if model_type == "mochi" else 1, stride)
    if target:
        if target < count or target < minimum or (target - minimum) % step:
            raise ValueError(f"target_frame_count must be >= {max(count, minimum)} and on the {minimum}+{step}*n {model_type} grid")
        return target
    return minimum + (max(0, count - minimum) + step - 1) // step * step


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
                io.Combo.Input("type", options=list(VIDEO_MASK_SPECS), default="minimax"),
                io.Int.Input("target_frame_count", default=0, min=0, max=16385,
                             tooltip="0 rounds up. Wan/Hunyuan: 4*n+1; LTX: 8*n+1; Mochi: 6*n+1 (minimum 7); H3: 17*n+5."),
                io.Mask.Input("mask", optional=True),
            ],
            outputs=[io.Image.Output("image"), io.Mask.Output("mask"), io.Int.Output("width"),
                     io.Int.Output("height"), io.Int.Output("length"), io.Int.Output("input_length")],
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
            width, height, length, count,
        )
