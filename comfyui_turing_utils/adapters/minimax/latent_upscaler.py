"""Learned 3D latent upscaling for MiniMax H3 video latents."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.nested_tensor
import comfy.model_patcher
import comfy.ops
import comfy.utils
import folder_paths

from ...log import get_logger
from ...profiling import WORKFLOW_TIMELINE


LOG = get_logger("minimax.upscale")


LATENTS_MEAN = (
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
)

LATENTS_STD = (
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235596,
    3.0496184825897217,
    2.1088054180145264,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523,
)

_PRECISION_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

H3_VIDEO_VAE_SPATIAL_DOWNSAMPLE = 16


def _normalization(channels: int, *, device=None, dtype=None):
    return comfy.ops.disable_weight_init.GroupNorm(
        32,
        channels,
        device=device,
        dtype=dtype,
    )


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels: int, emb_channels: int, *, device=None, dtype=None):
        super().__init__()
        ops = comfy.ops.disable_weight_init
        self.in_layers = nn.Sequential(
            _normalization(channels, device=device, dtype=dtype),
            nn.SiLU(),
            ops.Conv3d(channels, channels, 3, padding=1, device=device, dtype=dtype),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            ops.Linear(emb_channels, 2 * channels, device=device, dtype=dtype),
        )
        self.out_norm = _normalization(channels, device=device, dtype=dtype)
        # Identity keeps the checkpoint's out_layers.2 parameter names while
        # avoiding an inference-only Dropout module.
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Identity(),
            ops.Conv3d(channels, channels, 3, padding=1, device=device, dtype=dtype),
        )
        self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).to(h.dtype)
        while emb_out.ndim < h.ndim:
            emb_out = emb_out.unsqueeze(-1)
        scale, shift = emb_out.chunk(2, dim=1)
        h = self.out_norm(h) * (1.0 + scale) + shift
        return self.skip(x) + self.out_layers(h)


class TemporalConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int, *, device=None, dtype=None):
        super().__init__()
        ops = comfy.ops.disable_weight_init
        self.norm = _normalization(channels, device=device, dtype=dtype)
        self.dwconv = ops.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0),
            groups=channels,
            device=device,
            dtype=dtype,
        )
        self.pwconv = ops.Conv3d(channels, channels, 1, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dwconv(F.silu(self.norm(x)))
        return x + self.pwconv(h)


class H3LatentResizer3D(nn.Module):
    """Attention-free 3D backbone used by the public H3 latent upscaler weights."""

    def __init__(
        self,
        *,
        in_channels: int,
        in_blocks: int,
        out_blocks: int,
        channels: int,
        temporal_every: int,
        temporal_kernel: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        ops = comfy.ops.disable_weight_init
        self.conv_in = ops.Conv3d(
            in_channels,
            channels,
            3,
            padding=1,
            device=device,
            dtype=dtype,
        )
        self.embed = nn.Sequential(
            ops.Linear(1, 64, device=device, dtype=dtype),
            nn.SiLU(),
            ops.Linear(64, 64, device=device, dtype=dtype),
        )
        self.in_blocks = self._make_blocks(
            in_blocks,
            channels,
            temporal_every,
            temporal_kernel,
            device=device,
            dtype=dtype,
        )
        self.out_blocks = self._make_blocks(
            out_blocks,
            channels,
            temporal_every,
            temporal_kernel,
            device=device,
            dtype=dtype,
        )
        self.norm_out = _normalization(channels, device=device, dtype=dtype)
        self.conv_out = ops.Conv3d(
            channels,
            in_channels,
            3,
            padding=1,
            device=device,
            dtype=dtype,
        )
        self.register_buffer(
            "latent_mean",
            torch.tensor(LATENTS_MEAN, dtype=dtype or torch.float32).view(1, -1, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "latent_std",
            torch.tensor(LATENTS_STD, dtype=dtype or torch.float32).view(1, -1, 1, 1, 1),
            persistent=False,
        )

    @staticmethod
    def _make_blocks(
        count: int,
        channels: int,
        temporal_every: int,
        temporal_kernel: int,
        *,
        device=None,
        dtype=None,
    ) -> nn.ModuleList:
        blocks = nn.ModuleList()
        for index in range(count):
            blocks.append(ResBlockEmb3D(channels, 64, device=device, dtype=dtype))
            if temporal_every > 0 and index % temporal_every == 0:
                blocks.append(
                    TemporalConv(
                        channels,
                        temporal_kernel,
                        device=device,
                        dtype=dtype,
                    )
                )
        return blocks

    def get_dtype(self):
        return self.conv_in.weight.dtype

    @staticmethod
    def _run_blocks(blocks: nn.ModuleList, x: torch.Tensor, emb: torch.Tensor):
        for block in blocks:
            if isinstance(block, ResBlockEmb3D):
                x = block(x, emb.expand(x.shape[0], -1))
            else:
                x = block(x)
        return x

    def forward(
        self,
        latent: torch.Tensor,
        scale: float,
        target_size: tuple[int, int, int],
    ) -> torch.Tensor:
        x = (latent - self.latent_mean) / self.latent_std
        scale_emb = x.new_tensor([[float(scale) - 1.0]])
        emb = self.embed(scale_emb)
        x = self._run_blocks(self.in_blocks, self.conv_in(x), emb)
        x = F.interpolate(x, size=target_size, mode="trilinear", align_corners=False)
        x = self._run_blocks(self.out_blocks, x, emb)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x * self.latent_std + self.latent_mean


@dataclass(frozen=True)
class H3LatentUpscaleArchitecture:
    in_channels: int
    in_blocks: int
    out_blocks: int
    channels: int
    temporal_every: int
    temporal_kernel: int


def _block_indices(state_dict: dict[str, torch.Tensor], prefix: str):
    residual = set()
    temporal = set()
    for key in state_dict:
        match = re.match(rf"{prefix}\.(\d+)\.in_layers\.", key)
        if match:
            residual.add(int(match.group(1)))
        match = re.match(rf"{prefix}\.(\d+)\.dwconv\.weight$", key)
        if match:
            temporal.add(int(match.group(1)))
    return residual, temporal


def _infer_temporal_every(residual: set[int], temporal: set[int]) -> int:
    if not residual:
        raise ValueError("checkpoint has no 3D residual blocks")
    count = len(residual)
    for every in range(0, count + 1):
        expected_residual = set()
        expected_temporal = set()
        module_index = 0
        for block_index in range(count):
            expected_residual.add(module_index)
            module_index += 1
            if every > 0 and block_index % every == 0:
                expected_temporal.add(module_index)
                module_index += 1
        if residual == expected_residual and temporal == expected_temporal:
            return every
    raise ValueError(
        "checkpoint uses an unsupported 3D residual/temporal block arrangement"
    )


def detect_h3_latent_upscaler_architecture(
    state_dict: dict[str, torch.Tensor],
) -> H3LatentUpscaleArchitecture:
    conv = state_dict.get("conv_in.weight")
    if not torch.is_tensor(conv) or conv.ndim != 5:
        raise ValueError("checkpoint is not a supported 3D H3 latent upscaler")
    in_channels = int(conv.shape[1])
    channels = int(conv.shape[0])
    if in_channels != 24:
        raise ValueError(f"H3 latent upscaler must use 24 input channels, got {in_channels}")
    if channels % 32:
        raise ValueError(f"H3 latent upscaler width must be divisible by 32, got {channels}")

    in_residual, in_temporal = _block_indices(state_dict, "in_blocks")
    out_residual, out_temporal = _block_indices(state_dict, "out_blocks")
    in_every = _infer_temporal_every(in_residual, in_temporal)
    out_every = _infer_temporal_every(out_residual, out_temporal)
    if in_every != out_every:
        raise ValueError("input and output stages use different temporal block layouts")
    if any(".q.weight" in key or ".k.weight" in key for key in state_dict):
        raise ValueError("attention-enabled H3 latent upscaler checkpoints are unsupported")

    temporal_kernel = 5
    temporal_keys = [key for key in state_dict if key.endswith(".dwconv.weight")]
    if temporal_keys:
        kernels = {int(state_dict[key].shape[2]) for key in temporal_keys}
        if len(kernels) != 1:
            raise ValueError("checkpoint contains inconsistent temporal kernel sizes")
        temporal_kernel = kernels.pop()
        if temporal_kernel < 1 or temporal_kernel % 2 == 0:
            raise ValueError(f"temporal kernel must be a positive odd number, got {temporal_kernel}")

    return H3LatentUpscaleArchitecture(
        in_channels=in_channels,
        in_blocks=len(in_residual),
        out_blocks=len(out_residual),
        channels=channels,
        temporal_every=in_every,
        temporal_kernel=temporal_kernel,
    )


def _extract_state_dict(loaded) -> dict[str, torch.Tensor]:
    if isinstance(loaded, dict) and isinstance(loaded.get("model"), dict):
        loaded = loaded["model"]
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError("H3 latent upscaler checkpoint does not contain a state dictionary")
    if any(key.startswith("upscaler.") for key in loaded):
        loaded = {
            key.removeprefix("upscaler."): value
            for key, value in loaded.items()
            if key.startswith("upscaler.")
        }
    if not loaded or not all(isinstance(key, str) for key in loaded):
        raise ValueError("H3 latent upscaler checkpoint has an invalid state dictionary")
    return loaded


def _automatic_dtype(device: torch.device, state_dict: dict[str, torch.Tensor]):
    model_params = sum(value.numel() for value in state_dict.values() if torch.is_tensor(value))
    if comfy.model_management.should_use_fp16(device, model_params=model_params):
        return torch.float16
    if comfy.model_management.should_use_bf16(device, model_params=model_params):
        return torch.bfloat16
    return torch.float32


def load_h3_latent_upscaler(model_name: str, precision: str):
    model_path = folder_paths.get_full_path_or_raise("latent_upscale_models", model_name)
    loaded = comfy.utils.load_torch_file(model_path, safe_load=True)
    state_dict = _extract_state_dict(loaded)
    architecture = detect_h3_latent_upscaler_architecture(state_dict)
    load_device = comfy.model_management.get_torch_device()
    dtype = (
        _automatic_dtype(load_device, state_dict)
        if precision == "auto"
        else _PRECISION_DTYPES[precision]
    )
    for key, value in tuple(state_dict.items()):
        if torch.is_tensor(value) and value.is_floating_point() and value.dtype != dtype:
            state_dict[key] = value.to(dtype=dtype)

    model = H3LatentResizer3D(
        **architecture.__dict__,
        device="meta",
        dtype=dtype,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=True, assign=True)
    if missing or unexpected:
        raise ValueError(
            f"H3 latent upscaler checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    patcher = comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=load_device,
        offload_device=comfy.model_management.unet_offload_device(),
    )
    LOG.info(
        "Loaded MiniMax H3 3D latent upscaler %s: blocks=%d+%d width=%d temporal_every=%d kernel=%d dtype=%s",
        model_name,
        architecture.in_blocks,
        architecture.out_blocks,
        architecture.channels,
        architecture.temporal_every,
        architecture.temporal_kernel,
        dtype,
    )
    return patcher


def estimate_upscale_memory(
    video: torch.Tensor,
    target_height: int,
    target_width: int,
    channels: int,
    dtype: torch.dtype,
) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    source_voxels = int(video.shape[0] * video.shape[2] * video.shape[3] * video.shape[4])
    target_voxels = int(video.shape[0] * video.shape[2] * target_height * target_width)
    # Two neighboring feature volumes plus cuDNN workspace and normalized input.
    return max(source_voxels, target_voxels) * channels * element_size * 3 + (
        video.numel() * element_size * 2
    )


def _validate_video(video: torch.Tensor):
    if not torch.is_tensor(video) or video.ndim != 5 or int(video.shape[1]) != 24:
        shape = tuple(video.shape) if hasattr(video, "shape") else type(video).__name__
        raise ValueError(f"Expected H3 video latent [B,24,T,H,W], got {shape}")
    if int(video.shape[0]) < 1 or int(video.shape[2]) < 1:
        raise ValueError(f"H3 video latent dimensions must be non-empty, got {tuple(video.shape)}")
    return video


def _validate_audio(audio: torch.Tensor):
    if not torch.is_tensor(audio) or audio.ndim != 4 or tuple(audio.shape[1:3]) != (32, 2):
        shape = tuple(audio.shape) if hasattr(audio, "shape") else type(audio).__name__
        raise ValueError(f"Expected H3 audio latent [B,32,2,T], got {shape}")
    return audio


def _av_streams(samples):
    if not getattr(samples, "is_nested", False):
        return _validate_video(samples), None
    streams = list(samples.unbind())
    if len(streams) != 2:
        raise ValueError(f"Expected exactly two H3 latent streams, got {len(streams)}")
    video, audio = _validate_video(streams[0]), _validate_audio(streams[1])
    if int(video.shape[0]) != int(audio.shape[0]):
        raise ValueError("H3 video and audio latent batch sizes must match")
    return video, audio


def _target_axis(axis: int, scale: float) -> int:
    target = max(axis, int(round(axis * scale)))
    return max(2, (target + 1) // 2 * 2)


def _conditioning_keyframes(conditioning, source_height: int, source_width: int):
    keyframe_tensors = []
    seen = set()
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2 or not isinstance(entry[1], dict):
            raise ValueError("conditioning must contain [embedding, options] entries")
        keyframes = entry[1].get("minimax_keyframes")
        if keyframes is None:
            continue
        if not isinstance(keyframes, (list, tuple)):
            raise ValueError("minimax_keyframes must be a list")
        for keyframe in keyframes:
            if not isinstance(keyframe, dict) or "latent" not in keyframe:
                raise ValueError("each MiniMax H3 keyframe must contain a latent")
            tensor = _validate_video(keyframe["latent"])
            if tuple(tensor.shape[-2:]) != (source_height, source_width):
                raise ValueError(
                    "FL2AV keyframe latent spatial size must match the source video latent; "
                    f"got {tuple(tensor.shape[-2:])} and {(source_height, source_width)}"
                )
            tensor_id = id(tensor)
            if tensor_id not in seen:
                seen.add(tensor_id)
                keyframe_tensors.append(tensor)
    return keyframe_tensors


def _upscale_tensors(
    upscale_model,
    tensors: list[torch.Tensor],
    scale: float,
    target_height: int,
    target_width: int,
) -> dict[int, torch.Tensor]:
    model = upscale_model.model
    dtype = upscale_model.model_dtype()
    channels = int(model.conv_in.out_channels)
    groups = {}
    for tensor in tensors:
        groups.setdefault(tuple(tensor.shape[1:]), []).append(tensor)
    memory_required = max(
        sum(
            estimate_upscale_memory(
                tensor,
                target_height,
                target_width,
                channels,
                dtype,
            )
            for tensor in group
        )
        for group in groups.values()
    )
    device = upscale_model.load_device
    def execute():
        comfy.model_management.load_models_gpu(
            [upscale_model],
            memory_required=memory_required,
            force_full_load=True,
        )
        output_device = comfy.model_management.intermediate_device()
        outputs = {}
        for group in groups.values():
            batch_sizes = [int(tensor.shape[0]) for tensor in group]
            source = torch.cat(
                [tensor.to(device=device, dtype=dtype) for tensor in group],
                dim=0,
            )
            result = model(
                source,
                float(scale),
                (int(source.shape[2]), target_height, target_width),
            )
            for tensor, upscaled in zip(group, result.split(batch_sizes, dim=0)):
                outputs[id(tensor)] = upscaled.to(
                    device=output_device,
                    dtype=tensor.dtype,
                )
        return outputs

    return WORKFLOW_TIMELINE.call(
        "latent_upscale", device, execute
    )


def _sync_conditioning(conditioning, outputs: dict[int, torch.Tensor]):
    synced = []
    for embedding, options in conditioning:
        keyframes = options.get("minimax_keyframes")
        if keyframes is None:
            synced.append([embedding, options])
            continue
        new_options = options.copy()
        new_keyframes = []
        for keyframe in keyframes:
            new_keyframe = keyframe.copy()
            new_keyframe["latent"] = outputs[id(keyframe["latent"])]
            new_keyframes.append(new_keyframe)
        new_options["minimax_keyframes"] = new_keyframes
        new_options.pop("layout", None)
        synced.append([embedding, new_options])
    return synced


def _resize_video_mask(mask: torch.Tensor, target_height: int, target_width: int):
    if not torch.is_tensor(mask) or mask.ndim != 5:
        shape = tuple(mask.shape) if hasattr(mask, "shape") else type(mask).__name__
        raise ValueError(f"H3 video noise_mask must be 5D, got {shape}")
    original_dtype = mask.dtype
    source = mask if mask.is_floating_point() else mask.to(torch.float32)
    resized = F.interpolate(
        source,
        size=(int(mask.shape[2]), target_height, target_width),
        mode="nearest",
    )
    return resized.to(original_dtype)


def upscale_h3_latent(upscale_model, latent, conditioning, scale: float):
    """Upscale by a total spatial-pixel multiplier without rebuilding conditioning."""
    if scale < 1.0 or scale > 16.0:
        raise ValueError(f"scale must be between 1.0 and 16.0, got {scale}")
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("latent must be a LATENT dictionary")

    video, audio = _av_streams(latent["samples"])
    source_height, source_width = map(int, video.shape[-2:])
    if scale == 1.0:
        return (
            latent,
            conditioning,
            source_width * H3_VIDEO_VAE_SPATIAL_DOWNSAMPLE,
            source_height * H3_VIDEO_VAE_SPATIAL_DOWNSAMPLE,
        )
    # The public upscaler is conditioned on a linear H/W multiplier, while the
    # node exposes the more useful total spatial-pixel (and token) multiplier.
    linear_scale = math.sqrt(float(scale))
    target_height = _target_axis(source_height, linear_scale)
    target_width = _target_axis(source_width, linear_scale)
    keyframes = [] if conditioning is None else _conditioning_keyframes(conditioning, source_height, source_width)
    outputs = _upscale_tensors(
        upscale_model,
        [video, *keyframes],
        linear_scale,
        target_height,
        target_width,
    )

    output_latent = latent.copy()
    output_video = outputs[id(video)]
    if audio is None:
        output_latent["samples"] = output_video
    else:
        output_latent["samples"] = comfy.nested_tensor.NestedTensor((output_video, audio))

    noise_mask = latent.get("noise_mask")
    if noise_mask is not None:
        if getattr(noise_mask, "is_nested", False):
            streams = list(noise_mask.unbind())
            if len(streams) != 2:
                raise ValueError(f"Expected exactly two H3 noise-mask streams, got {len(streams)}")
            output_latent["noise_mask"] = comfy.nested_tensor.NestedTensor((
                _resize_video_mask(streams[0], target_height, target_width),
                streams[1],
            ))
        else:
            output_latent["noise_mask"] = _resize_video_mask(
                noise_mask,
                target_height,
                target_width,
            )

    output_conditioning = None if conditioning is None else _sync_conditioning(conditioning, outputs)
    return (
        output_latent,
        output_conditioning,
        target_width * H3_VIDEO_VAE_SPATIAL_DOWNSAMPLE,
        target_height * H3_VIDEO_VAE_SPATIAL_DOWNSAMPLE,
    )
