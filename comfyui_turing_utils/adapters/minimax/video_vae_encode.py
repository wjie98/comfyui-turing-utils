"""MiniMax H3 video VAE encode pipeline."""

from __future__ import annotations

import math

import torch

import comfy.model_management
import comfy.model_prefetch

from ...log import get_logger
from .video_vae import (
    TILE_OVERLAP,
    TILE_SIZE,
    _AUTO_ENCODE_TILE_BATCH_LIMIT,
    _TileProgress,
    _encode_memory_requirement,
    _select_tiles_per_batch,
    _spatial_tile_count,
    require_h3_video_vae,
    split_tiles,
)


LOG = get_logger("minimax.vae")


def _encoder_prefetch_stages(model):
    stages = [model.encoder.conv_in]
    for down in model.encoder.down:
        stages.extend(down.block)
        if hasattr(down, "downsample"):
            stages.append(down.downsample)
    stages.extend((model.encoder.conv_out, model.quant_conv))
    return stages


def _encode_moments(model, x, prefetch_dynamic_vbars=False):
    if not prefetch_dynamic_vbars:
        return model._encode_moments(x)

    stages = _encoder_prefetch_stages(model)
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(
        stages,
        x.device,
        {"prefetch_dynamic_vbars": bool(prefetch_dynamic_vbars)},
    )

    def run(module, value):
        comfy.model_prefetch.prefetch_queue_pop(
            prefetch_queue,
            value.device,
            module,
        )
        return module(value)

    h = run(model.encoder.conv_in, x)
    for down in model.encoder.down:
        for block in down.block:
            h = run(block, h)
        if hasattr(down, "downsample"):
            h = run(down.downsample, h)
    h = torch.nn.functional.silu(model.encoder.norm_out(h))
    h = run(model.encoder.conv_out, h)
    moments = run(model.quant_conv, h)
    comfy.model_prefetch.prefetch_queue_pop(
        prefetch_queue,
        moments.device,
        None,
    )
    return moments


def _tiled_encode(
    model,
    x,
    tile_size,
    tile_overlap,
    prefetch_dynamic_vbars=False,
    tiles_per_batch=1,
    progress=None,
):
    height, width = x.shape[-2:]
    y_idx, y_len, y_overlap = split_tiles(
        height, tile_size, tile_overlap, model.vae_ratio
    )
    x_idx, x_len, x_overlap = split_tiles(
        width, tile_size, tile_overlap, model.vae_ratio
    )

    descriptors = [
        (i, j, i_pos, i_len, j_pos, j_len)
        for i, (i_pos, i_len) in enumerate(zip(y_idx, y_len))
        for j, (j_pos, j_len) in enumerate(zip(x_idx, x_len))
    ]
    rows = [[None] * len(x_idx) for _ in y_idx]
    source_batch = x.shape[0]
    for start in range(0, len(descriptors), tiles_per_batch):
        group = descriptors[start : start + tiles_per_batch]
        inputs = [
            x[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
            for _i, _j, i_pos, i_len, j_pos, j_len in group
        ]
        batched = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=0)
        encoded = _encode_moments(model, batched, prefetch_dynamic_vbars)
        for (i, j, *_bounds), tile in zip(group, encoded.split(source_batch, dim=0)):
            rows[i][j] = tile
        if progress is not None:
            progress.update(len(group))

    latent_y_overlap = [value // model.vae_ratio for value in y_overlap]
    latent_x_overlap = [value // model.vae_ratio for value in x_overlap]
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = model.blend(
                    rows[i - 1][j], tile, latent_y_overlap[i - 1], dim=-2
                )
            if j > 0:
                tile = model.blend(row[j - 1], tile, latent_x_overlap[j - 1], dim=-1)
            if i < len(rows) - 1:
                tile = tile[..., : -latent_y_overlap[i], :]
            if j < len(row) - 1:
                tile = tile[..., :, : -latent_x_overlap[j]]
            result_row.append(tile)
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


def _encode_clip(
    model,
    clip,
    tile_size,
    tile_overlap,
    prefetch_dynamic_vbars=False,
    tiles_per_batch=1,
    progress=None,
):
    return _tiled_encode(
        model,
        model._normalize_pixels(clip),
        tile_size,
        tile_overlap,
        prefetch_dynamic_vbars,
        tiles_per_batch,
        progress,
    )


def _prepare_encoder_clip(clip, process_input, device, compute_dtype):
    """Prepare one encoder clip without widening an existing FP16 pixel store."""

    if clip.dtype == compute_dtype:
        clip = clip.to(device=device, dtype=compute_dtype)
        return process_input(clip)
    return process_input(clip.float()).to(device=device, dtype=compute_dtype)


def _encode_temporal_device(
    model,
    pixels,
    process_input,
    device,
    compute_dtype,
    tile_size,
    tile_overlap,
    prefetch_dynamic_vbars,
    tiles_per_batch,
    progress,
):
    """Normalize and transfer one temporal clip at a time."""

    z_list = []
    for start in range(0, pixels.shape[2], model.clip_length):
        clip = pixels[:, :, start : start + model.clip_length]
        if clip.shape[2] < model.clip_length:
            pad = clip[:, :, -1:].repeat(
                1,
                1,
                model.clip_length - clip.shape[2],
                1,
                1,
            )
            clip = torch.cat((clip, pad), dim=2)
        clip = _prepare_encoder_clip(
            clip,
            process_input,
            device,
            compute_dtype,
        )
        z_list.append(
            _encode_clip(
                model,
                clip,
                tile_size,
                tile_overlap,
                prefetch_dynamic_vbars,
                tiles_per_batch,
                progress,
            )
        )
    z = torch.cat(z_list, dim=2)
    return z[:, :, : -model.token_drop] if model.token_drop > 0 else z


def _encode_temporal_buffered(
    model,
    pixels,
    process_input,
    compute_dtype,
    device,
    tile_size,
    tile_overlap,
    prefetch_dynamic_vbars,
    tiles_per_batch,
    progress,
):
    copy_stream = torch.cuda.Stream(device=device)
    staging = [None, None]
    device_clips = [None, None]
    copy_done = [None, None]
    compute_done = [None, None]
    normalize_on_device = pixels.dtype == compute_dtype

    def prepare(clip_index, slot):
        if copy_done[slot] is not None:
            copy_done[slot].synchronize()
        start = clip_index * model.clip_length
        clip = pixels[:, :, start : start + model.clip_length]
        if clip.shape[2] < model.clip_length:
            pad = clip[:, :, -1:].repeat(1, 1, model.clip_length - clip.shape[2], 1, 1)
            clip = torch.cat((clip, pad), dim=2)
        # Preserve a compute-dtype input and apply VAE normalization after the
        # transfer. Other input dtypes follow ComfyUI's FP32 normalization path.
        clip = (
            clip.contiguous()
            if normalize_on_device
            else process_input(clip.float())
        )
        if (
            staging[slot] is None
            or staging[slot].shape != clip.shape
            or staging[slot].dtype != clip.dtype
        ):
            try:
                staging[slot] = torch.empty_like(clip, device="cpu", pin_memory=True)
            except RuntimeError as error:
                raise _PinnedBufferUnavailable from error
            device_clips[slot] = torch.empty(
                clip.shape,
                dtype=compute_dtype,
                device=device,
            )
        staging[slot].copy_(clip)
        if compute_done[slot] is not None:
            copy_stream.wait_event(compute_done[slot])
        event = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            device_clips[slot].copy_(staging[slot], non_blocking=True)
            event.record(copy_stream)
        copy_done[slot] = event

    count = math.ceil(pixels.shape[2] / model.clip_length)
    prepare(0, 0)
    z_list = []
    current_stream = torch.cuda.current_stream(device)
    for clip_index in range(count):
        slot = clip_index % 2
        current_stream.wait_event(copy_done[slot])
        clip = device_clips[slot]
        if normalize_on_device:
            clip = process_input(clip)
        z_list.append(
            _encode_clip(
                model,
                clip,
                tile_size,
                tile_overlap,
                prefetch_dynamic_vbars,
                tiles_per_batch,
                progress,
            )
        )
        done = torch.cuda.Event()
        done.record(current_stream)
        compute_done[slot] = done
        if clip_index + 1 < count:
            prepare(clip_index + 1, 1 - slot)
    z = torch.cat(z_list, dim=2)
    return z[:, :, : -model.token_drop] if model.token_drop > 0 else z


class _PinnedBufferUnavailable(RuntimeError):
    pass


def _prepare_encode_pixels(vae, pixels):
    if pixels.ndim == 4:
        pixels = vae.vae_encode_crop_pixels(pixels).movedim(-1, 1)
        return pixels.movedim(1, 0).unsqueeze(0)
    if pixels.ndim != 5:
        raise ValueError(
            "MiniMax H3 video pixels must be [T,H,W,C] or [B,T,H,W,C], "
            f"got {tuple(pixels.shape)}"
        )

    # ComfyUI's generic crop helper treats every dimension between batch and
    # channels as spatial.  For a batched video that includes the frame axis,
    # so crop H/W explicitly and leave time untouched.
    if vae.crop_input:
        ratio = vae.spacial_compression_encode()
        for dim in (-3, -2):
            extent = int(pixels.shape[dim])
            cropped = extent // ratio * ratio
            if cropped != extent:
                pixels = pixels.narrow(dim, (extent - cropped) // 2, cropped)
    channels = int(pixels.shape[-1])
    if channels > vae.output_channels:
        pixels = pixels[..., : vae.output_channels]
    elif channels < vae.output_channels:
        raise ValueError(
            f"MiniMax H3 video pixels require {vae.output_channels} channels, got {channels}"
        )
    return pixels.movedim(-1, 1)


def encode_video(vae, pixels):
    model = require_h3_video_vae(vae)
    pixels = _prepare_encode_pixels(vae, pixels)
    compute_dtype = vae.vae_dtype
    tile_size = TILE_SIZE
    tile_count = _spatial_tile_count(
        pixels.shape[-2],
        pixels.shape[-1],
        tile_size,
    )
    batch_tiles, memory = _select_tiles_per_batch(
        vae,
        tile_count,
        lambda count: _encode_memory_requirement(
            vae,
            pixels.shape,
            tile_size,
            count,
            compute_dtype,
        ),
        _AUTO_ENCODE_TILE_BATCH_LIMIT,
    )
    tile_overlap = TILE_OVERLAP
    comfy.model_management.load_models_gpu(
        [vae.patcher], memory_required=memory, force_full_load=vae.disable_offload
    )

    progress = None
    try:
        prefetch_dynamic_vbars = vae.patcher.is_dynamic()
        temporal_clips = (
            1
            if pixels.shape[2] == 1
            else math.ceil(pixels.shape[2] / model.clip_length)
        )
        progress = _TileProgress(
            tile_count * temporal_clips,
            vae.device,
            "H3 VAE Encode",
        )
        if pixels.shape[2] == 1:
            x = _prepare_encoder_clip(
                pixels,
                vae.process_input,
                vae.device,
                compute_dtype,
            )
            moments = _encode_clip(
                model,
                x,
                tile_size,
                tile_overlap,
                prefetch_dynamic_vbars,
                batch_tiles,
                progress,
            )[:, :, -1:]
        elif pixels.device.type == "cpu" and torch.cuda.is_available():
            try:
                moments = _encode_temporal_buffered(
                    model,
                    pixels,
                    vae.process_input,
                    compute_dtype,
                    vae.device,
                    tile_size,
                    tile_overlap,
                    prefetch_dynamic_vbars,
                    batch_tiles,
                    progress,
                )
            except _PinnedBufferUnavailable:
                comfy.model_management.synchronize()
                LOG.warning(
                    "H3 VAE could not allocate pinned encoder buffers; using synchronous FP32 pixel copies"
                )
                moments = _encode_temporal_device(
                    model,
                    pixels,
                    vae.process_input,
                    vae.device,
                    compute_dtype,
                    tile_size,
                    tile_overlap,
                    prefetch_dynamic_vbars,
                    batch_tiles,
                    progress,
                )
        else:
            moments = _encode_temporal_device(
                model,
                pixels,
                vae.process_input,
                vae.device,
                compute_dtype,
                tile_size,
                tile_overlap,
                prefetch_dynamic_vbars,
                batch_tiles,
                progress,
            )
        mean = torch.chunk(moments.float(), 2, dim=1)[0]
        latent_mean = model.latents_mean.view(1, -1, 1, 1, 1).to(mean)
        latent_std = model.latents_std.view(1, -1, 1, 1, 1).to(mean)
        return ((mean - latent_mean) / latent_std).to(
            device=vae.output_device, dtype=vae.vae_output_dtype()
        )
    finally:
        if progress is not None:
            progress.finish()


__all__ = ["encode_video"]
