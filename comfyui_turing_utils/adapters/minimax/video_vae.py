"""MiniMax H3 video VAE decode pipeline and shared tile primitives."""

from __future__ import annotations

import hashlib
import math
import queue
import threading

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import comfy.memory_management
import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.quant_ops
import comfy.rmsnorm
import comfy.utils
from comfy.ldm.minimax import vae as h3_vae
from comfy.ldm.modules.attention import AttentionTensorContainer

from ...attention import make_attention_override
from ...attention.integration import execute_projected_attention
from ...attention.protocol import (
    ATTENTION_EXECUTOR_KEY,
    QKTransformSpec,
    RMSNormSpec,
    RotaryEmbeddingSpec,
)
from ...hardware import is_supported_tensor_core_device
from ...kernel_api import load_turing_sage
from ...log import get_logger


LOG = get_logger("minimax.vae")
TILE_SIZE = 256
TILE_OVERLAP = 64
_AUTO_DECODE_TILE_BATCH_LIMIT = 16
_AUTO_ENCODE_TILE_BATCH_LIMIT = 16
_DECODER_BYTES_PER_FP16_TOKEN = 64 * 1024
_ENCODER_BYTES_PER_FP16_PIXEL_FRAME = 1800
_ENCODER_FIXED_WORKSPACE = 256 * 1024**2
_DECODE_SAFETY_FACTOR = 1.08
_ENCODE_SAFETY_FACTOR = 1.05
_MULTIBAND_DOWNSAMPLE = 8
_MULTIBAND_HIGH_WEIGHT_POWER = 4
_MULTIBAND_FRAME_CHUNK = 4


def require_h3_video_vae(vae):
    vae.throw_exception_if_invalid()
    model = vae.first_stage_model
    if not isinstance(model, h3_vae.MiniMaxH3VideoVAE):
        raise ValueError("This node requires a MiniMax H3 video VAE")
    return model


def split_tiles(input_len: int, tile_size: int, overlap_min: int, ratio: int):
    if tile_size >= input_len:
        return [0], [input_len], []

    count = math.ceil(input_len / tile_size)
    while True:
        overlaps = [overlap_min] * (count - 1)
        remaining = tile_size * count - sum(overlaps) - input_len
        if remaining >= 0:
            break
        count += 1

    for i in range(remaining // ratio):
        overlaps[i % (count - 1)] += ratio

    starts = [0]
    for i in range(count - 1):
        starts.append(starts[-1] + tile_size - overlaps[i])
    return starts, [tile_size] * count, overlaps


def _spatial_tile_count(height, width, tile_size=TILE_SIZE):
    return len(split_tiles(height, tile_size, TILE_OVERLAP, 16)[0]) * len(
        split_tiles(width, tile_size, TILE_OVERLAP, 16)[0]
    )


def _decode_memory_requirement(
    vae,
    latent_shape,
    tiles_per_batch,
    dtype,
):
    model = vae.first_stage_model
    height = latent_shape[-2] * model.vae_ratio
    width = latent_shape[-1] * model.vae_ratio
    tile_height = min(height, TILE_SIZE)
    tile_width = min(width, TILE_SIZE)
    bounded_shape = (
        latent_shape[0],
        latent_shape[1],
        latent_shape[2],
        tile_height // model.vae_ratio,
        tile_width // model.vae_ratio,
    )
    official_estimate = int(vae.memory_used_decode(bounded_shape, dtype))

    if latent_shape[2] == 1:
        resident_tokens = 1
        resident_frames = 1
    else:
        resident_tokens = model.tokens_chunk_size + model.token_overlap
        resident_frames = resident_tokens * model.vae_ratio_t
    sequence = (
        resident_tokens
        * (tile_height // model.vae_ratio)
        * (tile_width // model.vae_ratio)
        + 1
        + model.decoder.num_register_tokens
    )
    dtype_scale = comfy.model_management.dtype_size(dtype) / 2.0
    # Dense H3 decoder blocks peak around 64 KiB per FP16 token once QKV and
    # the gated MLP workspace overlap. The complete decoded chunk also exists
    # as a compute-dtype canvas while FP32 finalized pixels are copied out.
    transformer_workspace = (
        latent_shape[0]
        * sequence
        * _DECODER_BYTES_PER_FP16_TOKEN
        * dtype_scale
        * tiles_per_batch
    )
    decoder_dim = model.decoder.transformer_blocks[0].scale1.numel()
    unique_tokens = (
        latent_shape[0] * resident_tokens * latent_shape[-2] * latent_shape[-1]
    )
    # The fixed decoder keeps one global image state plus global QKV and
    # attention workspaces throughout all Transformer blocks.
    compute_bytes = comfy.model_management.dtype_size(dtype)
    transformer_workspace += unique_tokens * decoder_dim * (
        compute_bytes * 6 + 4
    )
    # Final pixel projection gathers only the active batch of 256px windows;
    # no duplicated all-window Transformer state is retained.
    projection_tokens = (
        latent_shape[0]
        * tiles_per_batch
        * resident_tokens
        * (tile_height // model.vae_ratio)
        * (tile_width // model.vae_ratio)
    )
    transformer_workspace += projection_tokens * decoder_dim * compute_bytes * 2
    pixel_elements = (
        latent_shape[0] * model.decoder.out_channels * resident_frames * height * width
    )
    pixel_workspace = pixel_elements * (comfy.model_management.dtype_size(dtype) + 4)
    # Multiband uses two FP32 accumulation canvases.  Its full-frame low-pass
    # temporaries are bounded to a few frames at a time.
    pixel_workspace += (
        pixel_elements * (8 - comfy.model_management.dtype_size(dtype))
        + latent_shape[0]
        * model.decoder.out_channels
        * min(resident_frames, _MULTIBAND_FRAME_CHUNK)
        * height
        * width
        * 8
        + 2 * height * width * 4
    )
    structural_estimate = int(
        (transformer_workspace + pixel_workspace) * _DECODE_SAFETY_FACTOR
    )
    # ComfyUI's estimate describes one internally tiled sample. Preserve its
    # fixed allowance and scale only the structural per-tile workspace here.
    return max(official_estimate, structural_estimate)


def _encode_memory_requirement(
    vae,
    pixel_shape,
    tile_size,
    tiles_per_batch,
    dtype,
):
    model = vae.first_stage_model
    batch, channels, frames, height, width = pixel_shape
    clip_frames = min(frames, model.clip_length)
    tile_height = min(height, tile_size)
    tile_width = min(width, tile_size)
    bounded_shape = (
        batch,
        channels,
        clip_frames,
        tile_height,
        tile_width,
    )
    official_estimate = int(vae.memory_used_encode(bounded_shape, dtype))

    dtype_scale = comfy.model_management.dtype_size(dtype) / 2.0
    # The causal CNN's high-resolution feature pyramid dominates the encoder.
    # This coefficient is deliberately conservative and was measured against
    # the full 2.6B-parameter H3 VAE at 256, 400, and 480 pixel tiles.
    convolution_workspace = (
        batch
        * clip_frames
        * tile_height
        * tile_width
        * _ENCODER_BYTES_PER_FP16_PIXEL_FRAME
        * dtype_scale
        * tiles_per_batch
    )
    buffer_count = 2 if frames > model.clip_length else 1
    input_buffers = (
        buffer_count
        * batch
        * channels
        * clip_frames
        * height
        * width
        * comfy.model_management.dtype_size(dtype)
    )
    structural_estimate = int(
        (convolution_workspace + input_buffers + _ENCODER_FIXED_WORKSPACE)
        * _ENCODE_SAFETY_FACTOR
    )
    return max(official_estimate, structural_estimate)


def _tile_memory_budget(vae):
    try:
        available = vae.patcher.get_free_memory(vae.device)
    except (AttributeError, RuntimeError):
        available = comfy.model_management.get_free_memory(vae.device)
    return max(
        0,
        int(available - comfy.model_management.extra_reserved_memory()),
    )


def _select_tiles_per_batch(
    vae,
    tile_count,
    memory_estimator,
    auto_limit,
):
    budget = _tile_memory_budget(vae)
    selected = 1
    for candidate in range(2, min(tile_count, auto_limit) + 1):
        if memory_estimator(candidate) > budget:
            break
        selected = candidate
    estimate = memory_estimator(selected)
    log = LOG.warning if estimate > budget else LOG.info
    log(
        "MiniMax H3 VAE auto tile batch selected %d/%d: estimated %.0f MiB, available %.0f MiB%s",
        selected,
        tile_count,
        estimate / 1024**2,
        budget / 1024**2,
        " (one tile exceeds the current budget)" if estimate > budget else "",
    )
    return selected, estimate


class _TileProgress:
    def __init__(self, total, device=None, description="H3 VAE"):
        self.total = int(total)
        self.bar = comfy.utils.ProgressBar(self.total)
        self.terminal = tqdm(
            total=self.total,
            desc=description,
            disable=not comfy.utils.PROGRESS_BAR_ENABLED,
        )
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.pending = None
        self.worker = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            self.pending = queue.Queue()
            self.worker = threading.Thread(
                target=self._consume,
                name="h3-vae-tile-progress",
                daemon=True,
            )
            self.worker.start()

    def _consume(self):
        while True:
            item = self.pending.get()
            if item is None:
                return
            event, count = item
            try:
                event.synchronize()
                self.bar.update(count)
                self.terminal.update(count)
            except RuntimeError:
                LOG.exception("H3 VAE tile progress event failed")

    def update(self, count):
        count = int(count)
        if self.worker is None:
            self.bar.update(count)
            self.terminal.update(count)
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        self.pending.put((event, count))

    def finish(self):
        if self.worker is not None:
            self.pending.put(None)
            self.worker.join()
            self.worker = None
        self.terminal.close()


def _norm_weight(module, name, reference):
    norm = getattr(module, name)
    weight = norm.weight
    if weight is not None:
        return weight
    cache_name = f"_turing_utils_{name}_unit_norm"
    cached = getattr(module, cache_name, None)
    if (
        cached is None
        or cached.device != reference.device
        or cached.dtype != reference.dtype
        or cached.numel() != module.dim_head
    ):
        cached = torch.ones(
            module.dim_head,
            device=reference.device,
            dtype=reference.dtype,
        )
        setattr(module, cache_name, cached)
    return cached


def _projected_attention(
    module,
    query,
    key,
    value,
    rotary_pos_emb,
    transformer_options,
):
    executor = transformer_options.get(ATTENTION_EXECUTOR_KEY)
    if callable(executor):
        query_norm = _norm_weight(module, "norm_q", query)
        key_norm = _norm_weight(module, "norm_k", key)
        rot_dim = int(rotary_pos_emb.shape[-3] * 2) if rotary_pos_emb is not None else 0
        transform = QKTransformSpec(
            query_norm=RMSNormSpec(query_norm, float(module.norm_q.eps), "head"),
            key_norm=RMSNormSpec(key_norm, float(module.norm_k.eps), "head"),
            rotary=RotaryEmbeddingSpec(
                rotary_pos_emb,
                rot_dim,
                "split_half" if rotary_pos_emb is not None else "none",
            ),
        )
        outcome = execute_projected_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            heads=module.heads,
            qk_transform=transform,
            transformer_options=transformer_options,
            container_factory=AttentionTensorContainer,
        )
        if outcome.supported:
            return outcome.output.nan_to_num_(0.0)

    query = comfy.rmsnorm.rms_norm(query, module.norm_q.weight, module.norm_q.eps)
    key = comfy.rmsnorm.rms_norm(key, module.norm_k.weight, module.norm_k.eps)

    if rotary_pos_emb is not None:
        rot = rotary_pos_emb.shape[-3] * 2
        query[..., :rot], key[..., :rot] = comfy.quant_ops.ck.apply_rope_split_half(
            query[..., :rot], key[..., :rot], rotary_pos_emb
        )

    query = AttentionTensorContainer(query.transpose(1, 2))
    key = AttentionTensorContainer(key.transpose(1, 2))
    value = AttentionTensorContainer(value.transpose(1, 2))
    out = h3_vae.optimized_attention(
        query,
        key,
        value,
        module.heads,
        skip_reshape=True,
        transformer_options=transformer_options,
    )
    return out.nan_to_num_(0.0)


def _apply_split_half_rope(value, rotary_pos_emb):
    if rotary_pos_emb is None:
        return value
    rot = int(rotary_pos_emb.shape[-3] * 2)
    apply_one = getattr(comfy.quant_ops.ck, "apply_rope_split_half1", None)
    if callable(apply_one):
        rotated = apply_one(value[..., :rot], rotary_pos_emb)
    else:
        source = (
            value[..., :rot]
            .reshape(*value.shape[:-1], 2, -1)
            .movedim(-2, -1)
            .unsqueeze(-2)
            .to(rotary_pos_emb.dtype)
        )
        rotated = (
            rotary_pos_emb[..., 0] * source[..., 0]
            + rotary_pos_emb[..., 1] * source[..., 1]
        )
        rotated = rotated.movedim(-1, -2).reshape(*value.shape[:-1], rot)
        rotated = rotated.to(value.dtype)
    if rot == value.shape[-1]:
        return rotated
    return torch.cat((rotated, value[..., rot:]), dim=-1)


def _asymmetric_projected_attention(
    module,
    query,
    key,
    value,
    query_rotary,
    key_rotary,
    transformer_options,
    qk_normalized=False,
):
    """Execute core-query/full-halo attention without padding either side.

    Prepared attention normally fuses a common Q/K RoPE table.  Shared-core Q
    is a subset of the window while K spans the full window, so normalize and
    rotate them independently before handing the true asymmetric tensors to
    the selected SDPA/Sage/W8A8 backend.
    """

    if not qk_normalized:
        query = comfy.rmsnorm.rms_norm(query, module.norm_q.weight, module.norm_q.eps)
        key = comfy.rmsnorm.rms_norm(key, module.norm_k.weight, module.norm_k.eps)
    query = _apply_split_half_rope(query, query_rotary)
    key = _apply_split_half_rope(key, key_rotary)
    out = h3_vae.optimized_attention(
        AttentionTensorContainer(query.transpose(1, 2)),
        AttentionTensorContainer(key.transpose(1, 2)),
        AttentionTensorContainer(value.transpose(1, 2)),
        module.heads,
        skip_reshape=True,
        transformer_options=transformer_options,
    )
    return out.nan_to_num_(0.0)


def _attention_forward(
    module,
    x,
    rotary_pos_emb,
    transformer_options,
):
    batch_size, seq_len, _ = x.shape
    qkv = module.to_qkv(x).view(batch_size, seq_len, -1, 3 * module.dim_head)
    query, key, value = torch.chunk(qkv, 3, dim=-1)
    out = _projected_attention(
        module,
        query,
        key,
        value,
        rotary_pos_emb,
        transformer_options,
    )
    return module.to_out(out)


def _attention_options(
    attention,
    device,
    *,
    prefetch_dynamic_vbars=False,
):
    override = make_attention_override(attention, device)
    options = {
        "optimized_attention_override": override,
        "prefetch_dynamic_vbars": bool(prefetch_dynamic_vbars),
    }
    executor = getattr(override, "prepared_attention_executor", None)
    if callable(executor):
        options[ATTENTION_EXECUTOR_KEY] = executor
    return options


def _clear_attention_caches(decoder):
    for block in decoder.transformer_blocks:
        attention = block.attn
        for name in (
            "_turing_utils_norm_q_unit_norm",
            "_turing_utils_norm_k_unit_norm",
        ):
            if hasattr(attention, name):
                delattr(attention, name)


def _fused_swiglu_eligible(linear):
    weight = linear.weight
    return bool(
        not comfy.model_management.in_training
        and isinstance(weight, comfy.ops.QuantizedTensor)
        and weight._layout_cls == "TensorWiseINT8Layout"
        and not getattr(weight._params, "transposed", False)
    )


class _SharedWindowLayout:
    def __init__(self, model, latent_t, latent_h, latent_w, device):
        self.latent_t = int(latent_t)
        self.latent_h = int(latent_h)
        self.latent_w = int(latent_w)
        y_idx, y_len, y_overlap = split_tiles(
            latent_h * model.vae_ratio,
            TILE_SIZE,
            TILE_OVERLAP,
            model.vae_ratio,
        )
        x_idx, x_len, x_overlap = split_tiles(
            latent_w * model.vae_ratio,
            TILE_SIZE,
            TILE_OVERLAP,
            model.vae_ratio,
        )
        self.y_idx = [value // model.vae_ratio for value in y_idx]
        self.y_len = [value // model.vae_ratio for value in y_len]
        self.y_overlap = [value // model.vae_ratio for value in y_overlap]
        self.x_idx = [value // model.vae_ratio for value in x_idx]
        self.x_len = [value // model.vae_ratio for value in x_len]
        self.x_overlap = [value // model.vae_ratio for value in x_overlap]
        self.descriptors = [
            (i, j, yi, yl, xi, xl)
            for i, (yi, yl) in enumerate(zip(self.y_idx, self.y_len))
            for j, (xi, xl) in enumerate(zip(self.x_idx, self.x_len))
        ]
        self.window_h = self.y_len[0]
        self.window_w = self.x_len[0]
        self.window_tokens = self.latent_t * self.window_h * self.window_w
        self.image_tokens = self.latent_t * self.latent_h * self.latent_w

        token_indices = []
        for _i, _j, yi, yl, xi, xl in self.descriptors:
            temporal = torch.arange(self.latent_t, device=device)[:, None, None]
            rows = torch.arange(yi, yi + yl, device=device)[None, :, None]
            columns = torch.arange(xi, xi + xl, device=device)[None, None, :]
            indices = (
                temporal * self.latent_h * self.latent_w
                + rows * self.latent_w
                + columns
            )
            token_indices.append(indices.reshape(-1))
        self.token_indices = torch.stack(token_indices)

    @staticmethod
    def _axis_overlap_weights(length, starts, lengths, device):
        """Build a full-overlap, normalized cosine partition of unity.

        Every covered token remains a query in every covering window.  This is
        deliberately more expensive than assigning almost the whole overlap
        to one owner, but avoids changing the attention context abruptly over
        a one-token ownership boundary.  Normalization also handles the
        three-way overlaps produced when fixed 256px windows cover a short
        image dimension.
        """
        weights = torch.zeros(len(starts), length, dtype=torch.float32, device=device)
        for index, (start, extent) in enumerate(zip(starts, lengths)):
            positions = torch.arange(
                extent,
                dtype=torch.float32,
                device=device,
            ).add_(0.5)
            window_weight = torch.ones_like(positions)
            if index > 0:
                overlap = starts[index - 1] + lengths[index - 1] - start
                if overlap > 0:
                    phase = (positions / overlap).clamp_(0.0, 1.0)
                    window_weight.mul_(0.5 - 0.5 * torch.cos(math.pi * phase))
            if index + 1 < len(starts):
                overlap = start + extent - starts[index + 1]
                if overlap > 0:
                    phase = ((extent - positions) / overlap).clamp_(0.0, 1.0)
                    window_weight.mul_(0.5 - 0.5 * torch.cos(math.pi * phase))
            weights[index, start : start + extent] = window_weight
        return weights / weights.sum(dim=0, keepdim=True).clamp_min_(1e-12)

    def query_groups(self, minimum_weight=0.0):
        device = self.token_indices.device
        y_weights = self._axis_overlap_weights(
            self.latent_h,
            self.y_idx,
            self.y_len,
            device,
        )
        x_weights = self._axis_overlap_weights(
            self.latent_w,
            self.x_idx,
            self.x_len,
            device,
        )
        spatial_weights_by_window = torch.stack(
            [
                y_weights[i, :, None] * x_weights[j, None, :]
                for i, j, *_unused in self.descriptors
            ]
        )
        minimum_weight = max(0.0, float(minimum_weight))
        if minimum_weight > 0.0:
            retained = spatial_weights_by_window >= minimum_weight
            # Always retain the strongest owner so aggressive experimental
            # thresholds cannot leave a token uncovered.
            strongest = spatial_weights_by_window.argmax(dim=0, keepdim=True)
            retained.scatter_(0, strongest, True)
            spatial_weights_by_window = spatial_weights_by_window.masked_fill(
                ~retained,
                0.0,
            )
            spatial_weights_by_window.div_(
                spatial_weights_by_window.sum(dim=0, keepdim=True).clamp_min_(1e-12)
            )

        global_indices_by_window = []
        local_indices_by_window = []
        weights_by_window = []
        spatial_weight_sum = torch.zeros(
            self.latent_h,
            self.latent_w,
            dtype=torch.float32,
            device=device,
        )
        for window_index, (i, j, yi, _yl, xi, _xl) in enumerate(self.descriptors):
            spatial_weights = spatial_weights_by_window[window_index]
            coordinates = torch.nonzero(spatial_weights > 0.0, as_tuple=False)
            rows, columns = coordinates[:, 0], coordinates[:, 1]
            token_weights = spatial_weights[rows, columns]
            spatial_weight_sum.index_put_(
                (rows, columns), token_weights, accumulate=True
            )
            temporal = torch.arange(self.latent_t, device=device)[:, None]
            spatial_global = rows * self.latent_w + columns
            spatial_local = (rows - yi) * self.window_w + (columns - xi)
            global_indices_by_window.append(
                (
                    temporal * self.latent_h * self.latent_w + spatial_global[None, :]
                ).reshape(-1)
            )
            local_indices_by_window.append(
                (
                    temporal * self.window_h * self.window_w + spatial_local[None, :]
                ).reshape(-1)
            )
            weights_by_window.append(token_weights.repeat(self.latent_t))

        if not torch.allclose(
            spatial_weight_sum,
            torch.ones_like(spatial_weight_sum),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError("shared-core overlap weights do not sum to one")

        windows_by_count = {}
        for window_index, global_indices in enumerate(global_indices_by_window):
            windows_by_count.setdefault(int(global_indices.numel()), []).append(
                window_index
            )
        grouped = {}
        for query_tokens, windows in windows_by_count.items():
            grouped[query_tokens] = (
                torch.tensor(windows, dtype=torch.long, device=device),
                torch.stack([global_indices_by_window[index] for index in windows]),
                torch.stack([local_indices_by_window[index] for index in windows]),
                torch.stack([weights_by_window[index] for index in windows]),
            )
        return grouped

    @property
    def window_count(self):
        return len(self.descriptors)


class _SharedSpatialPlan:
    """Immutable window and RoPE metadata for one decode-local spatial shape."""

    def __init__(self, model, latent_t, latent_h, latent_w, device, dtype):
        decoder = model.decoder
        self.layout = _SharedWindowLayout(
            model,
            latent_t,
            latent_h,
            latent_w,
            device,
        )
        suffix_tokens = 1 + decoder.num_register_tokens
        image_ids = h3_vae.create_token_ids(
            (latent_t, self.layout.window_h, self.layout.window_w),
            device,
            dtype,
        )
        suffix_ids = torch.zeros(
            (1, suffix_tokens, 3),
            dtype=image_ids.dtype,
            device=device,
        )
        self.rotary = decoder.pos_embed(torch.cat((image_ids, suffix_ids), dim=1))
        self.suffix_rotary_indices = torch.arange(
            self.layout.window_tokens,
            self.layout.window_tokens + suffix_tokens,
            device=device,
        )
        self._query_groups = {}
        self._overlap_batch_maps = {}
        self._overlap_backend_logged = False
        self.overlap_accumulate = None
        if is_supported_tensor_core_device(device):
            try:
                turing_sage = load_turing_sage()
                if turing_sage.overlap_accumulate_available():
                    self.overlap_accumulate = turing_sage.overlap_accumulate_compiled
            except (AttributeError, ImportError, OSError, RuntimeError):
                pass

    def query_groups(self, minimum_weight):
        key = float(minimum_weight)
        groups = self._query_groups.get(key)
        if groups is None:
            groups = self.layout.query_groups(key)
            self._query_groups[key] = groups
        return groups

    def overlap_batch_maps(self, minimum_weight, tiles_per_batch):
        """Return compact inverse maps for ordered streaming accumulation."""
        key = (float(minimum_weight), int(tiles_per_batch))
        cached = self._overlap_batch_maps.get(key)
        if cached is not None:
            return cached

        maps = {}
        query_groups = self.query_groups(minimum_weight)
        for core_tokens in sorted(query_groups):
            (
                grouped_windows,
                grouped_global_indices,
                _grouped_local_indices,
                grouped_weights,
            ) = query_groups[core_tokens]
            for group_start in range(
                0,
                grouped_windows.numel(),
                tiles_per_batch,
            ):
                window_count = min(
                    tiles_per_batch,
                    grouped_windows.numel() - group_start,
                )
                group_slice = slice(group_start, group_start + window_count)
                global_indices = grouped_global_indices[group_slice]
                weights = grouped_weights[group_slice]
                output_indices = torch.unique(global_indices, sorted=True)
                row_indices = torch.searchsorted(
                    output_indices,
                    global_indices.reshape(-1),
                )
                column_indices = (
                    torch.arange(window_count, device=global_indices.device)[:, None]
                    .expand_as(global_indices)
                    .reshape(-1)
                )
                local_indices = torch.full(
                    (output_indices.numel(), window_count),
                    -1,
                    dtype=torch.int32,
                    device=global_indices.device,
                )
                overlap_weights = torch.zeros(
                    (output_indices.numel(), window_count),
                    dtype=torch.float32,
                    device=global_indices.device,
                )
                query_positions = (
                    torch.arange(
                        core_tokens,
                        dtype=torch.int32,
                        device=global_indices.device,
                    )[None, :]
                    .expand_as(global_indices)
                    .reshape(-1)
                )
                local_indices[row_indices, column_indices] = query_positions
                overlap_weights[row_indices, column_indices] = weights.reshape(-1)
                maps[(core_tokens, group_start)] = (
                    output_indices.to(torch.int32).contiguous(),
                    local_indices.contiguous(),
                    overlap_weights.contiguous(),
                )
        self._overlap_batch_maps[key] = maps
        return maps


def _shared_spatial_plan(model, x, cache=None):
    key = (
        int(x.shape[2]),
        int(x.shape[3]),
        int(x.shape[4]),
        x.device.type,
        x.device.index,
        x.dtype,
    )
    if cache is None:
        return _SharedSpatialPlan(model, *key[:3], x.device, x.dtype)
    plan = cache.get(key)
    if plan is None:
        plan = _SharedSpatialPlan(model, *key[:3], x.device, x.dtype)
        cache[key] = plan
    return plan


def _feed_forward(module, value):
    if _fused_swiglu_eligible(module.w2):
        output = comfy.ops.linear_input_act(module.w2, module.w1(value), "swiglu")
    else:
        output = module(value)
    if output.shape != value.shape:
        raise RuntimeError(
            f"H3 VAE feed-forward returned {tuple(output.shape)} for input "
            f"{tuple(value.shape)}"
        )
    return output


def _latent_fingerprint(latent):
    value = latent.detach()
    if value.device.type == "cpu":
        payload = value.contiguous().view(torch.uint8).numpy().tobytes()
    else:
        flat = value.reshape(-1)
        stride = max(1, flat.numel() // 256)
        payload = (
            flat[::stride][:256]
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _reshape_decoder_patches(
    decoder,
    output,
    batch,
    latent_t,
    latent_h,
    latent_w,
):
    output = output.view(
        batch,
        latent_t,
        latent_h,
        latent_w,
        decoder.out_channels,
        decoder.patch_size_t,
        decoder.patch_size,
        decoder.patch_size,
    )
    output = output.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return output.reshape(
        batch,
        decoder.out_channels,
        latent_t * decoder.patch_size_t,
        latent_h * decoder.patch_size,
        latent_w * decoder.patch_size,
    )


def _shared_core_multiband_decoder_forward(
    model,
    x,
    transformer_options,
    tiles_per_batch,
    progress,
    spatial_cache=None,
    overlap_query_threshold=0.0,
    final_full_overlap_blocks=36,
):
    decoder = model.decoder
    batch, _, latent_t, latent_h, latent_w = x.shape
    # Geometry, RoPE, query groups, and inverse overlap maps are immutable for
    # one latent shape. Reuse them only through the decode-local cache; model
    # weights and outputs never enter this plan.
    plan = _shared_spatial_plan(model, x, spatial_cache)
    layout = plan.layout
    h = decoder.x_embedder(x.flatten(2).transpose(1, 2))
    dim = h.shape[-1]
    suffix_tokens = 1 + decoder.num_register_tokens
    suffix = torch.cat(
        (
            comfy.ops.cast_to_input(decoder.register_tokens, h)
            .view(1, 1, decoder.num_register_tokens, dim)
            .expand(batch, layout.window_count, -1, -1),
            torch.zeros(
                batch,
                layout.window_count,
                1,
                dim,
                dtype=h.dtype,
                device=h.device,
            ),
        ),
        dim=2,
    ).contiguous()
    rotary = plan.rotary
    suffix_rotary_indices = plan.suffix_rotary_indices
    linear_chunk = max(layout.window_tokens, tiles_per_batch * layout.window_tokens)
    blocks = list(decoder.transformer_blocks)
    overlap_query_threshold = max(0.0, float(overlap_query_threshold))
    final_full_overlap_blocks = min(
        max(0, int(final_full_overlap_blocks)), len(blocks)
    )
    final_full_overlap_start = len(blocks) - final_full_overlap_blocks
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(
        blocks,
        h.device,
        transformer_options,
    )

    for block_index, block in enumerate(blocks):
        comfy.model_prefetch.prefetch_queue_pop(
            prefetch_queue,
            h.device,
            block,
        )
        attention = block.attn
        # Q/K RMSNorm is token-local. Normalize each globally projected image
        # token once before overlapping windows gather it, instead of repeating
        # the same normalization for every window membership.
        query_image = torch.empty(
            batch,
            layout.image_tokens,
            attention.heads,
            attention.dim_head,
            dtype=h.dtype,
            device=h.device,
        )
        key_image = torch.empty_like(query_image)
        value_image = torch.empty_like(query_image)
        for token_start in range(0, layout.image_tokens, linear_chunk):
            token_end = min(token_start + linear_chunk, layout.image_tokens)
            normed = comfy.rmsnorm.rms_norm(
                h[:, token_start:token_end],
                block.norm1.weight,
                block.norm1.eps,
            )
            projected = attention.to_qkv(normed).view(
                batch,
                token_end - token_start,
                attention.heads,
                3 * attention.dim_head,
            )
            projected_q, projected_k, projected_v = torch.chunk(projected, 3, dim=-1)
            query_image[:, token_start:token_end].copy_(
                comfy.rmsnorm.rms_norm(
                    projected_q,
                    attention.norm_q.weight,
                    attention.norm_q.eps,
                )
            )
            key_image[:, token_start:token_end].copy_(
                comfy.rmsnorm.rms_norm(
                    projected_k,
                    attention.norm_k.weight,
                    attention.norm_k.eps,
                )
            )
            value_image[:, token_start:token_end].copy_(projected_v)

        # Suffix/register tokens remain independent per window, but their
        # linear projections do not depend on the compact image-query length.
        # Evaluate each suffix linear once across all windows so overlap
        # pruning cannot fragment it into many tiny GEMMs.
        suffix_flat = suffix.reshape(
            batch * layout.window_count,
            suffix_tokens,
            dim,
        )
        suffix_normed = comfy.rmsnorm.rms_norm(
            suffix_flat,
            block.norm1.weight,
            block.norm1.eps,
        )
        suffix_qkv = attention.to_qkv(suffix_normed).view(
            batch,
            layout.window_count,
            suffix_tokens,
            attention.heads,
            3 * attention.dim_head,
        )
        suffix_q, suffix_k, suffix_v = torch.chunk(suffix_qkv, 3, dim=-1)
        suffix_q = comfy.rmsnorm.rms_norm(
            suffix_q,
            attention.norm_q.weight,
            attention.norm_q.eps,
        )
        suffix_k = comfy.rmsnorm.rms_norm(
            suffix_k,
            attention.norm_k.weight,
            attention.norm_k.eps,
        )
        suffix_attention = torch.zeros_like(suffix)

        block_threshold = (
            0.0
            if block_index >= final_full_overlap_start
            else overlap_query_threshold
        )
        query_groups = plan.query_groups(block_threshold)
        streaming_overlap = (
            plan.overlap_accumulate
            if h.dtype in (torch.float16, torch.bfloat16)
            else None
        )
        overlap_batch_maps = (
            plan.overlap_batch_maps(block_threshold, tiles_per_batch)
            if streaming_overlap is not None
            else None
        )
        if streaming_overlap is not None and not plan._overlap_backend_logged:
            LOG.info(
                "MiniMax H3 VAE streaming deterministic FP32 overlap kernel active"
            )
            plan._overlap_backend_logged = True
        # Attention results from overlapping windows represent different local
        # contexts. Accumulate the deterministic partition-of-unity blend in
        # FP32 and in window order; this is the numerically validated path.
        image_attention = torch.zeros(
            h.shape,
            dtype=torch.float32,
            device=h.device,
        )
        for core_tokens in sorted(query_groups):
            (
                grouped_windows,
                grouped_global_indices,
                grouped_local_indices,
                grouped_weights,
            ) = query_groups[core_tokens]
            for group_start in range(0, grouped_windows.numel(), tiles_per_batch):
                window_tensor = grouped_windows[
                    group_start : group_start + tiles_per_batch
                ]
                group_slice = slice(group_start, group_start + window_tensor.numel())
                window_count = window_tensor.numel()
                key_indices = layout.token_indices.index_select(0, window_tensor)
                query_global_indices = grouped_global_indices[group_slice]
                query_local_indices = grouped_local_indices[group_slice]
                query_weights = grouped_weights[group_slice]

                window_key = key_image[:, key_indices].reshape(
                    batch * window_count,
                    layout.window_tokens,
                    attention.heads,
                    attention.dim_head,
                )
                window_value = value_image[:, key_indices].reshape(
                    batch * window_count,
                    layout.window_tokens,
                    attention.heads,
                    attention.dim_head,
                )
                query_q = query_image[:, query_global_indices]
                query_q = query_q.reshape(
                    batch * window_count,
                    core_tokens,
                    attention.heads,
                    attention.dim_head,
                )
                suffix_q_group = suffix_q.index_select(1, window_tensor).reshape(
                    batch * window_count,
                    suffix_tokens,
                    attention.heads,
                    attention.dim_head,
                )
                suffix_k_group = suffix_k.index_select(1, window_tensor).reshape(
                    batch * window_count,
                    suffix_tokens,
                    attention.heads,
                    attention.dim_head,
                )
                suffix_v_group = suffix_v.index_select(1, window_tensor).reshape(
                    batch * window_count,
                    suffix_tokens,
                    attention.heads,
                    attention.dim_head,
                )
                query = torch.cat((query_q, suffix_q_group), dim=1)
                key = torch.cat(
                    (
                        window_key,
                        suffix_k_group,
                    ),
                    dim=1,
                )
                value = torch.cat(
                    (window_value, suffix_v_group),
                    dim=1,
                )

                query_rotary_indices = torch.cat(
                    (
                        query_local_indices,
                        suffix_rotary_indices.expand(window_count, -1),
                    ),
                    dim=1,
                )
                query_rotary = rotary[0][query_rotary_indices]
                query_rotary = (
                    query_rotary.unsqueeze(0)
                    .expand(batch, *query_rotary.shape)
                    .reshape(
                        batch * window_count,
                        core_tokens + suffix_tokens,
                        *query_rotary.shape[2:],
                    )
                )
                key_rotary = rotary.expand(batch * window_count, *rotary.shape[1:])
                attended = _asymmetric_projected_attention(
                    attention,
                    query,
                    key,
                    value,
                    query_rotary,
                    key_rotary,
                    transformer_options,
                    qk_normalized=True,
                ).view(
                    batch,
                    window_count,
                    core_tokens + suffix_tokens,
                    dim,
                )

                if streaming_overlap is not None:
                    output_indices, local_map, weight_map = overlap_batch_maps[
                        (core_tokens, group_start)
                    ]
                    streaming_overlap(
                        attended[:, :, :core_tokens],
                        local_map,
                        weight_map,
                        output_indices,
                        image_attention,
                    )
                else:
                    for local_index, global_indices in enumerate(
                        query_global_indices
                    ):
                        weighted = attended[:, local_index, :core_tokens].mul(
                            query_weights[local_index].view(1, -1, 1)
                        )
                        # Every index is unique inside one window, and windows
                        # are consumed serially here. Avoid CUDA index_add
                        # atomics: their reduction order is not guaranteed.
                        accumulated = image_attention.index_select(
                            1, global_indices
                        )
                        accumulated.add_(weighted)
                        image_attention.index_copy_(
                            1, global_indices, accumulated
                        )

                suffix_attention.index_copy_(
                    1,
                    window_tensor,
                    attended[:, :, core_tokens:],
                )
        suffix_attention_flat = suffix_attention.reshape(
            batch * layout.window_count,
            suffix_tokens,
            dim,
        )
        suffix_flat.addcmul_(
            attention.to_out(suffix_attention_flat),
            comfy.ops.cast_to_input(block.scale1, suffix_flat),
        )
        suffix_normed = comfy.rmsnorm.rms_norm(
            suffix_flat,
            block.norm2.weight,
            block.norm2.eps,
        )
        suffix_flat.addcmul_(
            _feed_forward(block.ff, suffix_normed),
            comfy.ops.cast_to_input(block.scale2, suffix_flat),
        )
        del query_image, key_image, value_image
        del suffix_q, suffix_k, suffix_v, suffix_attention
        for token_start in range(0, layout.image_tokens, linear_chunk):
            token_end = min(token_start + linear_chunk, layout.image_tokens)
            h_slice = h[:, token_start:token_end]
            h_slice.addcmul_(
                attention.to_out(
                    image_attention[:, token_start:token_end].to(h.dtype)
                ),
                comfy.ops.cast_to_input(block.scale1, h_slice),
            )
            normed = comfy.rmsnorm.rms_norm(
                h_slice,
                block.norm2.weight,
                block.norm2.eps,
            )
            h_slice.addcmul_(
                _feed_forward(block.ff, normed),
                comfy.ops.cast_to_input(block.scale2, h_slice),
            )
        del image_attention

    comfy.model_prefetch.prefetch_queue_pop(
        prefetch_queue,
        h.device,
        None,
    )

    return _project_shared_state_windows(
        decoder,
        h,
        layout,
        tiles_per_batch,
        progress,
    )


def _project_shared_state_windows(
    decoder,
    h,
    layout,
    tiles_per_batch,
    progress,
):
    batch = h.shape[0]
    dim = h.shape[-1]
    pixel_y_idx = [value * decoder.patch_size for value in layout.y_idx]
    pixel_y_len = [value * decoder.patch_size for value in layout.y_len]
    pixel_x_idx = [value * decoder.patch_size for value in layout.x_idx]
    pixel_x_len = [value * decoder.patch_size for value in layout.x_len]
    assembler = _MultibandPixelAssembler(
        pixel_y_idx,
        pixel_y_len,
        pixel_x_idx,
        pixel_x_len,
        layout.latent_h * decoder.patch_size,
        layout.latent_w * decoder.patch_size,
        h.device,
    )
    for window_start in range(0, layout.window_count, tiles_per_batch):
        window_end = min(window_start + tiles_per_batch, layout.window_count)
        window_count = window_end - window_start
        window_indices = layout.token_indices[window_start:window_end]
        image_states = h[:, window_indices].reshape(
            batch * window_count, layout.window_tokens, dim
        )
        projected = decoder.proj_out(decoder.norm_out(image_states))
        decoded = _reshape_decoder_patches(
            decoder,
            projected,
            batch * window_count,
            layout.latent_t,
            layout.window_h,
            layout.window_w,
        ).view(
            batch,
            window_count,
            decoder.out_channels,
            layout.latent_t * decoder.patch_size_t,
            layout.window_h * decoder.patch_size,
            layout.window_w * decoder.patch_size,
        )
        for local_index, window_index in enumerate(range(window_start, window_end)):
            assembler.add(window_index, decoded[:, local_index])
        if progress is not None:
            progress.update(window_count)
    return assembler.finish()


def _decode_spatial(
    model,
    z,
    transformer_options,
    tiles_per_batch,
    progress,
    spatial_cache=None,
    overlap_query_threshold=0.0,
    final_full_overlap_blocks=36,
):
    z = model.post_quant_conv(z)
    return _shared_core_multiband_decoder_forward(
        model,
        z,
        transformer_options,
        tiles_per_batch,
        progress,
        spatial_cache,
        overlap_query_threshold,
        final_full_overlap_blocks,
    )


def _axis_multiband_weight(
    index,
    starts,
    lengths,
    dtype,
    device,
):
    length = lengths[index]
    weight = torch.ones(length, dtype=dtype, device=device)
    if index > 0:
        overlap = starts[index - 1] + lengths[index - 1] - starts[index]
        phase = (torch.arange(overlap, dtype=dtype, device=device) + 0.5) / overlap
        weight[:overlap].mul_(0.5 - 0.5 * torch.cos(math.pi * phase))
    if index + 1 < len(starts):
        overlap = starts[index] + lengths[index] - starts[index + 1]
        phase = (torch.arange(overlap, dtype=dtype, device=device) + 0.5) / overlap
        weight[-overlap:].mul_(
            torch.flip(0.5 - 0.5 * torch.cos(math.pi * phase), dims=(0,))
        )
    return weight


def _multiband_window_weights(
    y_index,
    x_index,
    y_starts,
    y_lengths,
    x_starts,
    x_lengths,
    dtype,
    device,
):
    y_weight = _axis_multiband_weight(y_index, y_starts, y_lengths, dtype, device)
    x_weight = _axis_multiband_weight(x_index, x_starts, x_lengths, dtype, device)
    low = y_weight[:, None] * x_weight[None, :]
    high = low.pow(_MULTIBAND_HIGH_WEIGHT_POWER)
    return low, high


def _multiband_denominators(
    descriptors,
    y_starts,
    y_lengths,
    x_starts,
    x_lengths,
    height,
    width,
    dtype,
    device,
):
    low_sum = torch.zeros(height, width, dtype=dtype, device=device)
    high_sum = torch.zeros_like(low_sum)
    for i, j, _zi, _zl, _zj, _zw in descriptors:
        low, high = _multiband_window_weights(
            i,
            j,
            y_starts,
            y_lengths,
            x_starts,
            x_lengths,
            dtype,
            device,
        )
        y = y_starts[i]
        x = x_starts[j]
        low_sum[y : y + y_lengths[i], x : x + x_lengths[j]].add_(low)
        high_sum[y : y + y_lengths[i], x : x + x_lengths[j]].add_(high)
    tiny = torch.finfo(dtype).tiny
    return low_sum.clamp_min_(tiny), high_sum.clamp_min_(tiny)


def _spatial_lowpass(tile):
    batch, channels, frames, height, width = tile.shape
    frame_batch = tile.permute(0, 2, 1, 3, 4).reshape(
        batch * frames, channels, height, width
    )
    low_height = max(1, math.ceil(height / _MULTIBAND_DOWNSAMPLE))
    low_width = max(1, math.ceil(width / _MULTIBAND_DOWNSAMPLE))
    low = F.interpolate(
        frame_batch,
        size=(low_height, low_width),
        mode="area",
    )
    low = F.interpolate(
        low,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return low.view(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


class _MultibandPixelAssembler:
    def __init__(
        self,
        y_starts,
        y_lengths,
        x_starts,
        x_lengths,
        height,
        width,
        device,
    ):
        self.y_starts = y_starts
        self.y_lengths = y_lengths
        self.x_starts = x_starts
        self.x_lengths = x_lengths
        self.height = height
        self.width = width
        self.columns = len(x_starts)
        descriptors = [
            (i, j, 0, 0, 0, 0)
            for i in range(len(y_starts))
            for j in range(len(x_starts))
        ]
        self.low_sum, self.high_sum = _multiband_denominators(
            descriptors,
            y_starts,
            y_lengths,
            x_starts,
            x_lengths,
            height,
            width,
            torch.float32,
            device,
        )
        self.low_canvas = None
        self.high_canvas = None

    def add(self, window_index, tile):
        i, j = divmod(window_index, self.columns)
        y, x = self.y_starts[i], self.x_starts[j]
        if self.low_canvas is None:
            self.low_canvas = torch.zeros(
                *tile.shape[:-2],
                self.height,
                self.width,
                dtype=torch.float32,
                device=tile.device,
            )
            self.high_canvas = torch.zeros_like(self.low_canvas)
        low_weight, high_weight = _multiband_window_weights(
            i,
            j,
            self.y_starts,
            self.y_lengths,
            self.x_starts,
            self.x_lengths,
            torch.float32,
            tile.device,
        )
        low_weight = (
            low_weight
            / self.low_sum[y : y + self.y_lengths[i], x : x + self.x_lengths[j]]
        )
        high_weight = (
            high_weight
            / self.high_sum[y : y + self.y_lengths[i], x : x + self.x_lengths[j]]
        )
        tile = tile.float()
        self.low_canvas[
            ..., y : y + tile.shape[-2], x : x + tile.shape[-1]
        ].addcmul_(tile, low_weight)
        self.high_canvas[
            ..., y : y + tile.shape[-2], x : x + tile.shape[-1]
        ].addcmul_(tile, high_weight)

    def finish(self):
        # Split frequencies only after both complete canvases exist.  The old
        # tile-local split used different boundary conditions for every tile;
        # because low and high bands also had different feather weights, even
        # identical overlapping pixels did not reconstruct identically and
        # produced a characteristic pair of seam lines.  A global split keeps
        # one sampling phase and has the useful invariant
        # low_canvas == high_canvas => output == either canvas.
        for frame_start in range(0, self.high_canvas.shape[2], _MULTIBAND_FRAME_CHUNK):
            frame_end = min(
                frame_start + _MULTIBAND_FRAME_CHUNK,
                self.high_canvas.shape[2],
            )
            frame_slice = slice(frame_start, frame_end)
            low = _spatial_lowpass(self.low_canvas[:, :, frame_slice])
            high = self.high_canvas[:, :, frame_slice]
            high.sub_(_spatial_lowpass(high)).add_(low)
        self.low_canvas = None
        return self.high_canvas


class _PixelWriter:
    def __init__(self, output, model, device):
        self.output = output
        self.model = model
        self.write_pos = 0
        self.double_buffer = (
            output.device.type == "cpu"
            and device.type == "cuda"
            and torch.cuda.is_available()
        )
        self.copy_stream = None
        self.staging = [None, None]
        self.pending = [None, None]
        self.next_slot = 0
        if self.double_buffer:
            self.copy_stream = torch.cuda.Stream(device=device)

    def _flush(self, index):
        entry = self.pending[index]
        if entry is None:
            return
        staging, done, start, frames, _source = entry
        done.synchronize()
        self.output[:, :, start : start + frames].copy_(staging[:, :, :frames])
        self.pending[index] = None

    def write(self, part):
        part_frames = part.shape[2]
        copy_frames = min(part_frames, max(0, self.output.shape[2] - self.write_pos))
        if copy_frames <= 0:
            return
        part = part[:, :, :copy_frames]
        part = self.model._finalize_pixels(part)
        part = part.to(self.output.dtype)
        start = self.write_pos
        self.write_pos += copy_frames
        if not self.double_buffer:
            self.output[:, :, start : start + copy_frames].copy_(part)
            return

        index = self.next_slot
        self._flush(index)
        staging = self.staging[index]
        if (
            staging is None
            or staging.shape != part.shape
            or staging.dtype != part.dtype
        ):
            try:
                staging = torch.empty_like(part, device="cpu", pin_memory=True)
            except RuntimeError:
                self._flush(0)
                self._flush(1)
                self.double_buffer = False
                self.copy_stream = None
                self.staging = [None, None]
                LOG.warning(
                    "H3 VAE could not allocate a pinned decoder buffer; using synchronous pixel copies"
                )
                self.output[:, :, start : start + copy_frames].copy_(part)
                return
            self.staging[index] = staging
        ready = torch.cuda.Event()
        done = torch.cuda.Event()
        torch.cuda.current_stream(part.device).record_event(ready)
        self.copy_stream.wait_event(ready)
        with torch.cuda.stream(self.copy_stream):
            staging.copy_(part, non_blocking=True)
            done.record(self.copy_stream)
        part.record_stream(self.copy_stream)
        # Keep the source alive until the side-stream D2H copy completes. This
        # also prevents the caching allocator from recycling its storage for
        # the next decoded chunk.
        self.pending[index] = (staging, done, start, copy_frames, part)
        self.next_slot = 1 - index
        self._flush(self.next_slot)

    def finish(self):
        self._flush(0)
        self._flush(1)
        return self.output


def _decode_temporal(
    model,
    z,
    transformer_options,
    tiles_per_batch,
    progress,
    output_device=None,
    output_dtype=torch.float32,
    overlap_query_threshold=0.0,
    final_full_overlap_blocks=36,
):
    chunk_dec = model.tokens_chunk_size * model.vae_ratio_t
    split_count = int(model.token_drop > 0) + 1
    if output_device is None:
        output_device = comfy.model_management.intermediate_device()
    output = torch.empty(
        model.decode_output_shape(z.shape),
        dtype=output_dtype,
        device=output_device,
    )
    writer = _PixelWriter(output, model, z.device)
    spatial_cache = {}

    pad_tokens, num_chunks = model._decode_temporal_chunks(z.shape[2])
    if pad_tokens > 0:
        pad_z = z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)
        z = torch.cat((z, pad_z), dim=2)

    dec_overlap = None
    for i in range(num_chunks):
        start = i * model.tokens_chunk_size
        end = start + model.tokens_chunk_size + model.token_overlap
        clip_z = z[:, :, start:end]
        clip_dec = _decode_spatial(
            model,
            clip_z,
            transformer_options,
            tiles_per_batch,
            progress,
            spatial_cache,
            overlap_query_threshold,
            final_full_overlap_blocks,
        )

        for j in range(split_count):
            frame_start = j * chunk_dec
            frame_end = min(frame_start + chunk_dec, clip_dec.shape[2])
            part = clip_dec[:, :, frame_start:frame_end]
            part = part[:, :, model.frame_pre_padding :]
            if j == 0:
                if dec_overlap is not None:
                    part = model.blend(dec_overlap, part, model.frame_overlap, dim=-3)
                    dec_overlap = None
                writer.write(part)
            else:
                dec_overlap = part.contiguous()

        if i == num_chunks - 1 and dec_overlap is not None:
            writer.write(dec_overlap)
            dec_overlap = None
    return writer.finish()


def decode_video(
    vae,
    latent,
    attention="sdpa",
    *,
    output_device=None,
    overlap_query_threshold=0.0,
    final_full_overlap_blocks=36,
):
    model = require_h3_video_vae(vae)
    # H3 advertises FP16/FP32 to ComfyUI and defaults to FP16 on supported
    # NVIDIA GPUs, including Turing.  Follow the dtype used to load this VAE so
    # an explicit global --fp32-vae override cannot create mixed-dtype modules.
    compute_dtype = vae.vae_dtype
    if output_device is None:
        output_device = vae.output_device
    output_device = torch.device(output_device)
    output_dtype = vae.vae_output_dtype()
    block_count = len(model.decoder.transformer_blocks)
    overlap_query_threshold = float(overlap_query_threshold)
    if not 0.0 <= overlap_query_threshold < 1.0:
        raise ValueError(
            "H3 VAE overlap_query_threshold must be in [0, 1), got "
            f"{overlap_query_threshold}"
        )
    final_full_overlap_blocks = int(final_full_overlap_blocks)
    if not 0 <= final_full_overlap_blocks <= block_count:
        raise ValueError(
            "H3 VAE final_full_overlap_blocks must be between 0 and "
            f"{block_count}, got {final_full_overlap_blocks}"
        )
    tile_count = _spatial_tile_count(
        latent.shape[-2] * model.vae_ratio,
        latent.shape[-1] * model.vae_ratio,
    )
    tile_tokens = (
        min(latent.shape[-2] * model.vae_ratio, TILE_SIZE) // model.vae_ratio
    ) * (min(latent.shape[-1] * model.vae_ratio, TILE_SIZE) // model.vae_ratio)
    duplicate_ratio = tile_count * tile_tokens / (
        latent.shape[-2] * latent.shape[-1]
    )
    prefetch_dynamic_vbars = vae.patcher.is_dynamic()
    LOG.info(
        "H3 VAE shared-core multiband decoder active: windows=%d duplicate_spatial_ratio=%.2fx overlap_threshold=%.4f final_full_overlap_blocks=%d weight_lifecycle=%s",
        tile_count,
        duplicate_ratio,
        overlap_query_threshold,
        final_full_overlap_blocks,
        "comfy_block_prefetch" if prefetch_dynamic_vbars else "resident",
    )
    storage_ptr = latent.untyped_storage().data_ptr()
    LOG.info(
        "H3 VAE decode input: shape=%s dtype=%s device=%s storage=0x%x fingerprint=%s",
        tuple(latent.shape),
        latent.dtype,
        latent.device,
        storage_ptr,
        _latent_fingerprint(latent),
    )
    batch_tiles, memory = _select_tiles_per_batch(
        vae,
        tile_count,
        lambda count: _decode_memory_requirement(
            vae,
            latent.shape,
            count,
            compute_dtype,
        ),
        _AUTO_DECODE_TILE_BATCH_LIMIT,
    )
    comfy.model_management.load_models_gpu(
        [vae.patcher], memory_required=memory, force_full_load=vae.disable_offload
    )
    progress = None
    try:
        transformer_options = _attention_options(
            attention,
            vae.device,
            prefetch_dynamic_vbars=prefetch_dynamic_vbars,
        )
        z = latent.to(device=vae.device, dtype=compute_dtype)
        mean = model.latents_mean.view(1, -1, 1, 1, 1).to(z)
        std = model.latents_std.view(1, -1, 1, 1, 1).to(z)
        z = z * std + mean
        temporal_chunks = (
            1 if z.shape[2] == 1 else model._decode_temporal_chunks(z.shape[2])[1]
        )
        progress_units = tile_count * temporal_chunks
        progress = _TileProgress(
            progress_units,
            z.device,
            "H3 VAE Decode Tiles",
        )
        if z.shape[2] == 1:
            dec = _decode_spatial(
                model,
                z,
                transformer_options,
                batch_tiles,
                progress,
                None,
                overlap_query_threshold,
                final_full_overlap_blocks,
            )[:, :, -1:]
            dec = model._finalize_pixels(dec)
            return dec.to(
                device=output_device,
                dtype=output_dtype,
                copy=True,
            ).movedim(1, -1)
        dec = _decode_temporal(
            model,
            z,
            transformer_options,
            batch_tiles,
            progress,
            output_device,
            output_dtype,
            overlap_query_threshold,
            final_full_overlap_blocks,
        )
        return dec.movedim(1, -1)
    finally:
        if progress is not None:
            progress.finish()
        _clear_attention_caches(model.decoder)


def encode_video(vae, pixels):
    """Compatibility facade for the encode pipeline moved to its own module."""
    from .video_vae_encode import encode_video as implementation

    return implementation(vae, pixels)
