"""MiniMax H3 video VAE decode pipeline and shared tile primitives."""

from __future__ import annotations

import hashlib
import math
import queue
import threading
from contextlib import contextmanager

import torch
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
from ...log import get_logger
from ...quantization.dispatch import register_backend
from ...quantization.operator_scope import use_turing_operator_backend


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
    tile_height = min(height, model.tile_size)
    tile_width = min(width, model.tile_size)
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
        resident_frames = model.vae_ratio_t
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
    compute_bytes = comfy.model_management.dtype_size(dtype)
    tile_pixels = (
        latent_shape[0]
        * tiles_per_batch
        * model.decoder.out_channels
        * resident_tokens
        * model.vae_ratio_t
        * tile_height
        * tile_width
    )
    pixel_elements = (
        latent_shape[0] * model.decoder.out_channels * resident_frames * height * width
    )
    # Independent decoded tiles, the native stitching canvas/row tails, and
    # finalized FP32 pixels can coexist before asynchronous output copies finish.
    pixel_workspace = tile_pixels * compute_bytes + pixel_elements * (
        2 * compute_bytes + 8
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
    # ModelPatcher.get_free_memory also counts evictable Aimdo weights. Extra
    # batching must use idle memory, not borrow it from the next DiT/CLIP run.
    available = comfy.model_management.get_free_memory(vae.device)
    unloaded_weights = max(0, vae.patcher.model_size() - vae.patcher.loaded_size())
    return max(
        0,
        int(available - comfy.model_management.extra_reserved_memory() - unloaded_weights),
    )


def _load_vae_for_tiles(vae, tile_count, memory_estimator, auto_limit):
    # Only the unavoidable one-tile requirement may request model eviction.
    # Measure optional batching after loading, reserving not-yet-resident VAE
    # weights as well (DynamicVRAM can populate them on first use).
    comfy.model_management.load_models_gpu(
        [vae.patcher], memory_required=memory_estimator(1),
        force_full_load=vae.disable_offload,
    )
    return _select_tiles_per_batch(vae, tile_count, memory_estimator, auto_limit)


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
        "MiniMax H3 VAE auto tile batch selected %d/%d: estimated %.0f MiB, non-evicting budget %.0f MiB%s",
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


def _decoder_forward(decoder, x, transformer_options):
    batch, _, latent_t, latent_h, latent_w = x.shape
    h = decoder.x_embedder(x.flatten(2).transpose(1, 2))
    num_patches = h.shape[1]
    num_suffix = 1 + decoder.num_register_tokens
    h = torch.cat(
        (
            h,
            comfy.ops.cast_to_input(decoder.register_tokens, h).expand(batch, -1, -1),
            torch.zeros_like(h[:, :1]),
        ),
        dim=1,
    )
    image_ids = h3_vae.create_token_ids(
        (latent_t, latent_h, latent_w), x.device, x.dtype
    ).expand(batch, -1, -1)
    suffix_ids = torch.zeros(
        batch, num_suffix, 3, dtype=image_ids.dtype, device=x.device
    )
    rotary = decoder.pos_embed(torch.cat((image_ids, suffix_ids), dim=1))
    blocks = list(decoder.transformer_blocks)
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(
        blocks, h.device, transformer_options
    )
    for block in blocks:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, h.device, block)
        normed = comfy.rmsnorm.rms_norm(h, block.norm1.weight, block.norm1.eps)
        h.addcmul_(
            _attention_forward(block.attn, normed, rotary, transformer_options),
            comfy.ops.cast_to_input(block.scale1, h),
        )
        normed = comfy.rmsnorm.rms_norm(h, block.norm2.weight, block.norm2.eps)
        h.addcmul_(
            _feed_forward(block.ff, normed),
            comfy.ops.cast_to_input(block.scale2, h),
        )
    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, h.device, None)
    output = decoder.proj_out(decoder.norm_out(h))[:, :num_patches]
    return _reshape_decoder_patches(
        decoder, output, batch, latent_t, latent_h, latent_w
    )


def _decode_spatial(model, z, transformer_options, tiles_per_batch, progress):
    height, width = z.shape[-2] * model.vae_ratio, z.shape[-1] * model.vae_ratio
    y_idx, y_len, y_overlap = model.split_tiles(height)
    x_idx, x_len, x_overlap = model.split_tiles(width)
    descriptors = [
        (
            i, j, y // model.vae_ratio, h // model.vae_ratio,
            x // model.vae_ratio, w // model.vae_ratio,
        )
        for i, (y, h) in enumerate(zip(y_idx, y_len))
        for j, (x, w) in enumerate(zip(x_idx, x_len))
    ]
    source_batch = z.shape[0]
    canvas = None
    row_tails = []
    out_y = 0
    for start in range(0, len(descriptors), tiles_per_batch):
        group = descriptors[start : start + tiles_per_batch]
        inputs = [z[..., y : y + h, x : x + w] for _, _, y, h, x, w in group]
        batched = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=0)
        decoded = _decoder_forward(
            model.decoder, model.post_quant_conv(batched), transformer_options
        )
        for (i, j, *_bounds), tile in zip(group, decoded.split(source_batch, dim=0)):
            if j == 0:
                new_tails = []
                left_tail = None
                out_x = 0
            # Keep unblended neighbors, exactly as MiniMaxH3VideoVAE.tiled_decode.
            if i < len(y_idx) - 1:
                new_tails.append(tile[..., -y_overlap[i]:, :].clone())
            next_left_tail = (
                tile[..., :, -x_overlap[j]:].clone() if j < len(x_idx) - 1 else None
            )
            if i > 0:
                tile = model.blend(row_tails[j], tile, y_overlap[i - 1], dim=-2)
            if j > 0:
                tile = model.blend(left_tail, tile, x_overlap[j - 1], dim=-1)
            left_tail = next_left_tail
            if i < len(y_idx) - 1:
                tile = tile[..., :-y_overlap[i], :]
            if j < len(x_idx) - 1:
                tile = tile[..., :, :-x_overlap[j]]
            if canvas is None:
                canvas = torch.empty(
                    *tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device
                )
            canvas[
                ..., out_y : out_y + tile.shape[-2], out_x : out_x + tile.shape[-1]
            ].copy_(tile)
            out_x += tile.shape[-1]
            if j == len(x_idx) - 1:
                row_tails = new_tails
                out_y += tile.shape[-2]
        if progress is not None:
            progress.update(len(group))
        del tile, decoded, batched
    return canvas


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


@contextmanager
def _vae_operator_scope():
    if register_backend():
        with use_turing_operator_backend():
            yield
    else:
        yield


@_vae_operator_scope()
def decode_video(
    vae,
    latent,
    attention="sdpa",
    *,
    output_device=None,
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
    tile_count = len(model.split_tiles(latent.shape[-2] * model.vae_ratio)[0]) * len(
        model.split_tiles(latent.shape[-1] * model.vae_ratio)[0]
    )
    prefetch_dynamic_vbars = vae.patcher.is_dynamic()
    LOG.info(
        "H3 VAE independent-window decoder active: windows=%d attention=%s weight_lifecycle=%s",
        tile_count,
        attention,
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
    batch_tiles, _ = _load_vae_for_tiles(
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
