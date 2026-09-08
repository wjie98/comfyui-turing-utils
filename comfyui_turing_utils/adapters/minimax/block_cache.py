"""Trajectory-aware transformer-block caching for MiniMax H3."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import comfy.ldm.common_dit
import comfy.model_management
import comfy.model_prefetch
import comfy.patcher_extension
import torch
from comfy.ldm.minimax import model as minimax_model

from ...log import get_logger
from ..methods import weak_method


LOG = get_logger("minimax.cache")
CACHE_KEY = "turing_utils_minimax_h3_block_cache"
PATCH_KEY = "turing_utils_minimax_h3_block_cache"
FORWARD_PATCH_KEY = "diffusion_model._forward"
_CACHE_FORWARD_ATTR = "_turing_utils_minimax_block_cache_forward"

_MIB = 1 << 20
_CPU_TRANSFER_CHUNK_BYTES = 64 * _MIB
_SHORT_PROFILE_MAX_STEPS = 10
_SHORT_PROFILE_EDGE_BLOCKS = 6

_FORWARD_PARAMETERS = (
    "x",
    "timestep",
    "context",
    "transformer_options",
    "minimax_payload",
    "kwargs",
)


@dataclass(frozen=True, slots=True)
class CacheProfile:
    name: str
    threshold: float
    start_percent: float
    end_percent: float
    max_consecutive_hits: int
    max_hits: int | None
    edge_blocks: int


_PROFILES = {
    "standard": CacheProfile("standard", 0.08, 0.10, 0.90, 2, None, 0),
    "4-step LoRA": CacheProfile("4-step LoRA", 0.30, 0.20, 0.80, 1, 1, 6),
    "8-step LoRA": CacheProfile("8-step LoRA", 0.20, 0.25, 0.75, 1, 2, 6),
}
_PROFILE_NAMES = ("auto", *_PROFILES)


def _denoise_step_count(transformer_options) -> int | None:
    sample_sigmas = transformer_options.get("sample_sigmas")
    if sample_sigmas is None:
        return None
    try:
        return max(len(sample_sigmas) - 1, 0)
    except TypeError:
        return None


def _profile_block_range(
    profile: CacheProfile,
    block_count: int,
) -> tuple[int, int]:
    block_count = max(int(block_count), 0)
    edge = min(max(int(profile.edge_blocks), 0), block_count // 2)
    start, end = edge, block_count - edge
    # A profile must never create an empty span: there would be no block after
    # which the exact residual could be captured. Small synthetic models use
    # the full span; production H3 keeps the requested six edge blocks exact.
    return (0, block_count) if start >= end else (start, end)


class ResidualStore:
    """Own one exact cache tensor with bounded CPU/GPU staging."""

    def __init__(self, cache_device: str):
        if cache_device not in {"auto", "gpu", "cpu"}:
            raise ValueError(f"Unsupported block-cache device: {cache_device}")
        self.cache_device = cache_device
        self.tensor = None
        self.ready_event = None
        self.comfy_pinned = False
        self.storage_device = "none"
        self.last_storage_device = "none"

    def clear(self):
        if self.ready_event is not None:
            self.ready_event.synchronize()
        if self.comfy_pinned and self.tensor is not None:
            unpin_memory = getattr(
                comfy.model_management,
                "unpin_memory",
                None,
            )
            if callable(unpin_memory):
                unpin_memory(self.tensor)
        self.tensor = None
        self.ready_event = None
        self.comfy_pinned = False
        self.storage_device = "none"

    def compatible(self, hidden: torch.Tensor) -> bool:
        cached = self.tensor
        return (
            cached is not None
            and tuple(cached.shape) == tuple(hidden.shape)
            and cached.dtype == hidden.dtype
        )

    @staticmethod
    def _byte_size(tensor: torch.Tensor) -> int:
        return int(tensor.numel()) * int(tensor.element_size())

    def _keep_on_gpu(self, source: torch.Tensor, transformer_options) -> bool:
        if source.device.type != "cuda":
            return True
        if self.cache_device == "gpu":
            return True
        if self.cache_device == "cpu":
            return False
        byte_size = self._byte_size(source)
        free_memory = int(comfy.model_management.get_free_memory(source.device))
        reserve = int(comfy.model_management.minimum_inference_memory())
        # The GPU snapshot is an additional allocation. Honor ComfyUI's full
        # inference reserve before cloning; skipped blocks are not prefetched on
        # a hit, and the snapshot is released before a full middle-span pass.
        return free_memory > byte_size + reserve

    @staticmethod
    def _rows_per_chunk(tensor: torch.Tensor) -> int:
        if tensor.ndim == 0 or int(tensor.shape[0]) <= 1:
            return 1
        row_elements = max(int(tensor.numel()) // int(tensor.shape[0]), 1)
        row_bytes = row_elements * int(tensor.element_size())
        return max(_CPU_TRANSFER_CHUNK_BYTES // max(row_bytes, 1), 1)

    def capture(self, source: torch.Tensor, transformer_options):
        """Capture an immutable pre-span snapshot on the selected device."""

        self.clear()
        if self._keep_on_gpu(source, transformer_options):
            self.tensor = source.detach().clone()
            self.storage_device = self.tensor.device.type
            self.last_storage_device = self.storage_device
            return

        cached = torch.empty_like(source, device="cpu")
        pin_memory = getattr(comfy.model_management, "pin_memory", None)
        self.comfy_pinned = (
            bool(pin_memory(cached)) if callable(pin_memory) else False
        )
        rows = self._rows_per_chunk(source)
        for start in range(0, int(source.shape[0]), rows):
            end = min(start + rows, int(source.shape[0]))
            cached[start:end].copy_(
                source[start:end],
                non_blocking=self.comfy_pinned,
            )
        self.tensor = cached
        self.storage_device = "cpu"
        self.last_storage_device = self.storage_device
        if source.device.type == "cuda":
            self.ready_event = torch.cuda.Event()
            self.ready_event.record(torch.cuda.current_stream(source.device))

    def finish_residual(self, after: torch.Tensor):
        """Replace the captured snapshot with the exact ``after - before``."""

        cached = self.tensor
        if cached is None:
            raise RuntimeError("Block-cache snapshot is unavailable")
        if not self.compatible(after):
            raise RuntimeError("Block-cache snapshot changed shape or dtype")
        if cached.device == after.device:
            cached.neg_().add_(after)
            return
        if cached.device.type == "cpu" and after.device.type == "cuda":
            stream = torch.cuda.current_stream(after.device)
            if self.ready_event is not None:
                stream.wait_event(self.ready_event)
            rows = self._rows_per_chunk(cached)
            staging_shape = (min(rows, int(cached.shape[0])), *cached.shape[1:])
            staging = torch.empty(
                staging_shape,
                dtype=after.dtype,
                device=after.device,
            )
            for start in range(0, int(cached.shape[0]), rows):
                end = min(start + rows, int(cached.shape[0]))
                count = end - start
                chunk = staging[:count]
                chunk.copy_(cached[start:end], non_blocking=self.comfy_pinned)
                chunk.neg_().add_(after[start:end])
                cached[start:end].copy_(
                    chunk,
                    non_blocking=self.comfy_pinned,
                )
            self.ready_event = torch.cuda.Event()
            self.ready_event.record(stream)
            return
        cached.neg_().add_(
            after.detach().to(device=cached.device, dtype=cached.dtype)
        )

    def apply(self, hidden: torch.Tensor):
        cached = self.tensor
        if cached is None:
            raise RuntimeError("Block-cache residual is unavailable")
        if cached.device == hidden.device and cached.dtype == hidden.dtype:
            hidden.add_(cached)
            return
        if cached.device.type == "cpu" and hidden.device.type == "cuda":
            stream = torch.cuda.current_stream(hidden.device)
            if self.ready_event is not None:
                stream.wait_event(self.ready_event)
            rows = self._rows_per_chunk(cached)
            staging_shape = (min(rows, int(cached.shape[0])), *cached.shape[1:])
            staging = torch.empty(
                staging_shape,
                dtype=hidden.dtype,
                device=hidden.device,
            )
            for start in range(0, int(cached.shape[0]), rows):
                end = min(start + rows, int(cached.shape[0]))
                count = end - start
                chunk = staging[:count]
                chunk.copy_(cached[start:end], non_blocking=self.comfy_pinned)
                hidden[start:end].add_(chunk)
            self.ready_event = torch.cuda.Event()
            self.ready_event.record(stream)
            return
        hidden.add_(cached.to(device=hidden.device, dtype=hidden.dtype))


class MiniMaxH3BlockCache:
    def __init__(
        self,
        profile: CacheProfile,
        cache_device: str,
        block_count: int,
    ):
        self.profile = profile
        self.block_count = int(block_count)
        self.skip_start, self.skip_end = _profile_block_range(
            profile,
            self.block_count,
        )
        self.residual = ResidualStore(cache_device)
        self.reset()

    def reset(self):
        self.residual.clear()
        self.residual.last_storage_device = "none"
        self.current_percent = -1.0
        self.current_step_index = -1
        self.step_count = 0
        self.previous_signature = None
        self.shape_key = None
        self.schedule = None
        self.last_sigma = None
        self.accumulated_delta = 0.0
        self.consecutive_hits = 0
        self.full_steps = 0
        self.cache_hits = 0
        self.skipped_blocks = 0
        self.cache_ranges = ()
        self.full_run = False
        self.reject_counts = {}

    def finish(self):
        total = (self.full_steps + self.cache_hits) * self.block_count
        if total:
            LOG.info(
                "MiniMax H3 block cache: profile=%s device=%s acceleration=%.1f%% "
                "hits=%d full=%d",
                self.profile.name,
                self.residual.last_storage_device,
                100.0 * self.skipped_blocks / total,
                self.cache_hits,
                self.full_steps,
            )
        if self.reject_counts:
            details = ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(self.reject_counts.items())
            )
            LOG.debug("MiniMax H3 block cache diagnostics: %s", details)
        self.reset()

    def _reject(self, reason: str):
        self.reject_counts[reason] = self.reject_counts.get(reason, 0) + 1

    def set_cache_ranges(self, layout):
        segments = getattr(layout, "segments", ()) if layout is not None else ()
        self.cache_ranges = tuple(
            (kind, (int(start), int(end)))
            for start, end, kind in segments
            if int(end) > int(start)
        )

    def _step_info(self, transformer_options):
        sigmas = transformer_options.get("sigmas")
        if sigmas is None:
            return None
        current_flat = sigmas.flatten()
        if not current_flat.numel():
            return None
        current_sigma = float(current_flat[0].detach().float().item())

        if self.schedule is None:
            sample_sigmas = transformer_options.get("sample_sigmas")
            if sample_sigmas is None:
                return None
            values = sample_sigmas.detach().float().cpu().flatten().tolist()
            if not values:
                return None
            self.schedule = tuple(float(value) for value in values)
            self.step_count = max(len(self.schedule) - 1, 1)

        count = min(self.step_count, len(self.schedule))
        self.current_step_index = min(
            range(count), key=lambda index: abs(self.schedule[index] - current_sigma)
        )
        denominator = max(self.step_count - 1, 1)
        return current_sigma, self.current_step_index / denominator

    @staticmethod
    def _shape_key(hidden: torch.Tensor, cache_ranges):
        return (
            tuple(hidden.shape),
            hidden.dtype,
            hidden.device.type,
            hidden.device.index,
            tuple(cache_ranges),
        )

    @staticmethod
    def _signature(hidden: torch.Tensor, cache_ranges):
        ranges = cache_ranges or (("all", (0, int(hidden.shape[0]))),)
        feature_count = int(hidden.shape[1])
        sampled_features = min(feature_count, 64)
        feature_stride = max(feature_count // sampled_features, 1)
        row_budget = max(4096 // max(len(ranges) * sampled_features, 1), 1)
        parts = []
        for _, (start, end) in ranges:
            row_step = max(math.ceil((end - start) / row_budget), 1)
            part = hidden[start:end:row_step, ::feature_stride]
            parts.append(part[:row_budget, :sampled_features].reshape(-1))
        return torch.cat(parts).detach().float()

    def _clear_tensors(self):
        self.residual.clear()
        self.previous_signature = None
        self.accumulated_delta = 0.0
        self.consecutive_hits = 0
        self.full_run = False

    @staticmethod
    def _relative_tensor_change(current, reference) -> float:
        if current is None or reference is None or current.shape != reference.shape:
            return math.inf
        current_flat = current.detach().reshape(-1)
        reference_flat = reference.detach().reshape(-1)
        stride = max(current_flat.numel() // 4096, 1)
        current_sample = current_flat[::stride].float()
        reference_sample = reference_flat[::stride].float()
        denominator = reference_sample.abs().mean().clamp_min(1e-6)
        return float(((current_sample - reference_sample).abs().mean() / denominator).item())

    def complete_middle(self, after, _transformer_options):
        if not self.full_run:
            return
        if self.residual.tensor is not None:
            self.residual.finish_residual(after)
        self.full_run = False
        self.accumulated_delta = 0.0
        self.consecutive_hits = 0
        self.full_steps += 1

    def _start_full(self, hidden, transformer_options, *, store_residual: bool):
        self.residual.clear()
        self.full_run = True
        if store_residual:
            self.residual.capture(hidden, transformer_options)

    def prepare_middle(self, hidden, transformer_options, force_full=False) -> bool:
        step_info = self._step_info(transformer_options)
        if step_info is None:
            self._reject("missing_step_info")
            self._start_full(
                hidden,
                transformer_options,
                store_residual=False,
            )
            return False
        current_sigma, percent = step_info

        shape_key = self._shape_key(hidden, self.cache_ranges)
        if self.shape_key is not None and shape_key != self.shape_key:
            self._reject("shape_reset")
            self._clear_tensors()
        self.shape_key = shape_key
        if self.last_sigma is not None and current_sigma >= self.last_sigma:
            self._reject("discontinuity")
            self._clear_tensors()
        self.last_sigma = current_sigma
        self.current_percent = percent

        signature = self._signature(hidden, self.cache_ranges)
        delta = self._relative_tensor_change(signature, self.previous_signature)
        self.previous_signature = signature
        if math.isfinite(delta):
            self.accumulated_delta += delta

        eligible = True
        store_residual = True
        if force_full:
            self._reject("patch_overlap")
            eligible = False
            store_residual = False
        if not self.profile.start_percent <= percent <= self.profile.end_percent:
            self._reject("outside_percent")
            eligible = False
            if percent > self.profile.end_percent:
                store_residual = False
        if not self.residual.compatible(hidden):
            self._reject(
                "missing_residual"
                if self.residual.tensor is None
                else "residual_shape"
            )
            eligible = False
        if not math.isfinite(delta):
            self._reject("missing_delta")
            eligible = False

        if self.accumulated_delta > self.profile.threshold:
            self._reject("delta_threshold")
            eligible = False
        if self.consecutive_hits >= self.profile.max_consecutive_hits:
            self._reject("mcs")
            eligible = False
        if (
            self.profile.max_hits is not None
            and self.cache_hits >= self.profile.max_hits
        ):
            self._reject("hit_budget")
            eligible = False
            store_residual = False

        if not eligible:
            self._start_full(
                hidden,
                transformer_options,
                store_residual=store_residual,
            )
            return False

        self.residual.apply(hidden)
        self.full_run = False
        self.consecutive_hits += 1
        self.cache_hits += 1
        self.skipped_blocks += self.skip_end - self.skip_start
        return True


class MiniMaxH3BlockCacheGroup:
    def __init__(
        self,
        requested_profile: str,
        cache_device: str,
        block_count: int,
    ):
        if requested_profile not in _PROFILE_NAMES:
            raise ValueError(f"Unsupported block-cache profile: {requested_profile}")
        self.requested_profile = requested_profile
        self.cache_device = cache_device
        self.block_count = int(block_count)
        self.states = {}

    def _strategy(self, transformer_options):
        step_count = _denoise_step_count(transformer_options)
        if self.requested_profile == "auto":
            if step_count == 4:
                return "4-step LoRA"
            if step_count == 8:
                return "8-step LoRA"
            return "standard"
        if (
            self.requested_profile in {"4-step LoRA", "8-step LoRA"}
            and step_count is not None
            and step_count > _SHORT_PROFILE_MAX_STEPS
        ):
            return "standard"
        return self.requested_profile

    @staticmethod
    def _branch_key(transformer_options):
        uuids = transformer_options.get("uuids")
        if uuids:
            return "uuid", tuple(uuids)
        branches = transformer_options.get("cond_or_uncond")
        if branches is not None and len(branches) == 1:
            return "branch", tuple(branches)
        return ("default",)

    def state_for(self, transformer_options):
        strategy = self._strategy(transformer_options)
        key = (strategy, *self._branch_key(transformer_options))
        state = self.states.get(key)
        if state is not None:
            return state
        state = MiniMaxH3BlockCache(
            _PROFILES[strategy],
            self.cache_device,
            self.block_count,
        )
        self.states[key] = state
        LOG.info(
            "MiniMax H3 block cache selected profile=%s steps=%s requested=%s",
            strategy,
            _denoise_step_count(transformer_options),
            self.requested_profile,
        )
        return state

    def reset(self):
        for state in self.states.values():
            state.reset()
        self.states.clear()

    def finish(self):
        for state in self.states.values():
            state.finish()
        self.states.clear()


class _SamplingScope:
    def __init__(self, cache: MiniMaxH3BlockCacheGroup):
        self.cache = cache

    def __call__(self, executor, *args, **kwargs):
        self.cache.reset()
        try:
            return executor(*args, **kwargs)
        finally:
            self.cache.finish()


class _CleanupCache:
    def __init__(self, cache: MiniMaxH3BlockCacheGroup):
        self.cache = cache

    def __call__(self, *_args, **_kwargs):
        self.cache.reset()


def _run_block(
    model,
    hidden,
    index,
    t_emb,
    mod_segments,
    rope_freqs,
    transformer_options,
    blocks_replace,
):
    block = model.blocks[index]
    replacement = blocks_replace.get(("double_block", index))
    if replacement is None:
        return block(
            hidden,
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options=transformer_options,
        )

    def block_wrap(args):
        return {
            "img": block(
                args["img"],
                args["t_emb"],
                args["mod_segments"],
                args["rope_freqs"],
                transformer_options=args["transformer_options"],
            )
        }

    return replacement(
        {
            "img": hidden,
            "t_emb": t_emb,
            "mod_segments": mod_segments,
            "rope_freqs": rope_freqs,
            "transformer_options": transformer_options,
        },
        {"original_block": block_wrap},
    )["img"]


def _run_blocks(
    model,
    hidden,
    indices,
    t_emb,
    mod_segments,
    rope_freqs,
    transformer_options,
    blocks_replace,
    device,
    *,
    complete_after: int | None = None,
    complete_callback=None,
):
    indices = tuple(int(index) for index in indices)
    if not indices:
        return hidden
    blocks = [model.blocks[index] for index in indices]
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(
        blocks, device, transformer_options
    )
    for index, block in zip(indices, blocks):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
        hidden = _run_block(
            model,
            hidden,
            index,
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options,
            blocks_replace,
        )
        if index == complete_after and complete_callback is not None:
            complete_callback(hidden, transformer_options)
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)
    return hidden


def _span_has_replacement(blocks_replace, start, end):
    return any(("double_block", index) in blocks_replace for index in range(start, end))


def _cached_forward(
    self,
    x,
    timestep,
    context,
    transformer_options={},
    minimax_payload=None,
    **kwargs,
):
    cache_group = transformer_options.get(CACHE_KEY)
    if not isinstance(cache_group, MiniMaxH3BlockCacheGroup):
        raise RuntimeError("MiniMax H3 cached forward was called without its cache state")

    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype

    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = minimax_model.PackedLayout(
            text_len,
            latent_t,
            lat_h,
            lat_w,
            audio_t,
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
            frame_count=payload.get("frame_count"),
        )

    shift_v = float(
        transformer_options.get(
            "minimax_h3_sigma_shift_video", self.sigma_shift_video
        )
    )
    shift_a = float(
        transformer_options.get(
            "minimax_h3_sigma_shift_audio", self.sigma_shift_audio
        )
    )
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - minimax_model.time_shift_sigma(sigma_v, shift_v, shift_a))

    vis_aug = float(
        payload.get("visual_cond_noise_aug", minimax_model.VISUAL_COND_TIMESTEP)
    )
    aud_aug = float(
        payload.get("audio_cond_noise_aug", minimax_model.AUDIO_COND_TIMESTEP)
    )
    has_vis_cond = any(kind in ("cond", "ref_img") for _, _, kind in layout.segments)
    has_aud_cond = any(kind == "ref_audio" for _, _, kind in layout.segments)
    seg_t = {
        "text": t_v,
        "video": t_v,
        "audio": t_a,
        "cond": max(t_v, vis_aug),
        "ref_img": max(t_v, vis_aug),
        "ref_audio": max(t_a, aud_aug),
    }
    unique_t = sorted(
        {t_v, t_a}
        | ({seg_t["cond"]} if has_vis_cond else set())
        | ({seg_t["ref_audio"]} if has_aud_cond else set())
    )
    t_row = {value: index for index, value in enumerate(unique_t)}
    seg_tag = {
        "text": 1,
        "video": 0,
        "audio": 2,
        "cond": 0,
        "ref_img": 0,
        "ref_audio": 2,
    }

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for start, end, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for index in range(1, end - start + 1):
                if index == end - start or tags[index] != tags[run_start]:
                    mod_segments.append(
                        (
                            start + run_start,
                            start + index,
                            row_base + int(tags[run_start]),
                        )
                    )
                    run_start = index
        else:
            mod_segments.append((start, end, row_base + seg_tag[kind]))

    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = minimax_model.patchify_video(video_x.to(torch.float32), self.patch_size)
    audio_rows = minimax_model.pack_audio(audio_x.to(torch.float32))
    cond_video_rows = self._cond_video_rows(payload, device)
    cond_audio_rows = self._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(
            img_update.shape[0],
            video_rows.shape[1],
            dtype=torch.float32,
            device=device,
        )
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(
            audio_update.shape[0],
            audio_rows.shape[1],
            dtype=torch.float32,
            device=device,
        )
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    video_embed = self.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != self.hidden_size:
        text_states = self.token_refiner(
            self.condition_proj(text_states), transformer_options=transformer_options
        )

    hidden = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
    video_offset = 0
    audio_offset = 0
    for start, end, kind in layout.segments:
        count = end - start
        if kind == "text":
            hidden[start:end] = text_states
        elif kind in ("cond", "ref_img", "video"):
            hidden[start:end] = video_embed[video_offset : video_offset + count]
            video_offset += count
        else:
            hidden[start:end] = audio_embed[audio_offset : audio_offset + count]
            audio_offset += count

    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if self.use_adaln_curves:
        table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(
            table[i0], table[i0 + 1], (pos - i0).unsqueeze(1)
        )
    else:
        t_emb = self.time_embedder(t_vals).to(dtype)

    rope_freqs = minimax_model.rope_rotation_table(
        self.rope_freqs(layout.position_ids, device), dtype
    )

    cache = cache_group.state_for(transformer_options)
    cache.set_cache_ranges(layout)
    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    hidden = _run_blocks(
        self,
        hidden,
        range(0, cache.skip_start),
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options,
        blocks_replace,
        device,
    )
    force_full = _span_has_replacement(
        blocks_replace, cache.skip_start, cache.skip_end
    )
    if not cache.prepare_middle(hidden, transformer_options, force_full=force_full):
        hidden = _run_blocks(
            self,
            hidden,
            range(cache.skip_start, len(self.blocks)),
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options,
            blocks_replace,
            device,
            complete_after=cache.skip_end - 1,
            complete_callback=cache.complete_middle,
        )
    else:
        hidden = _run_blocks(
            self,
            hidden,
            range(cache.skip_end, len(self.blocks)),
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options,
            blocks_replace,
            device,
        )

    video_seg = next(
        (start, end, t_row[seg_t["video"]])
        for start, end, kind in layout.segments
        if kind == "video"
    )
    audio_seg = next(
        (start, end, t_row[seg_t["audio"]])
        for start, end, kind in layout.segments
        if kind == "audio"
    )
    video, audio = self.final_layer(hidden, t_emb, video_seg, audio_seg)
    video_out = minimax_model.unpatchify_video(
        video,
        latent_t,
        lat_h // 2,
        lat_w // 2,
        self.latents_dim,
        self.patch_size,
    )
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = minimax_model.unpack_audio(audio)
    return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]


def _compatible_forward(forward) -> bool:
    try:
        parameters = tuple(inspect.signature(forward).parameters)
    except (TypeError, ValueError):
        return False
    if parameters and parameters[0] == "self":
        parameters = parameters[1:]
    if parameters != _FORWARD_PARAMETERS:
        return False
    code = getattr(forward, "__code__", None)
    if code is None:
        return False
    required = {
        "PackedLayout",
        "patchify_video",
        "pack_audio",
        "time_shift_sigma",
        "rope_rotation_table",
        "make_prefetch_queue",
        "final_layer",
        "unpatchify_video",
        "unpack_audio",
    }
    return required.issubset(code.co_names)


def _make_cached_forward(diffusion_model):
    def forward(
        self,
        x,
        timestep,
        context,
        transformer_options={},
        minimax_payload=None,
        **kwargs,
    ):
        return _cached_forward(
            self,
            x,
            timestep,
            context,
            transformer_options=transformer_options,
            minimax_payload=minimax_payload,
            **kwargs,
        )

    setattr(forward, _CACHE_FORWARD_ATTR, True)
    return weak_method(forward, diffusion_model)


def install_minimax_block_cache(
    model,
    profile: str,
    cache_device: str,
):
    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, minimax_model.MiniMaxH3Model):
        raise ValueError("MiniMax H3 block cache only supports MiniMax H3 models")
    if len(diffusion_model.blocks) < 2:
        raise ValueError("MiniMax H3 block cache requires at least two transformer blocks")
    if not _compatible_forward(type(diffusion_model)._forward):
        raise RuntimeError(
            "MiniMax H3 block cache is disabled because MiniMaxH3Model._forward changed"
        )

    patched = model.clone()
    cache = MiniMaxH3BlockCacheGroup(
        profile,
        cache_device,
        len(diffusion_model.blocks),
    )
    transformer_options = patched.model_options.setdefault(
        "transformer_options", {}
    )
    transformer_options[CACHE_KEY] = cache
    existing_forward = getattr(patched, "object_patches", {}).get(FORWARD_PATCH_KEY)
    if existing_forward is not None:
        function = getattr(existing_forward, "__func__", existing_forward)
        if not getattr(function, _CACHE_FORWARD_ATTR, False):
            raise RuntimeError(
                "MiniMax H3 block cache cannot compose with another "
                "diffusion_model._forward object patch"
            )
    patched.add_object_patch(
        FORWARD_PATCH_KEY, _make_cached_forward(diffusion_model)
    )
    if hasattr(patched, "remove_wrappers_with_key"):
        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, PATCH_KEY
        )
    if hasattr(patched, "remove_callbacks_with_key"):
        patched.remove_callbacks_with_key(
            comfy.patcher_extension.CallbacksMP.ON_CLEANUP, PATCH_KEY
        )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        PATCH_KEY,
        _SamplingScope(cache),
    )
    patched.add_callback_with_key(
        comfy.patcher_extension.CallbacksMP.ON_CLEANUP,
        PATCH_KEY,
        _CleanupCache(cache),
    )
    return patched


__all__ = [
    "CACHE_KEY",
    "CacheProfile",
    "MiniMaxH3BlockCache",
    "MiniMaxH3BlockCacheGroup",
    "ResidualStore",
    "install_minimax_block_cache",
]
