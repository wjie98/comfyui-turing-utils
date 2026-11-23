"""Qwen Image 2.1 reference preparation and shared ConvRot QKV input."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import inspect

import torch

import comfy.model_management
import comfy.model_prefetch
import comfy.ops
import comfy.quant_ops
from .methods import OriginalMethod, weak_method
from ..log import get_logger
from ..quantization.fusions import convrot_w8_plain_tensors, convrot_w8_output_slice
from ..quantization.dispatch import quantize_convrot_int8_activation

try:
    from comfy.ldm.qwen_image21 import model as qwen
    from comfy.text_encoders.qwen_image21 import QwenImage21TEModel
    from comfy.ldm.wan.model_animate2 import PoseBranchCache
except ModuleNotFoundError as error:
    if error.name not in {"comfy.ldm.qwen_image21", "comfy.ldm.qwen_image21.model", "comfy.text_encoders.qwen_image21"}:
        raise
    qwen = QwenImage21TEModel = None


LOG = get_logger("qwen_image21")
OPTIONS_KEY = "turing_utils_qwen_image21"
_PREPARED_KEY = "turing_utils_qwen_image21_target_pe"
_PATCH_MARKER = "turing_utils_qwen_image21_forward"
_FORWARD_INPUTS = ("x", "timesteps", "context", "ref_latents", "image_slots", "transformer_options")
_ATTENTION_INPUTS = ("x", "pe", "attn_fn", "prefix_len", "transformer_options")


def _enabled(options):
    return options.get(OPTIONS_KEY, {}).get("enabled", True)


def _select_slot(self, key, create=True):
    if self._pending:
        for _, stream in self._pending.values():
            if stream is not None:
                stream.synchronize()
        self._pending = {}
    for index, slot in enumerate(self.slots):
        if slot["key"].shape == key.shape and torch.equal(slot["key"], key.to(slot["key"].device)):
            self.slots.append(self.slots.pop(index))
            self.slot = slot
            return True
    return PoseBranchCache.select(self, key, create=create)


def _make_model_forward(module, original):
    original = OriginalMethod.capture(original, module)

    def forward(self, x, timesteps, context, ref_latents=None, image_slots=None, transformer_options={}, **kwargs):
        patches = transformer_options.get("patches", {})
        hooked = (transformer_options.get("patches_replace", {}).get("dit")
                  or any(patches.get(name) for name in ("post_input", "single_block", "attn1_patch"))
                  or any(module._forward_hooks or module._forward_pre_hooks
                         for module in (self.img_in, *self.txt_in.modules())))
        if (not _enabled(transformer_options) or not self.prefix_cache_enabled
                or comfy.model_management.in_training or hooked
                or transformer_options.get("qwen_image21_cache", {}).get("device") == "off"):
            return original(self, x, timesteps, context, ref_latents, image_slots, transformer_options, **kwargs)

        batch, _, height, width = x.shape
        refs = list(ref_latents or [])
        slots = list(image_slots or [])
        prefix_len = context.shape[1] + sum(ref.shape[-2] * ref.shape[-1] for ref in refs)
        if not prefix_len:
            return original(self, x, timesteps, context, ref_latents, image_slots, transformer_options, **kwargs)
        key = qwen.prefix_cache_key(x, context, refs, slots)
        cache_bytes = 2 * len(self.transformer_blocks) * batch * prefix_len * self.inner_dim * x.element_size()
        if self.prefix_cache is not None:
            self.prefix_cache.select = weak_method(_select_slot, self.prefix_cache)
        cache, cached = self.select_prefix_cache(
            key, cache_bytes, x.device, transformer_options.get("qwen_image21_cache", {}))
        if cache is None or (cached and _PREPARED_KEY not in cache.slot):
            return original(self, x, timesteps, context, ref_latents, image_slots, transformer_options, **kwargs)

        if cached:
            hidden_states = self.img_in(x.flatten(2).transpose(1, 2))
            pe = cache.slot[_PREPARED_KEY]
        else:
            LOG.info("Qwen Image 2.1 reference cache: images=%d tokens=%d target_tokens=%d device=%s format=%s",
                     len(refs), prefix_len, height * width, cache.store_device, cache.dtype)
            hidden_states, pe, segments = self.build_sequence(x, context, refs, slots)
            prefix_states, hidden_states = hidden_states[:, :prefix_len], hidden_states[:, prefix_len:]
            prefix_pe, pe = pe[:, :prefix_len], pe[:, prefix_len:]
            cache.slot[_PREPARED_KEY] = pe.clone()

        t = ((timesteps * 1000).to(x.dtype) / 1000).to(x.dtype)
        temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]), x.dtype)
        scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)
        mod = (qwen._split_rows(scale1), qwen._split_rows(gate1.tanh()),
               qwen._split_rows(scale2), qwen._split_rows(gate2.tanh()), torch.zeros_like(scale1[:1, None]))

        transformer_options["total_blocks"] = len(self.transformer_blocks)
        transformer_options["block_type"] = "single"
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.transformer_blocks), x.device, transformer_options)
        comfy.model_prefetch.malloc_graph_begin(x.device)
        for i, block in enumerate(self.transformer_blocks):
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, malloc_scope="block")
            transformer_options["block_index"] = i
            if not cached:
                with comfy.model_prefetch.pause_malloc_graph():
                    prefix_attn = qwen.block_causal_attention(segments[:-1], transformer_options, cache, i, prefix_len)
                    prefix_states = block(prefix_states, mod, prefix_pe, prefix_attn, prefix_len, transformer_options)
            prefix = cache.take(i, x.device, x.dtype, batch)
            if cached:
                cache.prefetch(i + 1, x.device, x.dtype)
            prefix_k, prefix_v = prefix.unbind(1)
            attn_fn = qwen.prefix_cached_attention(prefix_k, prefix_v, transformer_options)
            hidden_states = block(hidden_states, mod, pe, attn_fn, 0, transformer_options)
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, malloc_scope="block")
        comfy.model_prefetch.malloc_graph_end()
        return self.proj_out(self.norm_out(hidden_states, temb[:-1])).transpose(1, 2).reshape(batch, self.out_channels, height, width)

    setattr(forward, _PATCH_MARKER, True)
    return weak_method(forward, module)


@contextmanager
def _cast_linear(linear, x):
    weight, bias, stream = comfy.ops.cast_bias_weight(linear, x, offloadable=True, compute_dtype=x.dtype, want_requant=True)
    try:
        yield weight, bias
    finally:
        comfy.ops.uncast_bias_weight(linear, weight, bias, stream)


def _make_attention_forward(module, original):
    original = OriginalMethod.capture(original, module)
    container = getattr(qwen, "AttentionTensorContainer", None)

    def forward(self, x, pe, attn_fn, prefix_len, transformer_options={}):
        linears = (self.to_q, self.to_k, self.to_v)
        if (not _enabled(transformer_options) or x.device.type != "cuda"
                or x.dtype != torch.bfloat16 or comfy.model_management.in_training
                or transformer_options.get("patches", {}).get("attn1_patch")
                or any(getattr(linear, "pre_quant_scale", None) is not None
                       or getattr(linear, "_full_precision_mm", False)
                       or "forward" in linear.__dict__
                       or linear._forward_hooks or linear._forward_pre_hooks for linear in linears)
                or any(convrot_w8_plain_tensors(linear.weight) is None for linear in linears)):
            return original(self, x, pe, attn_fn, prefix_len, transformer_options)
        with ExitStack() as stack:
            cast = [stack.enter_context(_cast_linear(linear, x)) for linear in linears]
            plain = [convrot_w8_plain_tensors(weight) for weight, _ in cast]
            if any(weight is None for weight in plain):
                return original(self, x, pe, attn_fn, prefix_len, transformer_options)
            groups = [getattr(weight._params, "convrot_groupsize", 256) for weight, _ in cast]
            if len(set(groups)) != 1:
                return original(self, x, pe, attn_fn, prefix_len, transformer_options)
            activation, activation_scale = quantize_convrot_int8_activation(x.reshape(-1, x.shape[-1]), groups[0])
            projected = [convrot_w8_output_slice(activation, activation_scale, weight, scale, bias,
                                               0, weight.shape[0], x.dtype)
                         for (weight, scale), (_, bias) in zip(plain, cast)]
        batch, tokens, _ = x.shape
        q, k, v = [value.view(batch, tokens, self.heads, -1) for value in projected]
        qw, _, qs = comfy.ops.cast_bias_weight(self.norm_q, q, offloadable=True)
        kw, _, ks = comfy.ops.cast_bias_weight(self.norm_k, k, offloadable=True)
        try:
            q, k = comfy.quant_ops.ck.rms_rope(q, k, pe, qw, kw, self.norm_q.eps)
        finally:
            comfy.ops.uncast_bias_weight(self.norm_q, qw, None, qs)
            comfy.ops.uncast_bias_weight(self.norm_k, kw, None, ks)
        if container is not None:
            q, k, v = container(q), container(k), container(v)
        return self.to_out[0](attn_fn(q, k, v, self.heads, preferred_attention=self.comfy_attention))

    setattr(forward, _PATCH_MARKER, True)
    return weak_method(forward, module)


def apply_qwen_image21_adapter(model, device):
    root = getattr(model.model, "diffusion_model", None)
    if qwen is None or type(root) is not qwen.QwenImage21Transformer2DModel:
        return 0
    current = model.get_model_object("diffusion_model._forward")
    parameters = tuple(inspect.signature(current).parameters)
    legacy = parameters == (*_FORWARD_INPUTS, "kwargs")
    if not legacy and parameters != (*_FORWARD_INPUTS, "prefix_cache", "kwargs"):
        LOG.warning("Qwen Image 2.1 optimization skipped: forward contract changed")
        return 0
    # Modern ComfyUI owns writable KV slots and their offload lifecycle. Its
    # native forward must run even when QKV projections use our quantization.
    if legacy and not getattr(current, _PATCH_MARKER, False):
        model.add_object_patch("diffusion_model._forward", _make_model_forward(root, current))
    count = 0
    for i, block in enumerate(root.transformer_blocks):
        name = f"diffusion_model.transformer_blocks.{i}.attn.forward"
        current_attention = model.get_model_object(name)
        if getattr(current_attention, _PATCH_MARKER, False):
            continue
        if tuple(inspect.signature(current_attention).parameters) != _ATTENTION_INPUTS:
            continue
        model.add_object_patch(name, _make_attention_forward(block.attn, current_attention))
        count += 1
    if count:
        cache_path = "reference preparation reuse" if legacy else "native cache handling"
        LOG.info("Qwen Image 2.1: shared W8A8 QKV input on %d blocks, %s", count, cache_path)
    return count


def configure(model, enabled):
    if qwen is None or type(getattr(model.model, "diffusion_model", None)) is not qwen.QwenImage21Transformer2DModel:
        raise ValueError("Qwen Image 2.1 optimization requires a Qwen Image 2.1 model")
    patched = model.clone()
    patched.model_options.setdefault("transformer_options", {})[OPTIONS_KEY] = {"enabled": bool(enabled)}
    apply_qwen_image21_adapter(patched, patched.load_device)
    return patched


def apply_qwen_image21_clip_adapter(clip):
    if QwenImage21TEModel is None or not isinstance(clip.cond_stage_model, QwenImage21TEModel) or clip.patcher.load_device.type != "cuda":
        return False
    key = "turing_utils_qwen21_quantized_encode"
    if clip.patcher.get_attachment(key):
        return False
    original = OriginalMethod.capture(clip.patcher.get_model_object("encode_token_weights"), clip.cond_stage_model)
    device = clip.patcher.load_device

    def encode(self, *args, **kwargs):
        with comfy.ops.use_quantized_matmul(self, device):
            return original(self, *args, **kwargs)

    clip.patcher.add_object_patch("encode_token_weights", weak_method(encode, clip.cond_stage_model))
    clip.patcher.set_attachments(key, True)
    return True
