"""Thin ComfyUI nodes for Bernini conditioning and context windows."""

from __future__ import annotations

import comfy.context_windows
import comfy.patcher_extension
import comfy.utils
import node_helpers
from comfy_api.latest import io

from ..adapters import bernini as service


class BerniniInpaintCondition(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsBerniniInpaintCondition",
            display_name="Bernini Inpaint Condition",
            category="Turing Utils/Bernini",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Image.Input("source_video"),
                io.Int.Input("width", default=832, min=16, max=8192, step=16),
                io.Int.Input("height", default=480, min=16, max=8192, step=16),
                io.Int.Input("length", default=81, min=1, max=8192, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Boolean.Input("source_as_context", default=False, tooltip="Also append the aligned source video as Bernini context tokens."),
                io.Mask.Input("mask", optional=True, tooltip="White is repainted and black is preserved. Omit for global repaint."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, source_video, width, height, length, batch_size,
                source_as_context=False, mask=None):
        length = int(length)
        if (length - 1) % 4 != 0:
            raise ValueError(f"Bernini length must be 4*n+1 real frames; got {length}")

        source_video, mask = service._align_source_video_and_mask(
            source_video, mask, length
        )
        source_video, mask = service._resize_source_video_and_mask(
            source_video, mask, width, height
        )

        source_latent = vae.encode(source_video)
        source_latent = comfy.utils.repeat_to_batch_size(source_latent, int(batch_size))
        latent = {"samples": source_latent}
        if mask is not None:
            latent["noise_mask"] = service._upper_bound_latent_mask(mask, source_latent)

        context = []
        roles = []
        if source_as_context:
            context.append(source_latent)
            roles.append("aligned")

        if context:
            values = {
                "context_latents": context,
                service._CONTEXT_ROLES_KEY: tuple(roles),
            }
            positive = node_helpers.conditioning_set_values(positive, values)
            negative = node_helpers.conditioning_set_values(negative, values)
        return io.NodeOutput(positive, negative, latent)


class BerniniContextWindowsCore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "context_length": (
                    "INT",
                    {
                        "default": 81,
                        "min": 1,
                        "max": 16385,
                        "step": 4,
                        "tooltip": "The length of the context window in real frames. Must be 4*n + 1.",
                    },
                ),
                "context_overlap": (
                    "INT",
                    {
                        "default": 28,
                        "min": 0,
                        "max": 16384,
                        "step": 4,
                        "tooltip": "The overlap of the context window in real frames.",
                    },
                ),
                "position_mode": (
                    ["absolute", "relative"],
                    {
                        "default": "absolute",
                        "tooltip": "Absolute keeps global latent-frame RoPE positions; relative matches official ComfyUI window-local positions.",
                    },
                ),
                "context_schedule": (
                    [
                        comfy.context_windows.ContextSchedules.STATIC_STANDARD,
                        comfy.context_windows.ContextSchedules.UNIFORM_STANDARD,
                        comfy.context_windows.ContextSchedules.UNIFORM_LOOPED,
                        comfy.context_windows.ContextSchedules.BATCHED,
                    ],
                    {
                        "default": comfy.context_windows.ContextSchedules.UNIFORM_STANDARD,
                        "tooltip": "Step-dependent scheduling algorithm for context windows.",
                    },
                ),
                "context_stride": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 32,
                        "tooltip": "The stride of the context window; only applicable to uniform schedules.",
                    },
                ),
                "closed_loop": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Whether to close the context window loop; only applicable to looped schedules.",
                    },
                ),
                "fuse_method": (
                    comfy.context_windows.ContextFuseMethods.LIST_STATIC,
                    {
                        "default": comfy.context_windows.ContextFuseMethods.PYRAMID,
                        "tooltip": "The method to use to fuse the context windows.",
                    },
                ),
                "freenoise": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Whether to apply FreeNoise noise shuffling, improves window blending.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "Turing Utils/Bernini"
    TITLE = "Bernini Context Windows"

    def apply(
        self,
        model,
        context_length: int,
        context_overlap: int,
        context_schedule: str,
        position_mode: str = "absolute",
        context_stride: int = 1,
        closed_loop: bool = False,
        fuse_method: str = comfy.context_windows.ContextFuseMethods.PYRAMID,
        freenoise: bool = True,
    ):
        if position_mode not in ("absolute", "relative"):
            raise ValueError(f"position_mode must be absolute or relative; got {position_mode}")
        latent_context_length, latent_context_overlap = service._validate_context_window_frames(
            context_length,
            context_overlap,
        )
        context_handler = service.BerniniScheduledContextHandler(
            context_schedule=comfy.context_windows.get_matching_context_schedule(context_schedule),
            fuse_method=comfy.context_windows.get_matching_fuse_method(fuse_method),
            context_length=latent_context_length,
            context_overlap=latent_context_overlap,
            context_stride=max(int(context_stride), 1),
            closed_loop=bool(closed_loop),
            dim=2,
            freenoise=bool(freenoise),
            cond_retain_index_list="",
            split_conds_to_windows=False,
            latent_retain_index_list="",
            causal_window_fix=True,
            turing_utils_absolute_positions=position_mode == "absolute",
        )
        comfy.patcher_extension.add_callback(
            comfy.context_windows.IndexListCallbacks.RESIZE_COND_ITEM,
            service._resize_bernini_context,
            context_handler.callbacks,
        )

        patched = model.clone()
        patched.model_options["context_handler"] = context_handler
        patched.model_options.setdefault("transformer_options", {})
        base_model = getattr(patched, "model", None)
        if base_model is not None and callable(getattr(base_model, "extra_conds", None)) and hasattr(patched, "add_object_patch"):
            patched.add_object_patch(
                "extra_conds", service._make_extra_conds_with_bernini_roles(base_model)
            )

        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
            "ContextWindows_prepare_sampling",
        )
        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
            "ContextWindows_sampler_sample",
        )
        patched.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            service._BERNINI_ROPE_WRAPPER_KEY,
        )

        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
            "ContextWindows_prepare_sampling",
            service._bernini_prepare_sampling_wrapper,
        )
        if freenoise:
            comfy.context_windows.create_sampler_sample_wrapper(patched)
        if position_mode == "absolute":
            service._install_bernini_absolute_rope_patch()
            patched.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                service._BERNINI_ROPE_WRAPPER_KEY,
                service._bernini_context_rope_wrapper,
            )

        service.LOG.info(
            "Bernini context windows enabled: schedule=%s, length=%s -> %s latent frames, "
            "overlap=%s -> %s latent frames, stride=%s, closed_loop=%s, "
            "position=%s, causal_window_fix=True, fuse=%s",
            context_schedule,
            context_length,
            latent_context_length,
            context_overlap,
            latent_context_overlap,
            context_stride,
            closed_loop,
            position_mode,
            fuse_method,
        )
        return (patched,)
