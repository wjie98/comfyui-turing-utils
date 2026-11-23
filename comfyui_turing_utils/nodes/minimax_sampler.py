"""Native dynamic controls for one final-canvas H3 sampling task."""

from __future__ import annotations

import comfy.samplers
import folder_paths
from comfy_api.latest import io

from . import INTERNAL_NODE_NOTE
from .minimax import H3LatentUpscaleModel
from ..adapters.minimax.image_sampling import sample_h3_image
from ..adapters.minimax.latent_upscaler import load_h3_latent_upscaler
from ..adapters.minimax.sampling_sigmas import (
    DEFAULT_SIGMAS,
    DEFAULT_SECOND_SIGMAS,
    SECOND_PASS_PRESETS,
)
from ..adapters.minimax.sampling import needs_upscaler, sample_h3


def _curve_input(name, *, second=False):
    return io.DynamicCombo.Input(
        name,
        display_name="second curve" if second else "main curve",
        options=(
            [
                io.DynamicCombo.Option(
                    "preset",
                    [
                        io.Combo.Input(
                            "preset",
                            display_name="dynamic preset",
                            options=[name for name, _ in SECOND_PASS_PRESETS],
                            default=SECOND_PASS_PRESETS[0][0],
                            tooltip="Remapped to the current MODEL's video shift. LBH examples retain the author's recipe in base time.",
                        )
                    ],
                )
            ]
            if second
            else []
        )
        + [
            io.DynamicCombo.Option(
                "scheduler",
                [
                    io.Int.Input(
                        "steps",
                        display_name="base steps",
                        default=3 if second else 4,
                        min=1,
                        max=10000,
                    ),
                    io.Combo.Input(
                        "scheduler",
                        options=comfy.samplers.KSampler.SCHEDULERS,
                        default="simple",
                    ),
                    io.Float.Input(
                        "denoise",
                        display_name="denoise",
                        default=0.5 if second else 1.0,
                        min=0,
                        max=1,
                        step=0.01,
                    ),
                    io.Int.Input(
                        "tail_steps",
                        display_name="low-noise tail steps",
                        default=0 if second else 1,
                        min=0,
                        max=1000,
                        tooltip="Add evenly spaced points in base time between the last nonzero sigma and zero. Set to 0 to keep the native schedule.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                "manual",
                [
                    io.String.Input(
                        "values",
                        display_name="sigmas",
                        multiline=True,
                        default=DEFAULT_SECOND_SIGMAS if second else DEFAULT_SIGMAS,
                        tooltip="Complete absolute sigma curve ending at zero. No shift, extra points or reordering are applied.",
                    )
                ],
            ),
        ],
    )


def _mode_input(internal=False):
    def stage_inputs():
        return [
            io.Float.Input(
                "lowres_scale",
                display_name="lowres scale",
                default=0.75,
                min=0.05,
                max=1,
                step=0.05,
                tooltip="Width and height scale relative to the final canvas. Time and audio are not resized.",
            ),
            *(
                []
                if internal
                else [
                    io.Combo.Input(
                        "model_name",
                        display_name="upscaler model",
                        default="none",
                        options=[
                            "none",
                            *folder_paths.get_filename_list("latent_upscale_models"),
                        ],
                    )
                ]
            ),
            io.Int.Input(
                "mask_grow",
                display_name="highres mask grow (patch rings)",
                default=0,
                min=0,
                max=4096,
                tooltip="Grow the final high-resolution mask by patch rings. Low-resolution coverage includes this expanded region when resizing.",
            ),
        ]

    progressive = stage_inputs()
    progressive.insert(
        1,
        io.Int.Input(
            "lowres_steps",
            display_name="lowres steps",
            default=3,
            min=1,
            max=10000,
            tooltip="Run this many initial main-curve steps at low resolution, leaving at least one high-resolution step. Requires Euler.",
        ),
    )
    two_pass = stage_inputs()
    two_pass.insert(1, _curve_input("second_sigmas", second=True))
    return io.DynamicCombo.Input(
        "mode",
        display_name="sampling mode",
        options=[
            io.DynamicCombo.Option("single", []),
            io.DynamicCombo.Option("progressive", progressive),
            io.DynamicCombo.Option("two_pass", two_pass),
        ],
        tooltip="single: one stage; progressive: change resolution within one curve; two_pass: complete the main curve, then independently re-noise and refine.",
    )


def _seed_input():
    return io.Int.Input(
        "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True
    )


def _schema(internal):
    return io.Schema(
        node_id=(
            "_TuringUtilsH3KSamplerApply"
            if internal
            else "TuringUtilsMiniMaxH3KSampler"
        ),
        display_name=(
            "MiniMax H3 KSampler Apply (Internal)"
            if internal
            else "MiniMax H3 KSampler"
        ),
        category="" if internal else "Turing Utils/MiniMax H3",
        is_dev_only=internal,
        description=(INTERNAL_NODE_NOTE if internal else "")
        + (
            "One CONDITIONING and one final-size original LATENT. Single completes the main curve; "
            "progressive shares it across low/high resolution; two_pass finishes it before an independent second curve. "
            "Stage keyframes and masks are derived automatically. Original high-resolution keep regions and audio masks "
            "are preserved. Progress and attention protection span the whole task. "
            "Older workflows with a second LATENT must be rewired to this final-canvas contract."
        ),
        inputs=[
            io.Model.Input("model"),
            io.Conditioning.Input("conditioning"),
            io.Latent.Input(
                "latent",
                tooltip="Original final-size canvas with the timeline, audio and canonical mask. Low-resolution stages are derived internally.",
            ),
            _seed_input(),
            io.Combo.Input(
                "sampler_name",
                options=comfy.samplers.KSampler.SAMPLERS,
                default="euler",
            ),
            _curve_input("sigmas"),
            _mode_input(internal),
            *(
                [H3LatentUpscaleModel.Input("upscale_model", optional=True, lazy=True)]
                if internal
                else []
            ),
        ],
        outputs=[io.Latent.Output(display_name="latent")],
        search_aliases=[] if internal else ["H3 sampler", "H3 inpaint", "H3 hires"],
    )


class _H3KSamplerApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(True)

    @classmethod
    def check_lazy_status(cls, latent, mode=None, upscale_model=None, **kwargs):
        mode = mode or {}
        # ComfyUI supplies (value, original input name) for nested lazy inputs.
        settings = {
            name: value[0] if isinstance(value, tuple) else value
            for name, value in mode.items()
            if name in {"mode", "lowres_scale"}
        }
        if upscale_model is None and needs_upscaler(
            latent, settings, model=kwargs.get("model")
        ):
            return ["upscale_model"]
        return []

    @classmethod
    def execute(cls, mode=None, upscale_model=None, **kwargs):
        return io.NodeOutput(
            sample_h3(mode=mode, upscale_model=upscale_model, **kwargs)
        )


class MiniMaxH3KSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema(False)

    @classmethod
    def execute(cls, mode=None, **kwargs):
        mode = dict(mode or {})
        name = mode.pop("model_name", "none")
        upscale_model = None
        if name != "none" and needs_upscaler(
            kwargs["latent"], mode, model=kwargs["model"]
        ):
            upscale_model = load_h3_latent_upscaler(name, "auto")
        return io.NodeOutput(
            sample_h3(mode=mode, upscale_model=upscale_model, **kwargs)
        )


class MiniMaxH3ImageKSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsMiniMaxH3ImageKSampler",
            display_name="MiniMax H3 Image KSampler",
            category="Turing Utils/MiniMax H3",
            description="Sample one H3 Image latent with the checkpoint's PDD step count, Euler and CFG 1.",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("conditioning"),
                io.Latent.Input("latent"),
                _seed_input(),
                io.Float.Input("denoise", default=1.0, min=0, max=1, step=0.01),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, model, conditioning, latent, seed, denoise=1.0):
        return io.NodeOutput(sample_h3_image(model, conditioning, latent, seed, denoise))
