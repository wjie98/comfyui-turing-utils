"""Public cards have canvas-only ports and cannot execute as ordinary nodes."""

import folder_paths
import comfy.samplers
from comfy_api.latest import io
from comfy_extras.nodes_resolution import AspectRatio
from ..nodes.attention import AttentionStrategy

from .graph import H3, SETTINGS, H3_SETTINGS


ImageAsset = io.Custom("TURING_CANVAS_IMAGE_ASSET")
VideoAsset = io.Custom("TURING_CANVAS_VIDEO_ASSET")
AudioAsset = io.Custom("TURING_CANVAS_AUDIO_ASSET")
CATEGORY = "Turing Utils/Canvas"


class CanvasCard(io.ComfyNode):
    @classmethod
    def execute(cls, **kwargs):
        raise ValueError("Use the material canvas card buttons; ordinary workflow execution is disabled")


def model_input(name, folder):
    return io.Combo.Input(name, options=["", *folder_paths.get_filename_list(folder)])


class CanvasSettings(CanvasCard):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=SETTINGS, display_name="Canvas Root", category=CATEGORY,
            inputs=[
                io.String.Input("work_directory", default="canvas/project", tooltip="Relative to this ComfyUI instance's output directory."),
                io.String.Input("cache_directory", default="canvas/project", advanced=True,
                    tooltip="Relative to this ComfyUI instance's .cache directory; only rebuildable previews, not weights."),
                io.Float.Input("max_megapixels", default=4.0, min=0.01),
            ], outputs=[])


class CanvasH3Settings(CanvasCard):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=H3_SETTINGS, display_name="Canvas H3 Settings", category=CATEGORY,
            inputs=[
                model_input("dit", "diffusion_models"), model_input("clip", "text_encoders"),
                model_input("video_vae", "vae"), model_input("audio_vae", "vae"),
                io.String.Input("loras", default="[]", socketless=True,
                    tooltip='JSON list: [{"name":"model.safetensors","strength":1.0}]'),
                io.Boolean.Input("force_int8_gemm", default=False, advanced=True),
                io.Combo.Input("attention", options=["w8a8", "sage", "sdpa"], advanced=True),
                io.DynamicCombo.Input("strategy", options=[
                    io.DynamicCombo.Option("disabled", []),
                    *AttentionStrategy.define_schema().inputs[1].options,
                ]),
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple"),
                io.Int.Input("steps", default=8, min=1, max=100),
                io.Boolean.Input("refiner", default=True,
                    tooltip="H3 Sigma Refiner: add one step by cosine redistribution of the existing low-noise tail (sigma <= 0.7). No eligible tail: unchanged."),
                io.Float.Input("shift_video", default=12, min=0.01),
                io.Float.Input("shift_audio", default=6, min=0.01),
                io.String.Input("chat_url", default="http://127.0.0.1:9200", advanced=True),
                io.String.Input("chat_model", default="", advanced=True),
                io.String.Input("chat_api_key_env", default="", advanced=True,
                    tooltip="Environment variable name only. No API secret is saved in the workflow."),
                io.String.Input("chat_system_prompt", default="Rewrite the user's intent into a precise video generation prompt. Preserve reference numbering. Return only the prompt.", multiline=True, advanced=True),
            ], outputs=[])


def asset_schema(node_id, title, output):
    inputs = [io.String.Input("asset_id", default="", advanced=True),
              io.String.Input("local_path", default="", advanced=True, tooltip="Upload fallback: path relative to this server instance's input directory.")]
    if output != ImageAsset:
        inputs += [io.Float.Input("start_seconds", default=0, min=0, step=0.01, round=0.001),
                   io.Float.Input("end_seconds", default=0, min=0, step=0.01, round=0.001, tooltip="End time in seconds; 0: to end")]
    if output != AudioAsset:
        inputs += [io.Float.Input("max_megapixels", default=0, min=0,
            tooltip="Maximum decoded megapixels; 0 inherits Canvas Root. Original file is unchanged.")]
    if output == VideoAsset:
        inputs += [io.Boolean.Input("include_audio", default=True)]
    outputs = [output.Output("image" if output == ImageAsset else "video" if output == VideoAsset else "audio")]
    return io.Schema(node_id=node_id, display_name=title, category=CATEGORY, inputs=inputs, outputs=outputs)


class CanvasImage(CanvasCard):
    @classmethod
    def define_schema(cls):
        return asset_schema("TuringCanvasImage", "Canvas Image", ImageAsset)


class CanvasVideo(CanvasCard):
    @classmethod
    def define_schema(cls):
        return asset_schema("TuringCanvasVideo", "Canvas Video", VideoAsset)


class CanvasAudio(CanvasCard):
    @classmethod
    def define_schema(cls):
        return asset_schema("TuringCanvasAudio", "Canvas Audio", AudioAsset)


class CanvasH3(CanvasCard):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=H3, display_name="Canvas H3 Generate", category=CATEGORY,
            inputs=[
                ImageAsset.Input("first_frame", optional=True), ImageAsset.Input("last_frame", optional=True),
                *[io.Autogrow.Input(kind + "s", optional=True, template=io.Autogrow.TemplatePrefix(
                    input=io.MultiType.Input(kind, types=[AudioAsset, VideoAsset]) if kind == "audio" else typ.Input(kind), prefix=kind + "_", min=0, max=64))
                  for kind, typ in (("image", ImageAsset), ("video", VideoAsset), ("audio", AudioAsset))],
                VideoAsset.Input("prefix", optional=True), VideoAsset.Input("target", optional=True),
                io.String.Input("user_prompt", default="", multiline=True),
                io.String.Input("model_prompt", default="", multiline=True),
                io.Combo.Input("aspect_ratio", options=[*[x.value for x in AspectRatio], "Custom"], default=AspectRatio.WIDESCREEN_H.value),
                io.Float.Input("megapixels", default=0.4, min=0.01, step=0.05),
                io.Int.Input("width", default=864, min=32, step=32),
                io.Int.Input("height", default=480, min=32, step=32),
                io.Float.Input("duration", default=5.0, min=0.21, max=150, step=0.1, round=0.001, tooltip="Seconds at 24 FPS; with target, its selected interval determines the duration. Internal padding is removed from the result."),
                io.Float.Input("denoise", default=1.0, min=0, max=1, step=0.01,
                    tooltip="KSampler denoise: 1 fully redraws, 0 skips sampling. Without target, lower values do not preserve an original image."),
                io.String.Input("filename_prefix", default="h3"),
                io.Boolean.Input("preserve_audio", default=True, advanced=True),
                io.Float.Input("start_seconds", default=0, min=0, step=0.01, round=0.001, advanced=True),
                io.Float.Input("end_seconds", default=0, min=0, step=0.01, round=0.001, advanced=True),
            ], outputs=[VideoAsset.Output("video")])


PUBLIC_NODES = {SETTINGS: CanvasSettings, "TuringCanvasImage": CanvasImage,
                "TuringCanvasVideo": CanvasVideo, "TuringCanvasAudio": CanvasAudio,
                H3: CanvasH3, H3_SETTINGS: CanvasH3Settings}
