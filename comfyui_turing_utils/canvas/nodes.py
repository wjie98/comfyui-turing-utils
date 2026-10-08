"""Public cards have canvas-only ports and cannot execute as ordinary nodes."""

import folder_paths
from comfy_api.latest import io

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
                io.Combo.Input("import_mode", options=["browser_upload", "local_copy"]),
                io.Float.Input("max_megapixels", default=2, min=0.01),
            ], outputs=[])


class CanvasH3Settings(CanvasCard):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=H3_SETTINGS, display_name="Canvas H3 Settings", category=CATEGORY,
            inputs=[
                model_input("dit", "diffusion_models"), model_input("clip", "text_encoders"),
                model_input("video_vae", "vae"), model_input("audio_vae", "vae"),
                io.String.Input("loras", default="[]", multiline=True, advanced=True,
                    tooltip='JSON list: [{"name":"model.safetensors","strength":1.0}]'),
                io.Combo.Input("attention", options=["w8a8", "sage", "sdpa"], advanced=True),
                io.DynamicCombo.Input("sol", options=[
                    io.DynamicCombo.Option("disabled", []),
                    io.DynamicCombo.Option("enabled", [
                        io.Float.Input("routing_threshold", default=1.0, min=0, max=1, step=0.01),
                        io.Int.Input("dense_prefix_steps", default=0, min=0),
                        io.Int.Input("dense_suffix_steps", default=0, min=0),
                        io.Int.Input("dense_prefix_layers", default=2, min=0),
                        io.Int.Input("dense_suffix_layers", default=0, min=0),
                        io.Boolean.Input("sparse_reference_image", default=False),
                        io.Boolean.Input("sparse_reference_video", default=True),
                        io.Boolean.Input("sparse_reference_audio", default=False),
                    ]),
                ]),
                io.Int.Input("steps", default=8, min=1, max=100, advanced=True),
                io.String.Input("sigmas", default="", advanced=True,
                    tooltip="Optional explicit sigma sequence, already shifted. Empty uses simple scheduler and steps."),
                io.Float.Input("shift_video", default=12, min=0.01, advanced=True),
                io.Float.Input("shift_audio", default=6, min=0.01, advanced=True),
                io.String.Input("chat_url", default="http://127.0.0.1:9200", advanced=True),
                io.String.Input("chat_model", default="", advanced=True),
                io.String.Input("chat_api_key_env", default="", advanced=True,
                    tooltip="Environment variable name only. No API secret is saved in the workflow."),
                io.String.Input("chat_system_prompt", default="Rewrite the user's intent into a precise video generation prompt. Preserve reference numbering. Return only the prompt.", multiline=True, advanced=True),
            ], outputs=[])


def asset_schema(node_id, title, output):
    inputs = [io.String.Input("asset_id", default="", advanced=True),
              io.String.Input("local_path", default="", tooltip="Local-copy mode: path relative to this server instance's input directory.")]
    if output != ImageAsset:
        inputs += [io.Float.Input("start_seconds", default=0, min=0),
                   io.Float.Input("duration_seconds", default=0, min=0, tooltip="0: to end")]
    if output == VideoAsset:
        inputs += [io.Float.Input("force_rate", default=0, min=0, tooltip="0: source FPS; H3 inputs are normalized to 24 FPS."),
                   io.Int.Input("custom_width", default=0, min=0), io.Int.Input("custom_height", default=0, min=0),
                   io.Int.Input("skip_first_frames", default=0, min=0),
                   io.Int.Input("frame_load_cap", default=0, min=0),
                   io.Int.Input("select_every_nth", default=1, min=1)]
    outputs = [output.Output("material")]
    if output == VideoAsset:
        outputs.append(AudioAsset.Output("soundtrack"))
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
                    input=typ.Input(kind), prefix=kind + "_", min=0, max=64))
                  for kind, typ in (("image", ImageAsset), ("video", VideoAsset), ("audio", AudioAsset))],
                VideoAsset.Input("prefix", optional=True), VideoAsset.Input("target", optional=True),
                io.String.Input("user_prompt", default="", multiline=True),
                io.String.Input("model_prompt", default="", multiline=True),
                io.Int.Input("width", default=832, min=32, step=32),
                io.Int.Input("height", default=480, min=32, step=32),
                io.Int.Input("frames", default=124, min=5, max=3600),
                io.Float.Input("denoise", default=1.0, min=0, max=1, step=0.01,
                    tooltip="KSampler denoise: 1 fully redraws, 0 skips sampling. Without target, lower values do not preserve an original image."),
                io.String.Input("filename_prefix", default="h3"),
                io.Boolean.Input("preserve_audio", default=True, advanced=True),
            ], outputs=[VideoAsset.Output("video"), AudioAsset.Output("audio")])


PUBLIC_NODES = {SETTINGS: CanvasSettings, "TuringCanvasImage": CanvasImage,
                "TuringCanvasVideo": CanvasVideo, "TuringCanvasAudio": CanvasAudio,
                H3: CanvasH3, H3_SETTINGS: CanvasH3Settings}
