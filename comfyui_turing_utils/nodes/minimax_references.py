"""ComfyUI schemas for composable MiniMax H3 references."""

from __future__ import annotations

from comfy_api.latest import io

from ..adapters.minimax.references import h3_latent_info
from ..adapters.minimax.reference_execution import (
    encode_keyframes,
    encode_images,
    encode_videos,
    encode_audios,
    encode_semantic,
    build_conditioning,
)


H3KeyframeReferenceType = io.Custom("TURING_UTILS_H3_KEYFRAME_REFERENCE")
H3ImageReferenceType = io.Custom("TURING_UTILS_H3_IMAGE_REFERENCE")
H3VideoReferenceType = io.Custom("TURING_UTILS_H3_VIDEO_REFERENCE")
H3AudioReferenceType = io.Custom("TURING_UTILS_H3_AUDIO_REFERENCE")
H3SemanticReferenceType = io.Custom("TURING_UTILS_H3_SEMANTIC_REFERENCE")


class H3KeyframeReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3KeyframeReference",
            display_name="H3 Keyframe Reference",
            category="Turing Utils/MiniMax H3",
            description=(
                "Encode three reusable H3 keyframes without assigning first/last "
                "roles. Each image_N input has a matching keyframe_N output. With an "
                "optional latent, images are cover-resized and cropped to its decoded "
                "pixel canvas."
            ),
            inputs=[
                io.Vae.Input("vae"),
                io.Latent.Input("latent", optional=True),
                *[
                    io.Image.Input(f"image_{index}", optional=True)
                    for index in range(3)
                ],
            ],
            outputs=[
                H3KeyframeReferenceType.Output(f"keyframe_{index}")
                for index in range(3)
            ],
        )

    @classmethod
    def execute(
        cls, vae, latent=None, image_0=None, image_1=None, image_2=None
    ) -> io.NodeOutput:
        return io.NodeOutput(*encode_keyframes(vae, latent, image_0, image_1, image_2))


class H3ImageReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3ImageReference",
            display_name="H3 Image Reference",
            category="Turing Utils/MiniMax H3",
            description=(
                "Encode dynamic H3 reference images without cropping or upscaling. "
                "A latent applies match-area sizing; without one, the short edge is "
                "limited by the megapixel budget."
            ),
            inputs=[
                io.Vae.Input("vae"),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.1,
                    max=16.0,
                    step=0.1,
                    tooltip="Maximum source area when latent is not connected; smaller images are not enlarged.",
                ),
                io.Latent.Input("latent", optional=True),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image"),
                        prefix="image_",
                        min=0,
                        max=32,
                    ),
                ),
            ],
            outputs=[H3ImageReferenceType.Output("image_reference")],
        )

    @classmethod
    def execute(cls, vae, megapixels=1.0, latent=None, images=None) -> io.NodeOutput:
        return io.NodeOutput(encode_images(vae, megapixels, latent, images))


class H3VideoReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3VideoReference",
            display_name="H3 Video Reference",
            category="Turing Utils/MiniMax H3",
            description=(
                "Encode dynamic H3 reference videos that were resampled to 24 FPS "
                "upstream, plus index-paired soundtracks. Qwen receives a 2 FPS view; "
                "the DiT receives the full VAE latent. A latent applies match-area "
                "sizing; without one, the megapixel budget limits source area."
            ),
            inputs=[
                io.Vae.Input("video_vae"),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.1,
                    max=16.0,
                    step=0.1,
                    tooltip="Maximum source area when latent is not connected; smaller videos are not enlarged.",
                ),
                io.Vae.Input("audio_vae", optional=True),
                io.Latent.Input("latent", optional=True),
                io.Autogrow.Input(
                    "videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "video",
                            tooltip="Consecutive frames already resampled to 24 FPS",
                        ),
                        prefix="video_",
                        min=0,
                        max=16,
                    ),
                ),
                io.Autogrow.Input(
                    "video_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("video_audio"),
                        prefix="video_audio_",
                        min=0,
                        max=16,
                    ),
                ),
            ],
            outputs=[H3VideoReferenceType.Output("video_reference")],
        )

    @classmethod
    def execute(
        cls,
        video_vae,
        megapixels=1.0,
        audio_vae=None,
        latent=None,
        videos=None,
        video_audios=None,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            encode_videos(
                video_vae, megapixels, audio_vae, latent, videos, video_audios
            )
        )


class H3AudioReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3AudioReference",
            display_name="H3 Audio Reference",
            category="Turing Utils/MiniMax H3",
            description="Encode a dynamic set of standalone H3 reference audio clips.",
            inputs=[
                io.Vae.Input("audio_vae"),
                io.Autogrow.Input(
                    "audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("audio"),
                        prefix="audio_",
                        min=0,
                        max=16,
                    ),
                ),
            ],
            outputs=[H3AudioReferenceType.Output("audio_reference")],
        )

    @classmethod
    def execute(cls, audio_vae, audios=None) -> io.NodeOutput:
        return io.NodeOutput(encode_audios(audio_vae, audios))


class H3SemanticReference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3SemanticReference",
            display_name="H3 Semantic Reference",
            category="Turing Utils/MiniMax H3",
            description=(
                "Run one exact Qwen3-VL multimodal encode over the prompt and selected "
                "H3 references. Reuse the semantic result independently of the "
                "reference selection and resolution in H3 Build Conditioning."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                H3KeyframeReferenceType.Input("first_frame", optional=True),
                H3KeyframeReferenceType.Input("last_frame", optional=True),
                H3ImageReferenceType.Input("image_reference", optional=True),
                H3VideoReferenceType.Input("video_reference", optional=True),
                H3AudioReferenceType.Input("audio_reference", optional=True),
            ],
            outputs=[H3SemanticReferenceType.Output("semantic_reference")],
        )

    @classmethod
    def execute(
        cls,
        clip,
        prompt: str,
        first_frame=None,
        last_frame=None,
        image_reference=None,
        video_reference=None,
        audio_reference=None,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            encode_semantic(
                clip,
                prompt,
                first_frame,
                last_frame,
                image_reference,
                video_reference,
                audio_reference,
            )
        )


class H3BuildConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3BuildConditioning",
            display_name="H3 Build Conditioning",
            category="Turing Utils/MiniMax H3",
            description=(
                "Combine a reusable Qwen semantic reference with current-resolution "
                "first-last-frame, image, video, and audio VAE references. These "
                "may differ from the semantic encoder's references; only connected "
                "keyframes must match the target latent's spatial grid."
            ),
            inputs=[
                H3SemanticReferenceType.Input("semantic_reference"),
                io.Latent.Input("latent"),
                H3KeyframeReferenceType.Input("first_frame", optional=True),
                H3KeyframeReferenceType.Input("last_frame", optional=True),
                H3ImageReferenceType.Input("image_reference", optional=True),
                H3VideoReferenceType.Input("video_reference", optional=True),
                H3AudioReferenceType.Input("audio_reference", optional=True),
            ],
            outputs=[io.Conditioning.Output("conditioning")],
        )

    @classmethod
    def execute(
        cls,
        semantic_reference,
        latent,
        first_frame=None,
        last_frame=None,
        image_reference=None,
        video_reference=None,
        audio_reference=None,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            build_conditioning(
                semantic_reference,
                latent,
                first_frame,
                last_frame,
                image_reference,
                video_reference,
                audio_reference,
            )
        )


class H3LatentInfo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsH3LatentInfo",
            display_name="H3 Latent Info",
            category="Turing Utils/MiniMax H3",
            description=(
                "Read the decoded pixel width, height, canonical frame count, and "
                "model FPS from an H3 video or nested AV latent without decoding it."
            ),
            inputs=[io.Latent.Input("latent")],
            outputs=[
                io.Int.Output("width"),
                io.Int.Output("height"),
                io.Int.Output("length"),
                io.Float.Output("fps"),
            ],
        )

    @classmethod
    def execute(cls, latent) -> io.NodeOutput:
        return io.NodeOutput(*h3_latent_info(latent))


__all__ = [
    "H3AudioReference",
    "H3BuildConditioning",
    "H3KeyframeReference",
    "H3ImageReference",
    "H3LatentInfo",
    "H3SemanticReference",
    "H3VideoReference",
]
