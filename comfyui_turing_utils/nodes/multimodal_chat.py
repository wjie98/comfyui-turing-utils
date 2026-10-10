"""Multimodal Prompt Chat node schema and execution."""

from __future__ import annotations

import asyncio
import json
import time

from comfy_api.latest import io
from ..prompt.chat import (
    DEFAULT_CHAT_OPTIONS,
    DEFAULT_SYSTEM_PROMPT,
    build_chat_options,
    build_user_content,
    build_chat_request,
    extract_chat_text,
    normalize_chat_endpoint,
    resolve_api_key,
    request_chat_completion,
)


def _chat_option_inputs():
    return [
        io.Boolean.Input(
            "disable_thinking",
            default=DEFAULT_CHAT_OPTIONS.disable_thinking,
            tooltip="Send chat_template_kwargs.enable_thinking=false. Disable this option if the server rejects that extension.",
        ),
        io.Float.Input(
            "temperature",
            default=DEFAULT_CHAT_OPTIONS.temperature,
            min=-1.0,
            max=2.0,
            step=0.05,
            tooltip="-1 omits temperature and uses the server default.",
        ),
        io.Int.Input(
            "max_output_tokens",
            default=DEFAULT_CHAT_OPTIONS.max_output_tokens,
            min=1,
            max=131072,
            step=1,
        ),
        io.Combo.Input(
            "image_detail",
            options=["auto", "low", "high"],
            default=DEFAULT_CHAT_OPTIONS.image_detail,
        ),
        io.Combo.Input(
            "image_format",
            options=["jpeg", "png"],
            default=DEFAULT_CHAT_OPTIONS.image_format,
        ),
        io.Int.Input(
            "jpeg_quality",
            default=DEFAULT_CHAT_OPTIONS.jpeg_quality,
            min=40,
            max=100,
            step=1,
        ),
        io.Int.Input(
            "max_image_edge",
            default=DEFAULT_CHAT_OPTIONS.max_image_edge,
            min=256,
            max=8192,
            step=64,
        ),
        io.Float.Input(
            "video_sample_fps",
            default=DEFAULT_CHAT_OPTIONS.video_sample_fps,
            min=0.1,
            max=24.0,
            step=0.1,
        ),
        io.Int.Input(
            "video_max_frames",
            default=DEFAULT_CHAT_OPTIONS.video_max_frames,
            min=1,
            max=64,
            step=1,
        ),
        io.Int.Input(
            "video_max_edge",
            default=DEFAULT_CHAT_OPTIONS.video_max_edge,
            min=256,
            max=4096,
            step=64,
        ),
        io.Int.Input(
            "timeout_seconds",
            default=DEFAULT_CHAT_OPTIONS.timeout_seconds,
            min=1,
            max=3600,
            step=1,
        ),
        io.Int.Input(
            "max_retries",
            default=DEFAULT_CHAT_OPTIONS.max_retries,
            min=0,
            max=5,
            step=1,
        ),
        io.Float.Input(
            "retry_backoff",
            default=DEFAULT_CHAT_OPTIONS.retry_backoff,
            min=0.0,
            max=30.0,
            step=0.1,
        ),
        io.String.Input(
            "extra_body_json",
            multiline=True,
            default=DEFAULT_CHAT_OPTIONS.extra_body_json,
        ),
    ]


def _inline_chat_inputs():
    inputs = _chat_option_inputs()
    quality = next(item for item in inputs if item.id == "jpeg_quality")
    result = []
    for item in inputs:
        if item.id == "jpeg_quality":
            continue
        if item.id == "image_format":
            quality.advanced = True
            item = io.DynamicCombo.Input(
                "image_format",
                options=[
                    io.DynamicCombo.Option("jpeg", [quality]),
                    io.DynamicCombo.Option("png", []),
                ],
                optional=True,
                extra_dict={"advanced": True},
            )
        else:
            item.optional = True
            item.advanced = True
        result.append(item)
    return result


class MultimodalPromptChat(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TuringUtilsMultimodalPromptChat",
            display_name="Multimodal Prompt Chat",
            category="Turing Utils/Prompt",
            description=(
                "Send one system/user turn to an OpenAI-compatible multimodal Chat "
                "Completions API. First/last frames receive explicit labels, images "
                "become <Picture N>, and 24 FPS IMAGE sequences are sampled into "
                "timestamped <Video N> frames. At least one of prompt or "
                "system_prompt must be non-empty."
            ),
            search_aliases=["LLM", "chat", "prompt enhance", "vision", "multimodal"],
            inputs=[
                io.String.Input(
                    "prompt", multiline=True, dynamic_prompts=True, default=""
                ),
                io.String.Input(
                    "system_prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default=DEFAULT_SYSTEM_PROMPT,
                ),
                io.String.Input(
                    "base_url",
                    default="https://api.openai.com",
                    tooltip="Root URL, a versioned URL, or the complete /chat/completions endpoint. Missing /v1 is added automatically.",
                ),
                io.String.Input(
                    "model",
                    default="",
                    tooltip="Exact model id exposed by the API server.",
                ),
                io.String.Input(
                    "api_key",
                    default="",
                    tooltip="Literal key, $NAME, or ${NAME}. Empty sends the placeholder key 'not-needed'. Literal keys are stored in the workflow.",
                ),
                io.Int.Input(
                    "cache_buster",
                    default=0,
                    min=0,
                    max=2**31 - 1,
                    step=1,
                    control_after_generate=True,
                    tooltip="Change this value to make ComfyUI issue a fresh request for otherwise identical inputs.",
                ),
                io.Image.Input(
                    "first_frame",
                    optional=True,
                    tooltip="Optional first frame, labeled <First Frame> instead of as a numbered reference picture.",
                ),
                io.Image.Input(
                    "last_frame",
                    optional=True,
                    tooltip="Optional last frame, labeled <Last Frame> instead of as a numbered reference picture.",
                ),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image"),
                        prefix="image_",
                        min=0,
                        max=16,
                    ),
                    tooltip="Optional reference images, labeled <Picture 1> onward in natural input order.",
                ),
                io.Autogrow.Input(
                    "videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "video",
                            tooltip="Consecutive video frames at 24 FPS, matching the H3 reference-conditioning convention.",
                        ),
                        prefix="video_",
                        min=0,
                        max=4,
                    ),
                    tooltip="Optional 24 FPS IMAGE frame sequences, uniformly sampled and labeled <Video 1> onward.",
                ),
                *_inline_chat_inputs(),
            ],
            outputs=[
                io.String.Output("enhanced_prompt"),
                io.String.Output("metadata_json"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        system_prompt: str,
        base_url: str,
        model: str,
        api_key: str,
        cache_buster: int = 0,
        first_frame=None,
        last_frame=None,
        images=None,
        videos=None,
        **chat_settings,
    ) -> io.NodeOutput:
        if not prompt.strip() and not system_prompt.strip():
            raise ValueError("prompt or system_prompt must not be empty")
        model = model.strip()
        if not model:
            raise ValueError("model must not be empty")
        chat_settings = dict(chat_settings)
        image_format = chat_settings.get("image_format")
        if isinstance(image_format, dict):
            encoding = image_format["image_format"]
            chat_settings["image_format"] = encoding
            if encoding == "jpeg":
                chat_settings["jpeg_quality"] = image_format.get(
                    "jpeg_quality", DEFAULT_CHAT_OPTIONS.jpeg_quality
                )
        options = build_chat_options(**chat_settings)

        endpoint = normalize_chat_endpoint(base_url)
        key = resolve_api_key(api_key)
        started = time.monotonic()
        user_content, media = await asyncio.to_thread(
            build_user_content,
            prompt,
            first_frame,
            last_frame,
            images,
            videos,
            options,
        )
        body = build_chat_request(
            model,
            system_prompt,
            user_content,
            options,
        )
        response = await asyncio.to_thread(
            request_chat_completion,
            endpoint,
            key,
            body,
            options.timeout_seconds,
            options.max_retries,
            options.retry_backoff,
        )
        text, choice = extract_chat_text(response)
        metadata = {
            "endpoint": endpoint,
            "model": response.get("model", model),
            "finish_reason": choice.get("finish_reason"),
            "usage": response.get("usage", {}),
            "media": media,
            "thinking_disabled": options.disable_thinking,
            "cache_buster": int(cache_buster),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        return io.NodeOutput(text, json.dumps(metadata, ensure_ascii=False, indent=2))
