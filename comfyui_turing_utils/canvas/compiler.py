"""One selected card becomes a private ordinary ComfyUI execution graph."""

import json
import secrets
import uuid

from .graph import H3, material_inputs, signature, validate_canvas, image_ports, reference_ports
from .schema import VERSION, generation_values, strategy_inputs


def compile_task(graph, task, project):
    nodes, settings = validate_canvas(graph)
    task = str(task)
    node = nodes[task]
    if node["type"] != H3:
        raise ValueError("Only generation cards can be queued")
    materials = material_inputs(node, nodes, project.state(), project)
    values = generation_values(node["values"])
    denoise = float(values.get("denoise", 1.0))
    if not 0 <= denoise <= 1:
        raise ValueError("denoise must be between 0 and 1")
    values["seed"] = secrets.randbits(63)
    prompt = {}

    def add(key, cls, **inputs):
        prompt[key] = {"class_type": cls, "inputs": inputs}
        return [key, 0]

    reads = {}
    for name, ref in materials.items():
        ref = {**ref, "max_megapixels": ref.get("max_megapixels") or settings.get("max_megapixels", 4)}
        if name in {"prefix", "target"}:
            ref["spatial_multiple"] = 32
        reads[name] = add("read_" + name, "_TuringCanvasRead", directory=project.directory,
            reference=json.dumps(ref), width=int(values.get("width", 0)) if name in {"prefix", "target"} else 0,
            height=int(values.get("height", 0)) if name in {"prefix", "target"} else 0)

    for name in (("dit", "clip", "video_vae", "audio_vae") if denoise else ("video_vae", "audio_vae")):
        if not settings.get(name):
            raise ValueError(f"Select {name} in Canvas Settings")
    model = add("dit", "TuringUtilsConvRotDiffusionModelLoader", unet_name=settings.get("dit", ""),
        force_int8_gemm=settings.get("force_int8_gemm", False), patch_attention=settings.get("attention", "w8a8"))
    loras = json.loads(settings.get("loras", "[]"))
    if not isinstance(loras, list):
        raise ValueError("LoRAs must be a JSON list")
    for index, lora in enumerate(loras):
        if not lora.get("on", True) or not float(lora.get("strength", 1)):
            continue
        model = add(f"lora_{index}", "LoraLoaderModelOnly", model=model,
            lora_name=lora["name"], strength_model=float(lora.get("strength", 1)))
    model = add("shift", "MiniMaxH3SigmaShift", model=model,
        shift_video=settings.get("shift_video", 12), shift_audio=settings.get("shift_audio", 6))
    attention_inputs = strategy_inputs(settings)
    if attention_inputs is not None:
        model = add("strategy", "TuringUtilsAttentionStrategy", model=model, **attention_inputs)
    clip = add("clip", "TuringUtilsConvRotCLIPLoader", clip_name=settings.get("clip", ""), type="minimax", force_int8_gemm=settings.get("force_int8_gemm", False), device="default")
    vae = add("vae", "VAELoader", vae_name=settings["video_vae"])
    audio_vae = add("audio_vae", "VAELoader", vae_name=settings["audio_vae"])
    prepare_values = {k: v for k, v in values.items() if k in
        {"width", "height", "frames", "preserve_audio"}}
    prep = {"settings": json.dumps(prepare_values), "vae": vae, "audio_vae": audio_vae}
    if "target" in reads:
        prep["images"] = reads["target"]
        prep["audio"] = [reads["target"][0], 1]
    if "prefix" in reads:
        prep["prefix_images"] = reads["prefix"]
        prep["prefix_audio"] = [reads["prefix"][0], 1]
    latent = add("prepare", "_TuringCanvasPrepare", **prep)
    refs = {}
    if image_ports(reads):
        refs["image_reference"] = add("image_ref", "TuringUtilsH3ImageReference", vae=vae, megapixels=1,
            latent=latent, **{f"images.image_{i}": reads[p] for i, p in enumerate(image_ports(reads))})
    if reference_ports(reads, "video"):
        refs["video_reference"] = add("video_ref", "TuringUtilsH3VideoReference", video_vae=vae,
            latent=latent, megapixels=1,
            **{f"videos.video_{i}": reads[p] for i, p in enumerate(reference_ports(reads, "video"))})
    if reference_ports(reads, "audio"):
        refs["audio_reference"] = add("audio_ref", "TuringUtilsH3AudioReference", audio_vae=audio_vae,
            **{f"audios.audio_{i}": [reads[p][0], 1] for i, p in enumerate(reference_ports(reads, "audio"))})
    for name in ("first_frame", "last_frame"):
        if name in reads:
            refs[name] = add(name, "TuringUtilsH3KeyframeReference", vae=vae, latent=latent,
                **{"images.image_0": reads[name]})
    text = values.get("model_prompt", "").strip() or values.get("user_prompt", "")
    semantic = add("semantic", "TuringUtilsH3SemanticReference", clip=clip, prompt=text, **refs)
    conditioning = add("conditioning", "TuringUtilsH3BuildConditioning", semantic_reference=semantic, latent=latent, **refs)
    guider = add("guider", "BasicGuider", model=model, conditioning=conditioning)
    sigmas = add("sigmas", "BasicScheduler", model=model, scheduler=settings.get("scheduler", "simple"), steps=settings.get("steps", 8), denoise=denoise)
    if settings.get("refiner", True):
        sigmas = add("refiner", "_TuringCanvasSigmaRefiner", sigmas=sigmas)
    noise = add("noise", "RandomNoise", noise_seed=values.get("seed", 0))
    sampler = add("sampler", "KSamplerSelect", sampler_name=settings.get("sampler_name", "euler"))
    # A run nonce forces inference, without invalidating material reads or encoders.
    noise = add("run_noise", "_TuringCanvasRunNoise", noise=noise, run_id=uuid.uuid4().hex)
    sampled = add("sample", "SamplerCustomAdvanced", noise=noise, guider=guider,
        sampler=sampler, sigmas=sigmas, latent_image=latent)
    if denoise == 0:
        sampled = latent
    streams = add("separate", "LTXVSeparateAVLatent", av_latent=sampled)
    images = add("decode_video", "TuringUtilsMiniMaxH3VideoVAEDecode", samples=streams, vae=vae, attention=settings.get("attention", "w8a8"))
    audio = add("decode_audio", "VAEDecodeAudio", samples=[streams[0], 1], vae=audio_vae)
    output = {"images": images, "audio": audio, "length": ["prepare", 1], "prefix": ["prepare", 2], "original_audio": ["prepare", 3]}
    snapshot = {"materials": materials, "values": values,
                "settings": {k: v for k, v in settings.items() if not k.startswith("chat_")}, "template": VERSION}
    add("publish", "_TuringCanvasPublish", directory=project.directory, task=task,
        signature=signature(materials), snapshot=json.dumps(snapshot), run_id=uuid.uuid4().hex, **output)
    # Remove disconnected model/conditioning branches when denoise is zero.
    needed = set()
    def visit(key):
        if key in needed:
            return
        needed.add(key)
        for value in prompt[key]["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and value[0] in prompt:
                visit(value[0])
    visit("publish")
    prompt = {key: value for key, value in prompt.items() if key in needed}
    return prompt
