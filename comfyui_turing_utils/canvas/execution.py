"""Hidden execution graph stages. They reuse existing model and VAE contracts."""

import json
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F
from comfy_api.latest import InputImpl, Types
from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV
from comfy_extras.nodes_audio import VAEEncodeAudio
from comfy_extras.nodes_lt import LTXVConcatAVLatent

from ..nodes.latent import SetVideoLatentNoiseMask
from ..nodes.minimax_vae import MiniMaxH3VideoVAEEncode
from ..nodes.video_padding import VideoFramesPadding, padded_frame_count
from ..nodes.video_sequence import VideoContinuationConcat, H3SetAudioPrefixNoiseMask
from .media import read_material
from .store import Project
from ..nodes.attention import _ATTENTION_STRATEGIES


class ReadAsset:
    CATEGORY = ""
    FUNCTION = "read"
    RETURN_TYPES = ("IMAGE", "AUDIO", "MASK")
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"directory": ("STRING",), "reference": ("STRING",),
            "width": ("INT", {"default": 0}), "height": ("INT", {"default": 0})}}

    def read(self, directory, reference, width=0, height=0):
        return read_material(Project(directory), json.loads(reference), width, height)


class PrepareH3:
    CATEGORY = ""
    FUNCTION = "prepare"
    RETURN_TYPES = ("LATENT", "INT", "INT", "AUDIO")
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"settings": ("STRING",), "vae": ("VAE",), "audio_vae": ("VAE",)},
                "optional": {"images": ("IMAGE",), "mask": ("MASK",), "audio": ("AUDIO",),
                             "prefix_images": ("IMAGE",), "prefix_audio": ("AUDIO",)}}

    def prepare(self, settings, vae, audio_vae, images=None, mask=None, audio=None, prefix_images=None, prefix_audio=None):
        cfg = json.loads(settings)
        width, height, count = cfg["width"], cfg["height"], cfg["frames"]
        mode, prefix = ("edit" if images is not None else "reference"), 0
        empty_body = images is None
        trim_info = None
        body_audio_present = audio is not None and cfg.get("preserve_audio", True)
        if prefix_images is not None:
            if images is None:
                images = prefix_images[-1:].repeat(count, 1, 1, 1)
            composed = VideoContinuationConcat.execute(prefix_images=prefix_images, prefix_audio=prefix_audio,
                body_images=images, body_audio=audio if body_audio_present else None, frame_rate=24, mode="concat").result
            images, mask, audio, trim_info = composed
            if prefix_audio is None and not body_audio_present:
                audio = None
            prefix = len(prefix_images)
            mode = "edit"
        if mode != "reference":
            count, height, width = images.shape[:3]
            if mask is None:
                mask = torch.ones(images.shape[:3], dtype=images.dtype, device=images.device)
            if mask.shape[0] == 1:
                mask = mask.expand(count, -1, -1)
            if mask.shape[0] != count:
                raise ValueError("Mask and target video intervals do not have matching frame counts")
            if mask.shape[1:] != (height, width):
                mask = F.interpolate(mask.unsqueeze(1), size=(height, width), mode="nearest").squeeze(1)
            padded = VideoFramesPadding.execute(type="minimax", image=images, mask=mask).result
            video = MiniMaxH3VideoVAEEncode().encode(padded[0], vae)[0]
            video = SetVideoLatentNoiseMask.execute(video, padded[1], "minimax").result[0]
            if empty_body:
                video["samples"] = video["samples"] * (1 - video["noise_mask"])
            length = padded[4]
        else:
            length = padded_frame_count(count, "minimax")
        empty = EmptyMiniMaxH3LatentAV.execute(width, height, length).result[0]
        empty_video, empty_audio = empty["samples"].unbind()
        if mode == "reference":
            video = {"samples": empty_video}
        preserved = audio if cfg.get("preserve_audio", True) or prefix_audio is not None else None
        if preserved is not None:
            waveform = preserved["waveform"]
            samples = round((5 + 17 * max(0, (length - 5 + 16) // 17)) * preserved["sample_rate"] / 24)
            waveform = F.pad(waveform[..., :samples], (0, max(0, samples - waveform.shape[-1])))
            sound = VAEEncodeAudio.execute(audio_vae, {**preserved, "waveform": waveform}).result[0]
            sound["noise_mask"] = torch.zeros_like(sound["samples"])
            if trim_info is not None and not body_audio_present:
                trim_info = {**trim_info, "total_audio_samples": samples}
                sound = H3SetAudioPrefixNoiseMask.execute(sound, trim_info, "protect_prefix_generate_body").result[0]
                preserved = None
        else:
            sound = {"samples": empty_audio}
        latent = LTXVConcatAVLatent.execute(video, sound).result[0]
        return latent, count, prefix, preserved


class Publish:
    CATEGORY = ""
    FUNCTION = "publish"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"directory": ("STRING",), "task": ("STRING",),
            "signature": ("STRING",), "snapshot": ("STRING",), "run_id": ("STRING",)},
            "optional": {"images": ("IMAGE",), "audio": ("AUDIO",), "original_audio": ("AUDIO",),
                         "mask": ("MASK",), "length": ("INT",), "prefix": ("INT",)}}

    def publish(self, directory, task, signature, snapshot, run_id, images=None,
                audio=None, original_audio=None, mask=None, length=0, prefix=0):
        project = Project(directory)
        cfg = json.loads(snapshot)
        if mask is not None:
            asset_id, path = project.reserve(".npy")
            with path.open("wb") as file:
                np.save(file, mask.detach().cpu().numpy(), allow_pickle=False)
            kind = "mask"
        else:
            images = images[prefix:length or None]
            audio = original_audio if original_audio is not None else audio
            if audio is not None:
                rate = audio["sample_rate"]
                waveform = audio["waveform"][..., round(prefix * rate / 24):round((prefix + len(images)) * rate / 24)]
                expected = round(len(images) * rate / 24)
                waveform = F.pad(waveform, (0, max(0, expected - waveform.shape[-1])))
                audio = {**audio, "waveform": waveform}
            asset_id, path = project.reserve(".mp4", prefix=cfg.get("values", {}).get("filename_prefix", "h3"))
            video = InputImpl.VideoFromComponents(Types.VideoComponents(images=images, audio=audio,
                frame_rate=Fraction(24)), bit_depth=8, color_space="sRGB")
            video.save_to(str(path), format=Types.VideoContainer.MP4, codec=Types.VideoCodec.H264, crf=19)
            kind = "video"
        project.register(asset_id, path, kind, path.name, {"inputs": cfg["materials"], "snapshot": cfg, "signature": signature})
        project.publish(task, asset_id, signature, cfg)
        return {"ui": {"canvas_result": [{"task": task, "asset": asset_id}]}, "result": (asset_id,)}


class RunNoise:
    CATEGORY = ""
    FUNCTION = "run"
    RETURN_TYPES = ("NOISE",)
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"noise": ("NOISE",), "run_id": ("STRING",)}}

    def run(self, noise, run_id):
        return (noise,)


class Sol:
    CATEGORY = ""
    FUNCTION = "apply"
    RETURN_TYPES = ("MODEL",)
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "settings": ("STRING",)}}

    def apply(self, model, settings):
        return (_ATTENTION_STRATEGIES["sol"](model, **json.loads(settings)),)


INTERNAL_NODES = {"_TuringCanvasRead": ReadAsset, "_TuringCanvasPrepare": PrepareH3,
                  "_TuringCanvasPublish": Publish, "_TuringCanvasRunNoise": RunNoise, "_TuringCanvasSol": Sol}
for _node in INTERNAL_NODES.values():
    _node.DEV_ONLY = True
