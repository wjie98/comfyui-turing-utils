"""H3 preparation used by the editable material-workspace example."""

import torch
import torch.nn.functional as F
import comfy.utils
from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV
from comfy_extras.nodes_audio import VAEEncodeAudio
from comfy_extras.nodes_lt import LTXVConcatAVLatent
from ..nodes.latent import SetVideoLatentNoiseMask
from ..nodes.minimax_vae import MiniMaxH3VideoVAEEncode
from ..nodes.video_padding import VideoFramesPadding, padded_frame_count
from ..nodes.video_sequence import VideoContinuationConcat, H3SetAudioPrefixNoiseMask


class PrepareH3:
    DEV_ONLY = True
    CATEGORY = ""
    FUNCTION = "prepare"
    RETURN_TYPES = ("LATENT", "INT", "INT", "AUDIO")
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"width": ("INT", {"default":864,"min":32,"step":32}),
                             "height": ("INT", {"default":480,"min":32,"step":32}),
                             "duration": ("FLOAT", {"default":5,"min":0.01,"step":0.1}),
                             "preserve_audio": ("BOOLEAN", {"default":True}),
                             "vae": ("VAE",), "audio_vae": ("VAE",)},
                "optional": {"images": ("IMAGE",), "mask": ("MASK",), "audio": ("AUDIO",),
                             "prefix_images": ("IMAGE",), "prefix_audio": ("AUDIO",)}}

    def prepare(self, width, height, duration, preserve_audio, vae, audio_vae, images=None, mask=None, audio=None, prefix_images=None, prefix_audio=None):
        count = max(1, round(duration * 24))
        width, height = max(32, round(width / 32) * 32), max(32, round(height / 32) * 32)
        def resize(value):
            if value is None or value.shape[1:3] == (height, width):
                return value
            return comfy.utils.common_upscale(value.movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)
        images, prefix_images = resize(images), resize(prefix_images)
        mode, prefix = ("edit" if images is not None else "reference"), 0
        empty_body = images is None
        trim_info = None
        body_audio_present = audio is not None and preserve_audio
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
        preserved = audio if preserve_audio or prefix_audio is not None else None
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




class SigmaRefiner:
    DEV_ONLY = True
    CATEGORY = ""
    FUNCTION = "refine"
    RETURN_TYPES = ("SIGMAS",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sigmas": ("SIGMAS",), "enabled": ("BOOLEAN", {"default": True})}}

    def refine(self, sigmas, enabled=True):
        if not enabled:
            return (sigmas,)
        # H3SigmaRefiner defaults: +1 step, <=0.7 tail, cosine, end at zero.
        values = sigmas.tolist()
        index = next((i for i, value in enumerate(values) if value <= 0.7), len(values))
        if index >= len(values) - 1:
            return (sigmas,)
        t = torch.linspace(0, 1, len(values) - index + 1, device=sigmas.device, dtype=torch.float32)
        tail = values[index] + (max(0, values[-1]) - values[index]) * (1 - torch.cos(t * torch.pi)) / 2
        tail = tail.to(sigmas.dtype)
        tail[0], tail[-1] = sigmas[index], sigmas[-1]
        return (torch.cat((sigmas[:index], tail)),)


class FinishH3:
    DEV_ONLY = True
    CATEGORY = ""
    FUNCTION = "finish"
    RETURN_TYPES = ("IMAGE", "AUDIO")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",), "length": ("INT",), "prefix": ("INT",)},
                "optional": {"audio": ("AUDIO",), "original_audio": ("AUDIO",)}}

    def finish(self, images, length, prefix, audio=None, original_audio=None):
        images = images[prefix:length or None]
        audio = original_audio if original_audio is not None else audio
        if audio is not None:
            rate = audio["sample_rate"]
            waveform = audio["waveform"][..., round(prefix * rate / 24):round((prefix + len(images)) * rate / 24)]
            expected = round(len(images) * rate / 24)
            waveform = F.pad(waveform, (0, max(0, expected - waveform.shape[-1])))
            audio = {**audio, "waveform": waveform}
        return images, audio
