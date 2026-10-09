"""Decode selected material only when a task actually executes."""

import av
import math
import numpy as np
import torch
from PIL import Image, ImageOps


def read_material(project, ref, width=0, height=0, max_frames=0):
    item = project.asset(ref["asset"])
    path = project.path(ref["asset"])
    meta = item.get("metadata", {})
    source_rate = float(ref.get("force_rate", 0)) or float(meta.get("fps", 24))
    if not math.isfinite(source_rate) or source_rate <= 0:
        raise ValueError("Video frame rate must be positive and finite")
    nth = max(1, int(ref.get("select_every_nth", 1)))
    start, duration = float(ref.get("start", 0)), float(ref.get("duration", 0))
    start += int(ref.get("skip_first_frames", 0)) / source_rate
    cap = int(ref.get("frame_load_cap", 0))
    if cap < 0 or int(ref.get("skip_first_frames", 0)) < 0:
        raise ValueError("Frame range must be non-negative")
    if cap:
        duration = min(duration or float("inf"), cap * nth / source_rate)
    if not np.isfinite(start) or not np.isfinite(duration) or start < 0 or duration < 0:
        raise ValueError("Material time range must be finite and non-negative")
    end = start + duration if duration else float("inf")
    def dimensions(w, h):
        dw, dh = width or int(ref.get("custom_width", 0)), height or int(ref.get("custom_height", 0))
        if dw < 0 or dh < 0:
            raise ValueError("Material dimensions must be non-negative")
        if not dw and not dh:
            dw, dh = w, h
        elif not dw:
            dw = max(1, round(w * dh / h))
        elif not dh:
            dh = max(1, round(h * dw / w))
        limit = float(ref.get("max_megapixels") or 4) * 1024 * 1024
        if limit <= 0 or not math.isfinite(limit):
            raise ValueError("max_megapixels must be positive and finite")
        scale = min(1, math.sqrt(limit / (dw * dh)))
        multiple = max(1, int(ref.get("spatial_multiple", 1)))
        if limit < multiple * multiple:
            raise ValueError("Pixel limit is smaller than one aligned spatial tile")
        dw, dh = (max(multiple, int(value * scale) // multiple * multiple) for value in (dw, dh))
        if dw * dh > limit:
            if dw >= dh:
                dw = int(limit / dh) // multiple * multiple
            else:
                dh = int(limit / dw) // multiple * multiple
        return dw, dh
    if item["kind"] == "mask":
        return None, None, torch.from_numpy(np.load(path, allow_pickle=False))
    if item["kind"] == "image":
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = ImageOps.fit(image, dimensions(image.width, image.height), Image.Resampling.LANCZOS)
            pixels = torch.from_numpy(np.asarray(image).copy()).float().unsqueeze(0) / 255
        return pixels, None, pixels[..., 0]
    frames = []
    if item["kind"] == "video" and not ref.get("slot", 0) and ref.get("modality") != "audio":
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if stream.duration is not None:
                end = min(end, float(stream.duration * stream.time_base))
            origin = float(stream.start_time * stream.time_base) if stream.start_time else 0
            container.seek(int((start + origin) / stream.time_base), stream=stream)
            next_time = start
            next_source = start
            previous = None
            for frame in container.decode(stream):
                time = float(frame.time or 0) - origin
                if time < start:
                    continue
                if time >= end:
                    break
                if time + 1e-7 < next_source:
                    continue
                next_source += nth / source_rate
                dw, dh = dimensions(frame.width, frame.height)
                rgb = frame.reformat(width=dw, height=dh, format="rgb24").to_ndarray()
                while next_time < time - 1e-7:
                    frames.append(previous if previous is not None else rgb)
                    next_time += 1 / 24
                    if max_frames and len(frames) >= max_frames:
                        break
                if max_frames and len(frames) >= max_frames:
                    break
                if next_time <= time + 1e-7:
                    frames.append(rgb)
                    next_time += 1 / 24
                previous = rgb
                if max_frames and len(frames) >= max_frames:
                    break
            while previous is not None and np.isfinite(end) and next_time < end - 1e-7 and (not max_frames or len(frames) < max_frames):
                frames.append(previous)
                next_time += 1 / 24
        if not frames:
            raise ValueError("Selected video interval contains no frames")
    if ref.get("modality") == "video" or not ref.get("include_audio", True):
        if ref.get("modality") == "audio" or ref.get("slot", 0):
            raise ValueError("Audio output is disabled for this video")
        images = torch.from_numpy(np.stack(frames)).float() / 255 if frames else None
        return images, None, images[..., 0] if images is not None else None
    chunks = []
    rate = 44100
    with av.open(str(path)) as container:
        if container.streams.audio:
            stream = container.streams.audio[0]
            origin = float(stream.start_time * stream.time_base) if stream.start_time else 0
            container.seek(int((start + origin) / stream.time_base), stream=stream)
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=rate)
            audio_end = min(end, start + len(frames) / 24) if frames else end
            cursor = start

            def collect(converted):
                nonlocal cursor
                t = float(converted.time) - origin if converted.time is not None else cursor
                data = converted.to_ndarray()
                cursor = t + data.shape[1] / rate
                lo = max(0, round((start - t) * rate))
                hi = min(data.shape[1], round((audio_end - t) * rate)) if np.isfinite(audio_end) else data.shape[1]
                if hi > lo:
                    chunks.append((max(0, round((t - start) * rate) + lo), data[:, lo:hi]))

            for frame in container.decode(stream):
                time = float(frame.time or 0) - origin
                for converted in resampler.resample(frame):
                    collect(converted)
                if time >= audio_end:
                    break
            for converted in resampler.resample(None):
                collect(converted)
    audio = None
    if chunks:
        length = round((audio_end - start) * rate) if np.isfinite(audio_end) else max(pos + data.shape[1] for pos, data in chunks)
        waveform = np.zeros((2, length), dtype=np.float32)
        for pos, data in chunks:
            waveform[:, pos:pos + data.shape[1]] = data[:, :max(0, length - pos)]
        audio = {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": rate}
    if (ref.get("slot", 0) or ref.get("modality") == "audio") and audio is None:
        raise ValueError("The selected material has no audio in this interval")
    images = torch.from_numpy(np.stack(frames)).float() / 255 if frames else None
    return images, audio, images[..., 0] if images is not None else None
