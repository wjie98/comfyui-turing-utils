"""Decode image and audio materials only when a task actually executes."""

import av
import math
import numpy as np
import torch
from PIL import Image, ImageOps


def read_image(path, max_megapixels):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        limit = max_megapixels * 1024 * 1024
        scale = min(1, math.sqrt(limit / (image.width * image.height)))
        dimensions = max(1, int(image.width * scale)), max(1, int(image.height * scale))
        image = ImageOps.fit(image, dimensions, Image.Resampling.LANCZOS)
        return torch.from_numpy(np.asarray(image).copy()).float().unsqueeze(0) / 255


def read_audio(path, start, end):
    end = end or float("inf")
    chunks = []
    rate = 44100
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("The selected material has no audio")
        stream = container.streams.audio[0]
        origin = float(stream.start_time * stream.time_base) if stream.start_time else 0
        container.seek(int((start + origin) / stream.time_base), stream=stream)
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=rate)
        cursor = start

        def collect(converted):
            nonlocal cursor
            t = float(converted.time) - origin if converted.time is not None else cursor
            data = converted.to_ndarray()
            cursor = t + data.shape[1] / rate
            lo = max(0, round((start - t) * rate))
            hi = min(data.shape[1], round((end - t) * rate)) if math.isfinite(end) else data.shape[1]
            if hi > lo:
                chunks.append((max(0, round((t - start) * rate) + lo), data[:, lo:hi]))

        for frame in container.decode(stream):
            time = float(frame.time or 0) - origin
            for converted in resampler.resample(frame):
                collect(converted)
            if time >= end:
                break
        for converted in resampler.resample(None):
            collect(converted)
    if not chunks:
        raise ValueError("The selected material has no audio in this interval")
    length = round((end - start) * rate) if math.isfinite(end) else max(pos + data.shape[1] for pos, data in chunks)
    waveform = np.zeros((2, length), dtype=np.float32)
    for pos, data in chunks:
        waveform[:, pos:pos + data.shape[1]] = data[:, :max(0, length - pos)]
    return {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": rate}
