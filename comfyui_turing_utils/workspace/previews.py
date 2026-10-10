"""Lazy poster generation with bounded decoding and in-flight deduplication.

Only still JPEGs are generated here. Video playback continues to use native
range-aware file responses and is never started by requesting a poster.
"""

import asyncio
import hashlib
import math
from pathlib import Path
import uuid

import av
from PIL import Image, ImageOps


class PreviewStore:
    def __init__(self, directory: Path, concurrency: int = 2):
        self.directory = directory
        self.slots = asyncio.Semaphore(concurrency)
        self.pending: dict[str, asyncio.Task] = {}

    async def get(self, project_root, asset_id, path, kind, start=0):
        if kind not in {"image", "video"}:
            raise ValueError("This material has no image preview")
        if not math.isfinite(start) or start < 0:
            raise ValueError("Invalid thumbnail time")
        path = Path(path)
        stat = await asyncio.to_thread(path.stat)
        identity = (
            f"{project_root}:{asset_id}:{stat.st_size}:{stat.st_mtime_ns}:{start:.3f}"
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()
        preview = self.directory / (digest + ".jpg")
        if await asyncio.to_thread(preview.is_file):
            return preview

        task = self.pending.get(digest)
        if task is None:
            task = asyncio.create_task(self._render_limited(path, kind, start, preview))
            self.pending[digest] = task

            def release(done):
                self.pending.pop(digest, None)
                # A disconnected HTTP caller must not cancel shared decoding or
                # leave its later exception as an unobserved background error.
                if not done.cancelled():
                    done.exception()

            task.add_done_callback(release)
        await asyncio.shield(task)
        return preview

    async def _render_limited(self, path, kind, start, preview):
        async with self.slots:
            await asyncio.to_thread(render_poster, path, kind, start, preview)


def render_poster(path, kind, start, preview):
    if kind == "image":
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    else:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            origin = (
                float(stream.start_time * stream.time_base) if stream.start_time else 0
            )
            container.seek(int((origin + start) / stream.time_base), stream=stream)
            frame = next(
                (
                    frame
                    for frame in container.decode(video=0)
                    if float(frame.time or 0) - origin >= start
                ),
                None,
            )
            if frame is None:
                raise ValueError("No frame at selected time")
            image = frame.to_image().convert("RGB")

    image.thumbnail((512, 512))
    preview.parent.mkdir(parents=True, exist_ok=True)
    temporary = preview.with_name(uuid.uuid4().hex + ".partial")
    try:
        image.save(temporary, format="JPEG", quality=75)
        temporary.replace(preview)
    finally:
        temporary.unlink(missing_ok=True)
