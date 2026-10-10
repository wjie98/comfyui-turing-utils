"""Poster requests must not become unbounded or eager video playback."""

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image

from comfyui_turing_utils.workspace import previews


class MaterialPreviewsTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.png"
        Image.new("RGB", (1280, 720), "blue").save(self.source)
        self.store = previews.PreviewStore(self.root / "posters")

    def tearDown(self):
        self.temporary.cleanup()

    async def test_deduplicated_poster_is_bounded_and_reused(self):
        with mock.patch.object(
            previews, "render_poster", wraps=previews.render_poster
        ) as render:
            results = await asyncio.gather(
                *[
                    self.store.get(self.root, "image", self.source, "image")
                    for _ in range(8)
                ]
            )
            self.assertEqual(len(set(results)), 1)
            with Image.open(results[0]) as image:
                self.assertEqual(image.size, (512, 288))
            await self.store.get(self.root, "image", self.source, "image")
            render.assert_called_once()
            self.assertFalse(self.store.pending)

    async def test_invalid_times_and_kinds_never_decode(self):
        with mock.patch.object(previews, "render_poster") as render:
            for time in (-1, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    await self.store.get(self.root, "image", self.source, "image", time)
            with self.assertRaises(ValueError):
                await self.store.get(self.root, "audio", self.source, "audio")
            render.assert_not_called()

    async def test_disconnect_does_not_cancel_shared_render(self):
        started, finish = asyncio.Event(), asyncio.Event()

        async def render(*args):
            started.set()
            await finish.wait()

        with mock.patch.object(self.store, "_render_limited", side_effect=render) as op:
            caller = asyncio.create_task(
                self.store.get(self.root, "image", self.source, "image")
            )
            await started.wait()
            caller.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await caller
            self.assertEqual(len(self.store.pending), 1)
            other = asyncio.create_task(
                self.store.get(self.root, "image", self.source, "image")
            )
            # Let the second caller join before releasing the shared task.
            await asyncio.sleep(0.01)
            finish.set()
            await other
            op.assert_awaited_once()
            self.assertFalse(self.store.pending)
