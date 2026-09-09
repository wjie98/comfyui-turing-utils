from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.adapters.minimax.compat import (  # noqa: E402
    keyframe_condition_rows,
    make_packed_layout,
    packed_layout_is_legacy,
)


class _LegacyLayout:
    def __init__(
        self,
        text_len,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        keyframes=None,
        refs=None,
        frame_count=None,
    ):
        self.values = (
            text_len,
            latent_t,
            latent_h,
            latent_w,
            audio_t,
            keyframes,
            refs,
            frame_count,
        )


class _CurrentLayout:
    def __init__(
        self,
        text_len,
        latent_t,
        latent_h,
        latent_w,
        audio_t,
        keyframes=None,
        refs=None,
    ):
        self.values = (
            text_len,
            latent_t,
            latent_h,
            latent_w,
            audio_t,
            keyframes,
            refs,
        )


class MiniMaxCompatibilityTest(unittest.TestCase):
    def test_layout_builder_passes_frame_count_only_to_legacy_comfy(self):
        payload = {"keyframes": [object()], "refs": [], "frame_count": 22}
        legacy = make_packed_layout(_LegacyLayout, 3, 2, 4, 6, 8, payload)
        current = make_packed_layout(_CurrentLayout, 3, 2, 4, 6, 8, payload)

        self.assertTrue(packed_layout_is_legacy(_LegacyLayout))
        self.assertFalse(packed_layout_is_legacy(_CurrentLayout))
        self.assertEqual(legacy.values[-1], 22)
        self.assertEqual(len(current.values), 7)

    def test_keyframe_rows_follow_each_layout_generation(self):
        keyframes = [
            {
                "latent": torch.empty(1, 24, 2, 4, 6),
                "audio_latent": torch.empty(1, 32, 2, 5),
            },
            {"latent": torch.empty(1, 24, 1, 4, 6)},
        ]

        self.assertEqual(
            keyframe_condition_rows(_LegacyLayout, keyframes, 12),
            (24, 0),
        )
        self.assertEqual(
            keyframe_condition_rows(_CurrentLayout, keyframes, 12),
            (36, 10),
        )


if __name__ == "__main__":
    unittest.main()
