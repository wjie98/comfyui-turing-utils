from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils.adapters import dynamic_vram  # noqa: E402


class FakePatcher:
    def __init__(self, *, dynamic: bool):
        self.dynamic = dynamic
        self.wrappers = {}

    def is_dynamic(self):
        return self.dynamic

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault((wrapper_type, key), []).append(wrapper)

    def get_wrappers(self, wrapper_type, key):
        return self.wrappers.get((wrapper_type, key), [])


class DynamicVramFenceTest(unittest.TestCase):
    def test_installs_once_only_for_dynamic_cuda_models(self):
        dynamic = FakePatcher(dynamic=True)
        static = FakePatcher(dynamic=False)

        with mock.patch(
            "comfyui_turing_utils.adapters.dynamic_vram.torch.cuda.is_available",
            return_value=True,
        ):
            self.assertTrue(
                dynamic_vram.install_dynamic_vram_sample_fence(
                    dynamic, torch.device("cuda", 0)
                )
            )
            self.assertFalse(
                dynamic_vram.install_dynamic_vram_sample_fence(
                    dynamic, torch.device("cuda", 0)
                )
            )
            self.assertFalse(
                dynamic_vram.install_dynamic_vram_sample_fence(
                    static, torch.device("cuda", 0)
                )
            )
            self.assertFalse(
                dynamic_vram.install_dynamic_vram_sample_fence(
                    FakePatcher(dynamic=True), torch.device("cpu")
                )
            )

        self.assertEqual(len(dynamic.wrappers), 1)
        self.assertFalse(static.wrappers)

    def test_synchronizes_only_at_sampler_boundaries(self):
        events = []

        class Executor:
            def __call__(self, *args, **kwargs):
                events.append("sample")
                return "ok"

        wrapper = dynamic_vram.make_dynamic_vram_sample_fence(torch.device("cuda", 0))
        with (
            mock.patch(
                "comfyui_turing_utils.adapters.dynamic_vram.torch.cuda.synchronize",
                side_effect=lambda device: events.append(("sync", device)),
            ) as synchronize,
            mock.patch.object(
                dynamic_vram.CUDA_PHASE_PROFILER,
                "report_after_synchronize",
                side_effect=lambda: events.append("profile"),
            ) as report,
        ):
            result = wrapper(Executor())

        self.assertEqual(result, "ok")
        self.assertEqual(
            events,
            [
                ("sync", torch.device("cuda", 0)),
                "sample",
                ("sync", torch.device("cuda", 0)),
                "profile",
            ],
        )
        self.assertEqual(synchronize.call_count, 2)
        report.assert_called_once_with()

    def test_synchronizes_after_sampler_failure(self):
        class Executor:
            def __call__(self, *args, **kwargs):
                raise RuntimeError("sample failed")

        wrapper = dynamic_vram.make_dynamic_vram_sample_fence(torch.device("cuda", 0))
        with (
            mock.patch(
                "comfyui_turing_utils.adapters.dynamic_vram.torch.cuda.synchronize"
            ) as synchronize,
            mock.patch.object(
                dynamic_vram.CUDA_PHASE_PROFILER,
                "report_after_synchronize",
            ) as report,
        ):
            with self.assertRaisesRegex(RuntimeError, "sample failed"):
                wrapper(Executor())

        self.assertEqual(synchronize.call_count, 2)
        report.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
