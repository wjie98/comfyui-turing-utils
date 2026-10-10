from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from comfyui_turing_utils import profiling  # noqa: E402
from comfyui_turing_utils.profiling import (  # noqa: E402
    CudaPhaseProfiler,
    WorkflowTimeline,
)


class CudaPhaseProfilerTest(unittest.TestCase):
    def test_disabled_profiler_creates_no_cuda_events(self):
        profiler = CudaPhaseProfiler(0)
        function = mock.Mock(return_value="output")
        with mock.patch("torch.cuda.Event") as event:
            output = profiler.call("phase", function, 1, keyword=2)

        self.assertEqual(output, "output")
        function.assert_called_once_with(1, keyword=2)
        event.assert_not_called()
        self.assertFalse(profiler.records)

    def test_runtime_metadata_distinguishes_native_cubin_from_ptx_fallback(self):
        with (
            mock.patch.object(
                profiling,
                "attention_kernel_architectures",
                return_value=("sm75+ptx", "sm86"),
            ),
            mock.patch.object(
                profiling, "attention_runtime_profile_schema", return_value=1
            ),
            mock.patch.object(profiling, "kernel_version", return_value="0.31.0"),
            mock.patch.object(profiling.torch.cuda, "current_device", return_value=0),
            mock.patch.object(
                profiling.torch.cuda,
                "get_device_capability",
                return_value=(8, 6),
            ),
            mock.patch.object(
                profiling.torch.cuda,
                "get_device_name",
                return_value="NVIDIA GeForce RTX 3070",
            ),
        ):
            result = profiling._runtime_profile_metadata()

        self.assertEqual(result["device_sm"], "sm86")
        self.assertEqual(result["compiled_attention"], "sm75+ptx,sm86")
        self.assertTrue(result["native_arch"])
        self.assertEqual(result["profile_schema"], 1)

    def test_dynamic_vram_profile_defers_sync_until_sampler_boundary(self):
        profiler = CudaPhaseProfiler(1)
        start = mock.Mock()
        end = mock.Mock()
        start.elapsed_time.return_value = 2.5
        profiler.defer_to_sampler_boundary()

        with (
            mock.patch.object(
                profiling.torch.cuda,
                "Event",
                side_effect=(start, end),
            ),
            mock.patch.object(
                profiling,
                "_runtime_profile_metadata",
                return_value={
                    "kernel": "0.31.0",
                    "compiled_attention": "sm86",
                    "profile_schema": 1,
                    "device": "RTX 3070",
                    "device_sm": "sm86",
                    "native_arch": True,
                },
            ),
        ):
            self.assertEqual(profiler.call("phase", lambda: "output"), "output")
            profiler.complete_attention((1, 56, 60186, 128))

            end.synchronize.assert_not_called()
            self.assertTrue(profiler.pending_report)
            self.assertFalse(profiler.enabled)
            self.assertTrue(profiler.report_after_synchronize())

        start.elapsed_time.assert_called_once_with(end)
        self.assertFalse(profiler.pending_report)
        self.assertTrue(profiler.reported)

    def test_non_dynamic_profile_keeps_bounded_synchronization(self):
        profiler = CudaPhaseProfiler(1)
        start = mock.Mock()
        end = mock.Mock()
        start.elapsed_time.return_value = 1.0

        with (
            mock.patch.object(
                profiling.torch.cuda,
                "Event",
                side_effect=(start, end),
            ),
            mock.patch.object(
                profiling,
                "_runtime_profile_metadata",
                return_value={
                    "kernel": "0.31.0",
                    "compiled_attention": "sm86",
                    "profile_schema": 1,
                    "device": "RTX 3070",
                    "device_sm": "sm86",
                    "native_arch": True,
                },
            ),
        ):
            profiler.call("phase", lambda: None)
            profiler.complete_attention((1, 56, 60186, 128))

        end.synchronize.assert_called_once_with()
        self.assertTrue(profiler.reported)

    def test_profiles_independent_attention_and_mlp_shape_buckets(self):
        profiler = CudaPhaseProfiler(1, bucket_limit=4)
        profiler.defer_to_sampler_boundary()
        events = [mock.Mock() for _ in range(8)]
        for event in events[::2]:
            event.elapsed_time.return_value = 1.0

        with mock.patch.object(profiling.torch.cuda, "Event", side_effect=events):
            for kind, shape, phase in (
                ("attention", (1, 56, 60186, 128), "attention.execute"),
                ("mlp", (60186, 5376), "minimax.mlp.swiglu_fc2"),
                ("attention", (1, 56, 127275, 128), "attention.execute"),
                ("mlp", (127275, 5376), "minimax.mlp.swiglu_fc2_tile"),
            ):
                self.assertTrue(profiler.begin_operation(kind, shape, path="test"))
                profiler.call(phase, lambda: None)
                profiler.complete_operation(kind, shape)

        self.assertEqual(len(profiler._buckets), 4)
        self.assertTrue(all(bucket.pending for bucket in profiler._buckets.values()))
        self.assertFalse(profiler.enabled)

    def test_reported_bucket_suppresses_events_until_scope_completion(self):
        profiler = CudaPhaseProfiler(1, bucket_limit=1)
        profiler.defer_to_sampler_boundary()
        start = mock.Mock()
        end = mock.Mock()
        start.elapsed_time.return_value = 1.0
        with mock.patch.object(profiling.torch.cuda, "Event", side_effect=(start, end)):
            profiler.begin_operation("attention", (1, 1, 64, 64))
            profiler.call("attention.execute", lambda: None)
            profiler.complete_operation("attention", (1, 1, 64, 64))
        with mock.patch.object(profiling.torch.cuda, "Event") as event:
            self.assertFalse(profiler.begin_operation("mlp", (64, 64), path="full"))
            profiler.call("mlp", lambda: None)
            profiler.complete_operation("mlp", (64, 64))
        event.assert_not_called()


class WorkflowTimelineTest(unittest.TestCase):
    def test_disabled_timeline_has_zero_cuda_observation_overhead(self):
        timeline = WorkflowTimeline(False)
        function = mock.Mock(return_value="output")
        with (
            mock.patch.object(profiling.torch.cuda, "Event") as event,
            mock.patch.object(profiling.torch.cuda, "synchronize") as synchronize,
        ):
            output = timeline.call(
                "latent_upscale", torch.device("cuda", 0), function, 1, value=2
            )

        self.assertEqual(output, "output")
        function.assert_called_once_with(1, value=2)
        event.assert_not_called()
        synchronize.assert_not_called()

    def test_summary_timeline_is_one_info_line(self):
        timeline = WorkflowTimeline(True)
        window = mock.Mock()
        window.wall_start = 10.0
        window.cuda_start.elapsed_time.return_value = 12.5
        window.label = "sampler"
        window.counters = {}
        with (
            mock.patch.object(profiling, "profile_level", return_value=1),
            mock.patch.object(profiling.time, "perf_counter", return_value=10.020),
            mock.patch.object(profiling.torch.cuda, "memory_allocated", return_value=0),
            mock.patch.object(profiling.torch.cuda, "memory_reserved", return_value=0),
            mock.patch.object(
                profiling.torch.cuda, "max_memory_allocated", return_value=1024**3
            ),
            self.assertLogs("comfyui-turing-utils", level="INFO") as logs,
        ):
            self.assertTrue(timeline.finish_after_synchronize(window))
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(logs.records[0].levelname, "INFO")
        self.assertIn("sampler: wall=0.02s CUDA=0.01s peak=1.00 GiB", logs.output[0])
        self.assertNotIn("counters", logs.output[0])

    def test_enabled_timeline_records_one_bounded_cuda_window(self):
        timeline = WorkflowTimeline(True)
        start = mock.Mock()
        end = mock.Mock()
        start.elapsed_time.return_value = 12.5

        def function():
            timeline.sample("dynamic_reclaims", 2)
            return "output"

        with (
            mock.patch.object(profiling.torch.cuda, "Event", side_effect=(start, end)),
            mock.patch.object(profiling.torch.cuda, "synchronize") as synchronize,
            mock.patch.object(profiling.torch.cuda, "reset_peak_memory_stats") as reset,
            mock.patch.object(
                profiling.torch.cuda,
                "memory_allocated",
                side_effect=(100 * 1024**2, 120 * 1024**2),
            ),
            mock.patch.object(
                profiling.torch.cuda,
                "memory_reserved",
                side_effect=(140 * 1024**2, 160 * 1024**2),
            ),
            mock.patch.object(
                profiling.torch.cuda,
                "max_memory_allocated",
                return_value=180 * 1024**2,
            ),
            mock.patch.object(
                profiling.time, "perf_counter", side_effect=(10.0, 10.020)
            ),
            mock.patch.object(profiling, "profile_level", return_value=2),
            self.assertLogs("comfyui-turing-utils", level="INFO") as logs,
        ):
            output = timeline.call("latent_upscale", torch.device("cuda", 0), function)

        self.assertEqual(output, "output")
        self.assertEqual(synchronize.call_count, 2)
        reset.assert_called_once_with(torch.device("cuda", 0))
        start.record.assert_called_once_with()
        end.record.assert_called_once_with()
        start.elapsed_time.assert_called_once_with(end)
        self.assertIsNone(timeline._active)
        message = "\n".join(logs.output)
        self.assertIn("label=latent_upscale", message)
        self.assertIn("cuda=12.500 ms", message)
        self.assertIn("dynamic_reclaims=2", message)


if __name__ == "__main__":
    unittest.main()
