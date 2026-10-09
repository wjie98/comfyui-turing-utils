from __future__ import annotations

import unittest
import ast
from pathlib import Path
from unittest import mock

from comfyui_turing_utils.log import ROOT_LOGGER, get_logger, profile_level


class RuntimeLoggingTest(unittest.TestCase):
    def test_component_logger_has_filterable_name_and_visible_prefix(self):
        logger = get_logger("minimax.policy")

        with self.assertLogs(ROOT_LOGGER, level="INFO") as captured:
            logger.info("selection=full_fit rows=%d", 4096)

        self.assertEqual(logger.logger.name, f"{ROOT_LOGGER}.minimax.policy")
        self.assertIn(
            "[Turing/minimax.policy] selection=full_fit rows=4096",
            captured.output[0],
        )

    def test_empty_component_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            get_logger("...")

    def test_warnings_deduplicate_by_reason_not_shape(self):
        logger = get_logger("test.once")
        with self.assertLogs(ROOT_LOGGER, level="WARNING") as captured:
            logger.warning_once("fallback shape=%s", (1, 2))
            logger.warning_once("fallback shape=%s", (3, 4))
            logger.warning_once("different fallback")
        self.assertEqual(len(captured.records), 2)

    def test_profile_levels_and_default(self):
        self.addCleanup(profile_level.cache_clear)
        for value, expected in [(None, 0), ("0", 0), ("1", 1), ("2", 2)]:
            with self.subTest(value=value), mock.patch.dict("os.environ", {}, clear=True):
                if value is not None:
                    import os
                    os.environ["COMFYUI_TURING_UTILS_PROFILE"] = value
                profile_level.cache_clear()
                with self.assertNoLogs(ROOT_LOGGER, level="WARNING"):
                    self.assertEqual(profile_level(), expected)

    def test_invalid_profile_warns_once_and_disables(self):
        self.addCleanup(profile_level.cache_clear)
        with mock.patch.dict("os.environ", {"COMFYUI_TURING_UTILS_PROFILE": "99"}, clear=True):
            profile_level.cache_clear()
            with self.assertLogs(ROOT_LOGGER, level="WARNING") as captured:
                self.assertEqual(profile_level(), 0)
                self.assertEqual(profile_level(), 0)
            self.assertEqual(len(captured.records), 1)

    def test_retired_settings_report_names_without_values(self):
        self.addCleanup(profile_level.cache_clear)
        with mock.patch.dict("os.environ", {"SEC_DEBUG": "private-value",
                "COMFYUI_TURING_UTILS_PROFILE_CALLS": "17"}, clear=True):
            profile_level.cache_clear()
            with self.assertLogs(ROOT_LOGGER, level="WARNING") as captured:
                self.assertEqual(profile_level(), 0)
                profile_level()
            self.assertEqual(len(captured.records), 1)
            self.assertIn("SEC_DEBUG", captured.output[0])
            self.assertNotIn("private-value", captured.output[0])

    def test_runtime_environment_surface_stays_small(self):
        root = Path(__file__).resolve().parents[1] / "comfyui_turing_utils"
        names = set()
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call) or not node.args:
                    continue
                call = ast.unparse(node.func)
                if call not in {"os.getenv", "os.environ.get"}:
                    continue
                name = node.args[0]
                if isinstance(name, ast.Constant) and isinstance(name.value, str):
                    names.add(name.value)
        self.assertEqual(names, {
            "COMFYUI_TURING_UTILS_PROFILE",
            "COMFYUI_TURING_UTILS_H3_ACTIVATION_MODE",
        })

    def test_sec_debug_statistics_are_guarded(self):
        path = Path(__file__).resolve().parents[1] / "comfyui_turing_utils/vendor/sec/modeling_sec.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "mask_pixels"
                               for target in node.targets)]
        self.assertEqual(len(assignments), 1)
        guards = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and "profile_level() == 2" in ast.unparse(node.test)]
        self.assertTrue(any(assignments[0] in guard.body for guard in guards))


if __name__ == "__main__":
    unittest.main()
