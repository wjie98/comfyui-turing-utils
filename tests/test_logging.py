from __future__ import annotations

import unittest

from comfyui_turing_utils.log import ROOT_LOGGER, get_logger


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


if __name__ == "__main__":
    unittest.main()
