import runpy
import sys
import unittest
from pathlib import Path
from unittest import mock


class NVFP4ResourceAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scripts = Path(__file__).resolve().parents[1] / "kernel" / "scripts"
        with mock.patch.object(sys, "path", [str(scripts), *sys.path]):
            cls.describe = staticmethod(
                runpy.run_path(str(scripts / "audit_nvfp4_resources.py"))["describe"]
            )

    def test_weight_specializations(self):
        info = self.describe("nvfp4_convrot_s8_kernelILi7EE", 75)
        self.assertEqual(
            (info["groups_per_warp"], info["threads"], info["dynamic_shared"]),
            (7, 256, 0),
        )
        self.assertEqual(
            self.describe("nvfp4_convrot_s8_large_kernel", 75)["family"],
            "nvfp4_weight_large_k",
        )

    def test_unreachable_ampere_excluded_from_sm75(self):
        name = "DefaultGemmWithVisitor4Sm80GemmShapeILi128ELi256ELi64EEGemmShapeILi64ELi64ELi64EEbfloat16_tGemmIdentityThreadblockSwizzleILi1EEELi4E"
        self.assertIsNone(self.describe(name, 75))
        self.assertEqual(self.describe(name, 86)["dynamic_shared"], 98304)
        with self.assertRaisesRegex(ValueError, "stage count"):
            self.describe(name.replace("Li4E", "Li5E"), 86)

    def test_serial_shared_and_excluded_codebook(self):
        name = "GemmWithEpilogueVisitorGemmShapeILi128ELi256ELi64EEGemmShapeILi64ELi64ELi64EEbfloat16_t"
        self.assertEqual(self.describe(name, 75)["dynamic_shared"], 49152)
        self.assertIsNone(self.describe("TuringCodebookGemmKernel" + name, 75))
        self.assertIsNone(self.describe("integer_subbyte" + name, 75))

    def test_smaller_tile_resources(self):
        name = "GemmWithEpilogueVisitorGemmShapeILi128ELi128ELi64EEGemmShapeILi32ELi64ELi64EEbfloat16_t"
        info = self.describe(name, 75)
        self.assertEqual(info["threads"], 256)
        self.assertEqual(info["dynamic_shared"], 32768)

    def test_rowbuffer_includes_dynamic_storage(self):
        name = "bf16_rowbuffer_convrot_quantize_kernelILi1024ELb1ELb0ELb0E"
        info = self.describe(name, 75)
        self.assertEqual(info["threads"], 1024)
        self.assertEqual(info["dynamic_shared"], 61440)
