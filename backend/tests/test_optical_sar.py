"""
SatQuery AI — Optical + SAR Cross-Modal Remote-Sensing Unit Tests
Validates multi-sensor feature extraction, dual-stream neural fusion,
spatial cross-validation, query reasoning, and orchestrator routing.
"""
import io
import os
import sys
import json
import base64
import unittest
import torch
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from datasets.bigearthnet import BigEarthNetDataset
from core.optical_sar_analyzer import get_optical_sar_analyzer, OpticalSARAnalyzer
from core.orchestrator import orchestrate_crossmodal_satquery_request
from core.model_registry import MODEL_REGISTRY, run_optical_sar_analysis


class TestOpticalSARCrossModalAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.split_path = os.path.join(PROJECT_ROOT, "dataset", "bigearthnet", "splits", "test.json")
        with open(cls.split_path, "r") as f:
            cls.records = json.load(f)
        cls.dataset = BigEarthNetDataset(cls.records)
        cls.sample = cls.dataset[0]
        cls.analyzer = get_optical_sar_analyzer()

    def test_01_real_optical_sample_loads(self):
        """Test that real Sentinel-2 multispectral tensor loads with valid values."""
        s2 = self.sample["s2_tensor"]
        self.assertIsInstance(s2, torch.Tensor)
        self.assertEqual(s2.shape[0], 4, "Expected 4 bands: B02, B03, B04, B08")
        self.assertFalse(torch.isnan(s2).any(), "NaN values found in Sentinel-2 tensor")
        self.assertGreater(s2.max().item(), s2.min().item(), "Zero variance in optical tensor")

    def test_02_real_sar_sample_loads(self):
        """Test that real Sentinel-1 SAR tensor loads with valid values."""
        s1 = self.sample["s1_tensor"]
        self.assertIsInstance(s1, torch.Tensor)
        self.assertEqual(s1.shape[0], 2, "Expected 2 channels: VV, VH")
        self.assertFalse(torch.isnan(s1).any(), "NaN values found in Sentinel-1 tensor")
        self.assertGreater(s1.max().item(), s1.min().item(), "Zero variance in SAR tensor")

    def test_03_optical_dimensions_valid(self):
        """Test that optical image has valid spatial dimensions."""
        s2 = self.sample["s2_tensor"]
        h, w = s2.shape[1], s2.shape[2]
        self.assertGreaterEqual(h, 64)
        self.assertGreaterEqual(w, 64)

    def test_04_sar_dimensions_valid(self):
        """Test that SAR image has valid spatial dimensions."""
        s1 = self.sample["s1_tensor"]
        h, w = s1.shape[1], s1.shape[2]
        self.assertGreaterEqual(h, 64)
        self.assertGreaterEqual(w, 64)

    def test_05_optical_sar_pair_correspondence(self):
        """Test that optical and SAR pairs belong to matching spatial acquisitions."""
        self.assertTrue(os.path.exists(self.sample["s1_files"]["VV"]))
        self.assertTrue(os.path.exists(self.sample["s1_files"]["VH"]))
        self.assertTrue(os.path.exists(self.sample["s2_files"]["B02"]))
        self.assertEqual(self.sample["s1_tensor"].shape[1:], self.sample["s2_tensor"].shape[1:])

    def test_06_optical_analyzer_runs(self):
        """Test optical feature extractor produces NDVI, NDWI, and spectral evidence."""
        opt_res = self.analyzer.extract_optical_features(self.sample["s2_tensor"])
        self.assertIn("water_mask", opt_res)
        self.assertIn("builtup_mask", opt_res)
        self.assertIn("vegetation_mask", opt_res)
        self.assertIn("evidence", opt_res)
        self.assertIsInstance(opt_res["evidence"], list)
        self.assertGreater(len(opt_res["evidence"]), 0)

    def test_07_sar_analyzer_runs(self):
        """Test SAR feature extractor evaluates VV/VH backscatter and radar evidence."""
        sar_res = self.analyzer.extract_sar_features(self.sample["s1_tensor"])
        self.assertIn("vv_db", sar_res)
        self.assertIn("vh_db", sar_res)
        self.assertIn("water_mask", sar_res)
        self.assertIn("builtup_mask", sar_res)
        self.assertIn("evidence", sar_res)
        self.assertIsInstance(sar_res["evidence"], list)
        self.assertGreater(len(sar_res["evidence"]), 0)

    def test_08_fusion_component_runs(self):
        """Test dual-stream neural fusion and spatial cross-validation."""
        res = self.analyzer.analyze_cross_modal(
            self.sample["s2_tensor"],
            self.sample["s1_tensor"],
            "Use the optical and SAR images together to identify built-up and water-covered regions."
        )
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["task"], "optical_sar_cross_modal")
        self.assertIn("overlay", res)
        self.assertGreater(len(res["overlay"]), 1000)

    def test_09_crossmodal_result_structure_valid(self):
        """Test presence of all required auditable fields in cross-modal response."""
        res = self.analyzer.analyze_cross_modal(
            self.sample["s2_tensor"],
            self.sample["s1_tensor"],
            "Describe the scene."
        )
        required_keys = [
            "status", "task", "inputs", "modalities", "optical_evidence",
            "sar_evidence", "fused_evidence", "answer", "confidence",
            "overlay", "water_analysis", "built_up_analysis"
        ]
        for k in required_keys:
            self.assertIn(k, res, f"Missing key: {k}")
        self.assertEqual(res["modalities"], ["optical", "sar"])

    def test_10_built_up_query_works(self):
        """Test answering a specific built-up identification query."""
        res = self.analyzer.analyze_cross_modal(
            self.sample["s2_tensor"],
            self.sample["s1_tensor"],
            "Which regions are likely built-up?"
        )
        self.assertIsInstance(res["answer"], str)
        self.assertGreater(len(res["answer"]), 10)
        self.assertIn("built-up", res["answer"].lower())
        self.assertIn("built_up_analysis", res)

    def test_11_water_query_works(self):
        """Test answering a specific water identification query."""
        res = self.analyzer.analyze_cross_modal(
            self.sample["s2_tensor"],
            self.sample["s1_tensor"],
            "Which regions are water-covered?"
        )
        self.assertIsInstance(res["answer"], str)
        self.assertGreater(len(res["answer"]), 10)
        self.assertIn("water", res["answer"].lower())
        self.assertIn("water_analysis", res)

    def test_12_invalid_optical_input_rejected(self):
        """Test rejection of empty or corrupt optical input payload."""
        import asyncio
        res = asyncio.run(orchestrate_crossmodal_satquery_request("", "valid_sar_b64", "query"))
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["error"]["code"], "INVALID_OPTICAL_PAYLOAD")

    def test_13_invalid_sar_input_rejected(self):
        """Test rejection of empty or corrupt SAR input payload."""
        import asyncio
        res = asyncio.run(orchestrate_crossmodal_satquery_request("valid_optical_b64", "", "query"))
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["error"]["code"], "INVALID_SAR_PAYLOAD")

    def test_14_missing_modality_rejected(self):
        """Test rejection when question is empty."""
        import asyncio
        res = asyncio.run(orchestrate_crossmodal_satquery_request("opt", "sar", "   "))
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["error"]["code"], "EMPTY_QUESTION")

    def test_15_orchestrator_routes_optical_sar_request(self):
        """Test end-to-end orchestrator routing on valid base64 optical and SAR images."""
        import asyncio
        # Create small test optical and SAR base64 images
        opt_img = Image.new("RGB", (128, 128), color=(50, 120, 50))
        buf_opt = io.BytesIO()
        opt_img.save(buf_opt, format="PNG")
        b64_opt = base64.b64encode(buf_opt.getvalue()).decode()

        sar_img = Image.new("L", (128, 128), color=100)
        buf_sar = io.BytesIO()
        sar_img.save(buf_sar, format="PNG")
        b64_sar = base64.b64encode(buf_sar.getvalue()).decode()

        res = asyncio.run(orchestrate_crossmodal_satquery_request(
            b64_opt, b64_sar,
            "Use the optical and SAR images together to identify built-up and water-covered regions."
        ))
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["task"], "optical_sar_cross_modal")
        self.assertIn("optical_evidence", res)
        self.assertIn("sar_evidence", res)
        self.assertIn("fused_evidence", res)
        self.assertIn("overlay", res)
        self.assertIn("tools", res)

    def test_16_model_registry_crossmodal_entry(self):
        """Test that optical_sar_analyzer is registered in MODEL_REGISTRY."""
        self.assertIn("optical_sar_analyzer", MODEL_REGISTRY)
        entry = MODEL_REGISTRY["optical_sar_analyzer"]
        self.assertEqual(entry["name"], "Optical-SAR-Dual-Stream-Fusion (BigEarthNet Adapter)")
        self.assertIn("optical_analyzer", entry["metadata"])
        self.assertIn("sar_analyzer", entry["metadata"])
        self.assertIn("fusion_model", entry["metadata"])


if __name__ == "__main__":
    unittest.main()
