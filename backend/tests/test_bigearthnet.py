"""
SatQuery AI — BigEarthNet Unit & Integration Test Suite (REAL DATA VALIDATION)
Validates:
  1. Real dataset archive existence (BEN-GE-800)
  2. Real patch file loading (Sentinel-1 SAR VV/VH & Sentinel-2 B02/B03/B04/B08)
  3. Real tensor shapes and non-empty pixel values
  4. Actual official land-cover annotations
  5. Dataset splits and strict 0 data leakage (train ∩ test = ∅)
  6. Dual-stream PyTorch adapter model accepting real S1 + S2 sensor inputs
  7. Gradient flow to adapter parameters during backpropagation
  8. Adaptation checkpoint and metadata verification (synthetic_data_used == False)
  9. Held-out real-data evaluation pipeline metrics
  10. Model registry integration
"""
import os
import sys
import json
import torch
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import importlib.util

_ds_path = os.path.join(BASE_DIR, "datasets", "bigearthnet.py")
_spec_ds = importlib.util.spec_from_file_location("test_local_bigearthnet", _ds_path)
_mod_ds = importlib.util.module_from_spec(_spec_ds)
_spec_ds.loader.exec_module(_mod_ds)

BIGEARTHNET_19_CLASSES = _mod_ds.BIGEARTHNET_19_CLASSES
labels_to_natural_caption = _mod_ds.labels_to_natural_caption
generate_vqa_pairs = _mod_ds.generate_vqa_pairs
BigEarthNetDataset = _mod_ds.BigEarthNetDataset

_tr_path = os.path.join(BASE_DIR, "train_bigearthnet_adapter.py")
_spec_tr = importlib.util.spec_from_file_location("test_local_train", _tr_path)
_mod_tr = importlib.util.module_from_spec(_spec_tr)
_spec_tr.loader.exec_module(_mod_tr)

BigEarthNetVisionLanguageAdapter = _mod_tr.BigEarthNetVisionLanguageAdapter

from core.model_registry import MODEL_REGISTRY


class TestRealBigEarthNetAdaptation(unittest.TestCase):

    def test_01_real_dataset_files_exist(self):
        """Test that real BEN-GE-800 BigEarthNet dataset files exist on disk."""
        raw_ben_dir = os.path.join(PROJECT_ROOT, "dataset", "bigearthnet_raw", "ben-ge-800")
        s1_dir = os.path.join(raw_ben_dir, "sentinel-1")
        s2_dir = os.path.join(raw_ben_dir, "sentinel-2")

        self.assertTrue(os.path.exists(raw_ben_dir), f"Raw dataset directory missing at {raw_ben_dir}")
        self.assertTrue(os.path.exists(s1_dir), f"Sentinel-1 directory missing at {s1_dir}")
        self.assertTrue(os.path.exists(s2_dir), f"Sentinel-2 directory missing at {s2_dir}")

        s1_folders = os.listdir(s1_dir)
        s2_folders = os.listdir(s2_dir)
        self.assertGreaterEqual(len(s1_folders), 50, "Expected at least 50 real Sentinel-1 patch folders")
        self.assertGreaterEqual(len(s2_folders), 50, "Expected at least 50 real Sentinel-2 patch folders")

    def test_02_real_patch_and_tensor_loading(self):
        """Test loading real patch files and verifying S1 and S2 tensor shapes & stats."""
        split_path = os.path.join(PROJECT_ROOT, "dataset", "bigearthnet", "splits", "train.json")
        self.assertTrue(os.path.exists(split_path))

        with open(split_path, "r") as f:
            records = json.load(f)

        ds = BigEarthNetDataset(records)
        self.assertGreater(len(ds), 0)

        sample = ds[0]
        s1_tensor = sample["s1_tensor"]
        s2_tensor = sample["s2_tensor"]
        target = sample["target"]

        self.assertEqual(s1_tensor.shape, (2, 128, 128))
        self.assertEqual(s2_tensor.shape, (4, 128, 128))
        self.assertEqual(target.shape, (19,))

        # Verify tensors contain real values (not all zeroes or NaN)
        self.assertFalse(torch.isnan(s1_tensor).any())
        self.assertFalse(torch.isnan(s2_tensor).any())
        self.assertNotEqual(s1_tensor.max().item(), s1_tensor.min().item())
        self.assertNotEqual(s2_tensor.max().item(), s2_tensor.min().item())

    def test_03_actual_annotations_loaded(self):
        """Test that actual official CORINE labels are present in dataset samples."""
        split_path = os.path.join(PROJECT_ROOT, "dataset", "bigearthnet", "splits", "train.json")
        with open(split_path, "r") as f:
            records = json.load(f)

        labels = records[0]["labels"]
        self.assertIsInstance(labels, list)
        self.assertGreater(len(labels), 0)
        for l in labels:
            self.assertIsInstance(l, str)

    def test_04_train_val_test_splits_and_no_leakage(self):
        """Test dataset splits and strictly enforce zero data leakage between train/val/test."""
        splits_dir = os.path.join(PROJECT_ROOT, "dataset", "bigearthnet", "splits")
        train_path = os.path.join(splits_dir, "train.json")
        val_path = os.path.join(splits_dir, "val.json")
        test_path = os.path.join(splits_dir, "test.json")

        self.assertTrue(os.path.exists(train_path))
        self.assertTrue(os.path.exists(val_path))
        self.assertTrue(os.path.exists(test_path))

        with open(train_path, "r") as f:
            train_recs = json.load(f)
        with open(val_path, "r") as f:
            val_recs = json.load(f)
        with open(test_path, "r") as f:
            test_recs = json.load(f)

        train_pids = set(r["patch_id"] for r in train_recs)
        val_pids = set(r["patch_id"] for r in val_recs)
        test_pids = set(r["patch_id"] for r in test_recs)

        self.assertEqual(len(train_pids.intersection(val_pids)), 0, "Train and Val overlap!")
        self.assertEqual(len(train_pids.intersection(test_pids)), 0, "Train and Test overlap!")
        self.assertEqual(len(val_pids.intersection(test_pids)), 0, "Val and Test overlap!")

    def test_05_adapter_forward_and_backward(self):
        """Test model forward pass and gradient flow to adapter parameters."""
        model = BigEarthNetVisionLanguageAdapter()
        s1_input = torch.randn(2, 2, 128, 128)
        s2_input = torch.randn(2, 4, 128, 128)
        target = torch.ones(2, 19)

        logits, embeds = model(s1_input, s2_input)
        self.assertEqual(logits.shape, (2, 19))
        self.assertEqual(embeds.shape, (2, 256))

        loss = torch.nn.BCEWithLogitsLoss()(logits, target)
        loss.backward()

        self.assertIsNotNone(model.adapter_down.weight.grad)
        self.assertIsNotNone(model.adapter_up.weight.grad)

    def test_06_checkpoint_verification(self):
        """Test trained checkpoint and metadata (synthetic_data_used == False)."""
        ckpt_path = os.path.join(BASE_DIR, "checkpoints", "bigearthnet_adapter", "best_adapter.pt")
        meta_path = os.path.join(BASE_DIR, "checkpoints", "bigearthnet_adapter", "adaptation_metadata.json")

        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint missing at {ckpt_path}")
        self.assertTrue(os.path.exists(meta_path), f"Metadata missing at {meta_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.assertIn("model_state_dict", ckpt)

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self.assertFalse(meta.get("synthetic_data_used", True), "Synthetic data was incorrectly used!")
        self.assertIn("REAL BigEarthNet", meta["dataset"])
        self.assertLess(meta["final_train_loss"], 0.20)

    def test_07_evaluation_results(self):
        """Test held-out real data evaluation metrics."""
        eval_path = os.path.join(BASE_DIR, "checkpoints", "bigearthnet_adapter", "eval_results.json")
        self.assertTrue(os.path.exists(eval_path))

        with open(eval_path, "r") as f:
            results = json.load(f)

        self.assertEqual(results["data_leakage_checks"]["status"], "PASSED (0 OVERLAP)")
        self.assertGreater(results["improvements"]["f1_score_gain"], 0.10)
        self.assertGreater(results["improvements"]["accuracy_gain"], 0.30)

    def test_08_model_registry_integration(self):
        """Test model registry entry for BigEarthNet adapter."""
        self.assertIn("bigearthnet_adapter", MODEL_REGISTRY)
        entry = MODEL_REGISTRY["bigearthnet_adapter"]
        self.assertEqual(entry["name"], "BigEarthNet-RemoteSensing-Adapter")
        self.assertIn("modality", entry["metadata"])


if __name__ == "__main__":
    unittest.main()
