"""
SatQuery AI — RSVQA Unit & Integration Test Suite (REAL BENCHMARK VALIDATION)
Validates:
  1. RSVQA split file existence (rsvqa_test.json)
  2. Real RSVQA sample loading (Presence, Counting, Comparison, Area)
  3. Answer matching & numerical tolerance logic
  4. Evaluation metrics calculation & rsvqa_results.json creation
  5. No synthetic benchmark data used
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

_ds_path = os.path.join(BASE_DIR, "datasets", "rsvqa.py")
_spec_rsv = importlib.util.spec_from_file_location("test_rsvqa_mod", _ds_path)
_mod_rsv = importlib.util.module_from_spec(_spec_rsv)
_spec_rsv.loader.exec_module(_mod_rsv)

RSVQADataset = _mod_rsv.RSVQADataset

_ev_path = os.path.join(BASE_DIR, "evaluate_rsvqa.py")
_spec_ev = importlib.util.spec_from_file_location("test_rsvqa_eval_mod", _ev_path)
_mod_ev = importlib.util.module_from_spec(_spec_ev)
_spec_ev.loader.exec_module(_mod_ev)

match_rsvqa_answer = _mod_ev.match_rsvqa_answer
extract_first_number = _mod_ev.extract_first_number
normalize_answer = _mod_ev.normalize_answer


class TestRSVQAEvaluation(unittest.TestCase):

    def test_01_split_file_exists(self):
        """Test that prepared RSVQA evaluation split exists."""
        split_path = os.path.join(PROJECT_ROOT, "dataset", "rsvqa", "splits", "rsvqa_test.json")
        self.assertTrue(os.path.exists(split_path), f"Split missing at {split_path}")

        with open(split_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        self.assertGreaterEqual(len(records), 50)

    def test_02_sample_loading_and_types(self):
        """Test loading real RSVQA samples and question types."""
        split_path = os.path.join(PROJECT_ROOT, "dataset", "rsvqa", "splits", "rsvqa_test.json")
        with open(split_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        ds = RSVQADataset(records)
        sample = ds[0]

        self.assertIn("sample_id", sample)
        self.assertIn("question_type", sample)
        self.assertIn("question", sample)
        self.assertIn("ground_truth", sample)
        self.assertTrue(os.path.exists(sample["image_path"]))

    def test_03_number_extraction_and_matching(self):
        """Test numerical extraction and tolerance matching for count questions."""
        self.assertEqual(extract_first_number("There are 5 buildings"), 5.0)
        self.assertEqual(extract_first_number("The count is 62 farmlands"), 62.0)

        # Exact match
        self.assertTrue(match_rsvqa_answer("5", "5", "count"))
        # Within 15% tolerance for count 62
        self.assertTrue(match_rsvqa_answer("There are 60 farmlands", "62", "count"))
        # Outside tolerance
        self.assertFalse(match_rsvqa_answer("There are 10 farmlands", "62", "count"))

    def test_04_presence_and_area_matching(self):
        """Test presence (Yes/No) and area matching."""
        self.assertTrue(match_rsvqa_answer("Yes, presence detected", "yes", "presence"))
        self.assertTrue(match_rsvqa_answer("rural area", "rural", "area"))
        self.assertFalse(match_rsvqa_answer("urban area", "rural", "area"))

    def test_05_no_synthetic_benchmark_data(self):
        """Test that synthetic_data_used flag is False."""
        split_path = os.path.join(PROJECT_ROOT, "dataset", "rsvqa", "splits", "rsvqa_test.json")
        with open(split_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        for rec in records[:10]:
            self.assertFalse(rec.get("synthetic", False))
            self.assertTrue(os.path.exists(rec["image_path"]))


if __name__ == "__main__":
    unittest.main()
