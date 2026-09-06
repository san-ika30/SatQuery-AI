"""
SatQuery AI — Unit Tests for CDVQA Benchmark Dataset
Verifies:
  1. CDVQA dataset metadata file exists
  2. Real samples load without error
  3. T1 image loads cleanly
  4. T2 image loads cleanly
  5. T1 and T2 dimensions match
  6. Question exists
  7. Ground truth exists
  8. Synthetic data is NOT used (0% synthetic)
"""
import os
import sys
import pytest
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import importlib.util
cdvqa_path = os.path.join(BASE_DIR, "datasets", "cdvqa.py")
spec = importlib.util.spec_from_file_location("cdvqa_module", cdvqa_path)
cdvqa_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cdvqa_mod)
CDVQADataset = cdvqa_mod.CDVQADataset


def test_cdvqa_dataset_metadata_exists():
    dataset = CDVQADataset()
    assert os.path.exists(dataset.metadata_path), "CDVQA metadata file is missing!"
    assert len(dataset) >= 100, f"Expected at least 100 CDVQA samples, got {len(dataset)}"


def test_cdvqa_synthetic_data_prohibited():
    dataset = CDVQADataset()
    assert dataset.is_synthetic is False, "CDVQA dataset must be 100% real benchmark data!"


def test_cdvqa_sample_loading():
    dataset = CDVQADataset()
    sample = dataset[0]

    assert "sample_id" in sample
    assert "image_t1_path" in sample
    assert "image_t2_path" in sample
    assert "question" in sample
    assert "ground_truth" in sample
    assert sample["is_synthetic"] is False


def test_cdvqa_t1_image_validity():
    dataset = CDVQADataset()
    sample = dataset[0]
    img1 = Image.open(sample["image_t1_path"])
    assert img1 is not None
    assert img1.size[0] > 0 and img1.size[1] > 0


def test_cdvqa_t2_image_validity():
    dataset = CDVQADataset()
    sample = dataset[0]
    img2 = Image.open(sample["image_t2_path"])
    assert img2 is not None
    assert img2.size[0] > 0 and img2.size[1] > 0


def test_cdvqa_t1_t2_dimension_match():
    dataset = CDVQADataset()
    t1_img, t2_img = dataset.load_images(0)
    assert t1_img.size == t2_img.size, "T1 and T2 images must have matching dimensions!"
