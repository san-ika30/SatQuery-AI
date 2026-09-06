import os
import sys
import torch
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

import importlib.util
cdvqa_file_path = os.path.join(BASE_DIR, "backend", "datasets", "cdvqa.py")
spec = importlib.util.spec_from_file_location("cdvqa_dataset_module", cdvqa_file_path)
cdvqa_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cdvqa_mod)
CDVQADataset = cdvqa_mod.CDVQADataset

from core.change_analyzer import get_change_analyzer

dataset = CDVQADataset()
item = dataset[0]
print("Sample 0:", item)

t1_img, t2_img = dataset.load_images(0)
analyzer = get_change_analyzer()
model = analyzer._load_model()

print("Model:", model.__class__.__name__ if model else "None")

# Calculate pixel diff
arr1 = np.array(t1_img.resize((256, 256)), dtype=np.float32)
arr2 = np.array(t2_img.resize((256, 256)), dtype=np.float32)
diff = np.abs(arr1 - arr2).mean(axis=2)

print("Diff min/max/mean:", diff.min(), diff.max(), diff.mean())
print("Pixels with diff > 15:", np.sum(diff > 15), "/", diff.size, f"({np.mean(diff > 15)*100:.2f}%)")

res = analyzer.analyze_change(t1_img, t2_img, item["question"])
print("Change result:", res)
