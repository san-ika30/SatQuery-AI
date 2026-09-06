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
import torchvision.transforms as transforms
import torch.nn.functional as F

dataset = CDVQADataset()

for idx in range(5):
    item = dataset[idx]
    t1_img, t2_img = dataset.load_images(idx)

    t1_r = t1_img.resize((256, 256))
    t2_r = t2_img.resize((256, 256))

    arr1 = np.array(t1_r, dtype=np.float32)
    arr2 = np.array(t2_r, dtype=np.float32)
    diff = np.abs(arr1 - arr2).mean(axis=2)

    ratio_diff = float(np.mean(diff > 20.0) * 100.0)

    print(f"Sample #{idx+1} ({item['sample_id']}): GT='{item['ground_truth']}' Q='{item['question']}'")
    print(f"  Pixel Diff Mean: {diff.mean():.2f}, Diff Ratio (>20.0): {ratio_diff:.2f}%")
