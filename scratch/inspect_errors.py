import os
import sys
import json
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))
import importlib.util
spec = importlib.util.spec_from_file_location("cdvqa_mod", os.path.join(BASE_DIR, "backend", "datasets", "cdvqa.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ds = mod.CDVQADataset()

from scratch.test_cdvqa_full_eval import solve_cdvqa_sample

print("--- Inspecting 'largest_change' samples ---")
for idx in range(len(ds)):
    item = ds[idx]
    cat = item.get("change_category")
    if cat == "largest_change":
        t1, t2 = ds.load_images(idx)
        pred = solve_cdvqa_sample(t1, t2, item["question"], cat)
        print(f"Sample {idx}: Q: '{item['question']}' | GT: '{item['ground_truth']}' | Pred: '{pred}'")

print("\n--- Inspecting 'change_ratio' samples ---")
for idx in range(len(ds)):
    item = ds[idx]
    cat = item.get("change_category")
    if cat == "change_ratio":
        t1, t2 = ds.load_images(idx)
        pred = solve_cdvqa_sample(t1, t2, item["question"], cat)
        print(f"Sample {idx}: Q: '{item['question']}' | GT: '{item['ground_truth']}' | Pred: '{pred}'")
