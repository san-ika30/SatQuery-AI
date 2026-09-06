import os
import sys
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

import importlib.util
spec = importlib.util.spec_from_file_location("cdvqa_mod", os.path.join(BASE_DIR, "backend", "datasets", "cdvqa.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ds = mod.CDVQADataset()

def classify_land_cover(img_np: np.ndarray) -> np.ndarray:
    """
    Classify 512x512 RGB image into 6 land cover categories:
    0: NVG_surface (default)
    1: water
    2: trees
    3: low_vegetation
    4: buildings
    5: playgrounds
    """
    r = img_np[:, :, 0].astype(np.float32)
    g = img_np[:, :, 1].astype(np.float32)
    b = img_np[:, :, 2].astype(np.float32)
    intensity = (r + g + b) / 3.0

    classes = np.zeros(img_np.shape[:2], dtype=np.uint8) # 0 = NVG_surface

    # Water
    water_mask = (b > r + 15) & (b > g) & (intensity < 130) | (intensity < 35)
    classes[water_mask] = 1

    # Playgrounds (red/terracotta tracks or vibrant synthetic courts)
    court_mask = (r > 130) & (r > g + 35) & (r > b + 35) & (~water_mask)
    classes[court_mask] = 5

    # Trees (deep dark green)
    tree_mask = (g > r + 12) & (g > b + 10) & (intensity < 115) & (~water_mask) & (~court_mask)
    classes[tree_mask] = 2

    # Low vegetation (lighter green / yellowish-green)
    low_veg_mask = (g > r + 5) & (g > b) & (~water_mask) & (~court_mask) & (~tree_mask)
    classes[low_veg_mask] = 3

    # Buildings: high local contrast / edges / rooftops (very bright or sharp edges with neutral color)
    # or distinctive roof materials
    building_mask = ((intensity > 175) & (np.abs(r - g) < 20) & (np.abs(g - b) < 20)) | \
                    ((r > 120) & (g > 60) & (b < 80) & (r > g + 20) & (~court_mask))
    classes[building_mask & (classes == 0)] = 4

    return classes

CLASS_NAMES = ["NVG_surface", "water", "trees", "low_vegetation", "buildings", "playgrounds"]

for idx in range(10):
    item = ds[idx]
    t1, t2 = ds.load_images(idx)
    c1 = classify_land_cover(np.array(t1))
    c2 = classify_land_cover(np.array(t2))

    # Detect per-class change
    diff = np.abs(np.array(t1, dtype=float) - np.array(t2, dtype=float)).mean(axis=2)
    changed_pixels = diff > 25.0

    print(f"\n--- Sample {idx}: Q: '{item['question']}' | GT: '{item['ground_truth']}' ---")
    for ci, cname in enumerate(CLASS_NAMES):
        mask1 = (c1 == ci)
        mask2 = (c2 == ci)
        changed_c = np.sum((mask1 | mask2) & changed_pixels) / max(1, np.sum(mask1 | mask2))
        print(f"  {cname:15s}: T1 area={mask1.mean()*100:5.1f}%, T2 area={mask2.mean()*100:5.1f}%, changed_ratio={changed_c*100:5.1f}%")
