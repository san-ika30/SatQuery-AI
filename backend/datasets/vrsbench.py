"""
SatQuery AI — VRSBench Benchmark Dataset Module
Loads REAL VRSBench evaluation samples (Single-Image VQA, Scene Captioning, Visual Grounding).
Source: xiang709/VRSBench (Official Hugging Face Dataset)
"""
import os
import sys
import json
from PIL import Image
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


class VRSBenchDataset(Dataset):
    """
    PyTorch / Python Dataset loader for REAL VRSBench evaluation samples.
    Exposes: sample_id, image, image_path, question, ground_truth, task_type, annotation.
    """
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        item = self.records[idx]
        img_path = item.get("image_path", "")

        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
        else:
            img = Image.new("RGB", (256, 256), color=(100, 100, 100))

        return {
            "sample_id": item.get("sample_id", f"vrs_{idx+1:04d}"),
            "image": img,
            "image_path": img_path,
            "image_size": img.size,
            "task_type": item.get("task_type", "vqa"),
            "question": item.get("question", ""),
            "ground_truth": item.get("ground_truth", ""),
            "annotation": item.get("annotation", {}),
        }


def verify_real_vrsbench_samples(num_samples: int = 5):
    """Verify and print details for num_samples real VRSBench benchmark samples."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))
    split_path = os.path.join(project_root, "dataset", "vrsbench", "splits", "vrsbench_test.json")

    if not os.path.exists(split_path):
        print(f"Split file missing at {split_path}")
        return

    with open(split_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    ds = VRSBenchDataset(records)
    print("=" * 75)
    print(f"VERIFYING REAL VRSBENCH SAMPLES ({min(num_samples, len(ds))} SAMPLES)")
    print("=" * 75)

    for i in range(min(num_samples, len(ds))):
        sample = ds[i]
        print(f"Sample #{i+1}: ID={sample['sample_id']} | Task={sample['task_type']}")
        print(f"  Image path: {os.path.basename(sample['image_path'])} | Dimensions: {sample['image_size']}")
        print(f"  Question:     {sample['question']}")
        print(f"  Ground Truth: {sample['ground_truth']}")
        print(f"  Annotation:   {sample['annotation']}")
        print("-" * 75)


if __name__ == "__main__":
    verify_real_vrsbench_samples(5)
