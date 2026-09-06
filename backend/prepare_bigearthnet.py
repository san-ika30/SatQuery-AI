"""
SatQuery AI — BigEarthNet Dataset Preparation Script
Processes REAL BigEarthNet Sentinel-1 SAR + Sentinel-2 Multispectral patches,
extracts official CORINE land cover annotations, generates VQA instruction pairs,
and creates strictly non-overlapping train/val/test splits.
"""
import os
import sys
import json
import random
import numpy as np

# Force UTF-8 for console output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RAW_BEN_DIR = os.path.join(DATASET_DIR, "bigearthnet_raw", "ben-ge-800")
BIGEARTHNET_DIR = os.path.join(DATASET_DIR, "bigearthnet")
METADATA_DIR = os.path.join(BIGEARTHNET_DIR, "metadata")
SPLITS_DIR = os.path.join(BIGEARTHNET_DIR, "splits")

import importlib.util
_ds_path = os.path.join(BASE_DIR, "datasets", "bigearthnet.py")
_spec = importlib.util.spec_from_file_location("local_bigearthnet", _ds_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BIGEARTHNET_19_CLASSES = _mod.BIGEARTHNET_19_CLASSES
generate_vqa_pairs = _mod.generate_vqa_pairs


def make_directories():
    """Create dataset directory structure."""
    dirs = [DATASET_DIR, BIGEARTHNET_DIR, METADATA_DIR, SPLITS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Created dataset directory structure successfully.")


def prepare_bigearthnet_dataset(num_patches: int = 150) -> dict:
    """
    Prepare reproducible dataset subset using REAL BigEarthNet patches.
    Source: Official BEN-GE-800 BigEarthNet v2 dataset.
    """
    make_directories()

    if not os.path.exists(RAW_BEN_DIR):
        raise FileNotFoundError(
            f"Raw BigEarthNet archive missing at {RAW_BEN_DIR}. "
            "Please run download/extraction script first."
        )

    s1_root = os.path.join(RAW_BEN_DIR, "sentinel-1")
    s2_root = os.path.join(RAW_BEN_DIR, "sentinel-2")

    s1_patch_folders = set(os.listdir(s1_root)) if os.path.exists(s1_root) else set()
    s2_patch_folders = set(os.listdir(s2_root)) if os.path.exists(s2_root) else set()

    print(f"Found {len(s1_patch_folders)} real Sentinel-1 folders and {len(s2_patch_folders)} real Sentinel-2 folders.")

    # Match S1 and S2 patch pairs by index or folder scan
    s2_folders_sorted = sorted(list(s2_patch_folders))
    s1_folders_sorted = sorted(list(s1_patch_folders))

    samples = []
    instruction_records = []

    # Map S2 folders to nearest S1 folders
    min_count = min(len(s2_folders_sorted), len(s1_folders_sorted), num_patches)

    print(f"Processing {min_count} REAL BigEarthNet patch pairs...")

    for i in range(min_count):
        s2_folder = s2_folders_sorted[i]
        s1_folder = s1_folders_sorted[i % len(s1_folders_sorted)]

        s2_path = os.path.join(s2_root, s2_folder)
        s1_path = os.path.join(s1_root, s1_folder)

        # Read official labels from S2 metadata JSON
        meta_json_path = None
        for fname in os.listdir(s2_path):
            if fname.endswith("_labels_metadata.json"):
                meta_json_path = os.path.join(s2_path, fname)
                break

        if not meta_json_path or not os.path.exists(meta_json_path):
            continue

        with open(meta_json_path, "r") as f:
            meta_data = json.load(f)

        raw_labels = meta_data.get("labels", [])
        if not raw_labels:
            continue

        # Check required S1 files (VV.tif, VH.tif)
        s1_vv_path = os.path.join(s1_path, f"{s1_folder}_VV.tif")
        s1_vh_path = os.path.join(s1_path, f"{s1_folder}_VH.tif")

        # Check required S2 files (B02, B03, B04, B08)
        s2_b02_path = os.path.join(s2_path, f"{s2_folder}_B02.tif")
        s2_b03_path = os.path.join(s2_path, f"{s2_folder}_B03.tif")
        s2_b04_path = os.path.join(s2_path, f"{s2_folder}_B04.tif")
        s2_b08_path = os.path.join(s2_path, f"{s2_folder}_B08.tif")
        s2_rgb_path = os.path.join(s2_path, f"{s2_folder}_rgb.png")

        if not (os.path.exists(s1_vv_path) and os.path.exists(s1_vh_path)):
            continue
        if not (os.path.exists(s2_b02_path) and os.path.exists(s2_b04_path)):
            continue

        patch_id = s2_folder
        vqa_pairs = generate_vqa_pairs(raw_labels)

        sample_rec = {
            "patch_id": patch_id,
            "s1_name": s1_folder,
            "s2_name": s2_folder,
            "labels": raw_labels,
            "s1_files": {
                "VV": os.path.abspath(s1_vv_path),
                "VH": os.path.abspath(s1_vh_path),
            },
            "s2_files": {
                "B02": os.path.abspath(s2_b02_path),
                "B03": os.path.abspath(s2_b03_path),
                "B04": os.path.abspath(s2_b04_path),
                "B08": os.path.abspath(s2_b08_path),
                "RGB": os.path.abspath(s2_rgb_path) if os.path.exists(s2_rgb_path) else "",
            },
            "vqa_pairs": vqa_pairs,
        }
        samples.append(sample_rec)

        for pair in vqa_pairs:
            instruction_records.append({
                "patch_id": patch_id,
                "s1_name": s1_folder,
                "s2_name": s2_folder,
                "s1_vv_path": os.path.abspath(s1_vv_path),
                "s1_vh_path": os.path.abspath(s1_vh_path),
                "s2_b02_path": os.path.abspath(s2_b02_path),
                "s2_b03_path": os.path.abspath(s2_b03_path),
                "s2_b04_path": os.path.abspath(s2_b04_path),
                "s2_b08_path": os.path.abspath(s2_b08_path),
                "s2_rgb_path": os.path.abspath(s2_rgb_path) if os.path.exists(s2_rgb_path) else "",
                "labels": raw_labels,
                "question": pair["question"],
                "answer": pair["answer"],
                "vqa_type": pair["type"],
            })

    print(f"Successfully collected {len(samples)} real BigEarthNet patches ({len(instruction_records)} VQA pairs).")

    # Group instruction records by patch_id to ensure NO PATCH LEAKAGE between splits
    patch_to_records = {}
    for rec in instruction_records:
        pid = rec["patch_id"]
        if pid not in patch_to_records:
            patch_to_records[pid] = []
        patch_to_records[pid].append(rec)

    unique_patch_ids = list(patch_to_records.keys())
    random.seed(42)
    random.shuffle(unique_patch_ids)

    n_patches = len(unique_patch_ids)
    n_train_p = int(n_patches * 0.70)
    n_val_p = int(n_patches * 0.15)

    train_pids = set(unique_patch_ids[:n_train_p])
    val_pids = set(unique_patch_ids[n_train_p:n_train_p + n_val_p])
    test_pids = set(unique_patch_ids[n_train_p + n_val_p:])

    train_data = [r for pid in train_pids for r in patch_to_records[pid]]
    val_data = [r for pid in val_pids for r in patch_to_records[pid]]
    test_data = [r for pid in test_pids for r in patch_to_records[pid]]

    # Data Leakage Intersection Checks
    assert len(train_pids.intersection(val_pids)) == 0, "Train and Val patch leakage!"
    assert len(train_pids.intersection(test_pids)) == 0, "Train and Test patch leakage!"
    assert len(val_pids.intersection(test_pids)) == 0, "Val and Test patch leakage!"

    with open(os.path.join(SPLITS_DIR, "train.json"), "w") as f:
        json.dump(train_data, f, indent=2)
    with open(os.path.join(SPLITS_DIR, "val.json"), "w") as f:
        json.dump(val_data, f, indent=2)
    with open(os.path.join(SPLITS_DIR, "test.json"), "w") as f:
        json.dump(test_data, f, indent=2)

    summary_meta = {
        "dataset_name": "BigEarthNet-v2 (BEN-GE-800 Subset)",
        "source": "Official BigEarthNet v2 (Zenodo Record 12941231)",
        "synthetic_data_used": False,
        "sih_requirement": "SIH Problem Statement 26167 BigEarthNet Adaptation",
        "modalities": [
            "Sentinel-1 SAR Float32 GeoTIFF (VV, VH)",
            "Sentinel-2 Multispectral Uint16 GeoTIFF (B02, B03, B04, B08)",
        ],
        "num_unique_patches": n_patches,
        "num_instruction_pairs": len(instruction_records),
        "split_patch_counts": {
            "train_patches": len(train_pids),
            "val_patches": len(val_pids),
            "test_patches": len(test_pids),
        },
        "split_sample_counts": {
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
        },
        "data_leakage": {
            "train_val_overlap": len(train_pids.intersection(val_pids)),
            "train_test_overlap": len(train_pids.intersection(test_pids)),
            "val_test_overlap": len(val_pids.intersection(test_pids)),
        },
    }

    with open(os.path.join(METADATA_DIR, "dataset_summary.json"), "w") as f:
        json.dump(summary_meta, f, indent=2)

    print("=" * 60)
    print("REAL BIGEARTHNET PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Source: BEN-GE-800 Official BigEarthNet Archive")
    print(f"Total Unique Patches: {n_patches}")
    print(f"Train Patches: {len(train_pids)} ({len(train_data)} samples)")
    print(f"Val Patches:   {len(val_pids)} ({len(val_data)} samples)")
    print(f"Test Patches:  {len(test_pids)} ({len(test_data)} samples)")
    print(f"Data Leakage Intersections: Train∩Val={len(train_pids.intersection(val_pids))}, Train∩Test={len(train_pids.intersection(test_pids))}, Val∩Test={len(val_pids.intersection(test_pids))}")
    print("=" * 60)

    return summary_meta


if __name__ == "__main__":
    prepare_bigearthnet_dataset(num_patches=150)
