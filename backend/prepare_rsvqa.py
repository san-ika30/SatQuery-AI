"""
SatQuery AI — RSVQA Dataset Preparation Script
Downloads and prepares REAL RSVQA evaluation samples (Presence, Counting, Comparison, Area)
from official repositories 'dmarsili/RSVQA-LR-2k' and Zenodo Record 6344334.
"""
import io
import os
import sys
import json
import random
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RAW_RSVQA_DIR = os.path.join(DATASET_DIR, "rsvqa_raw")
RSVQA_DIR = os.path.join(DATASET_DIR, "rsvqa")
IMAGES_DIR = os.path.join(RSVQA_DIR, "images")
SPLITS_DIR = os.path.join(RSVQA_DIR, "splits")

os.makedirs(RAW_RSVQA_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)


def determine_q_type(q: str) -> str:
    """Classify RSVQA question type."""
    q_low = q.lower().strip()
    if any(k in q_low for k in ["how many", "count", "number of"]):
        return "count"
    elif any(k in q_low for k in ["is there", "are there", "presence", "does the"]):
        return "presence"
    elif any(k in q_low for k in ["more", "less", "equal", "greater", "fewer"]):
        return "comparison"
    else:
        return "area"


def prepare_rsvqa_dataset(max_samples: int = 150) -> dict:
    """
    Prepare REAL RSVQA benchmark subset for evaluation.
    Source: dmarsili/RSVQA-LR-2k / Zenodo Record 6344334
    """
    print("Preparing REAL RSVQA evaluation dataset...")

    parquet_path = hf_hub_download(
        repo_id="dmarsili/RSVQA-LR-2k",
        filename="data/validation-00000-of-00001.parquet",
        repo_type="dataset",
        local_dir=RAW_RSVQA_DIR,
    )

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} raw RSVQA dataset records.")

    rsvqa_records = []

    for idx in range(min(max_samples, len(df))):
        row = df.iloc[idx]
        img_dict = row.get("image")

        image_filename = f"rsvqa_sample_{idx+1:04d}.png"
        img_save_path = os.path.join(IMAGES_DIR, image_filename)

        # Extract image bytes or PIL image
        if not os.path.exists(img_save_path):
            if isinstance(img_dict, dict) and "bytes" in img_dict:
                img = Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB")
                img.save(img_save_path)
            elif isinstance(img_dict, Image.Image):
                img_dict.save(img_save_path)
            else:
                continue

        question_text = str(row.get("question", "")).strip()
        answer_text = str(row.get("answer", "")).strip()
        q_type = str(row.get("type", determine_q_type(question_text)))

        if not question_text or not answer_text:
            continue

        rsvqa_records.append({
            "sample_id": f"rsvqa_{idx+1:04d}",
            "image_path": os.path.abspath(img_save_path),
            "image_id": image_filename,
            "question": question_text,
            "ground_truth": answer_text,
            "question_type": q_type,
            "annotation": {
                "question_type": q_type,
                "dataset": "RSVQA-LR",
            },
        })

    # Save test split JSON
    out_path = os.path.join(SPLITS_DIR, "rsvqa_test.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rsvqa_records, f, indent=2)

    summary = {
        "dataset_name": "RSVQA-LR (Remote Sensing VQA Low-Resolution)",
        "source": "dmarsili/RSVQA-LR-2k (Zenodo Record 6344334)",
        "synthetic_data_used": False,
        "total_samples": len(rsvqa_records),
        "question_type_counts": {
            "presence": sum(1 for r in rsvqa_records if r["question_type"] == "presence"),
            "count": sum(1 for r in rsvqa_records if r["question_type"] == "count"),
            "comparison": sum(1 for r in rsvqa_records if r["question_type"] == "comparison"),
            "area": sum(1 for r in rsvqa_records if r["question_type"] == "area"),
        },
        "split_file": os.path.abspath(out_path),
    }

    print("=" * 60)
    print("RSVQA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"Total Prepared Samples: {summary['total_samples']}")
    print(f"Type Breakdown: Presence={summary['question_type_counts']['presence']}, Count={summary['question_type_counts']['count']}, Comparison={summary['question_type_counts']['comparison']}, Area={summary['question_type_counts']['area']}")
    print(f"Split File: {out_path}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    prepare_rsvqa_dataset(max_samples=150)
