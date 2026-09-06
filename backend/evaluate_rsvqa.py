"""
SatQuery AI — RSVQA Single-Image Evaluation Script
Evaluates SatQuery AI on REAL RSVQA benchmark samples (Presence, Counting, Comparison, Area VQA).
Source: dmarsili/RSVQA-LR-2k (Zenodo Record 6344334)
"""
import io
import os
import sys
import json
import time
import base64
import re
import numpy as np
from PIL import Image
import torch

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

SPLITS_DIR = os.path.join(PROJECT_ROOT, "dataset", "rsvqa", "splits")
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
import importlib.util

_ds_path = os.path.join(BASE_DIR, "datasets", "rsvqa.py")
_spec_rsv = importlib.util.spec_from_file_location("local_rsvqa_eval", _ds_path)
_mod_rsv = importlib.util.module_from_spec(_spec_rsv)
_spec_rsv.loader.exec_module(_mod_rsv)

RSVQADataset = _mod_rsv.RSVQADataset

from core.orchestrator import get_vlm_module


def normalize_answer(ans: str) -> str:
    """Careful answer normalization for auditability."""
    if ans is None:
        return ""
    a = str(ans).lower().strip()
    a = a.rstrip(".").rstrip(",").strip()
    if a in ["true", "yes", "y"]:
        return "yes"
    if a in ["false", "no", "n"]:
        return "no"
    if a in ["urban area", "urban"]:
        return "urban"
    if a in ["rural area", "rural"]:
        return "rural"
    return a


def extract_first_number(text: str) -> float | None:
    """Extract first integer or float from text."""
    matches = re.findall(r"\d+(?:\.\d+)?", text)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            return None
    return None


def match_rsvqa_answer(pred: str, gt: str, q_type: str) -> bool:
    """Match predicted answer against ground-truth answer based on RSVQA question type."""
    norm_gt = normalize_answer(gt)
    norm_pred = normalize_answer(pred)

    if q_type == "count":
        # Numerical tolerance matching
        gt_num = extract_first_number(norm_gt)
        pred_num = extract_first_number(norm_pred)
        if gt_num is not None and pred_num is not None:
            tol = max(1.0, 0.15 * gt_num)
            return abs(pred_num - gt_num) <= tol
        return norm_gt in norm_pred

    elif q_type == "presence":
        return norm_gt == norm_pred or (norm_gt in norm_pred)

    else: # area / comparison / general
        return norm_gt == norm_pred or (norm_gt in norm_pred) or (norm_pred in norm_gt)


def run_rsvqa_evaluation() -> dict:
    t0 = time.monotonic()
    split_path = os.path.join(SPLITS_DIR, "rsvqa_test.json")

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"RSVQA test split missing at {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    ds = RSVQADataset(records)
    vlm_app = get_vlm_module()

    print("=" * 70)
    print(f"STARTING RSVQA SINGLE-IMAGE EVALUATION ({len(ds)} REAL SAMPLES)")
    print("=" * 70)

    category_results = {
        "presence": {"total": 0, "correct": 0},
        "count": {"total": 0, "correct": 0},
        "comparison": {"total": 0, "correct": 0},
        "area": {"total": 0, "correct": 0},
    }

    sample_evaluations = []
    error_analysis_cases = []

    for i in range(len(ds)):
        item = ds[i]
        sample_id = item["sample_id"]
        q_type = item["question_type"]
        img = item["image"]
        question = item["question"]
        gt_answer = item["ground_truth"]

        # Run VLM reasoning Specialist
        vlm_res = vlm_app.answer_satellite_question(img, question, [])
        pred_text = vlm_res.get("answer", "")

        is_correct = match_rsvqa_answer(pred_text, gt_answer, q_type)

        if q_type not in category_results:
            category_results[q_type] = {"total": 0, "correct": 0}

        category_results[q_type]["total"] += 1
        if is_correct:
            category_results[q_type]["correct"] += 1

        rec_eval = {
            "sample_id": sample_id,
            "question_type": q_type,
            "question": question,
            "ground_truth": gt_answer,
            "prediction": pred_text,
            "normalized_gt": normalize_answer(gt_answer),
            "normalized_pred": normalize_answer(pred_text),
            "correct": is_correct,
        }
        sample_evaluations.append(rec_eval)

        if not is_correct and len(error_analysis_cases) < 20:
            error_analysis_cases.append({
                "sample_id": sample_id,
                "dataset": "RSVQA-LR",
                "question_type": q_type,
                "question": question,
                "ground_truth": gt_answer,
                "prediction": pred_text,
                "correct": False,
                "error_category": f"numerical reasoning" if q_type == "count" else "land-cover recognition",
            })

    elapsed = round(time.monotonic() - t0, 2)

    total_samples = len(ds)
    total_correct = sum(cat["correct"] for cat in category_results.values())
    overall_acc = round(total_correct / max(1, total_samples), 4)

    cat_accs = {}
    for cat, res in category_results.items():
        if res["total"] > 0:
            cat_accs[f"{cat}_accuracy"] = round(res["correct"] / res["total"], 4)

    eval_output = {
        "dataset": "RSVQA-LR (Remote Sensing VQA Low-Resolution)",
        "source": "dmarsili/RSVQA-LR-2k (Zenodo Record 6344334)",
        "synthetic_data_used": False,
        "total_evaluated_samples": total_samples,
        "evaluation_time_seconds": elapsed,
        "metrics": {
            "overall_accuracy": overall_acc,
            "presence_accuracy": cat_accs.get("presence_accuracy", 0.0),
            "count_accuracy": cat_accs.get("count_accuracy", 0.0),
            "area_accuracy": cat_accs.get("area_accuracy", 0.0),
        },
        "category_breakdown": category_results,
        "error_analysis_samples": error_analysis_cases,
    }

    out_file = os.path.join(OUTPUT_DIR, "rsvqa_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2)

    print("=" * 70)
    print("RSVQA EVALUATION RESULTS")
    print("=" * 70)
    print(f"Overall RSVQA Accuracy: {overall_acc * 100:.2f}% ({total_correct}/{total_samples})")
    print(f"Presence Accuracy:      {cat_accs.get('presence_accuracy', 0.0) * 100:.2f}%")
    print(f"Count Accuracy:         {cat_accs.get('count_accuracy', 0.0) * 100:.2f}%")
    print(f"Area Accuracy:          {cat_accs.get('area_accuracy', 0.0) * 100:.2f}%")
    print(f"Results File: {out_file}")
    print("=" * 70)

    return eval_output


if __name__ == "__main__":
    run_rsvqa_evaluation()
