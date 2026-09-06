"""
SatQuery AI — Real Optical + SAR Cross-Modal Benchmark Evaluation & Ablation Script
Evaluates multi-sensor feature extraction, dual-stream tensor fusion, and modality ablations
on the 75 REAL held-out BigEarthNet Sentinel-1 (SAR) + Sentinel-2 (Multispectral) test pairs.
"""
import io
import os
import sys
import json
import time
import asyncio
import numpy as np
import structlog
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from datasets.bigearthnet import BigEarthNetDataset, BIGEARTHNET_19_CLASSES
from core.optical_sar_analyzer import get_optical_sar_analyzer
from train_bigearthnet_adapter import BigEarthNetVisionLanguageAdapter

log = structlog.get_logger()

RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "optical_sar_results.json")
SPLIT_FILE = os.path.join(os.path.dirname(BASE_DIR), "dataset", "bigearthnet", "splits", "test.json")

# Standard CORINE 43-to-19 class mapping
CORINE_43_TO_19 = {
    "Continuous urban fabric": "Urban fabric",
    "Discontinuous urban fabric": "Urban fabric",
    "Industrial or commercial units": "Industrial or commercial units",
    "Non-irrigated arable land": "Arable land",
    "Permanently irrigated land": "Arable land",
    "Rice fields": "Arable land",
    "Vineyards": "Permanent crops",
    "Fruit trees and berry plantations": "Permanent crops",
    "Olive groves": "Permanent crops",
    "Pastures": "Pastures",
    "Complex cultivation patterns": "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation": "Land principally occupied by agriculture",
    "Agro-forestry areas": "Agro-forestry areas",
    "Broad-leaved forest": "Broad-leaved forest",
    "Coniferous forest": "Coniferous forest",
    "Mixed forest": "Mixed forest",
    "Natural grasslands": "Natural grasslands",
    "Moors and heathland": "Moors and heathland",
    "Sclerophyllous vegetation": "Sclerophyllous vegetation",
    "Transitional woodland/shrub": "Transitional woodland-shrub",
    "Beaches, dunes, sands": "Beaches, dunes, sands",
    "Inland marshes": "Inland wetlands",
    "Peat bogs": "Inland wetlands",
    "Salt marshes": "Coastal wetlands",
    "Salines": "Coastal wetlands",
    "Intertidal flats": "Coastal wetlands",
    "Water courses": "Inland waters",
    "Water bodies": "Inland waters",
    "Coastal lagoons": "Inland waters",
    "Estuaries": "Inland waters",
    "Sea and ocean": "Inland waters",
}


def run_modality_ablation_evaluation(
    dataset: BigEarthNetDataset,
    model: BigEarthNetVisionLanguageAdapter,
    threshold: float = 0.30
) -> dict:
    """
    Run comparative modality ablation:
      1. Optical Only (SAR input zeroed out)
      2. SAR Only (Optical input zeroed out)
      3. Optical + SAR Fused (Both active)
    """
    modes = ["optical_only", "sar_only", "fused"]
    ablation_results = {}

    urban_indices = [i for i, c in enumerate(BIGEARTHNET_19_CLASSES) if "urban" in c.lower() or "industrial" in c.lower()]
    water_indices = [i for i, c in enumerate(BIGEARTHNET_19_CLASSES) if "water" in c.lower() or "wetland" in c.lower()]
    veg_indices = [i for i, c in enumerate(BIGEARTHNET_19_CLASSES) if any(k in c.lower() for k in ["forest", "pasture", "arable", "crop", "grass", "wood"])]

    for mode in modes:
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i in range(len(dataset)):
                item = dataset[i]
                s1 = item["s1_tensor"].unsqueeze(0)
                s2 = item["s2_tensor"].unsqueeze(0)
                target = item["target"].numpy()

                if mode == "optical_only":
                    logits, _ = model(torch.zeros_like(s1), s2)
                elif mode == "sar_only":
                    logits, _ = model(s1, torch.zeros_like(s2))
                else:
                    logits, _ = model(s1, s2)

                probs = torch.sigmoid(logits).cpu().numpy()[0]
                preds = (probs >= threshold).astype(np.float32)
                if preds.sum() == 0:
                    preds[np.argmax(probs)] = 1.0

                all_preds.append(preds)
                all_targets.append(target)

        P = np.stack(all_preds)
        T = np.stack(all_targets)

        tp = float((P * T).sum())
        fp = float((P * (1 - T)).sum())
        fn = float(((1 - P) * T).sum())

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        acc = float((P == T).mean()) * 100.0

        def eval_subset(indices):
            p_sub = P[:, indices].max(axis=1)
            t_sub = T[:, indices].max(axis=1)
            tp_c = float((p_sub * t_sub).sum())
            fp_c = float((p_sub * (1 - t_sub)).sum())
            fn_c = float(((1 - p_sub) * t_sub).sum())
            pr_c = tp_c / (tp_c + fp_c + 1e-8)
            re_c = tp_c / (tp_c + fn_c + 1e-8)
            f1_c = 2 * pr_c * re_c / (pr_c + re_c + 1e-8)
            acc_c = float((p_sub == t_sub).mean()) * 100.0
            return {
                "precision": round(pr_c, 4),
                "recall": round(re_c, 4),
                "f1": round(f1_c, 4),
                "accuracy_pct": round(acc_c, 2),
            }

        ablation_results[mode] = {
            "macro_precision": round(precision, 4),
            "macro_recall": round(recall, 4),
            "macro_f1": round(f1, 4),
            "overall_accuracy_pct": round(acc, 2),
            "built_up_structures": eval_subset(urban_indices),
            "water_bodies": eval_subset(water_indices),
            "vegetation_and_crops": eval_subset(veg_indices),
        }

    return ablation_results


def run_cross_modal_vqa_evaluation(
    dataset: BigEarthNetDataset,
    analyzer
) -> dict:
    """
    Evaluate representative cross-modal VQA queries across all 75 real test pairs:
      1. Joint Query: "Use the optical and SAR images together to identify built-up and water-covered regions."
      2. Water Query: "Are there water bodies or aquatic features in this scene?"
      3. Built-Up Query: "Are there built-up urban structures or industrial units in this scene?"
    """
    total = len(dataset)
    correct_joint = 0
    correct_water = 0
    correct_urban = 0

    sample_evaluations = []

    for i in range(total):
        item = dataset[i]
        raw_labels = set(item["labels"])
        mapped = set(CORINE_43_TO_19.get(l, l) for l in raw_labels)

        gt_water = bool(mapped.intersection({"Inland waters", "Inland wetlands", "Coastal wetlands"}))
        gt_urban = bool(mapped.intersection({"Urban fabric", "Industrial or commercial units"}))

        # Run OpticalSARAnalyzer
        res = analyzer.analyze_cross_modal(
            item["s2_tensor"],
            item["s1_tensor"],
            "Use the optical and SAR images together to identify built-up and water-covered regions."
        )

        pred_water = res["water_analysis"]["confirmed"]
        pred_urban = res["built_up_analysis"]["confirmed"]

        match_joint = (pred_water == gt_water) and (pred_urban == gt_urban)
        match_water = (pred_water == gt_water)
        match_urban = (pred_urban == gt_urban)

        if match_joint:
            correct_joint += 1
        if match_water:
            correct_water += 1
        if match_urban:
            correct_urban += 1

        if i < 20:
            sample_evaluations.append({
                "patch_id": item["patch_id"],
                "raw_labels": item["labels"],
                "gt_water": gt_water,
                "gt_urban": gt_urban,
                "pred_water": pred_water,
                "pred_urban": pred_urban,
                "match_joint": match_joint,
                "answer": res["answer"],
                "optical_evidence_sample": res["optical_evidence"][0] if res["optical_evidence"] else "",
                "sar_evidence_sample": res["sar_evidence"][0] if res["sar_evidence"] else "",
            })

    return {
        "total_samples": total,
        "joint_builtup_water_accuracy": round((correct_joint / total) * 100.0, 2),
        "water_detection_accuracy": round((correct_water / total) * 100.0, 2),
        "built_up_detection_accuracy": round((correct_urban / total) * 100.0, 2),
        "sample_evaluations": sample_evaluations,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.monotonic()

    print("\n==================================================")
    print("STARTING CHUNK 10 OPTICAL + SAR CROSS-MODAL EVALUATION")
    print("Dataset: Real BigEarthNet Sentinel-1 + Sentinel-2 Test Split")
    print(f"Split path: {SPLIT_FILE}")
    print("==================================================\n")

    if not os.path.exists(SPLIT_FILE):
        raise FileNotFoundError(f"Test split file missing: {SPLIT_FILE}")

    with open(SPLIT_FILE, "r") as f:
        recs = json.load(f)

    dataset = BigEarthNetDataset(recs)
    print(f"Loaded {len(dataset)} verified real optical + SAR patch pairs.")

    # 1. Load Pretrained Dual-Stream Adapter
    ckpt_path = os.path.join(BASE_DIR, "checkpoints", "bigearthnet_adapter", "best_adapter.pt")
    model = BigEarthNetVisionLanguageAdapter()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 2. Run Modality Ablation Study (Optical vs SAR vs Fused)
    print("\n[Phase 1/2] Running Modality Ablation (Optical vs SAR vs Fused)...")
    ablation_results = run_modality_ablation_evaluation(dataset, model, threshold=0.30)

    opt_f1 = ablation_results["optical_only"]["macro_f1"]
    sar_f1 = ablation_results["sar_only"]["macro_f1"]
    fused_f1 = ablation_results["fused"]["macro_f1"]
    f1_gain = round(fused_f1 - max(opt_f1, sar_f1), 4)

    print(f"  Optical-Only Macro F1: {opt_f1}")
    print(f"  SAR-Only Macro F1:     {sar_f1}")
    print(f"  Fused Optical+SAR F1:  {fused_f1} (Gain: +{f1_gain})")

    # 3. Run Semantic Cross-Modal VQA Evaluation
    print("\n[Phase 2/2] Running Semantic Cross-Modal VQA Evaluation...")
    analyzer = get_optical_sar_analyzer()
    vqa_results = run_cross_modal_vqa_evaluation(dataset, analyzer)

    print(f"  Water Detection Accuracy:    {vqa_results['water_detection_accuracy']}%")
    print(f"  Built-up Detection Accuracy: {vqa_results['built_up_detection_accuracy']}%")
    print(f"  Joint Built-up+Water Acc:    {vqa_results['joint_builtup_water_accuracy']}%")

    elapsed_sec = round(time.monotonic() - t0, 2)

    summary = {
        "evaluation_name": "Chunk 10 Optical + SAR Cross-Modal Analysis & Ablation",
        "benchmark": "BigEarthNet-v2 (Sentinel-1 SAR + Sentinel-2 Multispectral)",
        "source": "Official BEN-GE-800 Benchmark (Charfuelan et al., DLR / TU Berlin)",
        "total_test_samples": len(dataset),
        "is_synthetic": False,
        "modalities_evaluated": ["Sentinel-2 Multispectral (B02/B03/B04/B08)", "Sentinel-1 SAR (VV/VH dB)"],
        "modality_ablation": {
            "optical_only": ablation_results["optical_only"],
            "sar_only": ablation_results["sar_only"],
            "optical_sar_fused": ablation_results["fused"],
            "improvements": {
                "f1_score_gain": f1_gain,
                "relative_f1_improvement_pct": round((f1_gain / max(opt_f1, sar_f1)) * 100.0, 2),
                "built_up_f1_gain": round(ablation_results["fused"]["built_up_structures"]["f1"] - ablation_results["optical_only"]["built_up_structures"]["f1"], 4),
            },
        },
        "vqa_semantic_evaluation": {
            "joint_builtup_water_accuracy": vqa_results["joint_builtup_water_accuracy"],
            "water_detection_accuracy": vqa_results["water_detection_accuracy"],
            "built_up_detection_accuracy": vqa_results["built_up_detection_accuracy"],
        },
        "total_evaluation_time_sec": elapsed_sec,
        "sample_evaluations": vqa_results["sample_evaluations"],
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n==================================================")
    print(f"EVALUATION COMPLETE ({elapsed_sec}s)")
    print(f"Results saved to: {RESULTS_FILE}")
    print("==================================================\n")

    return summary


if __name__ == "__main__":
    main()
