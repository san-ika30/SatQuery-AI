"""
SatQuery AI — BigEarthNet Remote-Sensing Adapter Evaluation Script
Evaluates and compares BASE MODEL (Unadapted Features) vs ADAPTED MODEL (BigEarthNet Real-Data Adapted Features)
on the held-out test split (75 samples across 23 held-out patches).
"""
import io
import os
import sys
import json
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

SPLITS_DIR = os.path.join(PROJECT_ROOT, "dataset", "bigearthnet", "splits")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints", "bigearthnet_adapter")

sys.path.insert(0, BASE_DIR)
import importlib.util

_ds_path = os.path.join(BASE_DIR, "datasets", "bigearthnet.py")
_spec_ds = importlib.util.spec_from_file_location("local_bigearthnet_eval", _ds_path)
_mod_ds = importlib.util.module_from_spec(_spec_ds)
_spec_ds.loader.exec_module(_mod_ds)

BIGEARTHNET_19_CLASSES = _mod_ds.BIGEARTHNET_19_CLASSES
BigEarthNetDataset = _mod_ds.BigEarthNetDataset

_tr_path = os.path.join(BASE_DIR, "train_bigearthnet_adapter.py")
_spec_tr = importlib.util.spec_from_file_location("local_train_adapter", _tr_path)
_mod_tr = importlib.util.module_from_spec(_spec_tr)
_spec_tr.loader.exec_module(_mod_tr)

BigEarthNetVisionLanguageAdapter = _mod_tr.BigEarthNetVisionLanguageAdapter


def compute_f1(preds_binary: np.ndarray, targets_binary: np.ndarray) -> float:
    """Compute Micro F1 Score across multi-label land cover predictions."""
    tp = np.sum((preds_binary == 1) & (targets_binary == 1))
    fp = np.sum((preds_binary == 1) & (targets_binary == 0))
    fn = np.sum((preds_binary == 0) & (targets_binary == 1))

    if tp + fp + fn == 0:
        return 1.0
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    if precision + recall == 0:
        return 0.0
    return float(2 * (precision * recall) / (precision + recall))


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict:
    """Evaluate loss, accuracy, F1 score, and embedding norm for a model on REAL patch tensors."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_embeds = []

    with torch.no_grad():
        for batch in dataloader:
            s1_t = batch["s1_tensor"].to(device)
            s2_t = batch["s2_tensor"].to(device)
            targets = batch["target"].to(device)

            logits, embeds = model(s1_t, s2_t)
            loss = criterion(logits, targets)

            total_loss += loss.item() * s1_t.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            preds_binary = (probs >= 0.30).astype(int)

            all_preds.append(preds_binary)
            all_targets.append(targets.cpu().numpy().astype(int))
            all_embeds.append(embeds.cpu().numpy())

    num_samples = len(dataloader.dataset)
    mean_loss = total_loss / num_samples

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_embeds = np.vstack(all_embeds)

    f1 = compute_f1(all_preds, all_targets)
    acc = float(np.mean(all_preds == all_targets))
    mean_embed_norm = float(np.mean(np.linalg.norm(all_embeds, axis=1)))

    return {
        "loss": round(float(mean_loss), 4),
        "f1_score": round(float(f1), 4),
        "accuracy": round(float(acc), 4),
        "mean_embedding_norm": round(float(mean_embed_norm), 4),
    }


def run_evaluation() -> dict:
    t0 = time.monotonic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_path = os.path.join(SPLITS_DIR, "train.json")
    val_path = os.path.join(SPLITS_DIR, "val.json")
    test_path = os.path.join(SPLITS_DIR, "test.json")
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_adapter.pt")

    if not os.path.exists(test_path):
        raise FileNotFoundError("Test split file missing at splits/test.json.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint missing at {ckpt_path}.")

    with open(train_path, "r") as f:
        train_records = json.load(f)
    with open(val_path, "r") as f:
        val_records = json.load(f)
    with open(test_path, "r") as f:
        test_records = json.load(f)

    # Data Leakage Verification
    train_pids = set(r["patch_id"] for r in train_records)
    val_pids = set(r["patch_id"] for r in val_records)
    test_pids = set(r["patch_id"] for r in test_records)

    train_val_overlap = len(train_pids.intersection(val_pids))
    train_test_overlap = len(train_pids.intersection(test_pids))
    val_test_overlap = len(val_pids.intersection(test_pids))

    assert train_val_overlap == 0, "DATA LEAKAGE ERROR: Train and Val contain overlapping patches!"
    assert train_test_overlap == 0, "DATA LEAKAGE ERROR: Train and Test contain overlapping patches!"
    assert val_test_overlap == 0, "DATA LEAKAGE ERROR: Val and Test contain overlapping patches!"

    test_ds = BigEarthNetDataset(test_records)

    def collate_fn(batch):
        return {
            "patch_id": [item["patch_id"] for item in batch],
            "s1_tensor": torch.stack([item["s1_tensor"] for item in batch], dim=0),
            "s2_tensor": torch.stack([item["s2_tensor"] for item in batch], dim=0),
            "target": torch.stack([item["target"] for item in batch], dim=0),
            "labels": [item["labels"] for item in batch],
            "question": [item["question"] for item in batch],
            "answer": [item["answer"] for item in batch],
        }

    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # 1. Base Model (Unadapted / Initialized Features)
    torch.manual_seed(123)
    base_model = BigEarthNetVisionLanguageAdapter().to(device)
    base_metrics = evaluate_model(base_model, test_loader, device)

    # 2. Adapted Model (Trained on Real BigEarthNet Data)
    adapted_model = BigEarthNetVisionLanguageAdapter().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    adapted_model.load_state_dict(ckpt["model_state_dict"])
    adapted_metrics = evaluate_model(adapted_model, test_loader, device)

    elapsed = round(time.monotonic() - t0, 2)

    eval_results = {
        "evaluation_dataset": "REAL BigEarthNet-v2 Held-Out Test Split",
        "num_test_patches": len(test_pids),
        "num_test_samples": len(test_ds),
        "data_leakage_checks": {
            "train_val_overlap": train_val_overlap,
            "train_test_overlap": train_test_overlap,
            "val_test_overlap": val_test_overlap,
            "status": "PASSED (0 OVERLAP)",
        },
        "evaluation_time_seconds": elapsed,
        "base_model_unadapted": base_metrics,
        "adapted_model_bigearthnet": adapted_metrics,
        "improvements": {
            "loss_reduction": round(base_metrics["loss"] - adapted_metrics["loss"], 4),
            "f1_score_gain": round(adapted_metrics["f1_score"] - base_metrics["f1_score"], 4),
            "accuracy_gain": round(adapted_metrics["accuracy"] - base_metrics["accuracy"], 4),
        },
    }

    eval_path = os.path.join(CHECKPOINT_DIR, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print("=" * 60)
    print("REAL BIGEARTHNET ADAPTATION EVALUATION REPORT")
    print("=" * 60)
    print(f"Test Held-Out Patches: {len(test_pids)} ({len(test_ds)} samples)")
    print(f"Data Leakage Check: Train∩Test={train_test_overlap}, Val∩Test={val_test_overlap} (PASSED)")
    print(f"BASE MODEL (Unadapted)  -> Loss: {base_metrics['loss']} | F1: {base_metrics['f1_score']} | Acc: {base_metrics['accuracy']}")
    print(f"ADAPTED MODEL (BigEarthNet)-> Loss: {adapted_metrics['loss']} | F1: {adapted_metrics['f1_score']} | Acc: {adapted_metrics['accuracy']}")
    print("-" * 60)
    print(f"Loss Reduction: {eval_results['improvements']['loss_reduction']}")
    print(f"F1 Score Gain:  +{eval_results['improvements']['f1_score_gain']}")
    print(f"Accuracy Gain:  +{eval_results['improvements']['accuracy_gain']}")
    print(f"Results File:   {eval_path}")
    print("=" * 60)

    return eval_results


if __name__ == "__main__":
    run_evaluation()
