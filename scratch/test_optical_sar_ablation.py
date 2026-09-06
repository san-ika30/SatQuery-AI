import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend"))

from datasets.bigearthnet import BigEarthNetDataset, BIGEARTHNET_19_CLASSES
from train_bigearthnet_adapter import BigEarthNetVisionLanguageAdapter

recs = json.load(open("dataset/bigearthnet/splits/test.json"))
ds = BigEarthNetDataset(recs)

model = BigEarthNetVisionLanguageAdapter()
ckpt = torch.load("backend/checkpoints/bigearthnet_adapter/best_adapter.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# We evaluate:
# 1. Optical only (s1 zeroed out)
# 2. SAR only (s2 zeroed out)
# 3. Optical + SAR fused (both active)

def evaluate_modality(mode="fused", threshold=0.30):
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            s1 = item["s1_tensor"].unsqueeze(0)
            s2 = item["s2_tensor"].unsqueeze(0)
            target = item["target"].numpy()
            
            if mode == "optical_only":
                logits, _ = model(torch.zeros_like(s1), s2)
            elif mode == "sar_only":
                logits, _ = model(s1, torch.zeros_like(s2))
            else: # fused
                logits, _ = model(s1, s2)
                
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            preds = (probs >= threshold).astype(np.float32)
            if preds.sum() == 0:
                preds[np.argmax(probs)] = 1.0
                
            all_preds.append(preds)
            all_targets.append(target)
            
    P = np.stack(all_preds)
    T = np.stack(all_targets)
    
    tp = (P * T).sum()
    fp = (P * (1 - T)).sum()
    fn = ((1 - P) * T).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = ((P == T).mean()) * 100.0
    
    # Also evaluate built-up and water specifically
    urban_indices = [i for i, c in enumerate(BIGEARTHNET_19_CLASSES) if "urban" in c.lower() or "industrial" in c.lower()]
    water_indices = [i for i, c in enumerate(BIGEARTHNET_19_CLASSES) if "water" in c.lower() or "wetland" in c.lower()]
    
    def cat_metric(indices):
        p_sub = P[:, indices].max(axis=1)
        t_sub = T[:, indices].max(axis=1)
        tp_c = (p_sub * t_sub).sum()
        fp_c = (p_sub * (1 - t_sub)).sum()
        fn_c = ((1 - p_sub) * t_sub).sum()
        pr_c = tp_c / (tp_c + fp_c + 1e-8)
        re_c = tp_c / (tp_c + fn_c + 1e-8)
        f1_c = 2 * pr_c * re_c / (pr_c + re_c + 1e-8)
        acc_c = (p_sub == t_sub).mean() * 100.0
        return {"precision": round(float(pr_c), 4), "recall": round(float(re_c), 4), "f1": round(float(f1_c), 4), "acc": round(float(acc_c), 2)}

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "accuracy": round(float(acc), 2),
        "built_up": cat_metric(urban_indices),
        "water": cat_metric(water_indices),
    }

print("OPTICAL ONLY:", evaluate_modality("optical_only"))
print("SAR ONLY:    ", evaluate_modality("sar_only"))
print("FUSED:       ", evaluate_modality("fused"))
