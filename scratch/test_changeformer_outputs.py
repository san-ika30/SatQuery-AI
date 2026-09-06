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
t1_img, t2_img = dataset.load_images(0)

analyzer = get_change_analyzer()
model = analyzer._load_model()

t1_resized = t1_img.resize((256, 256), Image.Resampling.BILINEAR)
t2_resized = t2_img.resize((256, 256), Image.Resampling.BILINEAR)

# Try different normalizations
norm1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

norm2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

t1_t = norm1(t1_resized).unsqueeze(0)
t2_t = norm1(t2_resized).unsqueeze(0)

with torch.no_grad():
    out = model(t1_t, t2_t)
    if isinstance(out, (tuple, list)):
        out = out[0]
    out_256 = F.interpolate(out, size=(256, 256), mode="bilinear", align_corners=True)
    prob = torch.softmax(out_256, dim=1).squeeze(0).cpu().numpy()
    arg = torch.argmax(out_256, dim=1).squeeze(0).cpu().numpy()

print("Output shape:", out.shape)
print("Prob min/max channel 0 (no change):", prob[0].min(), prob[0].max(), prob[0].mean())
print("Prob min/max channel 1 (change):", prob[1].min(), prob[1].max(), prob[1].mean())
print("Argmax sum:", arg.sum())

t1_t2 = norm2(t1_resized).unsqueeze(0)
t2_t2 = norm2(t2_resized).unsqueeze(0)
with torch.no_grad():
    out2 = model(t1_t2, t2_t2)
    if isinstance(out2, (tuple, list)):
        out2 = out2[0]
    out_256_2 = F.interpolate(out2, size=(256, 256), mode="bilinear", align_corners=True)
    prob2 = torch.softmax(out_256_2, dim=1).squeeze(0).cpu().numpy()
    arg2 = torch.argmax(out_256_2, dim=1).squeeze(0).cpu().numpy()

print("Norm2 Prob channel 1 mean:", prob2[1].mean(), "Argmax sum:", arg2.sum())
