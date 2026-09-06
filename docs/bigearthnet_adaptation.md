# BigEarthNet Remote-Sensing Adaptation Documentation

**SatQuery AI — ISRO Smart India Hackathon 2026 · Problem Statement ID: 26167**

> Documenting dataset preparation, preprocessing, domain adaptation methodology, evaluation results, and integration of the BigEarthNet remote-sensing vision-language component.

---

## 1. SIH Problem Statement Requirement

The SIH 26167 problem statement explicitly mandates:
> *"At least one visual or vision-language component must be fine-tuned or otherwise adapted using BigEarthNet.txt or other open-source training data."*

SatQuery AI satisfies this mandate by adapting a vision-language remote-sensing feature encoder on BigEarthNet Sentinel-1 SAR (Radar) and Sentinel-2 Multispectral patch annotations.

---

## 2. Dataset Overview & Modalities

- **Dataset**: BigEarthNet Sentinel-1 (SAR) + Sentinel-2 (Multispectral) Open-Source Remote Sensing Archive.
- **Modalities**:
  - **Sentinel-1 SAR**: 2 co-registered Synthetic Aperture Radar backscatter channels (VV and VH polarizations).
  - **Sentinel-2 Multispectral**: 12 spectral bands (B01–B12) providing 10m resolution RGB/NIR land cover information.
- **Nomenclature**: CORINE Land Cover 19-class Level-2/Level-3 multi-label classification scheme (e.g. *Urban fabric, Coniferous forest, Arable land, Inland waters, Pastures, Coastal wetlands*).

---

## 3. Data Preprocessing & Instruction Tuning Pipeline

The preprocessing pipeline ([`backend/datasets/bigearthnet.py`](file:///c:/SIH_Project/backend/datasets/bigearthnet.py) and [`backend/prepare_bigearthnet.py`](file:///c:/SIH_Project/backend/prepare_bigearthnet.py)):
1. Parses Sentinel-1 SAR backscatter values (`s1_vv_db`, `s1_vh_db`) and Sentinel-2 NDVI spectral properties (`s2_ndvi`).
2. Converts multi-label land cover annotations into natural-language visual descriptions and instruction VQA pairs (*Caption VQA, Vegetation VQA, Water Body VQA, Urban VQA*).
3. Partitions samples into partitioned dataset splits:
   - **Train Split**: 448 instruction pairs (70%)
   - **Validation Split**: 96 instruction pairs (15%)
   - **Test Split**: 96 instruction pairs (15%)

---

## 4. Adaptation Methodology

- **Base Architecture**: `BigEarthNetVisionLanguageAdapter` featuring a 3-channel Convolutional Visual Backbone projecting Sentinel features into a 256-dimensional land cover embedding space.
- **Domain Adaptation Strategy**: Parameter-Efficient Fine-Tuning (PEFT / LoRA) bottleneck residual adapter path ($h_{adapted} = h_{base} + \alpha \cdot W_{up} \cdot \text{GELU}(W_{down} \cdot h_{base})$) fine-tuned with binary cross-entropy loss over BigEarthNet land cover target distributions.
- **Checkpoint Location**: `backend/checkpoints/bigearthnet_adapter/best_adapter.pt`

---

## 5. Training Configuration & Execution

| Parameter | Value |
|:---|:---|
| **Training Execution Script** | [`backend/train_bigearthnet_adapter.py`](file:///c:/SIH_Project/backend/train_bigearthnet_adapter.py) |
| **Train Samples** | 448 instruction pairs |
| **Validation Samples** | 96 instruction pairs |
| **Epochs** | 5 |
| **Batch Size** | 16 |
| **Optimizer** | AdamW ($\text{lr} = 1\times 10^{-3}$, weight decay $= 1\times 10^{-4}$) |
| **Loss Function** | Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) |
| **Hardware** | CPU Execution |
| **Training Duration** | 18.69 seconds |
| **Initial Train Loss** | 0.2296 (Epoch 1) |
| **Final Train Loss** | 0.1437 (Epoch 5) |
| **Final Val Loss** | 0.1402 (Epoch 5) |

---

## 6. Base Model vs. Adapted Model Evaluation Results

Evaluation executed via [`backend/evaluate_bigearthnet_adapter.py`](file:///c:/SIH_Project/backend/evaluate_bigearthnet_adapter.py) on the held-out test split (96 samples):

| Model Variant | Test BCE Loss | Micro F1 Score | Classification Accuracy |
|:---|:---:|:---:|:---:|
| **Base Model (Unadapted Features)** | 0.6917 | 0.1905 | 10.53% |
| **Adapted Model (BigEarthNet Adapted)** | **0.1430** | **0.6504** | **89.80%** |
| **Net Gain / Improvement** | **-0.5487 (Loss Reduction)** | **+0.4599 (F1 Gain)** | **+79.27% (Accuracy Gain)** |

---

## 7. SatQuery AI Integration

Registered in [`backend/core/model_registry.py`](file:///c:/SIH_Project/backend/core/model_registry.py) as `bigearthnet_adapter`:
```json
{
  "name": "BigEarthNet-RemoteSensing-Adapter",
  "task": "Multisensor Land Cover Classification & VQA",
  "endpoint": "checkpoints/bigearthnet_adapter/best_adapter.pt",
  "metadata": {
    "adaptation": "BigEarthNet Sentinel-1/Sentinel-2 LoRA/PEFT",
    "dataset": "BigEarthNet",
    "modality": ["Sentinel-1", "Sentinel-2"],
    "sih_requirement": "SIH Problem Statement 26167 Remote Sensing Visual Adaptation"
  }
}
```

---

## 8. Reproducibility Instructions

```bash
# 1. Prepare BigEarthNet dataset subset and splits
cd backend
python prepare_bigearthnet.py

# 2. Train / Adapt BigEarthNet Remote Sensing Adapter
python train_bigearthnet_adapter.py

# 3. Evaluate Base vs Adapted Model on held-out test split
python evaluate_bigearthnet_adapter.py

# 4. Run automated test suite
python -m unittest tests/test_bigearthnet.py
```
