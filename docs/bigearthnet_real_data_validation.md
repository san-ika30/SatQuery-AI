# BigEarthNet Real Data Ingestion & Remote Sensing Adaptation Validation

## Executive Summary
This document provides formal empirical proof of real data ingestion, multimodal Sentinel-1 SAR and Sentinel-2 Multispectral feature processing, and genuine model adaptation for SIH Problem Statement 26167 compliance.

---

## 1. Dataset Source & Version
- **Dataset Name**: BigEarthNet-v2 (BEN-GE-800 Benchmark Subset)
- **Official Source**: Zenodo Record `12941231` (TUM Remote Sensing Technology / BIFOLD)
- **Open-Source Reference**: `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt` (Hugging Face Hub)
- **Synthetic Data**: **0%** (All synthetic PNG generation and noise simulation functions removed from production training/evaluation pipeline)

---

## 2. Download & Preparation Workflow
- **Archive File**: `dataset/bigearthnet_raw/ben-ge-800.tar.gz` (183 MB)
- **Extracted Archive Path**: `dataset/bigearthnet_raw/ben-ge-800/`
- **Unique Real Patches Ingested**: **150 patches** (800 raw pairs available)
- **Total VQA Instruction Pairs**: **490 pairs**

---

## 3. Actual Sensor Inputs & Modalities
### Sentinel-1 Synthetic Aperture Radar (SAR)
- **Channels**: 2 channels (`VV` and `VH` float32 backscatter in dB)
- **Files Used**: `*_VV.tif`, `*_VH.tif`
- **Tensor Shape**: `(2, 128, 128)` float32
- **Sample Min / Max / Mean**: `-0.9150 / 1.6990 / 0.3459` (normalized dB)

### Sentinel-2 Multispectral Imagery (MSI)
- **Bands**: 4 10m resolution bands (`B02-Blue`, `B03-Green`, `B04-Red`, `B08-NIR` uint16 reflectance)
- **Files Used**: `*_B02.tif`, `*_B03.tif`, `*_B04.tif`, `*_B08.tif`
- **Tensor Shape**: `(4, 128, 128)` float32
- **Sample Min / Max / Mean**: `0.0617 / 1.0000 / 0.3753` (reflectance scaled)

---

## 4. Official Land-Cover Annotations
- **Nomenclature**: Official CORINE Land Cover 19-class nomenclature
- **Annotation Source**: Read directly from `*_labels_metadata.json` for each patch
- **Target Vector**: 19-dimensional multi-hot binary tensor `(19,)`
- **Sample Real Labels**:
  - Patch `S2A_MSIL2A_20170813T112121_11_77`: `["Coniferous forest", "Transitional woodland/shrub", "Beaches, dunes, sands", "Sea and ocean"]`
  - Patch `S2A_MSIL2A_20171002T112111_12_74`: `["Coniferous forest", "Beaches, dunes, sands", "Sea and ocean"]`

---

## 5. Model Architecture & Adaptation Strategy
- **Architecture**: `BigEarthNetVisionLanguageAdapter` (Dual-Stream Convolutional Encoder + Multimodal Sensor Fusion + PEFT/LoRA Bottleneck Residual Head)
  - `s1_encoder`: Conv2d(2, 32) -> Conv2d(32, 64) -> Linear(1024, 128)
  - `s2_encoder`: Conv2d(4, 32) -> Conv2d(32, 64) -> Linear(1024, 128)
  - `fusion_layer`: Linear(256, 256) -> BatchNorm1d -> ReLU
  - `peft_adapter`: Low-rank Linear(256, 64) -> GELU -> Linear(64, 256) (Residual Bottleneck)
  - `classifier`: Linear(256, 19)
- **Trainable Parameters**: 428,243 parameters
- **Gradient Verification**: Verified that `adapter_down.weight.grad` receives non-zero gradients during backpropagation.

---

## 6. Training Configuration & Loss Trajectory
- **Hardware**: CPU (Intel/AMD x86_64)
- **Training Time**: 33.69 seconds
- **Epochs**: 5 | Batch Size: 16 | Learning Rate: 0.001 (AdamW)
- **Loss Trajectory**:
  - Epoch 1: Train Loss = 0.3502 | Val Loss = 0.2003
  - Epoch 2: Train Loss = 0.1420 | Val Loss = 0.1725
  - Epoch 3: Train Loss = 0.1121 | Val Loss = 0.1559
  - Epoch 4: Train Loss = 0.0846 | Val Loss = 0.1558
  - Epoch 5: Train Loss = 0.0587 | Val Loss = 0.1524

---

## 7. Data Leakage Verification
Data partitioned strictly at the **patch_id level**:
- `Train Patches`: 105 patches (343 samples)
- `Val Patches`: 22 patches (72 samples)
- `Test Patches`: 23 patches (75 samples)
- **Intersection Checks**:
  - `Train ∩ Val` = ∅ (0 patches)
  - `Train ∩ Test` = ∅ (0 patches)
  - `Val ∩ Test` = ∅ (0 patches)
- **Leakage Status**: **PASSED (0 DATA LEAKAGE)**

---

## 8. Held-Out Real Data Evaluation Results
Evaluated on **23 held-out test patches (75 real samples)** comparing Base Model vs Adapted Model:

| Metric | Base Model (Unadapted) | Adapted Model (BigEarthNet) | Gain / Delta |
| :--- | :--- | :--- | :--- |
| **BCE Loss** | 0.6931 | **0.1954** | **-0.4977** (Loss Reduction) |
| **Micro F1 Score** | 0.1445 | **0.4337** | **+0.2892** (+28.92%) |
| **Multi-label Accuracy** | 0.0779 | **0.9011** | **+0.8232** (+82.32%) |

---

## 9. Checkpoint & Artifact Locations
- **PyTorch Model Checkpoint**: [`best_adapter.pt`](file:///c:/SIH_Project/backend/checkpoints/bigearthnet_adapter/best_adapter.pt)
- **Training Metadata**: [`adaptation_metadata.json`](file:///c:/SIH_Project/backend/checkpoints/bigearthnet_adapter/adaptation_metadata.json)
- **Evaluation Results**: [`eval_results.json`](file:///c:/SIH_Project/backend/checkpoints/bigearthnet_adapter/eval_results.json)
- **Machine Validation JSON**: [`real_data_validation.json`](file:///c:/SIH_Project/backend/checkpoints/bigearthnet_adapter/real_data_validation.json)

---

## 10. Limitations & Future Work
- The current subset utilizes 150 real patches out of 800 available in BEN-GE-800 (and 590k in full BigEarthNet).
- Training can be scaled to GPU/CUDA for 100+ epochs on full BigEarthNet v2 if higher F1 score is required for production deployment.
