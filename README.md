# 🛰️ SatQuery AI

**ISRO Smart India Hackathon 2026 · Problem Statement ID: 26167**

> An agentic vision-language assistant for multimodal remote sensing image analysis through natural-language queries.

[![Frontend: Next.js](https://img.shields.io/badge/Frontend-Next.js%2015-black?logo=next.js)](https://nextjs.org)
[![Backend: FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Deploy: Vercel](https://img.shields.io/badge/Deploy-Vercel-black?logo=vercel)](https://vercel.com)
[![Deploy: HF Spaces](https://img.shields.io/badge/Deploy-HF%20Spaces-yellow?logo=huggingface)](https://huggingface.co/spaces)

---

## 🏗️ Multi-Microservice Architecture

```
Frontend (Next.js 15)
       │
       ▼  POST /api/satquery
Orchestrator Backend (FastAPI)
       │
       ├── Grounding Specialist (Grounding DINO + SAM)
       │       ├── Phrase Grounding Bounding Boxes
       │       └── Pixel-level Binary Masks & Overlay Generation
       │
       ├── VLM Specialist (Vision-Language Reasoning Engine)
       │       ├── Spatial Evidence Analysis (Regions, Center Distance)
       │       └── Object Counting, Location & Scene Reasoning
       │
       ├── ChangeFormer Specialist (Bi-temporal Satellite Change Detection)
       │       └── LEVIR-CD Pretrained Transformer Change Detection
       │
       ├── BigEarthNet Specialist (Remote-Sensing Domain Adapter)
       │       └── Sentinel-1 (SAR) + Sentinel-2 (Multispectral) Adapted Vision-Language Model
       │
       └── Optical + SAR Cross-Modal Specialist
               └── Dual-Stream Tensor Fusion & Mutual Spatial Cross-Validation Engine
```

---

## 🌲 BigEarthNet Remote-Sensing Adaptation

SIH Problem Statement 26167 requires adapting at least one visual component using open-source remote-sensing data:

> *"At least one visual or vision-language component must be fine-tuned or otherwise adapted using BigEarthNet.txt or other open-source training data."*

- **Dataset**: BigEarthNet Sentinel-1 SAR (VV/VH radar backscatter) + Sentinel-2 Multispectral (12 bands) patches across 19 CORINE Land Cover classes.
- **Methodology**: Parameter-Efficient Fine-Tuning (PEFT / LoRA) low-rank bottleneck residual adapter mapping Sentinel visual features to land cover instruction VQA space.
- **Training Script**: [`backend/train_bigearthnet_adapter.py`](file:///c:/SIH_Project/backend/train_bigearthnet_adapter.py)
- **Evaluation Script**: [`backend/evaluate_bigearthnet_adapter.py`](file:///c:/SIH_Project/backend/evaluate_bigearthnet_adapter.py)
- **Documentation**: Detailed dataset preparation, architecture, and reproducibility guide in [`docs/bigearthnet_adaptation.md`](file:///c:/SIH_Project/docs/bigearthnet_adaptation.md).

### Base vs. Adapted Model Evaluation Results

| Metric | Base Model (Unadapted) | Adapted Model (BigEarthNet) | Gain / Improvement |
|:---|:---:|:---:|:---:|
| **Held-Out Test BCE Loss** | 0.6917 | **0.1430** | **-0.5487 (Loss Reduction)** |
| **Multi-Label Micro F1** | 0.1905 | **0.6504** | **+0.4599 (F1 Gain)** |
| **Classification Accuracy** | 10.53% | **89.80%** | **+79.27% (Accuracy Gain)** |

---

## 📊 Single-Image Benchmark Evaluation (VRSBench & RSVQA)

SatQuery AI is quantitatively evaluated on official open-source single-image benchmarks specified in SIH Problem Statement 26167:

### 1. VRSBench (Visual Reasoning and Segmentation Benchmark)
- **Source**: `xiang709/VRSBench` (Official Hugging Face Dataset)
- **Evaluated Tasks**: Single-Image VQA, Scene Captioning, Visual Grounding (Grounding DINO + SAM)
- **Single-Image VQA Accuracy**: **38.33%**
- **Scene Captioning Keyword Recall**: **34.25%**
- **Visual Grounding Mean IoU / Accuracy (@IoU0.50)**: **0.4215** / **44.00%**
- **Documentation & Proof**: [`docs/vrsbench_evaluation.md`](file:///c:/SIH_Project/docs/vrsbench_evaluation.md)

### 2. RSVQA (Remote Sensing Visual Question Answering)
- **Source**: Zenodo Record `6344334` / `dmarsili/RSVQA-LR-2k` (Lobry et al.)
- **Evaluated Types**: Presence (Yes/No), Counting (Numeric Tolerance), Area / Land-Cover
- **Overall RSVQA Accuracy**: **39.33%**
  - **Presence (Yes/No)**: **57.89%**
  - **Count Questions**: **40.74%**
  - **Area Questions**: **10.26%**
### 3. CDVQA (Change Detection Visual Question Answering Benchmark)
- **Source**: `ljx620/CDVQA` (Official Hugging Face Dataset, Yuan et al. IEEE TGRS 2022)
- **Evaluated Tasks**: Bi-temporal Change Detection, Directional Increase/Decrease, Land-Cover Transitions, Change Ratio Quantification
- **Overall CDVQA Normalized Accuracy**: **61.33%** (92 / 150 real benchmark pairs)
  - **Largest Change Identification**: **75.00%** (9/12)
  - **Change To What (Transition)**: **69.23%** (9/13)
  - **Increase or Not (Expansion)**: **66.67%** (10/15)
  - **Change or Not (Binary Presence)**: **62.30%** (38/61)
  - **Change Ratio Quantification**: **57.89%** (11/19)
  - **Decrease or Not (Reduction)**: **55.56%** (10/18)
  - **Smallest Change Identification**: **41.67%** (5/12)
- **Documentation & Proof**: [`docs/cdvqa_evaluation.md`](file:///c:/SIH_Project/docs/cdvqa_evaluation.md) and [`docs/cdvqa_validation.md`](file:///c:/SIH_Project/docs/cdvqa_validation.md)

### 4. Cross-Modal Optical + SAR Analysis (BigEarthNet-v2 Sentinel-1 + Sentinel-2)
- **Source**: BigEarthNet-v2 / BEN-GE-800 Benchmark (Charfuelan et al., DLR / TU Berlin)
- **Evaluated Modalities**: Sentinel-2 MSI (B02/B03/B04/B08) + Sentinel-1 SAR (VV/VH C-band backscatter in dB)
- **Tasks**: Cross-Modal Built-Up & Water Surface Identification, Mutual Spatial Cross-Validation, Modality Ablation
- **Ablation Performance**:
  - **Optical-Only Macro F1**: **0.3318** (Built-up F1: 0.0000)
  - **SAR-Only Macro F1**: **0.1935** (Built-up F1: 0.0000)
  - **Optical + SAR Fused Macro F1**: **0.4235** (**+0.0917 gain / +27.64% relative improvement**; Built-up F1: **0.7273**)
- **VQA Semantic Accuracy**:
  - **Water Detection Accuracy**: **74.67%** (56/75)
  - **Built-Up Detection Accuracy**: **69.33%** (52/75)
  - **Joint Optical-SAR Accuracy**: **53.33%** (40/75)
- **Documentation & Proof**: [`docs/optical_sar_analysis.md`](file:///c:/SIH_Project/docs/optical_sar_analysis.md)

### 5. Agentic Orchestration & Specialist Tool Registry (Chunk 11)
- **Architecture**: Autonomous 6-stage pipeline (`QueryInterpreter` $\to$ `InputValidator` $\to$ `AgentPlanner` $\to$ `ToolRegistry` $\to$ `ToolExecutor` $\to$ `EvidenceCombiner`)
- **Intent Disambiguation**: 8 remote sensing intents (`single_image_description`, `single_image_vqa`, `object_grounding`, `bitemporal_change`, `optical_sar_crossmodal`, `optical_specific`, `sar_specific`, `complex_multitool`)
- **Tool Registry**: 7 registered specialist tools (`grounding_specialist`, `satellite_vlm`, `change_analyzer`, `optical_sar_analyzer`, `optical_analyzer`, `sar_analyzer`, `bigearthnet_adapter`)
- **Multi-Tool Sequencing**: Dynamic dependency resolution and context/bounding-box forwarding between specialist steps
- **Explainable Confidence**: Formulaic confidence aggregation with qualitative binning and explicit verifiable basis points
- **Auditable Execution Trace**: 6-step audit log recorded for every transaction
- **Documentation & Proof**: [`docs/agentic_orchestration.md`](file:///c:/SIH_Project/docs/agentic_orchestration.md) and [`docs/chunk11_sih_mapping.md`](file:///c:/SIH_Project/docs/chunk11_sih_mapping.md)

---

## 🔌 Unified API Contract (`POST /api/satquery`)

Supports **Single-Image VQA**, **Bi-Temporal Change Analysis**, and **Cross-Modal Optical + SAR Analysis**:

### Single-Image VQA Request Payload
```json
{
  "image": "<base64_image>",
  "question": "How many red boxes are visible?"
}
```

### Bi-Temporal Change Analysis Request Payload
```json
{
  "image_t1": "<base64_pre_change_image>",
  "image_t2": "<base64_post_change_image>",
  "question": "Did the areas of non-vegetated ground surface change?"
}
```

### Cross-Modal Optical + SAR Request Payload
```json
{
  "image_optical": "<base64_optical_geotiff_or_png>",
  "image_sar": "<base64_sar_geotiff_or_png>",
  "modality": "optical_sar",
  "question": "Use the optical and SAR images together to identify built-up and water-covered regions."
}
```

### Bi-Temporal Response Payload
```json
{
  "status": "success",
  "task": "bi_temporal_change_analysis",
  "inputs": 2,
  "answer": "Yes, bi-temporal analysis indicates significant change in NVG_surface areas. (Overall change: 56.6%).",
  "change_ratio": 56.58,
  "severity": "significant",
  "change_categories": ["NVG_surface_change", "trees_change", "low_vegetation_change", "playgrounds_change"],
  "overlay": "<base64_jet_colormap_heatmap_png>",
  "evidence": ["Bi-temporal change detection analysis completed using ChangeFormerV6 (LEVIR-CD)..."],
  "confidence": 0.92,
  "model": "ChangeFormerV6 (LEVIR-CD)",
  "tools": ["change_analyzer", "vlm_specialist"],
  "image_size": [512, 512],
  "execution_time_ms": 2042
}
```

### Single-Image Response Payload
```json
{
  "status": "success",
  "answer": "There are 2 red structures detected in the image.",
  "detections": [
    {
      "detection_index": 1,
      "label": "red box",
      "score": 0.89,
      "box": [58.3, 58.1, 161.7, 161.6],
      "sam_score": 0.98,
      "mask_area": 9996,
      "status": "segmented"
    }
  ],
  "overlay": "<base64_overlay_png>",
  "image_size": [512, 512],
  "execution_time_ms": 25103
}
```

### Health Check (`GET /health` & `GET /api/satquery/health`)
```json
{
  "status": "healthy",
  "services": {
    "grounding": "available",
    "vlm": "available",
    "changeformer": "available"
  }
}
```

---

## 🌐 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROUNDING_SERVICE_URL` | Hugging Face Space Grounding DINO + SAM endpoint | `https://sanika2006-satquery-grounding.hf.space/run/predict` |
| `VLM_SERVICE_URL` | Hugging Face Space VLM Reasoning endpoint | `https://sanika2006-satquery-vlm.hf.space/run/predict` |
| `HF_CHANGEFORMER_SPACE` | Hugging Face Space ChangeFormer endpoint | `sanika2006-satquery-changeformer` |
| `SUPABASE_URL` | Supabase Database URL | Required for DB logging |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase Service Role Key | Required for server DB ops |

---

## 🚀 Quick Start

### 1. Backend Server

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. BigEarthNet Adaptation & Validation

```bash
cd backend
# Dataset preparation & split partitioning
python prepare_bigearthnet.py

# Execute BigEarthNet domain adaptation training
python train_bigearthnet_adapter.py

# Evaluate Base vs Adapted Model
python evaluate_bigearthnet_adapter.py

# Run unit tests
python -m unittest tests/test_bigearthnet.py
```

### 3. Frontend Client

```bash
cd frontend
npm install
npm run dev
```

---

## 🧪 Verified Microservices

- **Grounding Specialist**: Pretrained Grounding DINO (`IDEA-Research/grounding-dino-tiny`) + Meta AI SAM (`sam_vit_b_01ec64.pth`).
- **VLM Specialist**: Vision-Language Satellite reasoning engine operating over Grounding DINO + SAM spatial evidence.
- **ChangeFormer Specialist**: Official `ChangeFormerV6` model loaded with 492MB `best_ckpt.pt` pretrained LEVIR-CD weights.
- **BigEarthNet Adapter**: PEFT/LoRA adapted vision-language feature encoder fine-tuned on Sentinel-1 SAR + Sentinel-2 multispectral data.
- **Optical + SAR Cross-Modal Specialist**: Physics-guided mutual cross-validation (specular reflection vs double-bounce corner reflection) + dual-stream tensor fusion.
- **Unified Orchestrator**: Multi-microservice coordinator exposing `POST /api/satquery` and `/health`.

---

## 👥 Team & Project

**ISRO Smart India Hackathon 2026 · Problem Statement ID 26167**  
Organisation: Indian Space Research Organisation (ISRO) / Dept. of Space  
Theme: Space Technology · Category: Software
