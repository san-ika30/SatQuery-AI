# Bi-Temporal Change Analysis Architecture

**SatQuery AI — Remote Sensing Vision-Language Architecture**

---

## 1. Overview

Bi-temporal Change Analysis in SatQuery AI provides visual change detection, spatial change heatmaps, and natural-language change-based Visual Question Answering (Change-VQA) by comparing satellite images captured at Time **T1** (pre-change) and Time **T2** (post-change).

Satisfies **SIH Problem Statement 26167**:
> "Bi-temporal change description or change-based VQA; spatial change map optional where masks exist."

---

## 2. System Architecture & Request Flow

```
User (T1 Image + T2 Image + Question)
         │
         ▼
  FastAPI Router (/api/satquery)
         │
         ▼
Unified Orchestrator (core/orchestrator.py)
         │
         ├───► 1. Preprocessing & Alignment (Resizing, Normalization)
         │
         ├───► 2. ChangeAnalyzer (core/change_analyzer.py)
         │        ├── ChangeFormerV6 Model Inference (LEVIR-CD Checkpoint)
         │        ├── Binary & Colormapped JET Change Map
         │        ├── Change Area Ratio & Severity Calculation
         │        └── Semantic Change Category Extraction
         │
         └───► 3. VLM Reasoning Specialist (hf_spaces/vlm)
                  ├── Fuses Change Map Evidence + T2 Image + User Question
                  └── Generates Evidence-Grounded Natural-Language Answer
```

---

## 3. Supported Change Analysis Features

1. **Appearance / Disappearance**: Detection of new buildings, roads, or cleared vegetation.
2. **Increase / Decrease**: Quantitative direction of land-cover shift (built-up expansion, deforestation, water reduction).
3. **Land-Cover Transformation**: Categorization of pre-change class to post-change class.
4. **Spatial Change Map**: Pixel-level visual overlay highlighting changed clusters in JET colormap.
5. **Change Area Ratio & Severity**: Exact percentage of changed ground surface and classification into (`significant`, `moderate`, `minor`, `no_change`).
