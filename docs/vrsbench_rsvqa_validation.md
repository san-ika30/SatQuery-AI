# VRSBench & RSVQA Real Data Authenticity & Evaluation Validation

## Executive Summary
This document provides formal empirical proof of real dataset ingestion, zero synthetic benchmark data usage, and single-image evaluation for SIH Problem Statement 26167 compliance across **VRSBench** and **RSVQA**.

---

## 1. VRSBench Dataset Authenticity
- **Dataset Name**: VRSBench (Visual Reasoning and Segmentation Benchmark for Remote Sensing)
- **Official Source**: `xiang709/VRSBench` (Official Hugging Face Repository by Xiang et al.)
- **Sample Count**: 150 real benchmark evaluation samples
- **Tasks Evaluated**:
  1. **Single-Image VQA**: 60 real question-answer pairs
  2. **Scene Captioning**: 40 real natural-language scene descriptions
  3. **Visual Grounding**: 50 real object referring expressions & ground-truth bounding boxes
- **Real Image Evidence**: Extracted from official `VRSBench_EVAL_vqa.json`, `VRSBench_EVAL_Cap.json`, `VRSBench_EVAL_referring.json` annotations and paired with real satellite patch imagery.
- **Synthetic Benchmark Data Used**: **NO**

---

## 2. RSVQA Dataset Authenticity
- **Dataset Name**: RSVQA-LR (Remote Sensing Visual Question Answering Benchmark - Low Resolution)
- **Official Source**: Zenodo Record `6344334` & `dmarsili/RSVQA-LR-2k` (Sylvain Lobry et al.)
- **Sample Count**: 150 real benchmark evaluation samples
- **Question Types Evaluated**:
  1. **Presence**: 57 questions (Yes/No land cover presence)
  2. **Counting**: 54 questions (Integer object counting)
  3. **Area / Land Cover**: 39 questions (Urban/Rural/Land-use categorization)
- **Real Image Evidence**: Extracted from `data/validation-00000-of-00001.parquet` (Zenodo Record `6344334`).
- **Synthetic Benchmark Data Used**: **NO**

---

## 3. Metric Summary Table

| Benchmark | Task / Question Type | Evaluated Samples | Key Metric | Metric Result |
| :--- | :--- | :--- | :--- | :--- |
| **RSVQA** | Presence (Yes/No) | 57 | Accuracy | **57.89%** |
| **RSVQA** | Counting | 54 | Numeric Accuracy (Tolerance 15%) | **40.74%** |
| **RSVQA** | Area / Land Cover | 39 | Categorical Accuracy | **10.26%** |
| **RSVQA** | **Overall** | **150** | **Overall VQA Accuracy** | **39.33%** |
| **VRSBench** | Single-Image VQA | 60 | VQA Accuracy | **38.33%** |
| **VRSBench** | Scene Captioning | 40 | Keyword Recall | **34.25%** |
| **VRSBench** | Visual Grounding | 50 | Mean IoU @ 0.50 | **0.4215** |
| **VRSBench** | Visual Grounding | 50 | Grounding Success Rate | **44.00%** |

---

## 4. Traceable Machine-Readable Artifacts
- **VRSBench Results JSON**: [`vrsbench_results.json`](file:///c:/SIH_Project/backend/evaluation_results/vrsbench_results.json)
- **RSVQA Results JSON**: [`rsvqa_results.json`](file:///c:/SIH_Project/backend/evaluation_results/rsvqa_results.json)
- **VRSBench Test Split**: [`vrsbench_test.json`](file:///c:/SIH_Project/dataset/vrsbench/splits/vrsbench_test.json)
- **RSVQA Test Split**: [`rsvqa_test.json`](file:///c:/SIH_Project/dataset/rsvqa/splits/rsvqa_test.json)
