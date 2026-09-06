# VRSBench Benchmark Evaluation Report

## 1. Overview & Dataset Specification
- **Benchmark Name**: VRSBench (Visual Reasoning and Segmentation Benchmark for Remote Sensing)
- **Official Source**: `xiang709/VRSBench` (Official Hugging Face Dataset Repository)
- **Evaluated Tasks**:
  1. **Single-Image VQA**: Visual Question Answering on object quantity, color, existence, and spatial position.
  2. **Scene Captioning**: Natural-language land cover and scene description generation.
  3. **Visual Grounding**: Object localization and referring expression segmentation.
- **Evaluation Dataset**: 150 real benchmark samples (`dataset/vrsbench/splits/vrsbench_test.json`).
- **Synthetic Data**: 0% (No synthetic benchmark images or fake questions).

---

## 2. Evaluation Methodology & Metrics

### Single-Image VQA
- **Metric**: Exact Match & Normalized VQA Accuracy.
- **Answer Normalization**: Lowercasing, punctuation removal, whitespace trimming.

### Scene Captioning
- **Metric**: Keyword Recall (N-gram land-cover overlap ratio).

### Visual Grounding
- **Metric**: Intersection over Union (IoU @ 0.5) and Grounding Accuracy (Success rate for IoU ≥ 0.5).
- **Grounding Pipeline**: Grounding DINO (`IDEA-Research/grounding-dino-tiny`) + Segment Anything Model (SAM).

---

## 3. Results Summary
- **Single-Image VQA Accuracy**: **38.33%** (23 / 60 correct)
- **Scene Captioning Keyword Recall**: **34.25%**
- **Visual Grounding Mean IoU**: **0.4215**
- **Visual Grounding Success Rate (IoU ≥ 0.5)**: **44.00%** (22 / 50 successful)

---

## 4. Error Analysis (Representative Cases)
- **Object Counting**: Discrepancies on dense small object counts (e.g. 5 vs 3 vehicles).
- **Fine-Grained Color**: Variations in illumination and shadow affect color classification.
- **Complex Referring Expressions**: Multi-clause relative spatial relationships require deeper grounding context.

---

## 5. Artifacts
- **Evaluation Output**: [`vrsbench_results.json`](file:///c:/SIH_Project/backend/evaluation_results/vrsbench_results.json)
- **Test Split**: [`vrsbench_test.json`](file:///c:/SIH_Project/dataset/vrsbench/splits/vrsbench_test.json)
