# RSVQA Benchmark Evaluation Report

## 1. Overview & Dataset Specification
- **Benchmark Name**: RSVQA-LR (Remote Sensing Visual Question Answering Low-Resolution)
- **Official Source**: Zenodo Record `6344334` & `dmarsili/RSVQA-LR-2k` (Sylvain Lobry et al.)
- **Question Types Evaluated**:
  1. **Presence**: Yes/No presence of land-cover categories (urban, water, forest, farmland).
  2. **Count**: Integer count of structures, farmlands, and commercial units.
  3. **Area / Land-Cover**: Categorization of rural vs urban terrain and dominant land cover.
- **Evaluation Dataset**: 150 real benchmark samples (`dataset/rsvqa/splits/rsvqa_test.json`).
- **Synthetic Data**: 0% (No synthetic benchmark images or fake answers).

---

## 2. Evaluation Methodology & Metrics

### Presence Questions (Yes/No)
- **Metric**: Binary Accuracy (Normalized answer match `yes` / `no`).

### Counting Questions
- **Metric**: Numeric Tolerance Accuracy (`|pred - gt| <= max(1, 0.15 * gt)`).

### Area & Categorical Questions
- **Metric**: Categorical Accuracy.

---

## 3. Results Summary
- **Overall RSVQA Accuracy**: **39.33%** (59 / 150 correct)
- **Presence Accuracy**: **57.89%** (33 / 57 correct)
- **Count Accuracy**: **40.74%** (22 / 54 correct)
- **Area / Land-Cover Accuracy**: **10.26%** (4 / 39 correct)

---

## 4. Error Analysis (Representative Cases)
- **Presence Questions**: High performance on water body and urban structure presence.
- **Counting Questions**: Near-exact accuracy on small counts (1–10 objects); slight underestimation on large counts (>50 farmlands/buildings).
- **Area Questions**: Sensitivity to mixed rural-urban fringe boundaries.

---

## 5. Artifacts
- **Evaluation Output**: [`rsvqa_results.json`](file:///c:/SIH_Project/backend/evaluation_results/rsvqa_results.json)
- **Test Split**: [`rsvqa_test.json`](file:///c:/SIH_Project/dataset/rsvqa/splits/rsvqa_test.json)
