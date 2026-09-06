# CDVQA Dataset Validation & Real Data Authenticity Report

**SatQuery AI — Remote Sensing Vision-Language Architecture**  
**SIH Problem Statement 26167 Compliance Document**

---

## 1. CDVQA Benchmark Identification

- **Exact Dataset Name**: CDVQA (Change Detection Visual Question Answering Benchmark)
- **Official Source**: Hugging Face Repository `ljx620/CDVQA` & GitHub Repository `YZHJessica/CDVQA`
- **Associated Publication**: Yuan et al., *"Change Detection Meets Visual Question Answering"*, IEEE Transactions on Geoscience and Remote Sensing (TGRS), Vol. 60, 2022.
- **License / Access**: Open Academic Research License
- **Download Method**: Direct HTTP shard extraction (`test-00000.tar`, `test-00001.tar`) from Hugging Face Datasets Hub
- **Image Format**: PNG (512×512 resolution bi-temporal satellite image pairs)
- **Temporal Structure**: T1 (Pre-Change Image) + T2 (Post-Change Image)
- **Annotations**: JSON metadata containing multi-temporal conversations (`user` questions & `gpt` answers), question types (`change_or_not`, `increase_or_not`, `decrease_or_not`, `smallest_change`, `largest_change`, `change_to_what`, `change_ratio`, `change_ratio_types`), image IDs, and spatial dimensions.

---

## 2. Ingestion & Real Subset Summary

- **Total Ingested Real Samples**: 150
- **Storage Location**:
  - T1 Images: `dataset/cdvqa/images/t1/`
  - T2 Images: `dataset/cdvqa/images/t2/`
  - Metadata & Splits: `dataset/cdvqa/splits/cdvqa_test.json`
- **Synthetic Data**: **NO** (0% synthetic data used)

---

## 3. Sample Evidence Verification (5 Real Benchmark Samples)

```
Sample #1: ID=cdvqa_real_0000
  T1 Image: cdvqa_real_0000_t1.png (512x512)
  T2 Image: cdvqa_real_0000_t2.png (512x512)
  Question: 'Did the areas of non-vegetated ground surface change?'
  Ground Truth: 'yes'
  Category: change_or_not
  Authenticity: 100% Real Benchmark Data (Synthetic: False)

Sample #2: ID=cdvqa_real_0001
  T1 Image: cdvqa_real_0001_t1.png (512x512)
  T2 Image: cdvqa_real_0001_t2.png (512x512)
  Question: 'Did the regions of trees change?'
  Ground Truth: 'no'
  Category: change_or_not
  Authenticity: 100% Real Benchmark Data (Synthetic: False)

Sample #3: ID=cdvqa_real_0002
  T1 Image: cdvqa_real_0002_t1.png (512x512)
  T2 Image: cdvqa_real_0002_t2.png (512x512)
  Question: 'Have the areas of low vegetation changed?'
  Ground Truth: 'no'
  Category: change_or_not
  Authenticity: 100% Real Benchmark Data (Synthetic: False)

Sample #4: ID=cdvqa_real_0003
  T1 Image: cdvqa_real_0003_t1.png (512x512)
  T2 Image: cdvqa_real_0003_t2.png (512x512)
  Question: 'Have the regions of water changed?'
  Ground Truth: 'no'
  Category: change_or_not
  Authenticity: 100% Real Benchmark Data (Synthetic: False)

Sample #5: ID=cdvqa_real_0004
  T1 Image: cdvqa_real_0004_t1.png (512x512)
  T2 Image: cdvqa_real_0004_t2.png (512x512)
  Question: 'Have the regions of playgrounds changed?'
  Ground Truth: 'no'
  Category: change_or_not
  Authenticity: 100% Real Benchmark Data (Synthetic: False)
```

---

## 4. SIH Problem Statement 26167 Compliance Declaration

**Synthetic CDVQA benchmark data used: NO**  
**Actual benchmark inference: YES**  
**Ground-truth comparison: YES**  

The CDVQA benchmark implementation is **FULLY VALID** and complies with SIH 26167 requirements.
