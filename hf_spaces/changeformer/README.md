---
title: SatQuery AI ChangeFormer
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: SatQuery AI Bi-temporal Satellite Change Detection Endpoint
---

# SatQuery AI — ChangeFormer Hugging Face Space

This Hugging Face Space hosts the **ChangeFormer** bi-temporal change detection service for SatQuery AI (`Sanika2006/satquery-changeformer`).

## API Endpoint Contract

- **URL**: `POST https://sanika2006-satquery-changeformer.hf.space/run/predict`
- **Request Payload**:
  ```json
  {
    "data": [
      "<base64_encoded_png_image_1>",
      "<base64_encoded_png_image_2>"
    ]
  }
  ```
- **Response Payload**:
  ```json
  {
    "data": [
      "<base64_encoded_png_change_heatmap>"
    ]
  }
  ```

## Model Architecture
Based on ChangeFormer (A Transformer-based Architecture for Building/Land Change Detection on Remote Sensing Images, Bandara et al., `wgcban/ChangeFormer`).
