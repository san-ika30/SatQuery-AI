---
title: SatQuery AI Grounding Specialist
emoji: 🎯
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Grounding DINO + Meta SAM Zero-Shot Detection and Segmentation Microservice
---

# SatQuery AI — Grounding & SAM Specialist Hugging Face Space

This Hugging Face Space hosts the **Grounding Specialist** microservice for SatQuery AI (`Sanika2006/satquery-grounding`), combining **Grounding DINO** (phrase detection) with **Meta AI SAM** (Segment Anything Model) for zero-shot satellite phrase grounding and fine-grained segmentation.

## API Endpoint Contract

- **Endpoint**: `POST https://sanika2006-satquery-grounding.hf.space/run/predict`
- **Request Payload**:
  ```json
  {
    "data": [
      "<base64_encoded_image>",
      "<natural_language_prompt>"
    ]
  }
  ```

- **Response Payload**:
  ```json
  {
    "data": [
      "<base64_encoded_overlay_png>",
      "<json_formatted_detections_string>"
    ]
  }
  ```

## Example JSON Detections Payload
```json
[
  {
    "detection_index": 1,
    "label": "red box",
    "score": 0.779,
    "box": [58.61, 58.36, 161.14, 161.27],
    "sam_score": 0.9938,
    "mask_area": 9996,
    "status": "segmented",
    "image_size": [512, 512]
  }
]
```

## How SatQuery AI Backend Calls This Endpoint
```python
import httpx
import base64

url = "https://sanika2006-satquery-grounding.hf.space/run/predict"
payload = {
    "data": [
        image_base64_string,
        "red buildings"
    ]
}

response = httpx.post(url, json=payload, timeout=60.0)
data = response.json()["data"]

overlay_b64 = data[0]
detections = json.loads(data[1])
```
