---
title: SatQuery AI VLM Reasoning Specialist
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: VLM Satellite Image Reasoning Microservice
---

# SatQuery AI — VLM Satellite Reasoning Hugging Face Space

This Hugging Face Space hosts the **VLM Reasoning Specialist** microservice for SatQuery AI (`Sanika2006/satquery-vlm`), interpreting satellite images alongside Grounding DINO detection evidence and SAM segmentation masks to answer complex visual and spatial questions.

## API Endpoint Contract

- **Endpoint**: `POST https://sanika2006-satquery-vlm.hf.space/run/predict`
- **Request Payload**:
  ```json
  {
    "data": [
      "<base64_encoded_image>",
      "<question_text>",
      "<json_formatted_detections_string>"
    ]
  }
  ```

- **Response Payload**:
  ```json
  {
    "data": [
      "<natural_language_answer_string>",
      "<structured_result_json_string>"
    ]
  }
  ```

## Example Usage
```python
import httpx
import json

url = "https://sanika2006-satquery-vlm.hf.space/run/predict"
payload = {
    "data": [
        image_base64_string,
        "How many red boxes are visible?",
        json_detections_string
    ]
}

response = httpx.post(url, json=payload, timeout=60.0)
data = response.json()["data"]

answer = data[0]
structured_result = json.loads(data[1])
```
