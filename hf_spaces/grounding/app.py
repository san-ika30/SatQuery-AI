"""
SatQuery AI — Grounding & Segmentation Specialist Endpoint
Space ID: Sanika2006/satquery-grounding

API Contract:
  Input Payload:
    {
      "data": [
        "<image_base64>",
        "<prompt_text>"
      ]
    }

  Output Payload:
    {
      "data": [
        "<overlay_base64>",
        "<json_detections_string>"
      ]
    }

Integrated Zero-Shot Phrase Grounding & Fine-Grained Segmentation microservice.
Combines Grounding DINO (phrase detection) with Meta AI SAM (Segment Anything Model)
to return bounding boxes, pixel-level binary masks, confidence scores, and mask overlays.
"""
import io
import os
import sys
import json
import base64
import numpy as np
from PIL import Image, ImageDraw
import gradio as gr

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from segment_anything import sam_model_registry, SamPredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
SAM_CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"

_dino_processor = None
_dino_model = None
_sam_predictor = None
_device = None


def get_device() -> str:
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def get_grounding_dino_model():
    """Load Grounding DINO processor and model lazily."""
    global _dino_processor, _dino_model
    if _dino_model is not None and _dino_processor is not None:
        return _dino_processor, _dino_model

    try:
        _dino_processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
        _dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID)
        _dino_model.to(device=get_device())
        _dino_model.eval()
        return _dino_processor, _dino_model
    except Exception as exc:
        raise RuntimeError(f"Failed to load Grounding DINO ({DINO_MODEL_ID}): {str(exc)}")


def get_sam_predictor() -> SamPredictor:
    """Load SAM model and predictor lazily."""
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor

    if not os.path.exists(SAM_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"SAM checkpoint file missing at '{SAM_CHECKPOINT_PATH}'. "
            "Please ensure sam_vit_b_01ec64.pth is present."
        )

    try:
        device = get_device()
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=device)
        _sam_predictor = SamPredictor(sam)
        return _sam_predictor
    except Exception as exc:
        raise RuntimeError(f"Failed to load SAM model: {str(exc)}")


def decode_base64_image(b64_str: str) -> Image.Image:
    """Decode base64 string (with or without data URI prefix) into a PIL Image."""
    if not b64_str or not isinstance(b64_str, str):
        raise ValueError("Invalid base64 image string.")

    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]

    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def encode_image_base64(img: Image.Image) -> str:
    """Encode PIL Image into a base64 PNG string."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_grounding_sam_pipeline(
    image: Image.Image,
    prompt: str,
    box_threshold: float = 0.20,
    text_threshold: float = 0.20,
) -> dict:
    """
    Full pipeline:
      1. Grounding DINO phrase grounding -> bounding boxes [xmin, ymin, xmax, ymax]
      2. Meta AI SAM -> pixel-level segmentation masks for each box
      3. Rendered segmentation overlay image
    """
    dino_proc, dino_model = get_grounding_dino_model()
    sam_predictor = get_sam_predictor()

    w, h = image.size
    formatted_prompt = prompt.strip()
    if not formatted_prompt.endswith("."):
        formatted_prompt = f"{formatted_prompt}."

    # Step 1: Grounding DINO detection
    inputs = dino_proc(images=image, text=formatted_prompt, return_tensors="pt").to(get_device())

    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_proc.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[(h, w)],
    )[0]

    raw_boxes = results["boxes"].cpu().numpy()
    dino_scores = results["scores"].cpu().numpy()
    raw_labels = results.get("text_labels", results.get("labels", []))

    if len(raw_boxes) == 0:
        return {
            "detections": [],
            "visualization": image,
            "mask_overlay_b64": encode_image_base64(image),
        }

    # Step 2: SAM Segmentation
    img_np = np.array(image.convert("RGB"))
    sam_predictor.set_image(img_np)

    detections = []
    overlay_np = img_np.copy()
    colors = [[255, 0, 0], [0, 150, 255], [255, 255, 0], [0, 255, 0], [255, 0, 255]]

    for idx, (box, score, label) in enumerate(zip(raw_boxes, dino_scores, raw_labels)):
        xmin = max(0.0, min(float(w), float(box[0])))
        ymin = max(0.0, min(float(h), float(box[1])))
        xmax = max(0.0, min(float(w), float(box[2])))
        ymax = max(0.0, min(float(h), float(box[3])))

        if xmax <= xmin or ymax <= ymin:
            continue

        box_np = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

        # SAM mask generation
        masks, sam_scores, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_np[None, :],
            multimask_output=False,
        )

        mask = masks[0]  # (h, w) bool
        sam_score = float(sam_scores[0])
        mask_area = int(np.sum(mask))

        color = colors[idx % len(colors)]
        overlay_np[mask] = (0.5 * overlay_np[mask] + 0.5 * np.array(color)).astype(np.uint8)

        detections.append({
            "detection_index": idx + 1,
            "label": str(label),
            "score": float(round(float(score), 4)),
            "box": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
            "sam_score": float(round(sam_score, 4)),
            "mask_area": mask_area,
            "status": "segmented",
            "image_size": [w, h],
        })

    vis_img = Image.fromarray(overlay_np)
    draw = ImageDraw.Draw(vis_img)

    for det in detections:
        xmin, ymin, xmax, ymax = det["box"]
        draw.rectangle([xmin, ymin, xmax, ymax], outline="yellow", width=3)
        draw.text((xmin + 4, ymin + 4), f"{det['label']} ({det['score']:.2f})", fill="yellow")

    return {
        "detections": detections,
        "visualization": vis_img,
        "mask_overlay_b64": encode_image_base64(vis_img),
    }


def predict(image_b64: str, prompt_text: str) -> tuple[str, str]:
    """
    Gradio API predict endpoint.
    Input Payload Contract:
      {"data": [image_b64, prompt_text]}
    Output Payload Contract:
      {"data": [overlay_b64, json_detections_string]}
    """
    if not image_b64 or not isinstance(image_b64, str):
        err_json = json.dumps({"error": "Base64 image string is empty or invalid."}, indent=2)
        raise gr.Error(err_json)

    if not prompt_text or not isinstance(prompt_text, str) or not prompt_text.strip():
        err_json = json.dumps({"error": "Prompt text cannot be empty."}, indent=2)
        raise gr.Error(err_json)

    try:
        img = decode_base64_image(image_b64)
    except Exception as exc:
        err_json = json.dumps({"error": f"Failed to decode base64 image: {str(exc)}"}, indent=2)
        raise gr.Error(err_json)

    try:
        pipeline_res = run_grounding_sam_pipeline(img, prompt_text.strip())
        overlay_b64 = pipeline_res["mask_overlay_b64"]
        detections_json = json.dumps(pipeline_res["detections"], indent=2)
        return overlay_b64, detections_json
    except Exception as exc:
        err_json = json.dumps({"error": f"Grounding + SAM pipeline execution failed: {str(exc)}"}, indent=2)
        raise gr.Error(err_json)


# Gradio Interface for Grounding Specialist Space
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Base64 Encoded Image", lines=3),
        gr.Textbox(label="Detection Prompt Text", lines=2, placeholder="e.g. red buildings, roads, vehicles"),
    ],
    outputs=[
        gr.Textbox(label="Base64 Overlay Image", lines=3),
        gr.Textbox(label="JSON Detections Payload", lines=6),
    ],
    title="SatQuery AI — Grounding & SAM Specialist API",
    description="Zero-Shot Grounding DINO detection + Meta AI SAM segmentation microservice endpoint.",
    api_name="predict",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
