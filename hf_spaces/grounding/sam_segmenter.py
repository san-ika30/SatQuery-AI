"""
SatQuery AI — Segment Anything Model (SAM) Module
Hosts the Meta AI SAM (Segment Anything) inference pipeline for spatial prompt segmentation.
Accepts original image and Grounding DINO bounding box coordinates [xmin, ymin, xmax, ymax]
to generate high-precision pixel-level binary segmentation masks.
"""
import os
import sys
import numpy as np
from PIL import Image
import torch

from segment_anything import sam_model_registry, SamPredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"

_sam_model = None
_sam_predictor = None
_sam_device = None


def get_sam_predictor() -> tuple[SamPredictor, str]:
    """Lazy initialization of SAM model and predictor."""
    global _sam_model, _sam_predictor, _sam_device

    if _sam_predictor is not None:
        return _sam_predictor, _sam_device

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"SAM checkpoint file missing at '{CHECKPOINT_PATH}'. "
            "Please ensure sam_vit_b_01ec64.pth is present."
        )

    _sam_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM] Loading SAM {MODEL_TYPE} model on device: {_sam_device}...")

    _sam_model = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    _sam_model.to(device=_sam_device)
    _sam_predictor = SamPredictor(_sam_model)

    print("[SAM] SAM Predictor successfully loaded!")
    return _sam_predictor, _sam_device


def segment_box_with_sam(
    image: Image.Image,
    box: list[float],
) -> dict:
    """
    Segment a single target object inside the given bounding box [xmin, ymin, xmax, ymax].
    Returns dict containing:
      - 'mask': boolean numpy array of shape (height, width) aligned with original image
      - 'mask_area': integer total mask pixel count
      - 'score': float SAM mask quality prediction score
      - 'box': input bounding box [xmin, ymin, xmax, ymax]
    """
    predictor, _ = get_sam_predictor()

    w, h = image.size
    img_np = np.array(image.convert("RGB"))

    # Encode image features once for SAM predictor
    predictor.set_image(img_np)

    box_np = np.array([float(box[0]), float(box[1]), float(box[2]), float(box[3])], dtype=np.float32)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box_np[None, :],
        multimask_output=False,
    )

    best_mask = masks[0]  # Shape: (h, w), dtype bool
    best_score = float(scores[0])
    mask_area = int(np.sum(best_mask))

    return {
        "mask": best_mask,
        "mask_area": mask_area,
        "score": best_score,
        "box": [round(float(b), 2) for b in box],
        "image_size": (w, h),
    }


def segment_multiple_boxes_with_sam(
    image: Image.Image,
    boxes: list[list[float]],
) -> list[dict]:
    """
    Segment multiple detected bounding boxes on the same image in a batch.
    Reuses cached image embeddings for optimal performance.
    """
    if not boxes:
        return []

    predictor, _ = get_sam_predictor()

    w, h = image.size
    img_np = np.array(image.convert("RGB"))

    # Set image embeddings ONCE for all bounding boxes in this image
    predictor.set_image(img_np)

    results = []
    for b in boxes:
        box_np = np.array([float(b[0]), float(b[1]), float(b[2]), float(b[3])], dtype=np.float32)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_np[None, :],
            multimask_output=False,
        )
        best_mask = masks[0]
        results.append({
            "mask": best_mask,
            "mask_area": int(np.sum(best_mask)),
            "score": float(scores[0]),
            "box": [round(float(coord), 2) for coord in b],
            "image_size": (w, h),
        })

    return results
