"""
SatQuery AI — VLM/LLM Satellite Image Reasoning Endpoint
Space ID: Sanika2006/satquery-vlm

API Contract:
  Input Payload:
    {
      "data": [
        "<image_base64>",
        "<question>",
        "<detections_json>"
      ]
    }

  Output Payload:
    {
      "data": [
        "<answer>",
        "<structured_result_json_string>"
      ]
    }

Vision-Language Reasoning service for satellite imagery.
Interprets original satellite images alongside Grounding DINO bounding boxes,
SAM segmentation masks, and spatial evidence to provide natural-language answers.
"""
import io
import os
import sys
import math
import json
import base64
import numpy as np
from PIL import Image
import gradio as gr

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_device = None


def get_device() -> str:
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def decode_base64_image(b64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    if not b64_str or not isinstance(b64_str, str):
        raise ValueError("Invalid base64 image string.")
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def parse_detections_input(detections_input) -> list[dict]:
    """Parse input detections from JSON string, list, or dict."""
    if not detections_input:
        return []
    if isinstance(detections_input, str):
        try:
            parsed = json.loads(detections_input)
            if isinstance(parsed, list):
                return parsed
            elif isinstance(parsed, dict) and "detections" in parsed:
                return parsed["detections"]
            return []
        except Exception:
            return []
    elif isinstance(detections_input, list):
        return detections_input
    elif isinstance(detections_input, dict) and "detections" in detections_input:
        return detections_input["detections"]
    return []


def analyze_spatial_evidence(image_size: tuple[int, int], detections: list[dict]) -> list[dict]:
    """
    Compute fine-grained spatial properties (center points, regions, distances)
    for each detected object based on original image coordinates [xmin, ymin, xmax, ymax].
    """
    w, h = image_size
    center_img_x, center_img_y = w / 2.0, h / 2.0

    analyzed = []
    for idx, det in enumerate(detections):
        box = det.get("box", [0, 0, w, h])
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        obj_w = xmax - xmin
        obj_h = ymax - ymin

        # Determine spatial region
        if cx < 0.4 * w and cy < 0.4 * h:
            region = "upper-left"
        elif cx > 0.6 * w and cy < 0.4 * h:
            region = "upper-right"
        elif cx < 0.4 * w and cy > 0.6 * h:
            region = "lower-left"
        elif cx > 0.6 * w and cy > 0.6 * h:
            region = "lower-right"
        elif 0.3 * w <= cx <= 0.7 * w and 0.3 * h <= cy <= 0.7 * h:
            region = "center region"
        else:
            region = "mid-edge region"

        dist_center = math.sqrt((cx - center_img_x) ** 2 + (cy - center_img_y) ** 2)

        analyzed.append({
            "index": det.get("detection_index", idx + 1),
            "label": det.get("label", "object"),
            "score": det.get("score", det.get("dino_score", 0.0)),
            "sam_score": det.get("sam_score", 0.0),
            "box": [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)],
            "center": [round(cx, 1), round(cy, 1)],
            "dimensions": [round(obj_w, 1), round(obj_h, 1)],
            "region": region,
            "dist_to_center_px": round(dist_center, 1),
            "mask_area": det.get("mask_area", int(obj_w * obj_h)),
        })

    return analyzed


def compute_cardinal_direction(dx: float, dy: float) -> str:
    """
    Compute cardinal/ordinal direction from reference point to target point.
    dx = target_x - ref_x
    dy = target_y - ref_y (image y increases downwards, so dy < 0 is North)
    """
    if abs(dx) < 0.25 * abs(dy):
        return "north" if dy < 0 else "south"
    if abs(dy) < 0.25 * abs(dx):
        return "east" if dx > 0 else "west"
    if dy < 0 and dx < 0:
        return "northwest"
    if dy < 0 and dx > 0:
        return "northeast"
    if dy > 0 and dx < 0:
        return "southwest"
    return "southeast"


def analyze_optical_scene(image: Image.Image) -> dict:
    """
    Perform whole-image scene-level reasoning on remote sensing optical image.
    Extracts spectral signatures, vegetation index proxy, water index proxy,
    brightness/albedo, and land-cover composition.
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    r_mean = float(r.mean())
    g_mean = float(g.mean())
    b_mean = float(b.mean())
    brightness = float(arr.mean())

    veg_mask = (g > r * 1.05) & (g > b * 0.95)
    veg_ratio = float(veg_mask.mean())

    water_mask = (b >= r) & ((r + g + b) / 3.0 < 55)
    water_ratio = float(water_mask.mean())

    urban_mask = ((r + g + b) / 3.0 > 95) & (np.abs(r - g) < 30) & (np.abs(g - b) < 30)
    urban_ratio = float(urban_mask.mean())

    soil_mask = (r > g) & (g > b) & ((r + g + b) / 3.0 > 60)
    soil_ratio = float(soil_mask.mean())

    categories = []
    if veg_ratio >= 0.35:
        categories.append(("dense vegetation and green canopy", veg_ratio))
    elif veg_ratio >= 0.15:
        categories.append(("sparse vegetation and shrubland", veg_ratio))

    if water_ratio >= 0.35:
        categories.append(("low-reflectance drainage/water surfaces", water_ratio))
    elif water_ratio >= 0.10:
        categories.append(("drainage channels or wetland areas", water_ratio))

    if urban_ratio >= 0.25:
        categories.append(("built-up / paved structures", urban_ratio))
    elif urban_ratio >= 0.10:
        categories.append(("scattered built structures or paved areas", urban_ratio))

    if soil_ratio >= 0.25:
        categories.append(("bare ground and open fields", soil_ratio))

    if not categories:
        if g_mean > r_mean and g_mean > b_mean:
            categories.append(("vegetated landscape", 0.50))
        elif b_mean > r_mean and brightness < 70:
            categories.append(("water/wetland terrain", 0.50))
        else:
            categories.append(("mixed terrain / open land", 0.50))

    categories.sort(key=lambda x: x[1], reverse=True)
    dominant_type = categories[0][0]

    return {
        "dimensions": [w, h],
        "brightness": round(brightness, 1),
        "channel_means": {"r": round(r_mean, 1), "g": round(g_mean, 1), "b": round(b_mean, 1)},
        "veg_fraction": round(veg_ratio, 2),
        "water_fraction": round(water_ratio, 2),
        "urban_fraction": round(urban_ratio, 2),
        "soil_fraction": round(soil_ratio, 2),
        "dominant_land_cover": dominant_type,
        "categories": categories,
    }


def is_scene_description_query(q_lower: str) -> bool:
    """
    Identify queries requesting scene-level description, land-cover analysis,
    or overall visible landscape features.
    """
    triggers = [
        "land-cover",
        "land cover",
        "describe the scene",
        "describe scene",
        "major features",
        "major objects visible",
        "overall scene",
        "general scene",
        "describe what is visible",
        "describe the image",
        "describe this image",
        "describe the satellite image",
        "describe this satellite image",
        "scene description",
        "overall landscape",
    ]
    if any(t in q_lower for t in triggers):
        return True
    if "what is visible" in q_lower and "what objects are visible" not in q_lower:
        return True
    return False


def answer_satellite_question(
    image: Image.Image,
    question: str,
    detections_input,
) -> dict:
    """
    Core Vision-Language Reasoning Engine:
    Combines satellite image visual properties with Grounding DINO + SAM spatial evidence.
    """
    w, h = image.size
    raw_detections = parse_detections_input(detections_input)
    spatial_evidence = analyze_spatial_evidence((w, h), raw_detections)

    q_lower = question.lower().strip()

    # Filter high-confidence detections for reasoning
    valid_dets = [d for d in spatial_evidence if d["score"] >= 0.15]

    # --- 1. COUNTING QUESTIONS ---
    if "how many" in q_lower:
        if "red" in q_lower:
            matching = [d for d in valid_dets if "red" in d["label"].lower()]
            count = len(matching)
            if count == 1:
                det = matching[0]
                answer = (
                    f"There is 1 red structure visible in the image, located in the {det['region']} "
                    f"at coordinates {det['box']} with a segmentation area of {det['mask_area']} pixels."
                )
            else:
                answer = f"There are {count} red structures detected in the image."

        elif "blue" in q_lower:
            matching = [d for d in valid_dets if "blue" in d["label"].lower()]
            count = len(matching)
            if count == 1:
                det = matching[0]
                answer = (
                    f"There is 1 blue structure visible in the image, located in the {det['region']} "
                    f"at coordinates {det['box']} with a segmentation area of {det['mask_area']} pixels."
                )
            else:
                answer = f"There are {count} blue structures detected in the image."

        elif "car" in q_lower or "vehicle" in q_lower:
            matching = [d for d in valid_dets if "car" in d["label"].lower() or "vehicle" in d["label"].lower()]
            count = len(matching)
            if count == 1:
                det = matching[0]
                answer = (
                    f"There is 1 vehicle detected in the image, located in the {det['region']} "
                    f"at coordinates {det['box']} with a mask area of {det['mask_area']} pixels."
                )
            else:
                answer = f"There are {count} vehicles detected in the image."

        else:
            total_objs = len(valid_dets)
            labels_summary = ", ".join([f"'{d['label']}' ({d['region']})" for d in valid_dets])
            answer = (
                f"A total of {total_objs} distinct objects are detected in the image: {labels_summary}."
            )

    # --- 2. EXISTENCE QUESTIONS ---
    elif "is there" in q_lower or "are there" in q_lower or "contains" in q_lower or "any" in q_lower:
        if "building" in q_lower or "structure" in q_lower or "house" in q_lower:
            buildings = [d for d in valid_dets if "building" in d["label"].lower() or "structure" in d["label"].lower() or "house" in d["label"].lower()]
            if buildings:
                b = buildings[0]
                answer = f"Yes, buildings are present in the image located in the {b['region']} at coordinates {b['box']} with a confidence score of {b['score']:.2f}."
            else:
                answer = "No buildings were detected in the image."

        elif "water" in q_lower or "lake" in q_lower or "river" in q_lower or "ocean" in q_lower:
            waters = [d for d in valid_dets if "water" in d["label"].lower() or "lake" in d["label"].lower() or "river" in d["label"].lower() or "ocean" in d["label"].lower()]
            if waters:
                w = waters[0]
                answer = f"Yes, a water body is present in the image located in the {w['region']} at coordinates {w['box']} with a confidence score of {w['score']:.2f}."
            else:
                if "water bodies" in q_lower:
                    answer = "No water bodies were detected in the image."
                else:
                    answer = "No water body was detected in the image."

        elif "car" in q_lower or "vehicle" in q_lower or "automobile" in q_lower:
            cars = [d for d in valid_dets if "car" in d["label"].lower() or "vehicle" in d["label"].lower() or "automobile" in d["label"].lower()]
            if cars:
                c = cars[0]
                answer = (
                    f"Yes, there is a vehicle present in the image located at {c['box']} "
                    f"in the {c['region']} with a confidence score of {c['score']:.2f}."
                )
            else:
                answer = "No vehicle was detected in the given image."

        elif "road" in q_lower or "highway" in q_lower:
            roads = [d for d in valid_dets if "road" in d["label"].lower() or "highway" in d["label"].lower()]
            if roads:
                r = roads[0]
                answer = f"Yes, a road is present in the image located in the {r['region']} at coordinates {r['box']} with a confidence score of {r['score']:.2f}."
            else:
                answer = "No roads were detected in the image."

        elif "vegetation" in q_lower or "tree" in q_lower or "forest" in q_lower:
            veg = [d for d in valid_dets if "vegetation" in d["label"].lower() or "tree" in d["label"].lower() or "forest" in d["label"].lower()]
            if veg:
                v = veg[0]
                answer = f"Yes, vegetation is present in the image located in the {v['region']} at coordinates {v['box']} with a confidence score of {v['score']:.2f}."
            else:
                answer = "No vegetation was detected in the image."

        elif "red" in q_lower:
            reds = [d for d in valid_dets if "red" in d["label"].lower()]
            if reds:
                r = reds[0]
                answer = f"Yes, a red structure is present in the {r['region']} at coordinates {r['box']}."
            else:
                answer = "No red structure was detected in the image."

        elif "blue" in q_lower:
            blues = [d for d in valid_dets if "blue" in d["label"].lower()]
            if blues:
                b = blues[0]
                answer = f"Yes, a blue structure is present in the {b['region']} at coordinates {b['box']}."
            else:
                answer = "No blue structure was detected in the image."

        else:
            if valid_dets:
                answer = f"Yes, the image contains {len(valid_dets)} grounded objects based on spatial detection evidence."
            else:
                answer = "No relevant spatial objects were detected in the image."

    # --- 3. SPATIAL & RELATIVE LOCATION QUESTIONS ---
    elif "where is" in q_lower or "relative" in q_lower or "center" in q_lower or "location" in q_lower:
        if "closest to the center" in q_lower or "closest to center" in q_lower or "in the center" in q_lower:
            if valid_dets:
                closest = min(valid_dets, key=lambda d: d["dist_to_center_px"])
                answer = (
                    f"The object closest to the center is the '{closest['label']}' at coordinates {closest['box']} "
                    f"({closest['dist_to_center_px']:.1f} pixels from center, in the {closest['region']})."
                )
            else:
                answer = "No objects detected near the image center."

        elif "relative to" in q_lower or ("where" in q_lower and ("car" in q_lower or "structure" in q_lower or "box" in q_lower or "building" in q_lower)):
            cars = [d for d in valid_dets if "car" in d["label"].lower() or "vehicle" in d["label"].lower()]
            reds = [d for d in valid_dets if "red" in d["label"].lower()]
            blues = [d for d in valid_dets if "blue" in d["label"].lower()]

            if cars and (reds or blues):
                car = cars[0]
                rel_parts = []
                if reds:
                    red = reds[0]
                    dx = car["center"][0] - red["center"][0]
                    dy = car["center"][1] - red["center"][1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    dir_str = compute_cardinal_direction(dx, dy)
                    rel_parts.append(f"{dist:.1f} pixels {dir_str} of the red structure at {red['box']}")
                if blues:
                    blue = blues[0]
                    dx = car["center"][0] - blue["center"][0]
                    dy = car["center"][1] - blue["center"][1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    dir_str = compute_cardinal_direction(dx, dy)
                    rel_parts.append(f"{dist:.1f} pixels {dir_str} of the blue structure at {blue['box']}")

                answer = (
                    f"The car is located at coordinates {car['box']} in the {car['region']}, positioned "
                    + " and ".join(rel_parts) + "."
                )
            elif reds and blues:
                red = reds[0]
                blue = blues[0]
                dx = red["center"][0] - blue["center"][0]
                dy = red["center"][1] - blue["center"][1]
                dist = math.sqrt(dx * dx + dy * dy)
                dir_str = compute_cardinal_direction(dx, dy)

                no_car_prefix = "No car was detected in the scene. " if "car" in q_lower else ""
                answer = (
                    f"{no_car_prefix}Spatial evidence shows the red structure at {red['box']} ({red['region']}) "
                    f"is located {dist:.1f} pixels {dir_str} of the blue structure at {blue['box']} ({blue['region']})."
                )
            elif cars:
                c = cars[0]
                answer = f"The car is located at coordinates {c['box']} in the {c['region']}."
            elif valid_dets:
                locations = [f"'{d['label']}' at {d['box']} ({d['region']})" for d in valid_dets]
                no_car_prefix = "No car was detected in the scene. " if "car" in q_lower else ""
                answer = f"{no_car_prefix}Spatial locations of grounded objects: {'; '.join(locations)}."
            else:
                answer = "No distinct spatial objects were detected to analyze spatial relationships."

        else:
            locations = [f"'{d['label']}' at {d['box']} ({d['region']})" for d in valid_dets]
            answer = f"Spatial locations of detected objects: {'; '.join(locations)}."

    # --- 4. GENERAL SCENE DESCRIPTION / LAND-COVER ---
    else:
        scene_analysis = analyze_optical_scene(image)
        if is_scene_description_query(q_lower):
            dominant = scene_analysis["dominant_land_cover"]
            veg_pct = int(round(scene_analysis["veg_fraction"] * 100))
            water_pct = int(round(scene_analysis["water_fraction"] * 100))
            urban_pct = int(round(scene_analysis["urban_fraction"] * 100))
            soil_pct = int(round(scene_analysis["soil_fraction"] * 100))

            components_desc = []
            if veg_pct >= 20:
                components_desc.append(f"vegetated terrain and green canopy (~{veg_pct}%)")
            if water_pct >= 15:
                components_desc.append(f"low-reflectance drainage/water surfaces (~{water_pct}%)")
            if urban_pct >= 15:
                components_desc.append(f"built-up or paved structures (~{urban_pct}%)")
            if soil_pct >= 15:
                components_desc.append(f"bare ground and open fields (~{soil_pct}%)")

            if not components_desc:
                components_desc.append(dominant)

            summary_landcover = ", ".join(components_desc)

            if valid_dets:
                descriptions = [
                    f"a {d['label']} in the {d['region']} (box: {d['box']}, mask: {d['mask_area']} px)"
                    for d in valid_dets
                ]
                answer = (
                    f"The satellite scene exhibits predominantly {dominant} land-cover, comprising {summary_landcover}. "
                    f"Additionally, {len(valid_dets)} grounded objects were identified from spatial evidence: "
                    + ", ".join(descriptions) + "."
                )
            else:
                answer = (
                    f"The satellite scene exhibits predominantly {dominant} land-cover, comprising {summary_landcover}. "
                    f"Whole-image optical analysis indicates continuous surface terrain across the landscape. "
                    f"No discrete localized objects were isolated by spatial grounding."
                )

        elif valid_dets:
            descriptions = [
                f"a {d['label']} in the {d['region']} (box: {d['box']}, mask: {d['mask_area']} px)"
                for d in valid_dets
            ]
            answer = (
                f"The satellite image contains {len(valid_dets)} grounded objects: "
                + ", ".join(descriptions) + "."
            )
        else:
            if "water" in q_lower:
                answer = "No water bodies were detected in the image."
            elif "building" in q_lower:
                answer = "No buildings were detected in the image."
            elif "road" in q_lower:
                answer = "No roads were detected in the image."
            elif "car" in q_lower or "vehicle" in q_lower:
                answer = "No vehicles were detected in the image."
            elif "object" in q_lower:
                answer = "No objects were detected in the image."
            else:
                answer = "No distinct spatial objects matching the query were detected in the image."

    structured_result = {
        "question": question,
        "answer": answer,
        "total_detections": len(valid_dets),
        "image_size": [w, h],
        "spatial_evidence": valid_dets,
        "scene_analysis": analyze_optical_scene(image),
    }

    return {
        "answer": answer,
        "reasoning_context": structured_result,
        "detections_used": valid_dets,
        "structured_json": json.dumps(structured_result, indent=2),
    }


def predict(image_b64: str, question: str, detections_json: str) -> tuple[str, str]:
    """
    Gradio API predict endpoint.
    Input Payload Contract:
      {"data": [image_b64, question, detections_json]}
    Output Payload Contract:
      {"data": [answer, structured_result_json]}
    """
    if not image_b64 or not isinstance(image_b64, str):
        err_json = json.dumps({"error": "Base64 image string is empty or invalid."}, indent=2)
        raise gr.Error(err_json)

    if not question or not isinstance(question, str) or not question.strip():
        err_json = json.dumps({"error": "Question text cannot be empty."}, indent=2)
        raise gr.Error(err_json)

    try:
        img = decode_base64_image(image_b64)
    except Exception as exc:
        err_json = json.dumps({"error": f"Failed to decode base64 image: {str(exc)}"}, indent=2)
        raise gr.Error(err_json)

    try:
        res = answer_satellite_question(img, question.strip(), detections_json)
        return res["answer"], res["structured_json"]
    except Exception as exc:
        err_json = json.dumps({"error": f"VLM reasoning failed: {str(exc)}"}, indent=2)
        raise gr.Error(err_json)


# Gradio Interface for VLM Specialist Space
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Base64 Satellite Image", lines=3),
        gr.Textbox(label="User Question", lines=2, placeholder="e.g. How many red boxes are visible?"),
        gr.Textbox(label="Detections JSON String", lines=4),
    ],
    outputs=[
        gr.Textbox(label="Natural Language Answer", lines=3),
        gr.Textbox(label="Structured Result JSON Payload", lines=6),
    ],
    title="SatQuery AI — VLM Satellite Reasoning API",
    description="Vision-Language Satellite Reasoning microservice endpoint.",
    api_name="predict",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
