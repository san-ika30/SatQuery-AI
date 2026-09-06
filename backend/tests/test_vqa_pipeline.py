"""
Tests for single-image VQA pipeline with Grounding DINO + SAM + VLM Specialist.
Verifies:
  A. 'Are there any buildings visible in the image?' (truthful presence and absence)
  B. 'Is there any water body visible in the image?' (truthful presence and absence)
  C. 'Identify the major objects visible in this image.' (describes actual detected objects)
  D. Detections are actually passed into VLM specialist
  E. No fabricated detection produced when Grounding DINO returns none
  F. Model name is truthful ("Satellite Spatial Reasoning Specialist") and confidence is evidence-based
"""
import pytest
import io
import base64
from PIL import Image
from unittest.mock import AsyncMock, patch

from core.model_registry import run_geochat
from schemas.models import TaskType


def _make_dummy_image_bytes() -> bytes:
    img = Image.new("RGB", (256, 256), color=(40, 120, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_vqa_building_detection_truthful_presence_and_absence():
    """Test A: 'Are there any buildings visible in the image?'"""
    from core.orchestrator import get_vlm_module
    answer_satellite_question = get_vlm_module().answer_satellite_question

    img = Image.new("RGB", (256, 256), color="green")

    # When building is detected
    dets_with_building = [
        {"label": "building", "score": 0.88, "box": [10.0, 10.0, 50.0, 50.0], "region": "upper-left", "mask_area": 1600}
    ]
    res_present = answer_satellite_question(img, "Are there any buildings visible in the image?", dets_with_building)
    assert "yes" in res_present["answer"].lower()
    assert "building" in res_present["answer"].lower()

    # When no building is detected (only water body)
    dets_no_building = [
        {"label": "water body", "score": 0.75, "box": [10.0, 10.0, 50.0, 50.0], "region": "upper-left", "mask_area": 1600}
    ]
    res_absent = answer_satellite_question(img, "Are there any buildings visible in the image?", dets_no_building)
    assert "no buildings were detected in the image" in res_absent["answer"].lower()


@pytest.mark.asyncio
async def test_vqa_water_body_truthful_presence_and_absence():
    """Test B: 'Is there any water body visible in the image?'"""
    from core.orchestrator import get_vlm_module
    answer_satellite_question = get_vlm_module().answer_satellite_question

    img = Image.new("RGB", (256, 256), color="blue")

    # When water body is detected
    dets_with_water = [
        {"label": "water body", "score": 0.82, "box": [20.0, 20.0, 100.0, 100.0], "region": "center region", "mask_area": 6400}
    ]
    res_present = answer_satellite_question(img, "Is there any water body visible in the image?", dets_with_water)
    assert "yes" in res_present["answer"].lower()
    assert "water body" in res_present["answer"].lower()

    # When no water body is detected (empty detections)
    res_absent = answer_satellite_question(img, "Is there any water body visible in the image?", [])
    assert "no water body was detected in the image" in res_absent["answer"].lower()


@pytest.mark.asyncio
async def test_vqa_identify_major_objects():
    """Test C: 'Identify the major objects visible in this image.'"""
    from core.orchestrator import get_vlm_module
    answer_satellite_question = get_vlm_module().answer_satellite_question

    img = Image.new("RGB", (256, 256), color="gray")

    dets = [
        {"label": "water body", "score": 0.85, "box": [10.0, 10.0, 80.0, 80.0], "region": "upper-left", "mask_area": 4900},
        {"label": "building", "score": 0.78, "box": [120.0, 120.0, 200.0, 200.0], "region": "lower-right", "mask_area": 6400},
    ]
    res = answer_satellite_question(img, "Identify the major objects visible in this image.", dets)
    ans = res["answer"].lower()
    assert "water body" in ans
    assert "building" in ans
    assert "2 grounded objects" in ans


@pytest.mark.asyncio
async def test_vqa_detections_passed_to_vlm_specialist():
    """Test D: Verify detections are actually passed into call_vlm_specialist."""
    img_bytes = _make_dummy_image_bytes()
    query = "Are there any buildings visible in the image?"

    mock_dets = [
        {"label": "building", "score": 0.85, "box": [10.0, 10.0, 50.0, 50.0], "region": "upper-left", "mask_area": 1600}
    ]

    with patch("core.orchestrator.call_grounding_specialist", new_callable=AsyncMock) as mock_grounding:
        with patch("core.orchestrator.call_vlm_specialist", new_callable=AsyncMock) as mock_vlm:
            mock_grounding.return_value = ("mock_overlay_b64", mock_dets)
            mock_vlm.return_value = ("Yes, buildings are present.", {"detections_used": mock_dets})

            out = await run_geochat(img_bytes, query, TaskType.vqa)

            # Assert Grounding DINO was called
            mock_grounding.assert_called_once()

            # Assert VLM specialist received the actual Grounding DINO detections
            mock_vlm.assert_called_once()
            called_args = mock_vlm.call_args[0]
            assert called_args[1] == query  # question preserved exactly
            assert called_args[2] == mock_dets  # detections passed in

            # Assert output properties
            assert out.model_name == "Satellite Spatial Reasoning Specialist"
            assert out.confidence == 0.85  # average of detection score
            assert out.bounding_boxes == mock_dets


@pytest.mark.asyncio
async def test_vqa_no_fabricated_detections_on_empty():
    """Test D: Verify no fabricated detection is produced when Grounding DINO returns none."""
    img_bytes = _make_dummy_image_bytes()
    query = "Is there any water body visible in the image?"

    with patch("core.orchestrator.call_grounding_specialist", new_callable=AsyncMock) as mock_grounding:
        mock_grounding.return_value = ("mock_overlay_b64", [])  # Grounding DINO finds nothing

        out = await run_geochat(img_bytes, query, TaskType.vqa)

        assert out.bounding_boxes is None
        assert "no water body was detected" in out.answer_text.lower()
        assert out.model_name == "Satellite Spatial Reasoning Specialist"
        assert out.confidence != 0.70  # removed hardcoded 0.70 confidence
        assert out.confidence == 0.75  # evidence-based absence certainty


@pytest.mark.asyncio
async def test_spatial_vqa_routing_invokes_grounding():
    """Test 8A & 8E: Object VQA invokes grounding when appropriate and models list reflects both."""
    from core.controller import classify_task
    from schemas.models import InputMode

    queries = [
        "Are there any water bodies in the image?",
        "Are there roads visible?",
        "Are there buildings?",
        "What objects are visible?",
    ]

    for q in queries:
        plan, method = await classify_task(q, InputMode.single)
        assert plan.task_type == TaskType.vqa
        assert "grounding_dino" in plan.models, f"grounding_dino missing for query: {q}"
        assert "geochat" in plan.models, f"geochat missing for query: {q}"


@pytest.mark.asyncio
async def test_vqa_truthful_absence_responses_for_all_categories():
    """Test 8D: Verify truthful absence responses for water, buildings, roads, objects."""
    from core.orchestrator import get_vlm_module
    answer_fn = get_vlm_module().answer_satellite_question
    img = Image.new("RGB", (256, 256), color="green")

    # Water query with empty detections
    res_water = answer_fn(img, "Are there any water bodies in the image?", [])
    assert "no water bodies were detected in the image" in res_water["answer"].lower()
    assert "distinct spatial objects were detected" not in res_water["answer"]

    # Road query with empty detections
    res_road = answer_fn(img, "Are there roads visible?", [])
    assert "no roads were detected in the image" in res_road["answer"].lower()
    assert "distinct spatial objects were detected" not in res_road["answer"]

    # Buildings query with empty detections
    res_bldg = answer_fn(img, "Are there buildings?", [])
    assert "no buildings were detected in the image" in res_bldg["answer"].lower()
    assert "distinct spatial objects were detected" not in res_bldg["answer"]

    # General objects query with empty detections
    res_obj = answer_fn(img, "What objects are visible?", [])
    assert "no objects were detected in the image" in res_obj["answer"].lower()
    assert "distinct spatial objects were detected" not in res_obj["answer"]


@pytest.mark.asyncio
async def test_scene_description_with_zero_grounding_detections():
    """Requirement 8A & 8C: Scene description does NOT return 'no objects detected' when GroundingDINO has zero boxes,
    and performs whole-image scene-level reasoning."""
    from core.orchestrator import get_vlm_module
    answer_fn = get_vlm_module().answer_satellite_question
    img = Image.new("RGB", (256, 256), color=(40, 160, 50))  # green vegetated terrain

    query = "Describe the land-cover and major objects visible in this image."
    res = answer_fn(img, query, [])

    ans = res["answer"].lower()
    # Must NOT claim 'no objects were detected'
    assert "no objects were detected in the image" not in ans
    # Must describe scene land-cover from optical evidence
    assert any(term in ans for term in ["land-cover", "vegetat", "canopy", "terrain"])
    # Must note truthful absence of discrete localized objects
    assert "discrete localized objects" in ans or "spatial grounding" in ans


@pytest.mark.asyncio
async def test_scene_description_no_fabricated_detections():
    """Requirement 8D: Ensure scene description never fabricates bounding boxes or detections."""
    from core.orchestrator import get_vlm_module
    answer_fn = get_vlm_module().answer_satellite_question
    img = Image.new("RGB", (256, 256), color="green")

    query = "Describe the scene and major features."
    res = answer_fn(img, query, [])

    assert res["reasoning_context"]["total_detections"] == 0
    assert res["reasoning_context"]["spatial_evidence"] == []
    assert res["detections_used"] == []
    assert "scene_analysis" in res["reasoning_context"]
    assert res["reasoning_context"]["scene_analysis"]["veg_fraction"] > 0.0


@pytest.mark.asyncio
async def test_scene_description_with_supplementary_detections():
    """Requirement 3: GroundingDINO detections are supplementary evidence when present."""
    from core.orchestrator import get_vlm_module
    answer_fn = get_vlm_module().answer_satellite_question
    img = Image.new("RGB", (256, 256), color="green")

    dets = [
        {"label": "building", "score": 0.88, "box": [50.0, 50.0, 100.0, 100.0], "region": "upper-left", "mask_area": 2500}
    ]
    query = "Describe the land-cover and major objects visible in this image."
    res = answer_fn(img, query, dets)

    ans = res["answer"].lower()
    assert "land-cover" in ans or "vegetat" in ans
    assert "building" in ans
    assert "grounded objects were identified" in ans


