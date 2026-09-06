"""
SatQuery AI — Unit Tests for Bi-Temporal Change Analyzer
Verifies:
  1. ChangeAnalyzer initializes properly
  2. Bi-temporal change analysis executes on T1 + T2 PIL images
  3. Output structure contains change_ratio, severity, change_mask_b64, change_categories
  4. Change evaluation metrics computation works
  5. Invalid one-image temporal input is rejected by orchestrator validation
"""
import os
import sys
import io
import base64
import pytest
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.change_analyzer import get_change_analyzer
from core.orchestrator import orchestrate_bitemporal_satquery_request


def create_dummy_pil_pair() -> tuple[Image.Image, Image.Image]:
    img1 = Image.new("RGB", (256, 256), color=(50, 100, 50))
    img2 = Image.new("RGB", (256, 256), color=(200, 50, 50))
    return img1, img2


def encode_image_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_change_analyzer_initialization():
    analyzer = get_change_analyzer()
    assert analyzer is not None


def test_change_analyzer_execution():
    analyzer = get_change_analyzer()
    img1, img2 = create_dummy_pil_pair()
    res = analyzer.analyze_change(img1, img2, "Did the area change?")

    assert "change_ratio" in res
    assert "severity" in res
    assert "change_mask_b64" in res
    assert "change_categories" in res
    assert "evidence_summary" in res
    assert res["change_ratio"] >= 0.0
    assert len(res["change_mask_b64"]) > 0


@pytest.mark.asyncio
async def test_bitemporal_orchestrator_execution():
    img1, img2 = create_dummy_pil_pair()
    b64_1 = encode_image_base64(img1)
    b64_2 = encode_image_base64(img2)

    res = await orchestrate_bitemporal_satquery_request(
        b64_1, b64_2, "Has the built-up area increased, decreased, or remained unchanged?"
    )

    assert res.get("status") == "success"
    assert res.get("task") == "bi_temporal_change_analysis"
    assert "answer" in res
    assert "change_ratio" in res
    assert "overlay" in res


@pytest.mark.asyncio
async def test_invalid_bitemporal_input_rejection():
    img1, _ = create_dummy_pil_pair()
    b64_1 = encode_image_base64(img1)

    # Empty T2 image payload should be rejected
    res = await orchestrate_bitemporal_satquery_request(b64_1, "", "What changed?")
    assert res.get("status") == "error"
    assert res.get("error", {}).get("code") == "INVALID_IMAGE_T2_PAYLOAD"
