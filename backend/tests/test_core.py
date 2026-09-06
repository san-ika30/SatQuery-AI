"""
SatQuery AI — Extended Backend Tests
Tests for preprocessor, controller, evidence integrator, and HTTP endpoints.
"""
import io
import json
import struct
import zlib
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# ── Preprocessor tests ─────────────────────────────────────────────────────────

def test_detect_modality_sar():
    from core.preprocessor import detect_modality
    from schemas.models import Modality
    assert detect_modality(1, "sentinel1_image.tif") == Modality.sar
    assert detect_modality(1, "risat_sar.tiff") == Modality.sar
    assert detect_modality(1, "image.tif") == Modality.sar  # single band default


def test_detect_modality_optical():
    from core.preprocessor import detect_modality
    from schemas.models import Modality
    assert detect_modality(3, "image.png") == Modality.optical
    assert detect_modality(4, "sentinel2_rgb.tif") == Modality.optical


def test_detect_modality_multispectral():
    from core.preprocessor import detect_modality
    from schemas.models import Modality
    assert detect_modality(13, "sentinel2_ms.tif") == Modality.multispectral
    assert detect_modality(6, "cartosat.tif") == Modality.multispectral


def test_normalize_band():
    import numpy as np
    from core.preprocessor import normalize_band
    band = np.array([[100, 200, 300], [400, 500, 600]], dtype=np.float32)
    result = normalize_band(band)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


def test_normalize_band_zero():
    """All-zero band should return all-zero uint8."""
    import numpy as np
    from core.preprocessor import normalize_band
    band = np.zeros((10, 10), dtype=np.float32)
    result = normalize_band(band)
    assert result.dtype == np.uint8
    assert (result == 0).all()


def test_validate_image_pair_bitemporal_ok():
    from core.preprocessor import validate_image_pair
    from schemas.models import ImageMetadata, Modality
    meta1 = ImageMetadata(filename="t1.tif", modality=Modality.optical,
                          width=512, height=512, band_count=3, dtype="uint8", crs="EPSG:4326")
    meta2 = ImageMetadata(filename="t2.tif", modality=Modality.optical,
                          width=512, height=512, band_count=3, dtype="uint8", crs="EPSG:4326")
    warnings = validate_image_pair(meta1, meta2, "bitemporal")
    assert len(warnings) == 0


def test_validate_image_pair_crs_mismatch():
    from core.preprocessor import validate_image_pair
    from schemas.models import ImageMetadata, Modality
    meta1 = ImageMetadata(filename="t1.tif", modality=Modality.optical,
                          width=512, height=512, band_count=3, dtype="uint8", crs="EPSG:4326")
    meta2 = ImageMetadata(filename="t2.tif", modality=Modality.optical,
                          width=512, height=512, band_count=3, dtype="uint8", crs="EPSG:32643")
    warnings = validate_image_pair(meta1, meta2, "bitemporal")
    assert any("CRS" in w for w in warnings)


def test_validate_image_pair_crossmodal_mismatch():
    from core.preprocessor import validate_image_pair
    from schemas.models import ImageMetadata, Modality
    meta1 = ImageMetadata(filename="opt.tif", modality=Modality.optical,
                          width=512, height=512, band_count=3, dtype="uint8")
    meta2 = ImageMetadata(filename="opt2.tif", modality=Modality.optical,
                          width=512, height=512, band_count=3, dtype="uint8")
    warnings = validate_image_pair(meta1, meta2, "crossmodal")
    assert len(warnings) > 0


def test_validate_image_pair_crossmodal_ok():
    from core.preprocessor import validate_image_pair
    from schemas.models import ImageMetadata, Modality
    meta1 = ImageMetadata(filename="opt.tif", modality=Modality.optical,
                          width=512, height=512, band_count=3, dtype="uint8", crs="EPSG:4326")
    meta2 = ImageMetadata(filename="sar.tif", modality=Modality.sar,
                          width=512, height=512, band_count=1, dtype="float32", crs="EPSG:4326")
    warnings = validate_image_pair(meta1, meta2, "crossmodal")
    assert len(warnings) == 0


def test_unsupported_extension():
    from core.preprocessor import geotiff_to_rgb_png, PreprocessingError
    with pytest.raises(PreprocessingError, match="Unsupported"):
        geotiff_to_rgb_png(b"fake", "image.bmp")


# ── Controller tests ──────────────────────────────────────────────────────────

def test_rule_classifier_grounding():
    from core.controller import _rule_based_classify
    from schemas.models import InputMode, TaskType
    plan = _rule_based_classify("Highlight the water body in this image.", InputMode.single)
    assert plan.task_type == TaskType.grounding
    assert "grounding_dino" in plan.models


def test_rule_classifier_caption():
    from core.controller import _rule_based_classify
    from schemas.models import InputMode, TaskType
    plan = _rule_based_classify("Describe the land cover in this satellite image.", InputMode.single)
    assert plan.task_type == TaskType.caption
    assert "geochat" in plan.models


def test_rule_classifier_vqa_default():
    from core.controller import _rule_based_classify
    from schemas.models import InputMode, TaskType
    plan = _rule_based_classify("What is the dominant vegetation type?", InputMode.single)
    assert plan.task_type == TaskType.vqa


def test_rule_classifier_bitemporal_change_vqa():
    from core.controller import _rule_based_classify
    from schemas.models import InputMode, TaskType
    plan = _rule_based_classify("Has the built-up area increased?", InputMode.bitemporal)
    assert plan.task_type == TaskType.change_vqa
    assert "changeformer" in plan.models


def test_rule_classifier_bitemporal_change_detection():
    from core.controller import _rule_based_classify
    from schemas.models import InputMode, TaskType
    plan = _rule_based_classify("Show me what changed between these two dates.", InputMode.bitemporal)
    assert plan.task_type == TaskType.change_detection


def test_rule_classifier_crossmodal():
    from core.controller import _rule_based_classify
    from schemas.models import InputMode, TaskType
    plan = _rule_based_classify("Use both images to find water.", InputMode.crossmodal)
    assert plan.task_type == TaskType.sar_fusion
    assert "sar_fusion_encoder" in plan.models


def test_rule_classifier_locate_keyword():
    from core.controller import _rule_based_classify
    from schemas.models import InputMode, TaskType
    plan = _rule_based_classify("Locate the airport in this image.", InputMode.single)
    assert plan.task_type == TaskType.grounding


def test_rule_classifier_find_keyword():
    from core.controller import _rule_based_classify
    from schemas.models import InputMode, TaskType
    plan = _rule_based_classify("Find all buildings in the scene.", InputMode.single)
    assert plan.task_type == TaskType.grounding


@pytest.mark.asyncio
async def test_classify_task_no_token():
    """Without HF_API_TOKEN, should use rule-based and return a TaskPlan."""
    from core.controller import classify_task
    from schemas.models import InputMode
    with patch.dict("os.environ", {"HF_API_TOKEN": ""}):
        plan, method = await classify_task("Describe the scene.", InputMode.single)
        assert method == "rule_based"
        assert plan.task_type is not None


# ── Evidence Integrator tests ─────────────────────────────────────────────────

def test_integrate_empty_outputs():
    from core.evidence_integrator import integrate_outputs
    from schemas.models import TaskPlan, TaskType, InputMode
    plan = TaskPlan(task_type=TaskType.vqa, input_mode=InputMode.single,
                    models=[], reasoning="")
    ev = integrate_outputs(plan, [])
    assert ev.confidence == 0.0
    assert "No model outputs" in ev.answer


def test_integrate_single_output():
    from core.evidence_integrator import integrate_outputs
    from schemas.models import TaskPlan, TaskType, InputMode, ModelOutput
    plan = TaskPlan(task_type=TaskType.vqa, input_mode=InputMode.single,
                    models=["geochat"], reasoning="")
    out = ModelOutput(model_name="GeoChat", task_type=TaskType.vqa,
                      answer_text="This is a forest area.", confidence=0.82)
    ev = integrate_outputs(plan, [out])
    assert ev.confidence == 0.82
    assert "forest" in ev.answer.lower()
    assert "GeoChat" in ev.models_used


def test_integrate_grounding_output():
    from core.evidence_integrator import integrate_outputs
    from schemas.models import TaskPlan, TaskType, InputMode, ModelOutput
    plan = TaskPlan(task_type=TaskType.grounding, input_mode=InputMode.single,
                    models=["grounding_dino"], reasoning="")
    boxes = [{"label": "river", "score": 0.9, "xmin": 10, "ymin": 20, "xmax": 100, "ymax": 200}]
    out = ModelOutput(model_name="GroundingDINO", task_type=TaskType.grounding,
                      answer_text="Detected regions:", confidence=0.9, bounding_boxes=boxes)
    ev = integrate_outputs(plan, [out])
    assert ev.confidence == 0.9
    assert len(ev.visual_evidence) > 0
    assert any(e["type"] == "bounding_boxes" for e in ev.visual_evidence)


def test_integrate_change_detection():
    from core.evidence_integrator import integrate_outputs
    from schemas.models import TaskPlan, TaskType, InputMode, ModelOutput
    plan = TaskPlan(task_type=TaskType.change_detection, input_mode=InputMode.bitemporal,
                    models=["changeformer"], reasoning="")
    out = ModelOutput(model_name="ChangeFormer", task_type=TaskType.change_detection,
                      answer_text="Change analysis complete.", confidence=0.75)
    ev = integrate_outputs(plan, [out], change_map_r2_key="sessions/abc/change_map.png")
    assert "change map" in ev.answer.lower()
    assert any(e["type"] == "change_map" for e in ev.visual_evidence)


def test_integrate_sar_fusion():
    from core.evidence_integrator import integrate_outputs
    from schemas.models import TaskPlan, TaskType, InputMode, ModelOutput
    plan = TaskPlan(task_type=TaskType.sar_fusion, input_mode=InputMode.crossmodal,
                    models=["sar_fusion_encoder"], reasoning="")
    out = ModelOutput(model_name="SAR Fusion", task_type=TaskType.sar_fusion,
                      answer_text="Optical-SAR fusion complete.", confidence=0.80)
    ev = integrate_outputs(plan, [out], fusion_composite_r2_key="sessions/abc/sar.png")
    assert any(e["type"] == "sar_composite" for e in ev.visual_evidence)


def test_integrate_multi_model():
    """Multiple model outputs should aggregate confidence."""
    from core.evidence_integrator import integrate_outputs
    from schemas.models import TaskPlan, TaskType, InputMode, ModelOutput
    plan = TaskPlan(task_type=TaskType.vqa, input_mode=InputMode.single,
                    models=["geochat", "blip2"], reasoning="")
    out1 = ModelOutput(model_name="GeoChat", task_type=TaskType.vqa,
                       answer_text="Forest region.", confidence=0.8)
    out2 = ModelOutput(model_name="BLIP2", task_type=TaskType.vqa,
                       answer_text="Dense vegetation.", confidence=0.6)
    ev = integrate_outputs(plan, [out1, out2])
    assert abs(ev.confidence - 0.7) < 0.001  # average of 0.8 and 0.6


def test_confidence_labels():
    from core.evidence_integrator import _confidence_label
    assert _confidence_label(0.90) == "Very High"
    assert _confidence_label(0.75) == "High"
    assert _confidence_label(0.55) == "Moderate"
    assert _confidence_label(0.35) == "Low"
    assert _confidence_label(0.10) == "Very Low"


# ── Report Generator tests ─────────────────────────────────────────────────────

def test_generate_json_report():
    from core.report_generator import generate_json_report
    from schemas.models import (AnalysisRequest, EvidencePackage, TaskType, InputMode,
                                ImageMetadata, Modality)
    meta = ImageMetadata(filename="test.tif", modality=Modality.optical,
                         width=512, height=512, band_count=3, dtype="uint8")
    req = AnalysisRequest(
        session_id="test-001", query="Describe the image.",
        input_mode=InputMode.single,
        image_keys=["sessions/test-001/img1_abc.png"],
        image_metadata=[meta],
    )
    ev = EvidencePackage(
        answer="This is a forest area.",
        confidence=0.82,
        task_type=TaskType.caption,
        models_used=["GeoChat-7B"],
        visual_evidence=[],
        execution_trace=[],
    )
    report_bytes = generate_json_report(req, ev)
    report = json.loads(report_bytes)
    assert report["satquery_ai_version"] == "1.0.0"
    assert report["result"]["confidence"] == 0.82
    assert "forest" in report["result"]["answer"]
    assert report["session"]["id"] == "test-001"


def test_generate_pdf_report():
    """PDF report should return non-empty bytes starting with %PDF."""
    from core.report_generator import generate_pdf_report
    from schemas.models import (AnalysisRequest, EvidencePackage, TaskType, InputMode,
                                ImageMetadata, Modality)
    meta = ImageMetadata(filename="test.tif", modality=Modality.optical,
                         width=256, height=256, band_count=3, dtype="uint8")
    req = AnalysisRequest(
        session_id="test-002", query="What is visible?",
        input_mode=InputMode.single,
        image_keys=["sessions/test-002/img1.png"],
        image_metadata=[meta],
    )
    ev = EvidencePackage(
        answer="Urban area detected.",
        confidence=0.75,
        task_type=TaskType.vqa,
        models_used=["BLIP2"],
        visual_evidence=[],
        execution_trace=[{"step": 1, "component": "Test", "status": "ok", "duration_ms": 100}],
    )
    pdf_bytes = generate_pdf_report(req, ev)
    assert len(pdf_bytes) > 0
    assert pdf_bytes[:4] == b"%PDF"


# ── Schemas tests ──────────────────────────────────────────────────────────────

def test_analysis_response_schema():
    from schemas.models import AnalysisResponse, AnalysisStatus
    resp = AnalysisResponse(session_id="test", status=AnalysisStatus.completed)
    d = resp.model_dump()
    assert d["status"] == "completed"
    assert d["session_id"] == "test"


def test_task_plan_schema():
    from schemas.models import TaskPlan, TaskType, InputMode
    plan = TaskPlan(
        task_type=TaskType.vqa,
        input_mode=InputMode.single,
        models=["geochat"],
        parameters={"max_tokens": 512},
        reasoning="VQA task detected.",
    )
    assert plan.task_type == TaskType.vqa
    assert plan.parameters["max_tokens"] == 512


def test_r2_client_no_config():
    """R2 client should return placeholder key when not configured."""
    from storage.r2_client import upload_bytes
    with patch.dict("os.environ", {"R2_ACCOUNT_ID": "", "R2_ACCESS_KEY_ID": "", "R2_SECRET_ACCESS_KEY": ""}):
        import importlib
        import storage.r2_client as r2
        r2._r2_client = None  # Reset singleton
        r2.R2_ACCOUNT_ID = ""
        key = r2.upload_bytes(b"test data", "test/key.png", "image/png")
        assert key.startswith("local/")


def test_get_public_url_local():
    from storage.r2_client import get_public_url
    url = get_public_url("local/sessions/abc/img.png")
    assert url.startswith("/api/local-files/")


def test_get_public_url_r2():
    from storage.r2_client import get_public_url
    import storage.r2_client as r2
    original = r2.R2_PUBLIC_URL
    try:
        r2.R2_PUBLIC_URL = "https://pub-abc.r2.dev"
        url = get_public_url("sessions/abc/img.png")
        assert url == "https://pub-abc.r2.dev/sessions/abc/img.png"
    finally:
        r2.R2_PUBLIC_URL = original


# ── In-memory session store tests ─────────────────────────────────────────────

def test_store_and_get_session():
    from db.session_store import store_session, get_session
    sid = "mem-store-001"
    store_session(sid, "single", "Describe the scene.", status="pending")
    s = get_session(sid)
    assert s is not None
    assert s["id"] == sid
    assert s["input_mode"] == "single"
    assert s["status"] == "pending"
    assert s["task_type"] is None


def test_update_session_in_memory():
    from db.session_store import store_session, update_session, get_session
    sid = "mem-store-002"
    store_session(sid, "bitemporal", "Any changes?", status="pending")
    update_session(sid, status="completed", task_type="change_detection")
    s = get_session(sid)
    assert s["status"] == "completed"
    assert s["task_type"] == "change_detection"


def test_store_image_with_bounds():
    from db.session_store import store_session, store_image, get_images
    sid = "mem-store-003"
    store_session(sid, "single", "Check bounds.", status="pending")
    store_image(sid, "sentinel2.tif", "optical", "local/img1.png",
                {"crs": "EPSG:4326", "bounds": [72.8, 18.9, 73.1, 19.2],
                 "width": 512, "height": 512, "band_count": 3})
    imgs = get_images(sid)
    assert len(imgs) == 1
    assert imgs[0]["bounds"] == [72.8, 18.9, 73.1, 19.2]
    assert imgs[0]["crs"] == "EPSG:4326"


def test_store_two_images_per_session():
    from db.session_store import store_session, store_image, get_images
    sid = "mem-store-006"
    store_session(sid, "bitemporal", "Multi-image test.", status="pending")
    store_image(sid, "t1.tif", "optical", "local/img1.png",
                {"crs": "EPSG:4326", "width": 512, "height": 512, "band_count": 3})
    store_image(sid, "t2.tif", "optical", "local/img2.png",
                {"crs": "EPSG:4326", "width": 512, "height": 512, "band_count": 3})
    imgs = get_images(sid)
    assert len(imgs) == 2
    assert imgs[0]["filename"] == "t1.tif"
    assert imgs[1]["filename"] == "t2.tif"


def test_store_result_and_get_result():
    from db.session_store import store_session, store_result, get_result
    sid = "mem-store-004"
    store_session(sid, "single", "VQA query.", status="pending")
    trace = [{"step": 1, "component": "Test", "status": "ok", "duration_ms": 50}]
    store_result(sid, "vqa", "GeoChat-7B", "Forest area detected.", 0.87,
                 None, "/api/local-files/reports/mem-store-004/report.pdf", trace)
    r = get_result(sid)
    assert r is not None
    assert r["task_type"] == "vqa"
    assert abs(r["confidence"] - 0.87) < 0.001
    assert len(r["execution_trace"]) == 1


def test_get_sessions_filter_by_status():
    from db.session_store import store_session, get_sessions
    store_session("filter-status-001", "single", "Filter test",
                  status="failed", task_type="vqa")
    rows, _ = get_sessions(status="failed")
    assert any(r["id"] == "filter-status-001" for r in rows)
    rows_ok, _ = get_sessions(status="completed")
    assert not any(r["id"] == "filter-status-001" for r in rows_ok)


def test_get_sessions_filter_by_task_type():
    from db.session_store import store_session, get_sessions
    store_session("filter-task-001", "single", "Grounding test",
                  status="completed", task_type="grounding")
    rows, _ = get_sessions(task_type="grounding")
    assert any(r["id"] == "filter-task-001" for r in rows)


def test_get_sessions_pagination():
    from db.session_store import store_session, get_sessions
    for i in range(6):
        store_session(f"page-test-{i:04d}", "single", f"Query {i}",
                      status="completed", task_type="vqa")
    rows, total = get_sessions(limit=3, offset=0, status="completed")
    assert len(rows) <= 3
    assert total >= 6


def test_get_stats_counts():
    from db.session_store import store_session, get_stats
    store_session("stats-001", "single", "Stats test",
                  status="completed", task_type="caption")
    stats = get_stats()
    assert stats["total_sessions"] >= 1
    assert "completed" in stats["by_status"]
    assert "caption" in stats["by_task_type"]


import pytest

@pytest.mark.asyncio
async def test_log_session_writes_to_memory_store():
    from db.supabase_client import log_session
    from db.session_store import get_session
    sid = "sb-fallback-001"
    await log_session(sid, "crossmodal", "SAR query.", status="pending")
    s = get_session(sid)
    assert s is not None
    assert s["id"] == sid
    assert s["input_mode"] == "crossmodal"


@pytest.mark.asyncio
async def test_update_session_status_writes_to_memory():
    from db.supabase_client import log_session, update_session_status
    from db.session_store import get_session
    sid = "sb-fallback-002"
    await log_session(sid, "single", "Status update test.", status="pending")
    await update_session_status(sid, "completed")
    s = get_session(sid)
    assert s["status"] == "completed"


@pytest.mark.asyncio
async def test_update_session_task_type_writes_to_memory():
    from db.supabase_client import log_session, update_session_task_type
    from db.session_store import get_session
    sid = "sb-fallback-003"
    await log_session(sid, "single", "Task type test.", status="routing")
    await update_session_task_type(sid, "grounding")
    s = get_session(sid)
    assert s["task_type"] == "grounding"


@pytest.mark.asyncio
async def test_log_result_writes_to_memory():
    from db.supabase_client import log_session, log_result
    from db.session_store import get_result
    sid = "sb-fallback-004"
    await log_session(sid, "single", "Result test.", status="pending")
    await log_result(sid, "caption", "GeoChat-7B", "Dense forest detected.", 0.91,
                     None, "/api/local-files/reports/sb-fallback-004/report.pdf",
                     [{"step": 1, "component": "Test", "status": "ok", "duration_ms": 100}])
    r = get_result(sid)
    assert r is not None
    assert r["task_type"] == "caption"
    assert abs(r["confidence"] - 0.91) < 0.001


@pytest.mark.asyncio
async def test_log_image_writes_bounds_to_memory():
    from db.supabase_client import log_session, log_image
    from db.session_store import get_images
    sid = "sb-fallback-005"
    await log_session(sid, "bitemporal", "Change check.", status="pending")
    await log_image(sid, "t1.tif", "optical", "local/sessions/t1.png",
                    {"crs": "EPSG:32643", "bounds": [77.0, 28.5, 77.5, 29.0],
                     "width": 256, "height": 256, "band_count": 3})
    imgs = get_images(sid)
    assert len(imgs) == 1
    assert imgs[0]["bounds"] == [77.0, 28.5, 77.5, 29.0]
    assert imgs[0]["modality"] == "optical"


def test_r2_local_write_creates_file(tmp_path, monkeypatch):
    """upload_bytes without R2 config writes bytes to LOCAL_STORAGE_DIR (P0 fix)."""
    import storage.r2_client as r2
    monkeypatch.setenv("LOCAL_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(r2, "R2_ACCOUNT_ID", "")
    monkeypatch.setattr(r2, "_r2_client", None)
    key = r2.upload_bytes(b"hello-world", "sessions/test-write/img1.png", "image/png")
    assert key == "local/sessions/test-write/img1.png"
    written = tmp_path / "sessions" / "test-write" / "img1.png"
    assert written.exists()
    assert written.read_bytes() == b"hello-world"
