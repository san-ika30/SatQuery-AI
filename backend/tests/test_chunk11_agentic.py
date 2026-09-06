"""
Chunk 11 — Agentic Orchestration & Model/Tool Registry Test Suite
Tests:
  1. Intent: single_image_description
  2. Tool selection: grounding_specialist for "Highlight the water body"
  3. Tool selection: change_analyzer for "What changed between these two dates?"
  4. Tool selection: change_analyzer for "Has the built-up area increased?"
  5. Tool selection: optical_sar_analyzer for optical+SAR joint query
  6. Input validation: MISSING_SAR_IMAGE error
  7. Input validation: MISSING_TEMPORAL_IMAGE error
  8. Tool selection: Grounding Specialist
  9. Multi-tool plan: complex multi-tool query
 10. Trace verification: execution_trace exists with valid steps
 11. Confidence verification: confidence has transparent basis
 12. Dynamic response verification: different inputs produce different answers (no hardcoding)
 13. Registry verification: 7 registered tools with capability lookup
 14. Optical-specific query: optical_analyzer planned
 15. SAR-specific query: sar_analyzer planned
"""
import io
import base64
import pytest
from PIL import Image

from core.model_registry import get_tool_registry
from core.query_interpreter import get_query_interpreter
from core.input_validator import get_input_validator
from core.agent_planner import get_agent_planner
from core.orchestrator import orchestrate_agentic_request


def _create_test_image_b64(color: tuple = (100, 150, 200), size: tuple = (128, 128)) -> str:
    """Generate a clean test image in base64 format."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TestChunk11AgenticOrchestration:

    # ── Test 1: Single Image Description Intent ───────────────────────────────
    def test_01_query_intent_description(self):
        qi = get_query_interpreter()
        query = "Describe the land-cover and major objects visible in this image."
        intent = qi.interpret(query)
        assert intent.intent == "single_image_description"
        assert "optical" in intent.modalities
        assert intent.temporal == "single"

    # ── Test 2: Tool Selection - Grounding Specialist ─────────────────────────
    def test_02_tool_selection_grounding_water(self):
        qi = get_query_interpreter()
        ap = get_agent_planner()
        query = "Highlight the water body."
        intent = qi.interpret(query)
        plan = ap.create_plan(query, intent, {})
        tools = [s.tool for s in plan.steps]
        assert "grounding_specialist" in tools

    # ── Test 3: Tool Selection - Change Analyzer ──────────────────────────────
    def test_03_tool_selection_bitemporal_change(self):
        qi = get_query_interpreter()
        ap = get_agent_planner()
        query = "What changed between these two dates?"
        intent = qi.interpret(query)
        assert intent.intent == "bitemporal_change"
        assert intent.requires_change_detection is True
        plan = ap.create_plan(query, intent, {"image_t1": "b64", "image_t2": "b64"})
        tools = [s.tool for s in plan.steps]
        assert "change_analyzer" in tools

    # ── Test 4: Tool Selection - Directional Change Query ─────────────────────
    def test_04_tool_selection_builtup_increase(self):
        qi = get_query_interpreter()
        ap = get_agent_planner()
        query = "Has the built-up area increased?"
        intent = qi.interpret(query)
        assert intent.intent == "bitemporal_change"
        plan = ap.create_plan(query, intent, {"image_t1": "b64", "image_t2": "b64"})
        tools = [s.tool for s in plan.steps]
        assert "change_analyzer" in tools

    # ── Test 5: Tool Selection - Optical + SAR Cross-Modal ────────────────────
    def test_05_tool_selection_optical_sar_joint(self):
        qi = get_query_interpreter()
        ap = get_agent_planner()
        query = "Use the optical and SAR images together to identify built-up and water-covered regions."
        intent = qi.interpret(query)
        assert intent.intent == "optical_sar_crossmodal"
        assert intent.requires_cross_modal is True
        assert "optical" in intent.modalities and "sar" in intent.modalities
        plan = ap.create_plan(query, intent, {"image_optical": "b64", "image_sar": "b64"})
        tools = [s.tool for s in plan.steps]
        assert "optical_sar_analyzer" in tools

    # ── Test 6: Missing SAR Image Rejection ───────────────────────────────────
    def test_06_missing_sar_image_rejection(self):
        qi = get_query_interpreter()
        iv = get_input_validator()
        query = "Use the optical and SAR images together to identify built-up and water-covered regions."
        intent = qi.interpret(query)
        # Provided optical but omitted SAR
        res = iv.validate_request(query, intent, {"image_optical": _create_test_image_b64()})
        assert res.is_valid is False
        assert res.error_code == "MISSING_SAR_IMAGE"
        assert "both optical and sar" in res.message.lower()

    # ── Test 7: Missing Temporal Image Rejection ──────────────────────────────
    def test_07_missing_temporal_image_rejection(self):
        qi = get_query_interpreter()
        iv = get_input_validator()
        query = "What changed between these two dates?"
        intent = qi.interpret(query)
        # Provided only 1 image
        res = iv.validate_request(query, intent, {"image_t1": _create_test_image_b64()})
        assert res.is_valid is False
        assert res.error_code == "MISSING_TEMPORAL_IMAGE"
        assert "two images" in res.message.lower()

    # ── Test 8: Grounding Query Tool Selection ────────────────────────────────
    def test_08_grounding_specialist_selection(self):
        qi = get_query_interpreter()
        ap = get_agent_planner()
        for q in ["Where are the buildings?", "Locate the roads.", "Highlight the vehicles."]:
            intent = qi.interpret(q)
            assert intent.intent == "object_grounding"
            plan = ap.create_plan(q, intent, {})
            tools = [s.tool for s in plan.steps]
            assert "grounding_specialist" in tools

    # ── Test 9: Complex Multi-Tool Query Planning ────────────────────────────
    def test_09_complex_multitool_query(self):
        qi = get_query_interpreter()
        ap = get_agent_planner()
        query = "Find the buildings, determine whether built-up area increased between the two dates, and explain the evidence using optical and SAR."
        intent = qi.interpret(query)
        assert intent.intent == "complex_multitool"
        assert intent.requires_change_detection is True
        assert intent.requires_cross_modal is True
        assert intent.requires_grounding is True

        plan = ap.create_plan(query, intent, {
            "image_t1": "b64",
            "image_t2": "b64",
            "image_optical": "b64",
            "image_sar": "b64",
        })
        tools = [s.tool for s in plan.steps]
        assert len(tools) >= 3
        assert "change_analyzer" in tools
        assert "optical_sar_analyzer" in tools
        assert "grounding_specialist" in tools
        assert plan.combiner_required is True

    # ── Test 10: Execution Trace Verification ────────────────────────────────
    @pytest.mark.asyncio
    async def test_10_execution_trace_generated(self):
        img_b64 = _create_test_image_b64()
        query = "Describe the land-cover and major objects visible in this image."
        res = await orchestrate_agentic_request(query, {"image": img_b64})

        assert res["status"] == "success"
        trace = res.get("execution_trace")
        assert trace is not None
        assert len(trace) >= 5

        actions = [t["action"] for t in trace]
        assert "query_interpretation" in actions
        assert "input_validation" in actions
        assert "tool_selection" in actions
        assert "tool_execution" in actions
        assert "evidence_combination" in actions
        assert "answer_generation" in actions

    # ── Test 11: Transparent Confidence & Evidence Basis ─────────────────────
    @pytest.mark.asyncio
    async def test_11_confidence_has_transparent_basis(self):
        img_b64 = _create_test_image_b64()
        query = "Is there water in this image?"
        res = await orchestrate_agentic_request(query, {"image": img_b64})

        assert res["status"] == "success"
        conf = res.get("confidence")
        assert isinstance(conf, (float, int))
        assert 0.0 <= conf <= 1.0

        details = res.get("confidence_details")
        assert details is not None
        assert "score" in details
        assert "label" in details
        assert "basis" in details
        assert len(details["basis"]) > 0

    # ── Test 12: Dynamic Non-Hardcoded Output ────────────────────────────────
    @pytest.mark.asyncio
    async def test_12_non_hardcoded_dynamic_response(self):
        img_blue = _create_test_image_b64(color=(20, 40, 180))
        img_green = _create_test_image_b64(color=(40, 200, 30))

        res1 = await orchestrate_agentic_request("Estimate vegetation using the optical image with NDVI.", {"image": img_green})
        res2 = await orchestrate_agentic_request("Identify water using NDWI in the optical image.", {"image": img_blue})

        assert res1["status"] == "success"
        assert res2["status"] == "success"
        assert res1["answer"] != res2["answer"]
        assert "vegetation" in res1["answer"].lower() or "ndvi" in res1["answer"].lower()
        assert "water" in res2["answer"].lower() or "ndwi" in res2["answer"].lower()

    # ── Test 13: Model / Tool Registry Schema & Capabilities ─────────────────
    def test_13_registry_capabilities_and_tools(self):
        registry = get_tool_registry()
        tools = registry.list_tools()
        assert len(tools) == 7

        tool_names = [t["name"] for t in tools]
        expected = [
            "grounding_specialist",
            "satellite_vlm",
            "change_analyzer",
            "optical_sar_analyzer",
            "optical_analyzer",
            "sar_analyzer",
            "bigearthnet_adapter",
        ]
        for exp in expected:
            assert exp in tool_names

        # Check capability search
        vqa_tools = registry.find_tools_by_capability("VQA")
        assert len(vqa_tools) >= 2

        ground_tools = registry.find_tools_by_capability("grounding")
        assert len(ground_tools) >= 1

    # ── Test 14: Optical-Specific Intent ─────────────────────────────────────
    def test_14_optical_specific_intent(self):
        qi = get_query_interpreter()
        ap = get_agent_planner()
        query = "Estimate vegetation using the optical image with NDVI."
        intent = qi.interpret(query)
        assert intent.intent == "optical_specific"
        plan = ap.create_plan(query, intent, {"image": "b64"})
        tools = [s.tool for s in plan.steps]
        assert "optical_analyzer" in tools

    # ── Test 15: SAR-Specific Intent ─────────────────────────────────────────
    def test_15_sar_specific_intent(self):
        qi = get_query_interpreter()
        ap = get_agent_planner()
        query = "Use SAR to identify built-up regions and analyze VV and VH backscatter."
        intent = qi.interpret(query)
        assert intent.intent == "sar_specific"
        plan = ap.create_plan(query, intent, {"image": "b64"})
        tools = [s.tool for s in plan.steps]
        assert "sar_analyzer" in tools
