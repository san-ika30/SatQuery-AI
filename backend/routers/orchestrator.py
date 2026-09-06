"""
SatQuery AI — Unified Orchestration Router
Provides POST /api/satquery and GET /api/satquery/health
"""
import os
import httpx
import structlog
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from core.orchestrator import (
    orchestrate_satquery_request,
    orchestrate_bitemporal_satquery_request,
    orchestrate_crossmodal_satquery_request,
    orchestrate_agentic_request,
    get_grounding_module,
    get_vlm_module,
    GROUNDING_SERVICE_URL,
    VLM_SERVICE_URL,
)

log = structlog.get_logger()
router = APIRouter()


class SatQueryRequest(BaseModel):
    image: Optional[str] = Field(None, description="Base64-encoded satellite image (single image or T1)")
    image_t1: Optional[str] = Field(None, description="Base64-encoded pre-change satellite image (T1)")
    image_t2: Optional[str] = Field(None, description="Base64-encoded post-change satellite image (T2)")
    image_optical: Optional[str] = Field(None, description="Base64-encoded optical satellite image")
    image_sar: Optional[str] = Field(None, description="Base64-encoded SAR satellite image")
    modality: Optional[str] = Field(None, description="Modality hint: 'optical', 'sar', 'crossmodal'")
    question: str = Field(..., description="Natural-language user query")


class SatQueryResponse(BaseModel):
    status: str
    task: Optional[str] = None
    query: Optional[str] = None
    interpreted_intent: Optional[dict] = None
    selected_tools: Optional[list[str]] = None
    inputs: Optional[int] = None
    modalities: Optional[list[str]] = None
    answer: Optional[str] = None
    detections: Optional[list[dict]] = None
    overlay: Optional[str] = None
    change_ratio: Optional[float] = None
    severity: Optional[str] = None
    change_categories: Optional[list[str]] = None
    evidence: Optional[list[str]] = None
    optical_evidence: Optional[list[str]] = None
    sar_evidence: Optional[list[str]] = None
    fused_evidence: Optional[list[str]] = None
    detected_classes: Optional[list[str]] = None
    water_analysis: Optional[dict] = None
    built_up_analysis: Optional[dict] = None
    confidence: Optional[float] = None
    confidence_details: Optional[dict] = None
    claims: Optional[list[dict]] = None
    execution_trace: Optional[list[dict]] = None
    model: Optional[str] = None
    tools: Optional[list[str]] = None
    image_size: Optional[list[int]] = None
    execution_time_ms: Optional[int] = None
    error: Optional[dict] = None


@router.post(
    "/satquery",
    response_model=SatQueryResponse,
    summary="Unified SatQuery AI Multi-Microservice Endpoint",
    description=(
        "Unified orchestrator endpoint connecting Frontend -> Agentic Orchestration Layer "
        "(Query Interpreter, Input Validator, Agent Planner, Model/Tool Registry, Tool Executor, Evidence Combiner)."
    ),
)
async def unified_satquery_endpoint(req: SatQueryRequest):
    inputs = {
        "image": req.image,
        "image_t1": req.image_t1,
        "image_t2": req.image_t2,
        "image_optical": req.image_optical,
        "image_sar": req.image_sar,
        "modality": req.modality,
    }
    result = await orchestrate_agentic_request(req.question, inputs)
    return _format_router_response(result)


def _format_router_response(result: dict) -> JSONResponse | dict:
    if result.get("status") in ("error", "input_error"):
        err_code = result.get("error", {}).get("code", "INTERNAL_ERROR")
        if err_code in (
            "INVALID_IMAGE_PAYLOAD", "INVALID_IMAGE_T1_PAYLOAD", "INVALID_IMAGE_T2_PAYLOAD",
            "INVALID_OPTICAL_PAYLOAD", "INVALID_SAR_PAYLOAD", "EMPTY_QUESTION", "INVALID_IMAGE_FORMAT",
            "MISSING_TEMPORAL_IMAGE", "MISSING_SAR_IMAGE"
        ):
            return JSONResponse(status_code=400, content=result)
        elif "UNAVAILABLE" in err_code:
            return JSONResponse(status_code=503, content=result)
        else:
            return JSONResponse(status_code=500, content=result)
    return result



@router.get(
    "/satquery/health",
    summary="Multi-Microservice Health Check",
    description="Check operational availability of Grounding Specialist, VLM Specialist, and ChangeFormer.",
)
async def satquery_health_check():
    services_status = {}

    # Check Grounding Specialist
    try:
        get_grounding_module()
        services_status["grounding"] = "available"
    except Exception as exc:
        services_status["grounding"] = f"unavailable: {str(exc)}"

    # Check VLM Specialist
    try:
        get_vlm_module()
        services_status["vlm"] = "available"
    except Exception as exc:
        services_status["vlm"] = f"unavailable: {str(exc)}"

    # Check ChangeFormer
    try:
        from core.model_registry import run_changeformer
        services_status["changeformer"] = "available"
    except Exception as exc:
        services_status["changeformer"] = f"unavailable: {str(exc)}"

    overall = "healthy" if services_status.get("grounding") == "available" and services_status.get("vlm") == "available" else "degraded"

    return {
        "status": overall,
        "services": services_status,
        "remote_endpoints": {
            "grounding_url": GROUNDING_SERVICE_URL,
            "vlm_url": VLM_SERVICE_URL,
        },
    }
