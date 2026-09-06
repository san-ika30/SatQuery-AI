"""
SatQuery AI — Health Check Router
"""
import os
import httpx
import structlog
from fastapi import APIRouter

from schemas.models import HealthResponse, ModelStatus

router = APIRouter()
log = structlog.get_logger()

_HF_TOKEN_RAW = os.getenv("HF_API_TOKEN", "")
# Treat placeholder / unset tokens as empty
HF_API_TOKEN = _HF_TOKEN_RAW if _HF_TOKEN_RAW.startswith("hf_") and len(_HF_TOKEN_RAW) > 10 else ""
HF_GEOCHAT_MODEL = os.getenv("HF_GEOCHAT_MODEL", "MBZUAI/GeoChat")
HF_BLIP2_MODEL = os.getenv("HF_BLIP2_MODEL", "Salesforce/blip2-opt-2.7b")
HF_GROUNDING_DINO_MODEL = os.getenv("HF_GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-tiny")
HF_CHANGEFORMER_SPACE = os.getenv("HF_CHANGEFORMER_SPACE", "")


async def _check_hf_model(model_id: str) -> bool:
    """Ping HF model to check availability."""
    if not HF_API_TOKEN:
        return False
    try:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {HF_API_TOKEN}"})
            return resp.status_code in (200, 503)  # 503 = loading = exists
    except Exception:
        return False


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint returning system and model availability."""
    geo_ok = await _check_hf_model(HF_GEOCHAT_MODEL)
    blip_ok = await _check_hf_model(HF_BLIP2_MODEL)
    gd_ok = await _check_hf_model(HF_GROUNDING_DINO_MODEL)

    models = [
        ModelStatus(
            name="GeoChat-7B",
            available=geo_ok,
            endpoint=HF_GEOCHAT_MODEL,
            task="VQA / Captioning (primary)",
        ),
        ModelStatus(
            name="BLIP2-OPT-2.7B",
            available=blip_ok,
            endpoint=HF_BLIP2_MODEL,
            task="VQA / Captioning (fallback)",
        ),
        ModelStatus(
            name="GroundingDINO-Tiny",
            available=gd_ok,
            endpoint=HF_GROUNDING_DINO_MODEL,
            task="Region Grounding",
        ),
        ModelStatus(
            name="ChangeFormer",
            available=bool(HF_CHANGEFORMER_SPACE),
            endpoint=HF_CHANGEFORMER_SPACE or "pixel-diff-fallback",
            task="Change Detection / Change-VQA",
        ),
        ModelStatus(
            name="SAR-Optical Dual Encoder",
            available=True,  # always available (composite fusion)
            endpoint="composite-channel-fusion",
            task="Optical–SAR Fusion",
        ),
    ]

    overall_status = "operational" if (geo_ok or blip_ok) else "degraded"

    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        models=models,
    )


@router.get("/models")
async def list_models():
    """List all registered specialist models."""
    return {
        "models": [
            {"id": "geochat", "name": "GeoChat-7B", "tasks": ["vqa", "caption", "change_vqa", "sar_fusion"],
             "description": "Remote-sensing fine-tuned VLM for VQA and captioning"},
            {"id": "grounding_dino", "name": "GroundingDINO-Tiny", "tasks": ["grounding"],
             "description": "Open-vocabulary object detection for text-guided region grounding"},
            {"id": "changeformer", "name": "ChangeFormer", "tasks": ["change_detection", "change_vqa"],
             "description": "Transformer-based bi-temporal change detection"},
            {"id": "sar_fusion_encoder", "name": "SAR-Optical Dual Encoder", "tasks": ["sar_fusion"],
             "description": "Channel-fusion encoder for joint optical–SAR analysis"},
        ]
    }
