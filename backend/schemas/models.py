"""
Pydantic schemas for SatQuery AI request/response models.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class InputMode(str, Enum):
    single = "single"
    bitemporal = "bitemporal"
    crossmodal = "crossmodal"


class TaskType(str, Enum):
    vqa = "vqa"
    caption = "caption"
    grounding = "grounding"
    change_detection = "change_detection"
    change_vqa = "change_vqa"
    sar_fusion = "sar_fusion"


class Modality(str, Enum):
    optical = "optical"
    sar = "sar"
    multispectral = "multispectral"


class AnalysisStatus(str, Enum):
    pending = "pending"
    preprocessing = "preprocessing"
    routing = "routing"
    executing = "executing"
    integrating = "integrating"
    completed = "completed"
    failed = "failed"


# ── Image Metadata ─────────────────────────────────────────────────────────────

class ImageMetadata(BaseModel):
    filename: str
    modality: Optional[Modality] = None
    acquisition_date: Optional[str] = None
    crs: Optional[str] = None
    bounds: Optional[list[float]] = None   # [west, south, east, north]
    width: int
    height: int
    band_count: int
    dtype: str


# ── Analysis Request ───────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    session_id: str
    query: str = Field(..., min_length=3, max_length=2000)
    input_mode: InputMode
    image_keys: list[str] = Field(..., min_length=1, max_length=2,
                                   description="R2 object keys for uploaded images")
    image_metadata: list[ImageMetadata]


# ── Task Plan (from Agentic Controller) ───────────────────────────────────────

class TaskPlan(BaseModel):
    task_type: TaskType
    input_mode: InputMode
    models: list[str]
    parameters: dict[str, Any] = {}
    reasoning: str


# ── Model Output ──────────────────────────────────────────────────────────────

class ModelOutput(BaseModel):
    model_name: str
    task_type: TaskType
    answer_text: Optional[str] = None
    confidence: Optional[float] = None
    bounding_boxes: Optional[list[dict]] = None   # [{label, x, y, w, h, score}]
    change_map_key: Optional[str] = None           # R2 key for overlay image
    raw_output: Optional[dict] = None


# ── Evidence Package ──────────────────────────────────────────────────────────

class EvidencePackage(BaseModel):
    answer: str
    confidence: float
    task_type: TaskType
    models_used: list[str]
    visual_evidence: list[dict] = []  # [{type, r2_key, description}]
    execution_trace: list[dict] = []  # [{step, model, status, duration_ms}]
    report_key: Optional[str] = None  # R2 key for downloadable report


# ── Analysis Response ─────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    session_id: str
    status: AnalysisStatus
    task_plan: Optional[TaskPlan] = None
    evidence: Optional[EvidencePackage] = None
    error: Optional[str] = None


# ── Health Check ──────────────────────────────────────────────────────────────

class ModelStatus(BaseModel):
    name: str
    available: bool
    endpoint: str
    task: str


class HealthResponse(BaseModel):
    status: str
    version: str
    models: list[ModelStatus]
