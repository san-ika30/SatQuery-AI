"""
SatQuery AI — Evidence Integrator
Merges outputs from multiple specialist models, computes aggregate confidence,
attaches visual evidence, and builds the execution trace.
"""
from __future__ import annotations
import time
import structlog
from typing import Optional

from schemas.models import (
    EvidencePackage,
    ModelOutput,
    TaskPlan,
    TaskType,
)

log = structlog.get_logger()


def _confidence_label(score: float) -> str:
    if score >= 0.85:
        return "Very High"
    if score >= 0.70:
        return "High"
    if score >= 0.50:
        return "Moderate"
    if score >= 0.30:
        return "Low"
    return "Very Low"


def _format_grounding_answer(output: ModelOutput) -> str:
    """Format grounding bounding boxes into a readable answer."""
    if not output.bounding_boxes:
        return output.answer_text or "No regions detected."

    lines = [output.answer_text or "Detected regions:"]
    for i, box in enumerate(output.bounding_boxes[:8], 1):
        lines.append(
            f"  {i}. {box['label']} — "
            f"[{box['xmin']}, {box['ymin']}, {box['xmax']}, {box['ymax']}] "
            f"(confidence: {box['score']:.2f})"
        )
    return "\n".join(lines)


def integrate_outputs(
    task_plan: TaskPlan,
    outputs: list[ModelOutput],
    change_map_r2_key: Optional[str] = None,
    fusion_composite_r2_key: Optional[str] = None,
    report_r2_key: Optional[str] = None,
    execution_trace: Optional[list[dict]] = None,
) -> EvidencePackage:
    """
    Merge all model outputs into a final EvidencePackage.
    """
    if not outputs:
        return EvidencePackage(
            answer="No model outputs were produced. Please check input images and retry.",
            confidence=0.0,
            task_type=task_plan.task_type,
            models_used=[],
            execution_trace=execution_trace or [],
        )

    # Aggregate answers
    valid_outputs = [o for o in outputs if o.answer_text]
    models_used = [o.model_name for o in outputs]

    # Compute aggregate confidence (weighted average)
    confs = [o.confidence for o in outputs if o.confidence is not None]
    agg_confidence = round(sum(confs) / len(confs), 3) if confs else 0.5

    # Format primary answer
    if task_plan.task_type == TaskType.grounding:
        primary = next((o for o in outputs if o.task_type == TaskType.grounding), None)
        answer = _format_grounding_answer(primary) if primary else "Grounding failed."

    elif task_plan.task_type in (TaskType.change_detection, TaskType.change_vqa):
        answers = [o.answer_text for o in valid_outputs if o.answer_text]
        answer = answers[0] if answers else "Change analysis complete."
        if change_map_r2_key:
            answer += (
                "\n\n📍 A visual change map has been generated and is attached below. "
                "Areas with higher change intensity appear in warmer colors (red/yellow)."
            )

    elif task_plan.task_type == TaskType.sar_fusion:
        answers = [o.answer_text for o in valid_outputs if o.answer_text]
        answer = answers[0] if answers else "Optical–SAR fusion analysis complete."
        if fusion_composite_r2_key:
            answer += (
                "\n\n🛰️ A side-by-side optical–SAR composite has been generated "
                "showing both modalities of the same scene."
            )

    else:  # VQA, Caption
        # Join all model answers
        if len(valid_outputs) == 1:
            answer = valid_outputs[0].answer_text
        else:
            parts = []
            for o in valid_outputs:
                parts.append(f"**{o.model_name}**: {o.answer_text}")
            answer = "\n\n".join(parts)

    # Build visual evidence list
    visual_evidence = []
    if change_map_r2_key and change_map_r2_key != "__pending_upload__":
        visual_evidence.append({
            "type": "change_map",
            "r2_key": change_map_r2_key,
            "description": "Bi-temporal change detection heatmap",
        })
    if fusion_composite_r2_key:
        visual_evidence.append({
            "type": "sar_composite",
            "r2_key": fusion_composite_r2_key,
            "description": "Optical–SAR side-by-side composite",
        })

    # Grounding bounding boxes as evidence
    for o in outputs:
        if o.bounding_boxes:
            visual_evidence.append({
                "type": "bounding_boxes",
                "boxes": o.bounding_boxes,
                "description": "Region grounding detections",
            })

    log.info(
        "evidence_integrated",
        task=task_plan.task_type.value,
        confidence=agg_confidence,
        n_models=len(outputs),
    )

    return EvidencePackage(
        answer=answer,
        confidence=agg_confidence,
        task_type=task_plan.task_type,
        models_used=models_used,
        visual_evidence=visual_evidence,
        execution_trace=execution_trace or [],
        report_key=report_r2_key,
    )
