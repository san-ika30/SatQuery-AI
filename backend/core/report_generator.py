"""
SatQuery AI — Report Generator
Generates PDF and JSON downloadable analysis reports.
"""
from __future__ import annotations
import io
import json
import time
from datetime import datetime, UTC
from typing import Optional

import structlog
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from schemas.models import AnalysisRequest, EvidencePackage

log = structlog.get_logger()

# Brand colours
NAVY = colors.HexColor("#0D1B2A")
CYAN = colors.HexColor("#00D4FF")
AMBER = colors.HexColor("#FFB703")
LIGHT = colors.HexColor("#E8F4F8")
GREY = colors.HexColor("#8892A4")


def generate_pdf_report(
    request: AnalysisRequest,
    evidence: EvidencePackage,
    change_map_bytes: Optional[bytes] = None,
    fusion_composite_bytes: Optional[bytes] = None,
) -> bytes:
    """
    Generate a PDF report for the analysis session.
    Returns raw PDF bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=22, textColor=NAVY, spaceAfter=6, alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=11, textColor=GREY, spaceAfter=12, alignment=TA_CENTER,
    )
    h1_style = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=14, textColor=NAVY, spaceBefore=16, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, textColor=colors.black, spaceAfter=8, leading=14,
    )
    answer_style = ParagraphStyle(
        "Answer", parent=styles["Normal"],
        fontSize=11, textColor=NAVY, spaceAfter=8, leading=16,
        backColor=LIGHT, borderPadding=(8, 8, 8, 8),
    )

    story = []

    # ── Header ───────────────────────────────────────────────────────────────
    story.append(Paragraph("🛰️ SatQuery AI", title_style))
    story.append(Paragraph("Remote Sensing Analysis Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=CYAN, spaceAfter=12))

    # ── Session Info ─────────────────────────────────────────────────────────
    story.append(Paragraph("Session Summary", h1_style))
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    info_data = [
        ["Session ID", request.session_id],
        ["Generated At", ts],
        ["Input Mode", request.input_mode.value.upper()],
        ["Task Type", evidence.task_type.value.upper()],
        ["Query", request.query],
        ["Confidence", f"{evidence.confidence * 100:.1f}% ({_conf_label(evidence.confidence)})"],
        ["Models Used", ", ".join(evidence.models_used)],
    ]
    t = Table(info_data, colWidths=[4 * cm, 13 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT),
        ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.5, GREY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # ── Answer ───────────────────────────────────────────────────────────────
    story.append(Paragraph("Analysis Result", h1_style))
    answer_clean = evidence.answer.replace("\n", "<br/>").replace("**", "")
    story.append(Paragraph(answer_clean, answer_style))
    story.append(Spacer(1, 12))

    # ── Visual Evidence ───────────────────────────────────────────────────────
    if change_map_bytes or fusion_composite_bytes:
        story.append(Paragraph("Visual Evidence", h1_style))
        img_bytes = change_map_bytes or fusion_composite_bytes
        if img_bytes:
            img_buf = io.BytesIO(img_bytes)
            rl_img = RLImage(img_buf, width=14 * cm, height=8 * cm, kind="proportional")
            story.append(rl_img)
            caption = "Change Detection Map" if change_map_bytes else "Optical–SAR Composite"
            story.append(Paragraph(f"<i>{caption}</i>", body_style))
        story.append(Spacer(1, 12))

    # ── Execution Trace ───────────────────────────────────────────────────────
    if evidence.execution_trace:
        story.append(Paragraph("Agentic Execution Trace", h1_style))
        trace_data = [["Step", "Component", "Status", "Duration"]]
        for step in evidence.execution_trace:
            trace_data.append([
                str(step.get("step", "")),
                step.get("model", step.get("component", "")),
                step.get("status", ""),
                f"{step.get('duration_ms', 0)} ms",
            ])
        t2 = Table(trace_data, colWidths=[1.5 * cm, 7 * cm, 4 * cm, 3.5 * cm])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID", (0, 0), (-1, -1), 0.5, GREY),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t2)
        story.append(Spacer(1, 12))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=GREY, spaceAfter=6))
    story.append(Paragraph(
        "Generated by SatQuery AI — ISRO SIH 2026 | "
        "For research and demonstration purposes only.",
        ParagraphStyle("Footer", parent=styles["Normal"],
                       fontSize=7, textColor=GREY, alignment=TA_CENTER)
    ))

    doc.build(story)
    return buf.getvalue()


def generate_json_report(
    request: AnalysisRequest,
    evidence: EvidencePackage,
) -> bytes:
    """Generate a machine-readable JSON report."""
    report = {
        "satquery_ai_version": "1.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "session": {
            "id": request.session_id,
            "query": request.query,
            "input_mode": request.input_mode.value,
            "images": [m.model_dump() for m in request.image_metadata],
        },
        "result": {
            "task_type": evidence.task_type.value,
            "answer": evidence.answer,
            "confidence": evidence.confidence,
            "confidence_label": _conf_label(evidence.confidence),
            "models_used": evidence.models_used,
        },
        "visual_evidence": evidence.visual_evidence,
        "execution_trace": evidence.execution_trace,
        "report_key": evidence.report_key,
    }
    return json.dumps(report, indent=2, default=str).encode("utf-8")


def _conf_label(score: float) -> str:
    if score >= 0.85:
        return "Very High"
    if score >= 0.70:
        return "High"
    if score >= 0.50:
        return "Moderate"
    if score >= 0.30:
        return "Low"
    return "Very Low"
