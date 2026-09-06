"""
SatQuery AI — Main Analysis Router
Orchestrates the full pipeline: upload → preprocess → classify → execute → integrate → report
"""
from __future__ import annotations
import asyncio
import json
import time
import uuid
import structlog
from typing import Optional, AsyncGenerator
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import Response, StreamingResponse

from core.preprocessor import geotiff_to_rgb_png, validate_image_pair, PreprocessingError
from core.controller import classify_task
from core.model_registry import run_geochat, run_grounding_dino, run_changeformer, run_sar_fusion
from core.evidence_integrator import integrate_outputs
from core.report_generator import generate_pdf_report, generate_json_report
from storage.r2_client import upload_image, upload_report_pdf, upload_report_json, get_public_url
from db.supabase_client import (
    log_session, update_session_status, update_session_task_type, log_image, log_result
)
from schemas.models import (
    AnalysisResponse, AnalysisStatus, InputMode,
    TaskType, ImageMetadata, AnalysisRequest,
)

router = APIRouter()
log = structlog.get_logger()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    query: str = Form(...),
    input_mode: str = Form(...),
    image1: UploadFile = File(...),
    image2: Optional[UploadFile] = File(None),
):
    """
    Main analysis endpoint.

    - image1: primary image (always required)
    - image2: second image for bi-temporal or cross-modal pairs
    - query: natural language question
    - input_mode: "single" | "bitemporal" | "crossmodal"
    """
    session_id = uuid.uuid4().hex
    t_start = time.monotonic()
    execution_trace: list[dict] = []

    def _trace(step: int, component: str, status: str, extra: dict = {}):
        elapsed = round((time.monotonic() - t_start) * 1000)
        entry = {"step": step, "component": component, "status": status,
                 "duration_ms": elapsed, **extra}
        execution_trace.append(entry)
        log.info("trace", **entry)

    # ── Validate input_mode ───────────────────────────────────────────────────
    try:
        mode = InputMode(input_mode)
    except ValueError:
        raise HTTPException(400, f"Invalid input_mode: {input_mode}. "
                                 "Must be 'single', 'bitemporal', or 'crossmodal'.")

    if mode in (InputMode.bitemporal, InputMode.crossmodal) and image2 is None:
        raise HTTPException(400, f"input_mode='{input_mode}' requires two images.")

    # ── Log session ───────────────────────────────────────────────────────────
    await log_session(session_id, mode.value, query, status="pending")
    _trace(1, "Session Init", "ok", {"session_id": session_id})

    # ── Read image bytes ──────────────────────────────────────────────────────
    img1_bytes = await image1.read()
    img2_bytes = await image2.read() if image2 else None
    _trace(2, "Image Read", "ok", {"files": [image1.filename, image2.filename if image2 else None]})

    # ── Preprocess ────────────────────────────────────────────────────────────
    await update_session_status(session_id, AnalysisStatus.preprocessing.value)
    try:
        png1_bytes, meta1 = geotiff_to_rgb_png(img1_bytes, image1.filename)
        _trace(3, "Preprocessor: Image 1", "ok",
               {"modality": meta1.modality.value if meta1.modality else "unknown",
                "size": f"{meta1.width}x{meta1.height}"})

        meta2 = None
        png2_bytes = None
        if img2_bytes and image2:
            png2_bytes, meta2 = geotiff_to_rgb_png(img2_bytes, image2.filename)
            _trace(4, "Preprocessor: Image 2", "ok",
                   {"modality": meta2.modality.value if meta2.modality else "unknown"})

            # Validate pair compatibility
            warnings = validate_image_pair(meta1, meta2, mode.value)
            if warnings:
                _trace(5, "Compatibility Check", "warning", {"warnings": warnings})
            else:
                _trace(5, "Compatibility Check", "ok")
        else:
            _trace(4, "Preprocessor: Image 2", "skipped")
            _trace(5, "Compatibility Check", "skipped")

    except PreprocessingError as exc:
        raise HTTPException(422, f"Image preprocessing failed: {exc}")

    # Upload original images to R2
    r2_key1 = upload_image(png1_bytes, session_id, "img1")
    r2_key2 = upload_image(png2_bytes, session_id, "img2") if png2_bytes else None
    await log_image(session_id, image1.filename, meta1.modality.value if meta1.modality else None,
                    r2_key1, meta1.model_dump())
    # Log image2 metadata when present (bitemporal / crossmodal sessions)
    if png2_bytes and meta2 and r2_key2 and image2:
        await log_image(session_id, image2.filename,
                        meta2.modality.value if meta2.modality else None,
                        r2_key2, meta2.model_dump())

    # ── Agentic Classification ────────────────────────────────────────────────
    await update_session_status(session_id, AnalysisStatus.routing.value)
    task_plan, classify_method = await classify_task(query, mode)
    await update_session_task_type(session_id, task_plan.task_type.value)
    _trace(6, f"Agentic Controller ({classify_method})", "ok",
           {"task_type": task_plan.task_type.value, "models": task_plan.models})

    # ── Model Execution ───────────────────────────────────────────────────────
    await update_session_status(session_id, AnalysisStatus.executing.value)

    model_outputs = []
    change_map_bytes: Optional[bytes] = None
    fusion_composite_bytes: Optional[bytes] = None

    task = task_plan.task_type

    # Route to correct specialist tool
    if task == TaskType.vqa:
        if "grounding_dino" in task_plan.models:
            import base64
            from core.orchestrator import call_grounding_specialist, derive_grounding_prompt
            b64_str = base64.b64encode(png1_bytes).decode("utf-8")
            grounding_prompt = derive_grounding_prompt(query)
            try:
                _, detections = await call_grounding_specialist(b64_str, grounding_prompt)
            except Exception as g_exc:
                log.warning("vqa_grounding_specialist_failed", error=str(g_exc))
                detections = []
            _trace(7, "GroundingDINO [Grounding]", "ok", {
                "n_boxes": len(detections),
                "labels": [d.get("label") for d in detections]
            })
            out = await run_geochat(png1_bytes, query, TaskType.vqa, detections=detections)
            model_outputs.append(out)
            _trace(8, out.model_name, "ok", {"confidence": out.confidence})
        else:
            out = await run_geochat(png1_bytes, query, TaskType.vqa)
            model_outputs.append(out)
            _trace(7, out.model_name, "ok", {"confidence": out.confidence})

    elif task == TaskType.caption:
        out = await run_geochat(png1_bytes, query or "Describe this satellite image.", TaskType.caption)
        model_outputs.append(out)
        _trace(7, out.model_name, "ok", {"confidence": out.confidence})

    elif task == TaskType.grounding:
        out = await run_grounding_dino(png1_bytes, query)
        model_outputs.append(out)
        _trace(7, f"GroundingDINO [Grounding]", "ok",
               {"n_boxes": len(out.bounding_boxes or [])})

    elif task in (TaskType.change_detection, TaskType.change_vqa):
        if png2_bytes is None:
            raise HTTPException(400, "Change analysis requires two images.")
        q = query if task == TaskType.change_vqa else None
        result_tuple = await run_changeformer(png1_bytes, png2_bytes, q)
        out, change_map_bytes = result_tuple
        model_outputs.append(out)
        _trace(7, f"ChangeFormer [{task.value}]", "ok", {"confidence": out.confidence})

    elif task == TaskType.sar_fusion:
        if png2_bytes is None:
            raise HTTPException(400, "SAR fusion analysis requires two images (optical + SAR).")
        result_tuple = await run_sar_fusion(png1_bytes, png2_bytes, query)
        out, fusion_composite_bytes = result_tuple
        model_outputs.append(out)
        _trace(7, f"SAR Fusion Encoder", "ok", {"confidence": out.confidence})

    # ── Upload Visual Evidence ────────────────────────────────────────────────
    change_map_r2_key: Optional[str] = None
    fusion_composite_r2_key: Optional[str] = None

    if change_map_bytes:
        change_map_r2_key = upload_image(change_map_bytes, session_id, "change_map")
        _trace(8, "R2 Upload: Change Map", "ok", {"key": change_map_r2_key})
    if fusion_composite_bytes:
        fusion_composite_r2_key = upload_image(fusion_composite_bytes, session_id, "sar_composite")
        _trace(8, "R2 Upload: SAR Composite", "ok", {"key": fusion_composite_r2_key})

    # ── Evidence Integration ──────────────────────────────────────────────────
    await update_session_status(session_id, AnalysisStatus.integrating.value)
    evidence = integrate_outputs(
        task_plan=task_plan,
        outputs=model_outputs,
        change_map_r2_key=change_map_r2_key,
        fusion_composite_r2_key=fusion_composite_r2_key,
        execution_trace=execution_trace,
    )
    _trace(9, "Evidence Integrator", "ok", {"confidence": evidence.confidence})

    # ── Generate Reports ──────────────────────────────────────────────────────
    req_obj = AnalysisRequest(
        session_id=session_id,
        query=query,
        input_mode=mode,
        image_keys=[r2_key1] + ([r2_key2] if r2_key2 else []),
        image_metadata=[meta1] + ([meta2] if meta2 else []),
    )

    try:
        pdf_bytes = generate_pdf_report(
            req_obj, evidence,
            change_map_bytes=change_map_bytes,
            fusion_composite_bytes=fusion_composite_bytes,
        )
        report_key = upload_report_pdf(pdf_bytes, session_id)
        evidence.report_key = report_key
        _trace(10, "Report Generator", "ok", {"key": report_key})
    except Exception as exc:
        log.warning("report_generation_failed", error=str(exc))
        _trace(10, "Report Generator", "failed", {"error": str(exc)})

    # Add public URLs to evidence
    for ev in evidence.visual_evidence:
        if "r2_key" in ev:
            ev["public_url"] = get_public_url(ev["r2_key"])
    if evidence.report_key:
        evidence.report_key = get_public_url(evidence.report_key)

    # ── Persist result ────────────────────────────────────────────────────────
    await update_session_status(session_id, AnalysisStatus.completed.value)
    await log_result(
        session_id,
        task_plan.task_type.value,
        ", ".join(evidence.models_used),
        evidence.answer,
        evidence.confidence,
        change_map_r2_key or fusion_composite_r2_key,
        evidence.report_key,
        execution_trace,
    )

    total_ms = round((time.monotonic() - t_start) * 1000)
    log.info("analysis_complete", session_id=session_id, total_ms=total_ms,
             task=task_plan.task_type.value)

    return AnalysisResponse(
        session_id=session_id,
        status=AnalysisStatus.completed,
        task_plan=task_plan,
        evidence=evidence,
    )


@router.get("/session/{session_id}/report.pdf")
async def download_pdf_report(session_id: str):
    """Download the PDF report for a session (if stored locally)."""
    # In production this redirects to R2 public URL
    raise HTTPException(404, "Report not found locally. Check R2 public URL.")


@router.get("/session/{session_id}/report.json")
async def download_json_report(session_id: str):
    """Placeholder for JSON report download."""
    raise HTTPException(404, "Report not found locally. Check R2 public URL.")


# ── SSE Streaming Analysis Endpoint ──────────────────────────────────────────

def _sse_event(event_type: str, data: dict) -> str:
    """Format a Server-Sent Events message."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@router.post("/analyze/stream")
async def analyze_stream(
    query: str = Form(...),
    input_mode: str = Form(...),
    image1: UploadFile = File(...),
    image2: Optional[UploadFile] = File(None),
):
    """
    SSE streaming analysis endpoint.
    Yields progress events as the pipeline executes.

    Events:
      - progress: {step, message, status} — pipeline step updates
      - result:   AnalysisResponse JSON    — final result
      - error:    {detail}                 — on failure
    """
    # Read all file bytes eagerly (before generator starts)
    img1_bytes = await image1.read()
    img2_bytes = await image2.read() if image2 else None
    img1_filename = image1.filename or "image1"
    img2_filename = image2.filename if image2 else None

    async def event_generator() -> AsyncGenerator[str, None]:
        session_id = uuid.uuid4().hex
        t_start = time.monotonic()
        execution_trace: list[dict] = []

        def _trace(step: int, component: str, status: str, extra: dict = {}):
            elapsed = round((time.monotonic() - t_start) * 1000)
            entry = {"step": step, "component": component, "status": status,
                     "duration_ms": elapsed, **extra}
            execution_trace.append(entry)
            log.info("sse_trace", **entry)

        async def _progress(step: int, msg: str, pct: int = 0):
            """Yield a progress SSE event."""
            yield _sse_event("progress", {"step": step, "message": msg, "pct": pct,
                                          "session_id": session_id})
            # Brief yield to allow the event to flush
            await asyncio.sleep(0)

        try:
            # ── Validate input_mode ───────────────────────────────────────
            try:
                mode = InputMode(input_mode)
            except ValueError:
                yield _sse_event("error", {"detail": f"Invalid input_mode: {input_mode}"})
                return

            if mode in (InputMode.bitemporal, InputMode.crossmodal) and img2_bytes is None:
                yield _sse_event("error", {"detail": f"input_mode='{input_mode}' requires two images."})
                return

            async for ev in _progress(1, "Initialising session…", 5):
                yield ev

            await log_session(session_id, mode.value, query, status="pending")
            _trace(1, "Session Init", "ok", {"session_id": session_id})

            # ── Preprocess ────────────────────────────────────────────────
            async for ev in _progress(2, "Preprocessing GeoTIFF bands…", 15):
                yield ev
            await update_session_status(session_id, AnalysisStatus.preprocessing.value)

            try:
                png1_bytes, meta1 = geotiff_to_rgb_png(img1_bytes, img1_filename)
                _trace(2, "Preprocessor: Image 1", "ok",
                       {"modality": meta1.modality.value if meta1.modality else "unknown",
                        "size": f"{meta1.width}x{meta1.height}"})

                meta2 = None
                png2_bytes = None
                if img2_bytes and img2_filename:
                    png2_bytes, meta2 = geotiff_to_rgb_png(img2_bytes, img2_filename)
                    _trace(3, "Preprocessor: Image 2", "ok",
                           {"modality": meta2.modality.value if meta2.modality else "unknown"})
                    warnings = validate_image_pair(meta1, meta2, mode.value)
                    if warnings:
                        _trace(4, "Compatibility Check", "warning", {"warnings": warnings})
                    else:
                        _trace(4, "Compatibility Check", "ok")
                else:
                    _trace(3, "Preprocessor: Image 2", "skipped")
                    _trace(4, "Compatibility Check", "skipped")

            except PreprocessingError as exc:
                yield _sse_event("error", {"detail": f"Image preprocessing failed: {exc}"})
                return

            r2_key1 = upload_image(png1_bytes, session_id, "img1")
            r2_key2 = upload_image(png2_bytes, session_id, "img2") if png2_bytes else None
            await log_image(session_id, img1_filename, meta1.modality.value if meta1.modality else None,
                            r2_key1, meta1.model_dump())
            # Log image2 metadata when present (bitemporal / crossmodal sessions)
            if png2_bytes and meta2 and r2_key2 and img2_filename:
                await log_image(session_id, img2_filename,
                                meta2.modality.value if meta2.modality else None,
                                r2_key2, meta2.model_dump())

            # ── Agentic Classification ────────────────────────────────────
            async for ev in _progress(3, "Running agentic task router (Mistral-7B)…", 30):
                yield ev
            await update_session_status(session_id, AnalysisStatus.routing.value)
            task_plan, classify_method = await classify_task(query, mode)
            await update_session_task_type(session_id, task_plan.task_type.value)
            _trace(5, f"Agentic Controller ({classify_method})", "ok",
                   {"task_type": task_plan.task_type.value, "models": task_plan.models})

            async for ev in _progress(4, f"Routing to {', '.join(task_plan.models)}…", 40):
                yield ev

            # ── Model Execution ───────────────────────────────────────────
            async for ev in _progress(5, f"Executing specialist model inference…", 55):
                yield ev
            await update_session_status(session_id, AnalysisStatus.executing.value)

            model_outputs = []
            change_map_bytes_sse: Optional[bytes] = None
            fusion_composite_bytes_sse: Optional[bytes] = None
            task = task_plan.task_type

            if task == TaskType.vqa:
                if "grounding_dino" in task_plan.models:
                    import base64
                    from core.orchestrator import call_grounding_specialist, derive_grounding_prompt
                    async for ev in _progress(5, "Executing GroundingDINO object detection…", 55):
                        yield ev
                    b64_str = base64.b64encode(png1_bytes).decode("utf-8")
                    grounding_prompt = derive_grounding_prompt(query)
                    try:
                        _, detections = await call_grounding_specialist(b64_str, grounding_prompt)
                    except Exception as g_exc:
                        log.warning("vqa_grounding_specialist_stream_failed", error=str(g_exc))
                        detections = []
                    _trace(6, "GroundingDINO [Grounding]", "ok", {
                        "n_boxes": len(detections),
                        "labels": [d.get("label") for d in detections]
                    })
                    async for ev in _progress(6, "Executing Satellite Spatial Reasoning Specialist…", 65):
                        yield ev
                    out = await run_geochat(png1_bytes, query, TaskType.vqa, detections=detections)
                    model_outputs.append(out)
                    _trace(7, out.model_name, "ok", {"confidence": out.confidence})
                else:
                    out = await run_geochat(png1_bytes, query, TaskType.vqa)
                    model_outputs.append(out)
                    _trace(6, out.model_name, "ok", {"confidence": out.confidence})

            elif task == TaskType.caption:
                out = await run_geochat(png1_bytes, query or "Describe this satellite image.", TaskType.caption)
                model_outputs.append(out)
                _trace(6, out.model_name, "ok", {"confidence": out.confidence})

            elif task == TaskType.grounding:
                out = await run_grounding_dino(png1_bytes, query)
                model_outputs.append(out)
                _trace(6, "GroundingDINO [Grounding]", "ok", {"n_boxes": len(out.bounding_boxes or [])})

            elif task in (TaskType.change_detection, TaskType.change_vqa):
                if png2_bytes is None:
                    yield _sse_event("error", {"detail": "Change analysis requires two images."})
                    return
                q = query if task == TaskType.change_vqa else None
                out, change_map_bytes_sse = await run_changeformer(png1_bytes, png2_bytes, q)
                model_outputs.append(out)
                _trace(6, f"ChangeFormer [{task.value}]", "ok", {"confidence": out.confidence})

            elif task == TaskType.sar_fusion:
                if png2_bytes is None:
                    yield _sse_event("error", {"detail": "SAR fusion requires two images."})
                    return
                out, fusion_composite_bytes_sse = await run_sar_fusion(png1_bytes, png2_bytes, query)
                model_outputs.append(out)
                _trace(6, "SAR Fusion Encoder", "ok", {"confidence": out.confidence})

            # ── Upload Visual Evidence ─────────────────────────────────────
            async for ev in _progress(6, "Uploading visual evidence to cloud storage…", 70):
                yield ev

            change_map_r2_key: Optional[str] = None
            fusion_composite_r2_key: Optional[str] = None

            if change_map_bytes_sse:
                change_map_r2_key = upload_image(change_map_bytes_sse, session_id, "change_map")
                _trace(7, "R2 Upload: Change Map", "ok", {"key": change_map_r2_key})
            if fusion_composite_bytes_sse:
                fusion_composite_r2_key = upload_image(fusion_composite_bytes_sse, session_id, "sar_composite")
                _trace(7, "R2 Upload: SAR Composite", "ok", {"key": fusion_composite_r2_key})

            # ── Evidence Integration ───────────────────────────────────────
            async for ev in _progress(7, "Integrating evidence and computing confidence…", 82):
                yield ev
            await update_session_status(session_id, AnalysisStatus.integrating.value)
            evidence = integrate_outputs(
                task_plan=task_plan,
                outputs=model_outputs,
                change_map_r2_key=change_map_r2_key,
                fusion_composite_r2_key=fusion_composite_r2_key,
                execution_trace=execution_trace,
            )
            _trace(8, "Evidence Integrator", "ok", {"confidence": evidence.confidence})

            # ── Generate Reports ───────────────────────────────────────────
            async for ev in _progress(8, "Generating PDF analysis report…", 92):
                yield ev

            req_obj = AnalysisRequest(
                session_id=session_id,
                query=query,
                input_mode=mode,
                image_keys=[r2_key1] + ([r2_key2] if r2_key2 else []),
                image_metadata=[meta1] + ([meta2] if meta2 else []),
            )

            try:
                pdf_bytes = generate_pdf_report(
                    req_obj, evidence,
                    change_map_bytes=change_map_bytes_sse,
                    fusion_composite_bytes=fusion_composite_bytes_sse,
                )
                report_key = upload_report_pdf(pdf_bytes, session_id)
                evidence.report_key = report_key
                _trace(9, "Report Generator", "ok", {"key": report_key})
            except Exception as exc:
                log.warning("sse_report_generation_failed", error=str(exc))
                _trace(9, "Report Generator", "failed", {"error": str(exc)})

            # Add public URLs
            for ev_item in evidence.visual_evidence:
                if "r2_key" in ev_item:
                    ev_item["public_url"] = get_public_url(ev_item["r2_key"])
            if evidence.report_key:
                evidence.report_key = get_public_url(evidence.report_key)

            # ── Persist result ─────────────────────────────────────────────
            await update_session_status(session_id, AnalysisStatus.completed.value)
            await log_result(
                session_id,
                task_plan.task_type.value,
                ", ".join(evidence.models_used),
                evidence.answer,
                evidence.confidence,
                change_map_r2_key or fusion_composite_r2_key,
                evidence.report_key,
                execution_trace,
            )

            total_ms = round((time.monotonic() - t_start) * 1000)
            log.info("sse_analysis_complete", session_id=session_id, total_ms=total_ms,
                     task=task_plan.task_type.value)

            response_data = AnalysisResponse(
                session_id=session_id,
                status=AnalysisStatus.completed,
                task_plan=task_plan,
                evidence=evidence,
            )

            async for ev in _progress(9, "Analysis complete!", 100):
                yield ev

            yield _sse_event("result", response_data.model_dump())

        except Exception as exc:
            log.error("sse_pipeline_error", error=str(exc))
            yield _sse_event("error", {"detail": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
