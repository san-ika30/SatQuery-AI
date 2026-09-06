"""
SatQuery AI — Supabase Database Client
Handles session logging, metadata persistence, and result storage.
Always writes to in-memory store as well so sessions survive even without Supabase.
"""
from __future__ import annotations
import os
import structlog
from typing import Optional, Any

from db.session_store import (
    store_session, update_session, store_image, store_result,
)

log = structlog.get_logger()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY", "")

_client = None


def _get_client():
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY or "xxxxxxxxx" in SUPABASE_URL:
            return None
        try:
            from supabase import create_client
            _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as exc:
            log.error("supabase_init_failed", error=str(exc))
            return None
    return _client


async def log_session(
    session_id: str,
    input_mode: str,
    query_text: str,
    status: str = "pending",
    task_type: Optional[str] = None,
) -> None:
    """Log a new analysis session (in-memory + Supabase if configured)."""
    # Always write to in-memory store
    store_session(session_id, input_mode, query_text, status, task_type)

    client = _get_client()
    if client is None:
        log.info("supabase_unavailable_using_memory_store")
        return
    try:
        payload: dict[str, Any] = {
            "id": session_id,
            "input_mode": input_mode,
            "query_text": query_text,
            "status": status,
        }
        if task_type is not None:
            payload["task_type"] = task_type
        client.table("sessions").insert(payload).execute()
    except Exception as exc:
        log.error("supabase_session_log_failed", error=str(exc))


async def update_session_status(session_id: str, status: str) -> None:
    """Update session status."""
    update_session(session_id, status=status)

    client = _get_client()
    if client is None:
        return
    try:
        client.table("sessions").update({"status": status}).eq("id", session_id).execute()
    except Exception as exc:
        log.error("supabase_session_update_failed", error=str(exc))


async def update_session_task_type(session_id: str, task_type: str) -> None:
    """Update the classified task_type for a session."""
    update_session(session_id, task_type=task_type)

    client = _get_client()
    if client is None:
        return
    try:
        client.table("sessions").update({"task_type": task_type}).eq("id", session_id).execute()
    except Exception as exc:
        log.error("supabase_task_type_update_failed", error=str(exc))


async def log_image(
    session_id: str,
    filename: str,
    modality: Optional[str],
    r2_key: str,
    metadata: dict,
) -> None:
    """Log uploaded image metadata."""
    store_image(session_id, filename, modality, r2_key, metadata)

    client = _get_client()
    if client is None:
        return
    try:
        client.table("images").insert({
            "session_id": session_id,
            "filename": filename,
            "modality": modality,
            "r2_key": r2_key,
            "crs": metadata.get("crs"),
            "bounds": metadata.get("bounds"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "band_count": metadata.get("band_count"),
        }).execute()
    except Exception as exc:
        log.error("supabase_image_log_failed", error=str(exc))


async def log_result(
    session_id: str,
    task_type: str,
    model_used: str,
    answer_text: str,
    confidence: float,
    overlay_r2_key: Optional[str],
    report_r2_key: Optional[str],
    execution_trace: list,
) -> None:
    """Log analysis result."""
    store_result(
        session_id, task_type, model_used, answer_text,
        confidence, overlay_r2_key, report_r2_key, execution_trace,
    )

    client = _get_client()
    if client is None:
        return
    try:
        import json
        client.table("results").insert({
            "session_id": session_id,
            "task_type": task_type,
            "model_used": model_used,
            "answer_text": answer_text,
            "confidence": confidence,
            "overlay_r2_key": overlay_r2_key,
            "report_r2_key": report_r2_key,
            "execution_trace": execution_trace,
        }).execute()
    except Exception as exc:
        log.error("supabase_result_log_failed", error=str(exc))
