"""
SatQuery AI — In-Memory Session Store
Fallback session/result store when Supabase is not configured.
Stores data in memory during the server lifetime (perfect for demos).
"""
from __future__ import annotations
import time
from typing import Optional, Any

# In-memory store: session_id → dict
_sessions: dict[str, dict] = {}
_results: dict[str, dict] = {}   # keyed by session_id
_images: dict[str, list] = {}    # keyed by session_id → list of image dicts


def store_session(
    session_id: str,
    input_mode: str,
    query_text: str,
    status: str = "pending",
    task_type: Optional[str] = None,
) -> None:
    from datetime import datetime, UTC
    _sessions[session_id] = {
        "id": session_id,
        "created_at": datetime.now(UTC).isoformat(),
        "input_mode": input_mode,
        "query_text": query_text,
        "status": status,
        "task_type": task_type,
    }


def update_session(session_id: str, **kwargs: Any) -> None:
    if session_id in _sessions:
        _sessions[session_id].update(kwargs)


def store_image(
    session_id: str,
    filename: str,
    modality: Optional[str],
    r2_key: str,
    metadata: dict,
) -> None:
    if session_id not in _images:
        _images[session_id] = []
    _images[session_id].append({
        "session_id": session_id,
        "filename": filename,
        "modality": modality,
        "r2_key": r2_key,
        "crs": metadata.get("crs"),
        "bounds": metadata.get("bounds"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "band_count": metadata.get("band_count"),
    })


def store_result(
    session_id: str,
    task_type: str,
    model_used: str,
    answer_text: str,
    confidence: float,
    overlay_r2_key: Optional[str],
    report_r2_key: Optional[str],
    execution_trace: list,
) -> None:
    _results[session_id] = {
        "session_id": session_id,
        "task_type": task_type,
        "model_used": model_used,
        "answer_text": answer_text,
        "confidence": confidence,
        "overlay_r2_key": overlay_r2_key,
        "report_r2_key": report_r2_key,
        "execution_trace": execution_trace,
    }


def get_sessions(
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None,
    task_type: Optional[str] = None,
) -> tuple[list[dict], int]:
    rows = list(_sessions.values())
    # Sort newest first
    rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    if status:
        rows = [r for r in rows if r.get("status") == status]
    if task_type:
        rows = [r for r in rows if r.get("task_type") == task_type]
    total = len(rows)
    return rows[offset: offset + limit], total


def get_session(session_id: str) -> Optional[dict]:
    return _sessions.get(session_id)


def get_result(session_id: str) -> Optional[dict]:
    return _results.get(session_id)


def get_images(session_id: str) -> list[dict]:
    return _images.get(session_id, [])


def get_stats() -> dict:
    rows = list(_sessions.values())
    by_status: dict[str, int] = {}
    by_task: dict[str, int] = {}
    for row in rows:
        s = row.get("status") or "unknown"
        t = row.get("task_type") or "unknown"
        by_status[s] = by_status.get(s, 0) + 1
        by_task[t] = by_task.get(t, 0) + 1
    return {
        "total_sessions": len(rows),
        "by_status": by_status,
        "by_task_type": by_task,
    }
