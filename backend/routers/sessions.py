"""
SatQuery AI — Sessions History Router
Provides endpoints to list past analysis sessions and fetch results.
Falls back to in-memory store when Supabase is not configured.
"""
from __future__ import annotations
import os
import structlog
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from db import session_store

log = structlog.get_logger()
router = APIRouter()

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


@router.get("/sessions")
async def list_sessions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: Optional[str] = Query(None),
    task_type: Optional[str] = Query(None),
):
    """
    List analysis sessions with pagination.

    Query params:
      - limit: number of sessions to return (1–100, default 20)
      - offset: pagination offset
      - status: filter by status (pending, preprocessing, routing, executing, integrating, completed, failed)
      - task_type: filter by task_type (vqa, caption, grounding, change_detection, change_vqa, sar_fusion)
    """
    client = _get_client()

    # ── In-memory fallback ─────────────────────────────────────────────────────
    if client is None:
        rows, total = session_store.get_sessions(limit, offset, status, task_type)
        note = "Using in-memory store (Supabase not configured)" if not SUPABASE_URL else None
        result = {"sessions": rows, "total": total, "limit": limit, "offset": offset}
        if note:
            result["note"] = note
        return result

    # ── Supabase ───────────────────────────────────────────────────────────────
    try:
        query = (
            client.table("sessions")
            .select("id, created_at, input_mode, query_text, task_type, status")
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
        )
        if status:
            query = query.eq("status", status)
        if task_type:
            query = query.eq("task_type", task_type)

        resp = query.execute()
        sessions = resp.data if resp.data else []

        count_query = client.table("sessions").select("id", count="exact")
        if status:
            count_query = count_query.eq("status", status)
        if task_type:
            count_query = count_query.eq("task_type", task_type)
        count_resp = count_query.execute()
        total = count_resp.count if hasattr(count_resp, "count") and count_resp.count else len(sessions)

        return {"sessions": sessions, "total": total, "limit": limit, "offset": offset}
    except Exception as exc:
        log.error("list_sessions_failed", error=str(exc))
        raise HTTPException(500, f"Failed to fetch sessions: {exc}")


@router.get("/sessions/stats/summary")
async def session_stats():
    """
    Aggregate statistics: total sessions, breakdown by task_type, status, etc.
    """
    client = _get_client()

    if client is None:
        stats = session_store.get_stats()
        return stats

    try:
        resp = client.table("sessions").select("status, task_type").execute()
        rows = resp.data or []

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
    except Exception as exc:
        log.error("session_stats_failed", error=str(exc))
        raise HTTPException(500, f"Failed to fetch stats: {exc}")


@router.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Fetch a single session with its result and image metadata.
    Falls back to in-memory store if Supabase is not configured.
    """
    client = _get_client()

    # ── In-memory fallback ─────────────────────────────────────────────────────
    if client is None:
        session = session_store.get_session(session_id)
        if not session:
            raise HTTPException(404, f"Session {session_id} not found")
        result = session_store.get_result(session_id)
        images = session_store.get_images(session_id)
        return {"session": session, "result": result, "images": images}

    # ── Supabase ───────────────────────────────────────────────────────────────
    try:
        sess_resp = (
            client.table("sessions")
            .select("*")
            .eq("id", session_id)
            .single()
            .execute()
        )
        if not sess_resp.data:
            raise HTTPException(404, f"Session {session_id} not found")

        session = sess_resp.data

        result_resp = (
            client.table("results")
            .select("*")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )
        result = result_resp.data[0] if result_resp.data else None

        images_resp = (
            client.table("images")
            .select("filename, modality, r2_key, width, height, band_count, crs")
            .eq("session_id", session_id)
            .execute()
        )
        images = images_resp.data if images_resp.data else []

        return {
            "session": session,
            "result": result,
            "images": images,
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.error("get_session_failed", error=str(exc), session_id=session_id)
        raise HTTPException(500, f"Failed to fetch session: {exc}")
