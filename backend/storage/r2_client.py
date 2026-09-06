"""
SatQuery AI — Cloudflare R2 Storage Client
Handles upload/download of images and reports.
Falls back to LOCAL_STORAGE_DIR when R2 is not configured, so that
/api/local-files/{path} can serve the file without needing cloud credentials.
"""
from __future__ import annotations
import os
import uuid
import structlog

try:
    import boto3
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    boto3 = None  # type: ignore
    BOTO3_AVAILABLE = False

log = structlog.get_logger()

R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "satquery-ai")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "")

_r2_client = None


def _get_client():
    global _r2_client
    if _r2_client is None:
        if not BOTO3_AVAILABLE or not R2_ACCOUNT_ID:
            return None
        from botocore.config import Config as BotoConfig
        _r2_client = boto3.client(
            "s3",
            endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            config=BotoConfig(signature_version="s3v4"),
            region_name="auto",
        )
    return _r2_client


# ── Local storage helpers ─────────────────────────────────────────────────────

def _local_storage_dir():
    """Return the local storage directory as a Path, creating it if needed."""
    from pathlib import Path
    d = Path(os.getenv("LOCAL_STORAGE_DIR", "./local_storage"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_local(object_key: str, data: bytes) -> str:
    """
    Write bytes to LOCAL_STORAGE_DIR/{object_key} and return a "local/" key.
    Creates any required subdirectories automatically.
    This is the fallback used when R2 is not configured or upload fails.
    The key returned matches what get_public_url() and main.py's
    /api/local-files/{path} route expect.
    """
    dest = _local_storage_dir() / object_key
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    log.info("local_file_written", path=str(dest), size=len(data))
    return f"local/{object_key}"


# ── Public API ────────────────────────────────────────────────────────────────

def upload_bytes(
    data: bytes,
    object_key: str,
    content_type: str = "application/octet-stream",
) -> str:
    """
    Upload bytes to R2 and return the object key.
    Falls back to LOCAL_STORAGE_DIR when R2 is not configured, so that
    /api/local-files/{path} can serve the file during local development.
    """
    client = _get_client()
    if client is None:
        log.info("r2_not_configured_writing_to_local_storage", key=object_key)
        return _write_local(object_key, data)

    try:
        client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=object_key,
            Body=data,
            ContentType=content_type,
        )
        log.info("r2_upload_success", key=object_key, size=len(data))
        return object_key
    except Exception as exc:
        log.error("r2_upload_failed_falling_back_to_local", error=str(exc), key=object_key)
        # Fallback: write locally so the session result is not lost
        return _write_local(object_key, data)


def get_public_url(object_key: str) -> str:
    """Return the public URL for an R2 object."""
    if object_key.startswith("local/"):
        return f"/api/local-files/{object_key[6:]}"
    if R2_PUBLIC_URL:
        return f"{R2_PUBLIC_URL.rstrip('/')}/{object_key}"
    return object_key


def upload_image(image_bytes: bytes, session_id: str, suffix: str = "img") -> str:
    uid = uuid.uuid4().hex[:8]
    key = f"sessions/{session_id}/{suffix}_{uid}.png"
    return upload_bytes(image_bytes, key, content_type="image/png")


def upload_report_pdf(pdf_bytes: bytes, session_id: str) -> str:
    key = f"reports/{session_id}/report.pdf"
    return upload_bytes(pdf_bytes, key, content_type="application/pdf")


def upload_report_json(json_bytes: bytes, session_id: str) -> str:
    key = f"reports/{session_id}/report.json"
    return upload_bytes(json_bytes, key, content_type="application/json")
