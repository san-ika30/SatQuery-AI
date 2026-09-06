"""
SatQuery AI — End-to-end HTTP test script.
Run with: python test_e2e.py
"""
import urllib.request
import json
import io
import uuid

from PIL import Image


def create_test_png(color=(80, 120, 60)) -> bytes:
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def multipart_body(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )
    for name, (filename, data, ctype) in files.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {ctype}\r\n\r\n".encode()
            + data
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def get(url, timeout=10):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def post_multipart(url, fields, files, timeout=90):
    body, ctype = multipart_body(fields, files)
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


BASE = "http://localhost:8000"

print("=" * 60)
print("SatQuery AI — End-to-End Verification")
print("=" * 60)

# 1. Root
print("\n[1] GET /")
r = get(f"{BASE}/")
print(f"  Service: {r['service']}, Version: {r['version']}, Status: {r['status']}")

# 2. Health
print("\n[2] GET /api/health")
r = get(f"{BASE}/api/health")
print(f"  Status: {r['status']}, Models: {len(r['models'])}")
for m in r["models"]:
    avail = "✅" if m["available"] else "❌ (fallback)"
    print(f"    {avail} {m['name']} — {m['task']}")

# 3. Sessions (empty)
print("\n[3] GET /api/sessions")
r = get(f"{BASE}/api/sessions")
print(f"  Total: {r['total']}, Note: {r.get('note', 'n/a')}")

# 4. Stats
print("\n[4] GET /api/sessions/stats/summary")
r = get(f"{BASE}/api/sessions/stats/summary")
print(f"  Total sessions: {r['total_sessions']}")

# 5. Full analyze pipeline
print("\n[5] POST /api/analyze (caption task, local fallback)")
img_bytes = create_test_png(color=(80, 120, 60))
try:
    r = post_multipart(
        f"{BASE}/api/analyze",
        fields={"query": "Describe the land cover visible in this satellite image.", "input_mode": "single"},
        files={"image1": ("test_satellite.png", img_bytes, "image/png")},
        timeout=90,
    )
    print(f"  Status:    {r['status']}")
    print(f"  Task:      {r.get('task_plan', {}).get('task_type', 'n/a')}")
    print(f"  Classify:  {r.get('task_plan', {}).get('reasoning', '')[:80]}")
    ev = r.get("evidence", {})
    print(f"  Models:    {ev.get('models_used', [])}")
    print(f"  Confidence:{ev.get('confidence', 0):.2f}")
    print(f"  Answer:    {ev.get('answer', '')[:200]}")
    print(f"  SessionID: {r.get('session_id', '')[:16]}...")
    session_id = r.get("session_id", "")
except Exception as e:
    print(f"  ERROR: {e}")
    session_id = ""

# 6. VQA grounding test
print("\n[6] POST /api/analyze (grounding task)")
img_bytes2 = create_test_png(color=(40, 80, 200))  # blue-ish (water)
try:
    r = post_multipart(
        f"{BASE}/api/analyze",
        fields={"query": "Locate the water body in this image.", "input_mode": "single"},
        files={"image1": ("blue_image.png", img_bytes2, "image/png")},
        timeout=90,
    )
    ev = r.get("evidence", {})
    print(f"  Status:    {r['status']}")
    print(f"  Task:      {r.get('task_plan', {}).get('task_type', 'n/a')}")
    print(f"  Models:    {ev.get('models_used', [])}")
    print(f"  Confidence:{ev.get('confidence', 0):.2f}")
    print(f"  Answer:    {ev.get('answer', '')[:200]}")
except Exception as e:
    print(f"  ERROR: {e}")

# 7. Sessions after analyses
print("\n[7] GET /api/sessions (after analyses)")
r = get(f"{BASE}/api/sessions")
print(f"  Total: {r['total']}")
for s in r.get("sessions", []):
    print(f"    [{s['status']}] {s['task_type']} — {s['query_text'][:60]}")

# 8. Stats after analyses
print("\n[8] GET /api/sessions/stats/summary (after analyses)")
r = get(f"{BASE}/api/sessions/stats/summary")
print(f"  Total sessions: {r['total_sessions']}")
print(f"  By status: {r['by_status']}")
print(f"  By task:   {r['by_task_type']}")

# 9. Fetch single session
if session_id:
    print(f"\n[9] GET /api/sessions/{session_id[:16]}...")
    try:
        r = get(f"{BASE}/api/sessions/{session_id}")
        print(f"  Session status: {r.get('session', {}).get('status', 'n/a')}")
        print(f"  Result confidence: {r.get('result', {}).get('confidence', 'n/a')}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n" + "=" * 60)
print("✅ End-to-end verification complete!")
print("=" * 60)
