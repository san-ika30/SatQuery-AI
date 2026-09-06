"""
SatQuery AI -- Supabase Connection Verification Script
Run from backend/ directory: python verify_supabase.py

Checks:
  1. Env vars loaded correctly (URL only, never prints the key)
  2. Supabase client can connect
  3. Tables are accessible (sessions / images / results)
  4. Can insert, read, and delete a test session without leaving residue
  5. In-memory fallback still works when client is None
"""
import os
import sys
import uuid
from pathlib import Path

# Windows: force UTF-8 output so the script doesn't crash on special chars
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load .env from project root (one level up)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY", "")

SEP = "-" * 60


def check(label: str, passed: bool, detail: str = "") -> bool:
    icon = "PASS" if passed else "FAIL"
    print(f"  [{icon}]  {label}")
    if detail:
        print(f"           {detail}")
    return passed


results: dict[str, bool] = {}

print(f"\n{SEP}")
print("  SatQuery AI -- Supabase Verification")
print(SEP)

# -- 1. Environment variables -------------------------------------------------
print("\n[1] Environment Variables")

url_ok = bool(SUPABASE_URL) and "xxxxxxxxxxx" not in SUPABASE_URL
key_ok = (
    bool(SUPABASE_KEY)
    and len(SUPABASE_KEY) > 50
    and SUPABASE_KEY != "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
)

results["env_url"] = check(
    "SUPABASE_URL is set and not a placeholder",
    url_ok,
    SUPABASE_URL if url_ok else "Still placeholder!",
)
results["env_key"] = check(
    "SUPABASE_SERVICE_ROLE_KEY is set (value hidden)",
    key_ok,
    f"Length: {len(SUPABASE_KEY)} chars" if key_ok else "Missing or placeholder!",
)

if not (url_ok and key_ok):
    print("\n  CANNOT PROCEED -- fix env vars first.")
    sys.exit(1)

# -- 2. Client initialisation -------------------------------------------------
print("\n[2] Supabase Client Initialisation")
client = None
try:
    from supabase import create_client
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    results["client_init"] = check("create_client() succeeded", True)
except Exception as exc:
    results["client_init"] = check("create_client() succeeded", False, str(exc))
    print("\n  Client init failed -- cannot continue.")
    sys.exit(1)

# -- 3. Table accessibility ---------------------------------------------------
print("\n[3] Table Accessibility")

for table in ("sessions", "images", "results"):
    try:
        resp = client.table(table).select("id", count="exact").limit(1).execute()
        n = resp.count if resp.count is not None else len(resp.data)
        results[f"table_{table}"] = check(
            f"Table '{table}' accessible", True, f"{n} existing row(s)"
        )
    except Exception as exc:
        results[f"table_{table}"] = check(f"Table '{table}' accessible", False, str(exc))

# -- 4. Insert / read / delete a test session ---------------------------------
print("\n[4] Session Insert / Read / Delete")
test_sid = f"verify-{uuid.uuid4().hex[:12]}"

try:
    client.table("sessions").insert({
        "id": test_sid,
        "input_mode": "single",
        "query_text": "[Automated verification -- safe to delete]",
        "status": "pending",
    }).execute()
    results["insert"] = check("INSERT into sessions", True, f"id={test_sid}")
except Exception as exc:
    results["insert"] = check("INSERT into sessions", False, str(exc))

try:
    resp = client.table("sessions").select("id,status").eq("id", test_sid).single().execute()
    got = resp.data
    results["read"] = check(
        "SELECT back the test session",
        got is not None and got.get("id") == test_sid,
        f"status={got.get('status')}" if got else "no row returned",
    )
except Exception as exc:
    results["read"] = check("SELECT back the test session", False, str(exc))

try:
    client.table("sessions").delete().eq("id", test_sid).execute()
    results["delete"] = check("DELETE test session (cleanup)", True)
except Exception as exc:
    results["delete"] = check("DELETE test session (cleanup)", False, str(exc))

# -- 5. In-memory fallback ----------------------------------------------------
print("\n[5] In-Memory Fallback (independent of Supabase)")
try:
    from db.session_store import store_session, get_session, update_session
    fsid = "fallback-verify-001"
    store_session(fsid, "single", "Fallback test.", status="pending")
    update_session(fsid, status="completed", task_type="vqa")
    s = get_session(fsid)
    ok = s is not None and s["status"] == "completed" and s["task_type"] == "vqa"
    results["fallback"] = check("In-memory store CRUD works independently", ok)
except Exception as exc:
    results["fallback"] = check("In-memory store CRUD works independently", False, str(exc))

# -- Summary ------------------------------------------------------------------
passed = sum(1 for v in results.values() if v)
total  = len(results)
print(f"\n{SEP}")
print(f"  Result: {passed}/{total} checks passed")
print(SEP)

for k, v in results.items():
    status = "PASS" if v else "FAIL"
    print(f"    {status:<6} {k}")

print()
if passed == total:
    print("  All checks PASSED -- Supabase integration is fully operational.")
    sys.exit(0)
else:
    failed = [k for k, v in results.items() if not v]
    print(f"  FAILED checks: {', '.join(failed)}")
    sys.exit(1)
