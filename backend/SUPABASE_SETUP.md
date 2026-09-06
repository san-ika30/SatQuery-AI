# Supabase Setup Guide — SatQuery AI Backend

## Overview

The backend uses Supabase (PostgreSQL) to persist analysis sessions, image metadata, and results.
When Supabase credentials are absent the application falls back transparently to an in-memory store,
so **nothing breaks without Supabase** — the fallback is for development and demos only.

---

## Step 1 — Create a Supabase Project

1. Go to [https://supabase.com](https://supabase.com) and sign in.
2. Click **New Project**.
3. Set a name (e.g. `satquery-ai`), choose a region close to your deployment, set a database password.
4. Wait ~2 minutes for provisioning.

---

## Step 2 — Run the Schema

1. In your Supabase dashboard, go to **Database → SQL Editor**.
2. Click **New Query**.
3. Paste the entire contents of `backend/db/schema.sql`.
4. Click **Run**.

Verify the tables were created:
```sql
SELECT COUNT(*) FROM sessions;
SELECT COUNT(*) FROM images;
SELECT COUNT(*) FROM results;
```
All three should return `0` (empty, no error).

---

## Step 3 — Collect Your Credentials

### Supabase URL
- Dashboard → **Settings → API**
- Field: **Project URL** (e.g. `https://abcdefghijkl.supabase.co`)

### Service-Role Key
- Dashboard → **Settings → API**
- Field: **service_role** (under "Project API keys")
- ⚠️ **This is a secret key — treat it like a password**

### Anon Key (optional)
- Same page: **anon public** key
- Only used as a fallback if the service-role key is not set

---

## Step 4 — Configure Environment Variables

Edit `backend/.env` (copy from `.env.example` if it doesn't exist):

```env
# Supabase — required for persistent storage
SUPABASE_URL=https://abcdefghijkl.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

> **Never commit `.env` to git.** The root `.gitignore` already excludes it.

---

## Step 5 — Verify the Connection

Start the backend:
```bash
cd backend
uvicorn main:app --reload --port 8000
```

Run an analysis and check the database. In Supabase SQL Editor:
```sql
SELECT id, input_mode, query_text, status, task_type, created_at
FROM sessions
ORDER BY created_at DESC
LIMIT 5;
```

Or use the API:
```bash
curl http://localhost:8000/api/sessions | python -m json.tool
```

If Supabase is connected you will see `"note"` is absent from the response.
If falling back to in-memory store the response includes `"note": "Using in-memory store..."`.

---

## Database Tables

| Table | Purpose |
|-------|---------|
| `sessions` | One row per analysis job. Tracks `status` through the pipeline and `task_type` after routing. |
| `images` | Up to 2 rows per session (image1 and image2). Stores filename, modality, CRS, bounds, dimensions, and the R2 object key. |
| `results` | One row per completed job. Stores answer text, confidence, overlay R2 key, PDF report public URL, and the full JSONB execution trace. |

---

## Environment Variables Reference

| Variable | Required | Where Used |
|----------|----------|-----------|
| `SUPABASE_URL` | For persistence | `db/supabase_client.py` |
| `SUPABASE_SERVICE_ROLE_KEY` | For persistence | `db/supabase_client.py` |
| `SUPABASE_ANON_KEY` | Fallback only | `db/supabase_client.py` |

> The frontend **never** uses the service-role key. The frontend only calls the FastAPI backend, which uses the service-role key server-side.

---

## Fallback Behaviour

If `SUPABASE_URL` is absent, empty, or still contains `"xxxxxxxxx"` (placeholder), the backend:
- Logs `supabase_unavailable_using_memory_store` at INFO level (not ERROR)
- Writes all session/image/result data to an in-memory Python dict
- Continues operating fully — analysis, streaming, reports, history all work
- Data is lost when the server restarts (expected for demos)

The `GET /api/sessions` response includes `"note": "Using in-memory store..."` when in fallback mode, which the frontend surfaces as a subtle indicator.

---

## Security Notes

| Rule | Reason |
|------|--------|
| Keep `SUPABASE_SERVICE_ROLE_KEY` server-side only | It bypasses all Row Level Security policies |
| Never expose it in frontend environment variables | Any `NEXT_PUBLIC_` variable is embedded in the browser bundle |
| Do not enable RLS unless you add user authentication | RLS is disabled in the schema — the service-role key bypasses it anyway |
| Rotate the key if it is ever committed to git | Supabase → Settings → API → Regenerate service_role key |

---

## Deploying to Hugging Face Spaces

In your HF Space's **Settings → Variables and secrets** add:

```
SUPABASE_URL          → https://abcdefghijkl.supabase.co
SUPABASE_SERVICE_ROLE_KEY → eyJ...
```

Mark `SUPABASE_SERVICE_ROLE_KEY` as a **Secret** (not a plain variable) so it is not visible in the Space logs.
