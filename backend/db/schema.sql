-- ============================================================
-- SatQuery AI — Supabase PostgreSQL + PostGIS Schema
-- Run this once in the Supabase SQL editor (Database → SQL Editor)
-- ============================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Sessions Table ─────────────────────────────────────────────────────────
-- One row per analysis job. Created at job start; status updated through
-- the pipeline: pending → preprocessing → routing → executing →
--               integrating → completed | failed
CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,          -- uuid4().hex from the backend
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    input_mode      TEXT NOT NULL CHECK (input_mode IN ('single','bitemporal','crossmodal')),
    query_text      TEXT NOT NULL,
    task_type       TEXT,                      -- populated after agentic routing
    status          TEXT DEFAULT 'pending' CHECK (
                        status IN ('pending','preprocessing','routing',
                                   'executing','integrating','completed','failed')
                    )
);

-- ── Images Table ───────────────────────────────────────────────────────────
-- One row per uploaded image (up to 2 per session for bi-temporal/crossmodal).
-- r2_key is the object key in Cloudflare R2 (or a "local/…" path in dev mode).
CREATE TABLE IF NOT EXISTS images (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    modality        TEXT CHECK (modality IN ('optical','sar','multispectral')),
    acquisition_date DATE,                    -- optional; not written by backend currently
    crs             TEXT,                     -- e.g. "EPSG:4326"
    bounds          FLOAT8[],                 -- [west, south, east, north] decimal degrees
    width           INT,
    height          INT,
    band_count      INT,
    r2_key          TEXT NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Results Table ──────────────────────────────────────────────────────────
-- One row per completed analysis (1:1 with sessions in practice).
-- execution_trace is stored as JSONB for flexible querying.
-- report_r2_key stores the public URL of the PDF report (not the raw R2 key).
-- overlay_r2_key stores the raw R2 / local key for the change-map or SAR composite.
CREATE TABLE IF NOT EXISTS results (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id      TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    task_type       TEXT NOT NULL,
    model_used      TEXT NOT NULL,            -- comma-separated model names
    answer_text     TEXT,
    confidence      FLOAT4,                   -- 0.0 – 1.0
    overlay_r2_key  TEXT,                     -- change map or SAR composite key
    report_r2_key   TEXT,                     -- public URL of PDF report
    execution_trace JSONB,                    -- [{step, component, status, duration_ms}, …]
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Indexes ────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_sessions_created_at  ON sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_status      ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_task_type   ON sessions(task_type);
CREATE INDEX IF NOT EXISTS idx_images_session       ON images(session_id);
CREATE INDEX IF NOT EXISTS idx_results_session      ON results(session_id);

-- ── Row Level Security ─────────────────────────────────────────────────────
-- RLS is disabled — the backend uses the service-role key which bypasses RLS.
-- Enable only if you introduce end-user authentication in future.
-- ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE images   ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE results  ENABLE ROW LEVEL SECURITY;

-- ── Smoke-test query (run after schema creation) ───────────────────────────
-- SELECT COUNT(*) FROM sessions;
-- SELECT COUNT(*) FROM images;
-- SELECT COUNT(*) FROM results;
