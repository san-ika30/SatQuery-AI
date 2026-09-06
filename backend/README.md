---
title: SatQuery AI Backend
emoji: 🛰️
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Agentic VLM pipeline for remote sensing image analysis
---

# 🛰️ SatQuery AI — FastAPI Backend

**ISRO Smart India Hackathon 2026 · Problem Statement ID: 26167**

Agentic vision-language assistant for multimodal remote sensing image analysis.
Deployed as a Hugging Face Spaces Docker container.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/api/health` | Model availability & system status |
| GET | `/api/models` | List registered specialist models |
| POST | `/api/analyze` | Full analysis pipeline (blocking) |
| POST | `/api/analyze/stream` | SSE streaming analysis pipeline |
| GET | `/api/sessions` | List past analysis sessions |
| GET | `/api/sessions/{id}` | Fetch a single session with result |
| GET | `/api/sessions/stats/summary` | Aggregate statistics |
| GET | `/api/local-files/{path}` | Serve locally stored images/reports |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc |

## Required Secrets (set in Space Settings → Variables and secrets)

| Secret | Description |
|--------|-------------|
| `HF_API_TOKEN` | Hugging Face API token for model inference |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase service role key |
| `R2_ACCOUNT_ID` | Cloudflare R2 account ID |
| `R2_ACCESS_KEY_ID` | Cloudflare R2 access key |
| `R2_SECRET_ACCESS_KEY` | Cloudflare R2 secret key |
| `R2_BUCKET_NAME` | R2 bucket name (default: satquery-ai) |
| `R2_PUBLIC_URL` | R2 public CDN URL |
| `CORS_ORIGINS` | Comma-separated allowed origins (e.g. your Vercel URL) |

All secrets have graceful fallbacks — the app runs without them (using local storage and rule-based classification).

## Model Stack

| Task | Model | Fallback |
|------|-------|---------|
| VQA / Captioning | GeoChat-7B | BLIP2-OPT-2.7B → local pixel analysis |
| Region Grounding | GroundingDINO-Tiny | Empty result |
| Change Detection | ChangeFormer (HF Space) | Pixel-difference heatmap |
| SAR Fusion | Dual-Encoder (composite) | Optical-only |
| Task Routing | Mistral-7B-Instruct | Rule-based keyword classifier |

## Local Development

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# → http://localhost:8000/docs
```
