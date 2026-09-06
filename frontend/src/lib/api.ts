/**
 * SatQuery AI — Typed API client for FastAPI backend
 */

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

export type InputMode = 'single' | 'bitemporal' | 'crossmodal'

export interface ModelStatus {
  name: string
  available: boolean
  endpoint: string
  task: string
}

export interface HealthResponse {
  status: string
  version: string
  models: ModelStatus[]
}

export interface TaskPlan {
  task_type: string
  input_mode: string
  models: string[]
  parameters: Record<string, unknown>
  reasoning: string
}

export interface BoundingBox {
  label: string
  score: number
  xmin: number
  ymin: number
  xmax: number
  ymax: number
}

export interface VisualEvidence {
  type: 'change_map' | 'sar_composite' | 'bounding_boxes'
  r2_key?: string
  public_url?: string
  description?: string
  boxes?: BoundingBox[]
}

export interface ExecutionStep {
  step: number
  component: string
  status: string
  duration_ms: number
  [key: string]: unknown
}

export interface EvidencePackage {
  answer: string
  confidence: number
  task_type: string
  models_used: string[]
  visual_evidence: VisualEvidence[]
  execution_trace: ExecutionStep[]
  report_key?: string
}

export interface AnalysisResponse {
  session_id: string
  status: string
  task_plan?: TaskPlan
  evidence?: EvidencePackage
  error?: string
}

export interface AnalysisError {
  detail: string
}

// ── Session History Types ─────────────────────────────────────────────────────

export interface SessionRow {
  id: string
  created_at: string
  input_mode: string
  query_text: string
  task_type: string | null
  status: string
}

export interface SessionDetail {
  session: Record<string, unknown>
  result: Record<string, unknown> | null
  images: Array<Record<string, unknown>>
}

export interface SessionsResponse {
  sessions: SessionRow[]
  total: number
  limit: number
  offset: number
  note?: string
}

export interface SessionStats {
  total_sessions: number
  by_status: Record<string, number>
  by_task_type: Record<string, number>
  note?: string
}

// ── API Functions ─────────────────────────────────────────────────────────────

export interface SatQueryUnifiedResponse {
  status: 'success' | 'error'
  task?: string
  inputs?: number
  answer?: string
  detections?: BoundingBox[]
  overlay?: string
  change_ratio?: number
  severity?: string
  change_categories?: string[]
  evidence?: string[]
  confidence?: number
  model?: string
  tools?: string[]
  image_size?: [number, number]
  execution_time_ms?: number
  error?: { code: string; message: string }
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BACKEND_URL}/api/health`, { cache: 'no-store' })
  if (!res.ok) throw new Error('Health check failed')
  return res.json()
}

export async function querySatQueryApi(
  imageFile: File,
  question: string,
  image2File?: File,
): Promise<SatQueryUnifiedResponse> {
  const fileToBase64 = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        const res = reader.result as string
        resolve(res.includes(',') ? res.split(',')[1] : res)
      }
      reader.onerror = reject
      reader.readAsDataURL(file)
    })

  const b64Image1 = await fileToBase64(imageFile)
  const payload: Record<string, unknown> = {
    question,
  }

  if (image2File) {
    const b64Image2 = await fileToBase64(image2File)
    payload.image_t1 = b64Image1
    payload.image_t2 = b64Image2
  } else {
    payload.image = b64Image1
  }

  const res = await fetch(`${BACKEND_URL}/api/satquery`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: 'SatQuery request failed' }))
    const msg = errData.detail?.message || errData.detail || `HTTP ${res.status}`
    throw new Error(msg)
  }

  return res.json()
}


export async function analyzeImages(
  query: string,
  inputMode: InputMode,
  image1: File,
  image2?: File,
  onProgress?: (message: string, pct?: number) => void,
): Promise<AnalysisResponse> {
  const formData = new FormData()
  formData.append('query', query)
  formData.append('input_mode', inputMode)
  formData.append('image1', image1)
  if (image2) formData.append('image2', image2)

  onProgress?.('Uploading images…', 2)

  const res = await fetch(`${BACKEND_URL}/api/analyze/stream`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok || !res.body) {
    // Fallback: try the non-streaming endpoint
    const fallbackRes = await fetch(`${BACKEND_URL}/api/analyze`, {
      method: 'POST',
      body: formData,
    })
    if (!fallbackRes.ok) {
      const err: AnalysisError = await fallbackRes.json().catch(() => ({ detail: 'Unknown error' }))
      throw new Error(err.detail || `HTTP ${fallbackRes.status}`)
    }
    return fallbackRes.json()
  }

  // SSE streaming
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // Process complete SSE messages from buffer
    const messages = buffer.split('\n\n')
    buffer = messages.pop() ?? ''  // Keep incomplete last chunk

    for (const msg of messages) {
      if (!msg.trim()) continue

      let eventType = 'message'
      let data = ''

      for (const line of msg.split('\n')) {
        if (line.startsWith('event: ')) eventType = line.slice(7).trim()
        else if (line.startsWith('data: ')) data = line.slice(6).trim()
      }

      if (!data) continue

      // Parse JSON outside try/catch so SSE error events always propagate
      let parsed: unknown
      try {
        parsed = JSON.parse(data)
      } catch {
        console.warn('SSE: failed to parse JSON chunk, skipping:', data.slice(0, 100))
        continue
      }

      const p = parsed as Record<string, unknown>

      if (eventType === 'progress') {
        onProgress?.(String(p.message ?? 'Processing…'), typeof p.pct === 'number' ? p.pct : undefined)
      } else if (eventType === 'result') {
        return p as unknown as AnalysisResponse
      } else if (eventType === 'error') {
        // Always throw — never swallow backend error events
        throw new Error(String(p.detail || p.message || 'Analysis failed on server'))
      }
    }
  }

  throw new Error('Stream ended without a result')
}


export function getPublicUrl(key: string): string {
  const base = process.env.NEXT_PUBLIC_R2_PUBLIC_URL || ''
  if (!base || key.startsWith('http')) return key
  return `${base.replace(/\/$/, '')}/${key}`
}

export function confidenceColor(score: number): string {
  if (score >= 0.85) return '#22c55e'   // green
  if (score >= 0.70) return '#84cc16'   // lime
  if (score >= 0.50) return '#eab308'   // yellow
  if (score >= 0.30) return '#f97316'   // orange
  return '#ef4444'                       // red
}

export function confidenceLabel(score: number): string {
  if (score >= 0.85) return 'Very High'
  if (score >= 0.70) return 'High'
  if (score >= 0.50) return 'Moderate'
  if (score >= 0.30) return 'Low'
  return 'Very Low'
}

export function taskTypeLabel(t: string): string {
  const map: Record<string, string> = {
    vqa: 'Visual Question Answering',
    caption: 'Scene Description',
    grounding: 'Region Grounding',
    change_detection: 'Change Detection',
    change_vqa: 'Change-based VQA',
    sar_fusion: 'Optical–SAR Fusion',
  }
  return map[t] || t
}

export async function fetchSessions(
  limit = 20,
  offset = 0,
  status?: string,
  taskType?: string,
): Promise<SessionsResponse> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) })
  if (status) params.set('status', status)
  if (taskType) params.set('task_type', taskType)
  const res = await fetch(`${BACKEND_URL}/api/sessions?${params}`, { cache: 'no-store' })
  if (!res.ok) throw new Error('Failed to fetch sessions')
  return res.json()
}

export async function fetchSession(sessionId: string): Promise<SessionDetail> {
  const res = await fetch(`${BACKEND_URL}/api/sessions/${sessionId}`, { cache: 'no-store' })
  if (!res.ok) throw new Error(`Session not found: ${sessionId}`)
  return res.json()
}

export async function fetchStats(): Promise<SessionStats> {
  const res = await fetch(`${BACKEND_URL}/api/sessions/stats/summary`, { cache: 'no-store' })
  if (!res.ok) throw new Error('Failed to fetch stats')
  return res.json()
}

export function inputModeLabel(mode: string): string {
  const map: Record<string, string> = {
    single: 'Single Image',
    bitemporal: 'Bi-Temporal',
    crossmodal: 'Optical + SAR',
  }
  return map[mode] || mode
}

export function statusColor(status: string): string {
  const map: Record<string, string> = {
    completed: '#34d399',
    failed: '#f87171',
    pending: '#64748b',
    preprocessing: '#00d4ff',
    routing: '#00d4ff',
    executing: '#ffb703',
    integrating: '#a78bfa',
  }
  return map[status] || '#64748b'
}
