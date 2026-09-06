'use client'

import { useState, useEffect } from 'react'
import { fetchHealth, type HealthResponse } from '@/lib/api'

interface ModelStatusPanelProps {
  /** If true, renders as a compact inline bar (for nav). Otherwise renders full card. */
  compact?: boolean
}

export default function ModelStatusPanel({ compact = false }: ModelStatusPanelProps) {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
      .finally(() => setLoading(false))
  }, [])

  const overallOk = health?.status === 'operational'

  if (compact) {
    return (
      <div className="relative">
        <button
          id="model-status-toggle"
          onClick={() => setExpanded(e => !e)}
          className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border transition-all ${
            loading
              ? 'border-space-600/60 text-slate-500'
              : overallOk
              ? 'border-emerald-400/40 text-emerald-400 bg-emerald-400/8 hover:bg-emerald-400/12'
              : 'border-amber-400/40 text-amber-400 bg-amber-400/8 hover:bg-amber-400/12'
          }`}
        >
          <div
            className={`w-1.5 h-1.5 rounded-full ${loading ? 'bg-slate-500' : overallOk ? 'bg-emerald-400' : 'bg-amber-400'} ${!loading ? 'animate-pulse' : ''}`}
          />
          {loading ? 'Checking…' : overallOk ? 'Models Online' : 'Degraded'}
          <span className="ml-0.5">{expanded ? '▲' : '▼'}</span>
        </button>

        {expanded && (
          <div className="absolute right-0 top-full mt-2 w-72 glass rounded-xl border border-cyan-400/15 shadow-xl z-50 p-4">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
              Model Status
            </div>
            {health?.models.map(m => (
              <div key={m.name} className="flex items-center justify-between py-1.5 border-b border-space-700/30 last:border-0">
                <div>
                  <div className="text-xs font-semibold text-white">{m.name}</div>
                  <div className="text-xs text-slate-500">{m.task}</div>
                </div>
                <div className={`flex items-center gap-1 text-xs ${m.available ? 'text-emerald-400' : 'text-amber-400'}`}>
                  <div className={`w-1.5 h-1.5 rounded-full ${m.available ? 'bg-emerald-400' : 'bg-amber-400'}`} />
                  {m.available ? 'Online' : 'Unavailable'}
                </div>
              </div>
            ))}
            {!health && !loading && (
              <p className="text-xs text-slate-500 text-center py-2">Backend unreachable</p>
            )}
          </div>
        )}
      </div>
    )
  }

  // Full card variant
  return (
    <div className="card space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-space font-bold text-white text-sm flex items-center gap-2">
          🤖 Model Status
        </h3>
        {!loading && health && (
          <span
            className={`badge text-xs ${overallOk ? 'badge-green' : 'badge-amber'}`}
          >
            {overallOk ? '● Operational' : '⚠ Degraded'}
          </span>
        )}
      </div>

      {loading ? (
        <div className="space-y-2">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="skeleton h-10 rounded-lg" />
          ))}
        </div>
      ) : !health ? (
        <p className="text-xs text-slate-500 text-center py-4">
          ⚠️ Backend unreachable. Start the FastAPI server to see model status.
        </p>
      ) : (
        <div className="space-y-2">
          {health.models.map(m => (
            <div
              key={m.name}
              className="flex items-start justify-between py-2 px-3 rounded-lg hover:bg-space-700/20 transition-colors"
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-0.5">
                  <div
                    className={`w-2 h-2 rounded-full flex-shrink-0 ${m.available ? 'bg-emerald-400' : 'bg-amber-400/80'}`}
                  />
                  <span className="text-xs font-semibold text-white">{m.name}</span>
                </div>
                <div className="text-xs text-slate-500 ml-4">{m.task}</div>
                <div className="text-xs font-mono text-slate-600 ml-4 truncate">{m.endpoint}</div>
              </div>
              <span
                className={`text-xs font-medium flex-shrink-0 ml-2 ${
                  m.available ? 'text-emerald-400' : 'text-amber-400'
                }`}
              >
                {m.available ? '✓ Online' : '○ Standby'}
              </span>
            </div>
          ))}
        </div>
      )}

      <div className="text-xs text-center text-slate-600 pt-1">
        v{health?.version ?? '—'} · All models use HF Inference API
      </div>
    </div>
  )
}
