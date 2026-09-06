'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { use } from 'react'
import {
  fetchSession, type SessionDetail,
  taskTypeLabel, inputModeLabel, confidenceColor, confidenceLabel, statusColor,
} from '@/lib/api'

function InfoRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex gap-3 py-2 border-b border-space-700/30 last:border-0">
      <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider w-32 flex-shrink-0 pt-0.5">
        {label}
      </span>
      <span className="text-sm text-slate-200 flex-1">{value}</span>
    </div>
  )
}

export default function SessionDetailPage({
  params,
}: {
  params: Promise<{ sessionId: string }>
}) {
  const { sessionId } = use(params)
  const [data, setData] = useState<SessionDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchSession(sessionId)
      .then(setData)
      .catch(e => setError(e instanceof Error ? e.message : 'Not found'))
      .finally(() => setLoading(false))
  }, [sessionId])

  return (
    <div className="min-h-screen bg-space-950">
      <div className="stars-bg" />
      <header className="glass border-b border-cyan-400/10 px-6 py-4 flex items-center justify-between sticky top-0 z-10">
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-xs">
            🛰️
          </div>
          <span className="font-space font-bold text-white group-hover:text-cyan-400 transition-colors">
            SatQuery <span className="gradient-text">AI</span>
          </span>
        </Link>
        <div className="flex items-center gap-3">
          <Link href="/results" className="btn-secondary text-sm py-2 px-4">
            ← History
          </Link>
          <Link href="/analyze" className="btn-primary text-sm py-2 px-4">
            + New Analysis
          </Link>
        </div>
      </header>

      <main className="relative z-10 max-w-4xl mx-auto px-4 sm:px-6 py-10 space-y-6">
        {loading ? (
          <div className="space-y-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="card skeleton h-32 rounded-xl" />
            ))}
          </div>
        ) : error ? (
          <div className="card border-red-500/30 bg-red-500/5 text-center py-16">
            <div className="text-4xl mb-4">❌</div>
            <h3 className="font-bold text-red-400 mb-2">Session Not Found</h3>
            <p className="text-sm text-slate-400">{error}</p>
            <Link href="/results" className="btn-secondary mt-6 inline-flex">
              ← Back to History
            </Link>
          </div>
        ) : data ? (
          <>
            {/* Page title */}
            <div>
              <div className="flex items-center gap-3 mb-1 flex-wrap">
                <h1 className="font-space text-2xl font-bold text-white">Session Detail</h1>
                <div
                  className="flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-full border"
                  style={{
                    color: statusColor(String(data.session.status || '')),
                    borderColor: `${statusColor(String(data.session.status || ''))}40`,
                    background: `${statusColor(String(data.session.status || ''))}12`,
                  }}
                >
                  {String(data.session.status || '').charAt(0).toUpperCase() + String(data.session.status || '').slice(1)}
                </div>
              </div>
              <p className="text-xs font-mono text-slate-500">{sessionId}</p>
            </div>

            {/* Session Info */}
            <div className="card">
              <h2 className="font-space font-bold text-white text-sm mb-4 flex items-center gap-2">
                <span>📋</span> Session Information
              </h2>
              <div>
                <InfoRow label="Session ID" value={<span className="font-mono">{sessionId}</span>} />
                <InfoRow
                  label="Created"
                  value={new Date(String(data.session.created_at || '')).toLocaleString('en-IN', {
                    dateStyle: 'full', timeStyle: 'medium',
                  })}
                />
                <InfoRow
                  label="Input Mode"
                  value={inputModeLabel(String(data.session.input_mode || ''))}
                />
                {Boolean(data.session.task_type) && (
                  <InfoRow
                    label="Task Type"
                    value={<span className="badge badge-cyan">{taskTypeLabel(String(data.session.task_type))}</span>}
                  />
                )}
                <InfoRow
                  label="Query"
                  value={
                    <span className="italic text-slate-300">
                      "{String(data.session.query_text || '')}"
                    </span>
                  }
                />
              </div>
            </div>

            {/* Analysis Result */}
            {data.result && (
              <div className="card">
                <h2 className="font-space font-bold text-white text-sm mb-4 flex items-center gap-2">
                  <span>✨</span> Analysis Result
                </h2>
                <div className="mb-4">
                  <InfoRow label="Model Used" value={String(data.result.model_used || '—')} />
                  {typeof data.result.confidence === 'number' && (
                    <InfoRow
                      label="Confidence"
                      value={
                        <div className="flex items-center gap-3">
                          <span style={{ color: confidenceColor(data.result.confidence as number) }}>
                            {((data.result.confidence as number) * 100).toFixed(1)}%
                          </span>
                          <span className="text-slate-500">
                            ({confidenceLabel(data.result.confidence as number)})
                          </span>
                          <div className="flex-1 confidence-bar">
                            <div
                              className="confidence-fill"
                              style={{
                                width: `${(data.result.confidence as number) * 100}%`,
                                background: `linear-gradient(90deg, ${confidenceColor(data.result.confidence as number)}, ${confidenceColor(Math.min((data.result.confidence as number) + 0.1, 1))})`,
                              }}
                            />
                          </div>
                        </div>
                      }
                    />
                  )}
                  {Boolean(data.result.report_r2_key) && (
                    <InfoRow
                      label="Report"
                      value={
                        <a
                          href={String(data.result.report_r2_key)}
                          download
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-cyan-400 hover:underline text-xs"
                        >
                          ⬇️ Download PDF Report
                        </a>
                      }
                    />
                  )}
                </div>

                {/* Answer */}
                {Boolean(data.result.answer_text) && (
                  <div>
                    <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                      Answer
                    </div>
                    <div className="glass-light rounded-xl p-4 text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
                      {String(data.result.answer_text)}
                    </div>
                  </div>
                )}

                {/* Execution Trace */}
                {Array.isArray(data.result.execution_trace) && (data.result.execution_trace as unknown[]).length > 0 && (
                  <div className="mt-4">
                    <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                      Execution Trace
                    </div>
                    <div className="space-y-1">
                      {(data.result.execution_trace as Array<Record<string, unknown>>).map((step, i) => (
                        <div key={i} className="trace-step">
                          <div
                            className="trace-dot"
                            style={{
                              background: step.status === 'ok' ? '#34d399'
                                : step.status === 'warning' ? '#ffb703'
                                : step.status === 'failed' ? '#f87171'
                                : step.status === 'skipped' ? '#64748b' : '#00d4ff',
                            }}
                          />
                          <div className="flex-1 min-w-0">
                            <span className="text-xs font-mono text-slate-500 mr-2">
                              {String(step.step || i + 1).padStart(2, '0')}
                            </span>
                            <span className="text-xs text-slate-300">{String(step.component || '')}</span>
                          </div>
                          <span className="text-xs font-mono text-slate-600">{String(step.duration_ms || 0)}ms</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Images */}
            {data.images.length > 0 && (
              <div className="card">
                <h2 className="font-space font-bold text-white text-sm mb-4 flex items-center gap-2">
                  <span>🗺️</span> Input Images
                </h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {data.images.map((img, i) => (
                    <div key={i} className="glass-light rounded-xl p-4">
                      <div className="flex items-center gap-2 mb-3">
                        <span className="text-xl">{img.modality === 'sar' ? '📡' : '🛰️'}</span>
                        <div>
                          <div className="text-sm font-semibold text-white truncate">
                            {String(img.filename || `Image ${i + 1}`)}
                          </div>
                          <div className="text-xs text-slate-400">
                            {String(img.modality || 'unknown')} · {String(img.width || '?')}×{String(img.height || '?')} · {String(img.band_count || '?')} bands
                          </div>
                        </div>
                      </div>
                      {Boolean(img.crs) && (
                        <div className="text-xs font-mono text-slate-500">CRS: {String(img.crs)}</div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : null}
      </main>
    </div>
  )
}
