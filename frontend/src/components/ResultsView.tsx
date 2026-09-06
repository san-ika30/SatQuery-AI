'use client'

import { useState } from 'react'
import type { AnalysisResponse } from '@/lib/api'
import { confidenceColor, confidenceLabel, taskTypeLabel, getPublicUrl } from '@/lib/api'

interface ResultsViewProps {
  response: AnalysisResponse
  isLoading: boolean
  loadingStatus?: string
  loadingPct?: number
}

function SkeletonResults() {
  return (
    <div className="space-y-4 animate-pulse">
      <div className="skeleton h-6 w-48 rounded" />
      <div className="skeleton h-4 w-full rounded" />
      <div className="skeleton h-4 w-5/6 rounded" />
      <div className="skeleton h-4 w-4/6 rounded" />
      <div className="skeleton h-32 w-full rounded-xl" />
    </div>
  )
}

export default function ResultsView({ response, isLoading, loadingStatus, loadingPct = 0 }: ResultsViewProps) {
  const [tab, setTab] = useState<'answer' | 'visual' | 'report'>('answer')

  if (isLoading) {
    return (
      <div className="card space-y-5">
        {/* Progress bar with real percentage */}
        <div className="space-y-2">
          <div className="flex justify-between items-center text-xs">
            <span className="text-cyan-400 animate-pulse">{loadingStatus || 'Processing your query…'}</span>
            <span className="font-mono text-slate-400">{loadingPct}%</span>
          </div>
          <div className="confidence-bar">
            <div
              className="confidence-fill"
              style={{
                width: `${loadingPct}%`,
                background: 'linear-gradient(90deg, #00d4ff, #ffb703)',
                transition: 'width 0.6s ease',
              }}
            />
          </div>
        </div>
        <SkeletonResults />
      </div>
    )
  }

  if (!response || response.status === 'failed') {
    return (
      <div className="card border-red-500/30 bg-red-500/5">
        <div className="flex items-center gap-3 mb-3">
          <span className="text-2xl">❌</span>
          <h3 className="font-bold text-red-400">Analysis Failed</h3>
        </div>
        <p className="text-sm text-slate-400">{response?.error || 'Unknown error occurred.'}</p>
        <p className="text-xs text-slate-500 mt-2">
          Check that your images are valid GeoTIFF/TIFF files and retry.
        </p>
      </div>
    )
  }

  const evidence = response.evidence
  if (!evidence) return null

  const hasVisual = evidence.visual_evidence.length > 0
  const hasReport = !!evidence.report_key

  return (
    <div className="card space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-2xl">✨</span>
            <h3 className="font-space font-bold text-white text-base">Analysis Complete</h3>
          </div>
          <div className="flex flex-wrap gap-2">
            <span className="badge badge-cyan">{taskTypeLabel(evidence.task_type)}</span>
            <span
              className="badge"
              style={{
                background: `${confidenceColor(evidence.confidence)}20`,
                color: confidenceColor(evidence.confidence),
                borderColor: `${confidenceColor(evidence.confidence)}50`,
              }}
            >
              {(evidence.confidence * 100).toFixed(0)}% Confidence · {confidenceLabel(evidence.confidence)}
            </span>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-space-700/50 pb-0">
        {[
          { id: 'answer' as const, label: '💬 Answer', show: true },
          { id: 'visual' as const, label: '🗺️ Visual Evidence', show: hasVisual },
          { id: 'report' as const, label: '📄 Report', show: hasReport },
        ]
          .filter(t => t.show)
          .map(t => (
            <button
              key={t.id}
              id={`tab-${t.id}`}
              onClick={() => setTab(t.id)}
              className={`px-4 py-2.5 text-sm font-medium rounded-t-lg transition-all ${
                tab === t.id
                  ? 'bg-space-700/50 text-cyan-400 border-b-2 border-cyan-400'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {t.label}
            </button>
          ))}
      </div>

      {/* Tab Content */}
      {tab === 'answer' && (
        <div className="space-y-4">
          <div className="glass-light rounded-xl p-5">
            <div className="text-sm text-slate-200 leading-relaxed whitespace-pre-wrap">
              {evidence.answer}
            </div>
          </div>

          {/* Confidence bar */}
          <div>
            <div className="flex justify-between text-xs text-slate-500 mb-1">
              <span>Model Confidence</span>
              <span>{(evidence.confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{
                  width: `${evidence.confidence * 100}%`,
                  background: `linear-gradient(90deg, ${confidenceColor(evidence.confidence)}, ${confidenceColor(Math.min(evidence.confidence + 0.1, 1))})`,
                }}
              />
            </div>
          </div>

          {/* Bounding boxes table (if grounding) */}
          {evidence.visual_evidence
            .filter(ev => ev.type === 'bounding_boxes' && ev.boxes)
            .map((ev, i) => (
              <div key={i}>
                <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                  Detected Regions
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-space-700/50">
                        <th className="text-left py-2 px-3 text-slate-500">#</th>
                        <th className="text-left py-2 px-3 text-slate-500">Label</th>
                        <th className="text-left py-2 px-3 text-slate-500">Confidence</th>
                        <th className="text-left py-2 px-3 text-slate-500">Bounding Box</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ev.boxes?.slice(0, 8).map((box, j) => (
                        <tr key={j} className="border-b border-space-700/30">
                          <td className="py-2 px-3 text-slate-500">{j + 1}</td>
                          <td className="py-2 px-3 text-white font-medium">{box.label}</td>
                          <td className="py-2 px-3" style={{ color: confidenceColor(box.score) }}>
                            {(box.score * 100).toFixed(1)}%
                          </td>
                          <td className="py-2 px-3 font-mono text-slate-400">
                            [{box.xmin}, {box.ymin}, {box.xmax}, {box.ymax}]
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
        </div>
      )}

      {tab === 'visual' && (
        <div className="space-y-4">
          {evidence.visual_evidence
            .filter(ev => ev.type !== 'bounding_boxes' && ev.public_url)
            .map((ev, i) => (
              <div key={i} className="space-y-2">
                <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  {ev.description || ev.type}
                </div>
                <div className="rounded-xl overflow-hidden border border-space-700/50">
                  <img
                    src={ev.public_url}
                    alt={ev.description || 'Visual evidence'}
                    className="w-full object-contain bg-space-900"
                    style={{ maxHeight: '400px' }}
                  />
                </div>
              </div>
            ))}

          {evidence.visual_evidence.filter(ev => ev.type !== 'bounding_boxes').length === 0 && (
            <div className="text-center text-slate-500 py-8 text-sm">
              No visual overlays generated for this task type.
            </div>
          )}
        </div>
      )}

      {tab === 'report' && evidence.report_key && (
        <div className="space-y-4">
          <div className="glass-light rounded-xl p-6 text-center space-y-4">
            <div className="text-4xl">📄</div>
            <h4 className="font-bold text-white">Analysis Report Ready</h4>
            <p className="text-sm text-slate-400">
              Download the full analysis report with execution trace, visual evidence, and confidence details.
            </p>
            <div className="flex justify-center gap-3">
              <a
                href={evidence.report_key}
                download
                target="_blank"
                rel="noopener noreferrer"
                id="download-pdf"
                className="btn-primary"
              >
                ⬇️ Download PDF
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
