'use client'

import { useMemo } from 'react'
import type { AnalysisResponse, ExecutionStep } from '@/lib/api'
import { confidenceColor, confidenceLabel, taskTypeLabel } from '@/lib/api'

interface ExecutionTraceProps {
  response: AnalysisResponse
}

const STATUS_ICON: Record<string, string> = {
  ok: '✅',
  warning: '⚠️',
  failed: '❌',
  skipped: '⏭️',
  running: '⏳',
}

const STATUS_COLOR: Record<string, string> = {
  ok: '#34d399',
  warning: '#ffb703',
  failed: '#f87171',
  skipped: '#64748b',
  running: '#00d4ff',
}

export default function ExecutionTrace({ response }: ExecutionTraceProps) {
  const { evidence, task_plan, session_id } = response

  const trace: ExecutionStep[] = evidence?.execution_trace ?? []
  const confidence = evidence?.confidence ?? 0

  const totalDuration = useMemo(() => {
    if (!trace.length) return 0
    return trace[trace.length - 1]?.duration_ms ?? 0
  }, [trace])

  return (
    <div className="card space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-space font-bold text-white text-base">
            🤖 Agentic Execution Trace
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Session: <span className="font-mono text-slate-400">{session_id.slice(0, 12)}…</span>
          </p>
        </div>
        <div className="text-right">
          <div className="text-xs text-slate-400">Total time</div>
          <div className="font-mono font-bold text-cyan-400 text-sm">{totalDuration}ms</div>
        </div>
      </div>

      {/* Task Plan Summary */}
      {task_plan && (
        <div className="glass-light rounded-xl p-4 space-y-3">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="badge badge-cyan">Task: {taskTypeLabel(task_plan.task_type)}</span>
            <span className="badge badge-amber">Mode: {task_plan.input_mode.toUpperCase()}</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {task_plan.models.map(m => (
              <span key={m} className="text-xs font-mono bg-space-700 text-slate-300 px-2 py-1 rounded">
                {m}
              </span>
            ))}
          </div>
          {task_plan.reasoning && (
            <p className="text-xs text-slate-400 italic border-l-2 border-cyan-400/30 pl-3">
              {task_plan.reasoning}
            </p>
          )}
        </div>
      )}

      {/* Confidence Meter */}
      <div>
        <div className="flex justify-between items-center mb-2">
          <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Aggregate Confidence
          </span>
          <span className="text-sm font-bold" style={{ color: confidenceColor(confidence) }}>
            {(confidence * 100).toFixed(1)}% · {confidenceLabel(confidence)}
          </span>
        </div>
        <div className="confidence-bar">
          <div
            className="confidence-fill"
            style={{
              width: `${confidence * 100}%`,
              background: `linear-gradient(90deg, ${confidenceColor(confidence)}, ${confidenceColor(Math.min(confidence + 0.1, 1))})`,
            }}
          />
        </div>
      </div>

      {/* Trace Steps */}
      <div>
        <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
          Pipeline Steps
        </div>
        <div className="space-y-1">
          {trace.map((step, i) => (
            <div key={i} className="trace-step group">
              <div
                className="trace-dot"
                style={{ background: STATUS_COLOR[step.status] || '#64748b' }}
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-slate-500 w-5 flex-shrink-0">
                    {String(step.step).padStart(2, '0')}
                  </span>
                  <span className="text-xs text-slate-300 truncate">{step.component}</span>
                  <span className="text-sm flex-shrink-0">
                    {STATUS_ICON[step.status] || '◦'}
                  </span>
                </div>
                {/* Extra details */}
                {Object.entries(step)
                  .filter(([k]) => !['step', 'component', 'status', 'duration_ms'].includes(k))
                  .slice(0, 2)
                  .map(([k, v]) => (
                    <div key={k} className="text-xs text-slate-500 font-mono ml-7 mt-0.5">
                      {k}: {typeof v === 'object' ? JSON.stringify(v) : String(v)}
                    </div>
                  ))}
              </div>
              <div className="text-xs font-mono text-slate-600 flex-shrink-0">
                {step.duration_ms}ms
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Models Used */}
      {evidence?.models_used && evidence.models_used.length > 0 && (
        <div>
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
            Models Executed
          </div>
          <div className="flex flex-wrap gap-2">
            {evidence.models_used.map(m => (
              <span key={m} className="badge badge-cyan text-xs">
                🤖 {m}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
