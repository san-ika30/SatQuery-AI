'use client'

import { useState, useEffect, useCallback } from 'react'
import Link from 'next/link'
import {
  fetchSessions, fetchStats,
  type SessionRow, type SessionStats,
  taskTypeLabel, inputModeLabel, statusColor, confidenceColor,
} from '@/lib/api'

const TASK_ICONS: Record<string, string> = {
  vqa: '💬',
  caption: '📝',
  grounding: '📍',
  change_detection: '🔄',
  change_vqa: '🔄',
  sar_fusion: '📡',
  unknown: '❓',
}

const MODE_ICONS: Record<string, string> = {
  single: '🖼️',
  bitemporal: '🔄',
  crossmodal: '📡',
}

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="card text-center">
      <div className="text-3xl font-bold text-white font-space mb-1">{value}</div>
      <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{label}</div>
      {sub && <div className="text-xs text-cyan-400 mt-1">{sub}</div>}
    </div>
  )
}

function SessionCard({ session }: { session: SessionRow }) {
  const color = statusColor(session.status)
  const taskIcon = TASK_ICONS[session.task_type || 'unknown'] || '❓'
  const modeIcon = MODE_ICONS[session.input_mode] || '🖼️'

  const formatted = new Date(session.created_at).toLocaleString('en-IN', {
    dateStyle: 'medium',
    timeStyle: 'short',
  })

  return (
    <Link
      href={`/results/${session.id}`}
      id={`session-${session.id.slice(0, 8)}`}
      className="card block hover:border-cyan-400/40 hover:shadow-[0_4px_24px_rgba(0,212,255,0.08)] transition-all duration-200 group"
    >
      <div className="flex items-start justify-between gap-4">
        {/* Left */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-2 flex-wrap">
            <span className="text-lg">{taskIcon}</span>
            <span className="badge badge-cyan text-xs">
              {taskTypeLabel(session.task_type || 'unknown')}
            </span>
            <span className="badge badge-amber text-xs">
              {modeIcon} {inputModeLabel(session.input_mode)}
            </span>
          </div>
          <p className="text-sm text-slate-200 leading-relaxed line-clamp-2 mb-3">
            {session.query_text}
          </p>
          <div className="flex items-center gap-3 text-xs text-slate-500">
            <span className="font-mono">{session.id.slice(0, 12)}…</span>
            <span>·</span>
            <span>{formatted}</span>
          </div>
        </div>

        {/* Right — Status */}
        <div className="flex flex-col items-end gap-2 flex-shrink-0">
          <div
            className="flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-full border"
            style={{ color, borderColor: `${color}40`, background: `${color}12` }}
          >
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
            {session.status.charAt(0).toUpperCase() + session.status.slice(1)}
          </div>
          <span className="text-xs text-slate-600 group-hover:text-cyan-400/70 transition-colors">
            View details →
          </span>
        </div>
      </div>
    </Link>
  )
}

export default function ResultsPage() {
  const [sessions, setSessions] = useState<SessionRow[]>([])
  const [stats, setStats] = useState<SessionStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [page, setPage] = useState(0)
  const [total, setTotal] = useState(0)
  const [filterStatus, setFilterStatus] = useState<string>('')
  const [filterTask, setFilterTask] = useState<string>('')
  const LIMIT = 10

  const load = useCallback(async (pg: number, status: string, task: string) => {
    setLoading(true)
    setError(null)
    try {
      const [sessData, statsData] = await Promise.all([
        fetchSessions(LIMIT, pg * LIMIT, status || undefined, task || undefined),
        fetchStats(),
      ])
      setSessions(sessData.sessions)
      setTotal(sessData.total)
      setStats(statsData)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load sessions')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load(page, filterStatus, filterTask) }, [load, page, filterStatus, filterTask])

  const handleFilter = (status: string, task: string) => {
    setFilterStatus(status)
    setFilterTask(task)
    setPage(0)
  }

  const TASK_FILTERS = ['', 'vqa', 'caption', 'grounding', 'change_detection', 'change_vqa', 'sar_fusion']
  const STATUS_FILTERS = ['', 'completed', 'failed', 'pending']

  const totalPages = Math.ceil(total / LIMIT)

  return (
    <div className="min-h-screen bg-space-950">
      {/* Stars BG */}
      <div className="stars-bg" />

      {/* Top bar */}
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
          <span className="badge badge-cyan text-xs hidden sm:flex">Session History</span>
          <Link href="/analyze" className="btn-primary text-sm py-2 px-4">
            + New Analysis
          </Link>
        </div>
      </header>

      <main className="relative z-10 max-w-5xl mx-auto px-4 sm:px-6 py-10 space-y-8">

        {/* Page header */}
        <div>
          <h1 className="font-space text-3xl font-bold text-white mb-2">
            Analysis <span className="gradient-text">History</span>
          </h1>
          <p className="text-slate-400 text-sm">
            Browse all past satellite image analysis sessions and their results.
          </p>
        </div>

        {/* Stats Grid */}
        {stats && (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <StatCard label="Total Sessions" value={stats.total_sessions} />
            <StatCard
              label="Completed"
              value={stats.by_status['completed'] ?? 0}
              sub={`${stats.total_sessions > 0 ? Math.round(((stats.by_status['completed'] ?? 0) / stats.total_sessions) * 100) : 0}% success rate`}
            />
            <StatCard
              label="Top Task"
              value={
                Object.entries(stats.by_task_type).sort((a, b) => b[1] - a[1])[0]?.[0]
                  ? taskTypeLabel(Object.entries(stats.by_task_type).sort((a, b) => b[1] - a[1])[0][0])
                  : '—'
              }
            />
            <StatCard
              label="Failed"
              value={stats.by_status['failed'] ?? 0}
            />
          </div>
        )}

        {/* Task type breakdown */}
        {stats && Object.keys(stats.by_task_type).length > 0 && (
          <div className="card">
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
              Task Distribution
            </div>
            <div className="space-y-2">
              {Object.entries(stats.by_task_type)
                .sort((a, b) => b[1] - a[1])
                .map(([task, count]) => {
                  const pct = stats.total_sessions > 0 ? (count / stats.total_sessions) * 100 : 0
                  return (
                    <div key={task} className="flex items-center gap-3">
                      <span className="text-sm w-48 text-slate-300 flex-shrink-0">
                        {TASK_ICONS[task] || '❓'} {taskTypeLabel(task)}
                      </span>
                      <div className="flex-1 confidence-bar">
                        <div
                          className="confidence-fill"
                          style={{
                            width: `${pct}%`,
                            background: 'linear-gradient(90deg, #00d4ff, #ffb703)',
                          }}
                        />
                      </div>
                      <span className="text-xs font-mono text-slate-400 w-8 text-right">{count}</span>
                    </div>
                  )
                })}
            </div>
          </div>
        )}

        {/* Filters */}
        <div className="flex flex-wrap gap-3 items-center">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 uppercase tracking-wider">Status:</span>
            {STATUS_FILTERS.map(s => (
              <button
                key={s || 'all'}
                id={`filter-status-${s || 'all'}`}
                onClick={() => handleFilter(s, filterTask)}
                className={`text-xs px-3 py-1.5 rounded-full border transition-all ${
                  filterStatus === s
                    ? 'border-cyan-400/60 bg-cyan-400/10 text-cyan-400'
                    : 'border-space-700/60 text-slate-400 hover:border-cyan-400/30'
                }`}
              >
                {s || 'All'}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500 uppercase tracking-wider">Task:</span>
            <select
              id="filter-task"
              value={filterTask}
              onChange={e => handleFilter(filterStatus, e.target.value)}
              className="text-xs px-3 py-1.5 rounded-full border border-space-700/60 bg-space-800 text-slate-300 outline-none focus:border-cyan-400/40"
            >
              {TASK_FILTERS.map(t => (
                <option key={t} value={t}>{t ? taskTypeLabel(t) : 'All Tasks'}</option>
              ))}
            </select>
          </div>
          {(filterStatus || filterTask) && (
            <button
              onClick={() => handleFilter('', '')}
              className="text-xs text-slate-500 hover:text-slate-300 underline"
            >
              Clear filters
            </button>
          )}
        </div>

        {/* Session List */}
        {loading ? (
          <div className="space-y-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="card skeleton h-28 rounded-xl" />
            ))}
          </div>
        ) : error ? (
          <div className="card border-red-500/30 bg-red-500/5 text-center py-12">
            <div className="text-4xl mb-4">⚠️</div>
            <h3 className="font-bold text-red-400 mb-2">Failed to Load Sessions</h3>
            <p className="text-sm text-slate-400">{error}</p>
            <p className="text-xs text-slate-500 mt-2">
              Make sure your backend is running and Supabase is configured.
            </p>
          </div>
        ) : sessions.length === 0 ? (
          <div className="card text-center py-16">
            <div className="text-6xl mb-6 animate-float">🛰️</div>
            <h3 className="font-space text-xl font-bold text-white mb-3">No Sessions Found</h3>
            <p className="text-slate-400 text-sm mb-6">
              {filterStatus || filterTask
                ? 'No sessions match your current filters.'
                : 'Your analysis sessions will appear here after you run your first analysis.'}
            </p>
            <Link href="/analyze" className="btn-primary">
              🚀 Start First Analysis
            </Link>
          </div>
        ) : (
          <div className="space-y-4">
            {sessions.map(session => (
              <SessionCard key={session.id} session={session} />
            ))}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between">
            <p className="text-xs text-slate-500">
              Showing {page * LIMIT + 1}–{Math.min((page + 1) * LIMIT, total)} of {total} sessions
            </p>
            <div className="flex gap-2">
              <button
                id="prev-page"
                onClick={() => setPage(p => Math.max(0, p - 1))}
                disabled={page === 0}
                className="btn-secondary text-xs py-2 px-4 disabled:opacity-40"
              >
                ← Prev
              </button>
              <span className="flex items-center text-xs text-slate-400 px-2">
                {page + 1} / {totalPages}
              </span>
              <button
                id="next-page"
                onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="btn-secondary text-xs py-2 px-4 disabled:opacity-40"
              >
                Next →
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
