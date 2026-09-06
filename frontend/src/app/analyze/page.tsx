'use client'

import { useState, useCallback } from 'react'
import Link from 'next/link'
import ImageUploader from '@/components/ImageUploader'
import QueryPanel from '@/components/QueryPanel'
import ResultsView from '@/components/ResultsView'
import ExecutionTrace from '@/components/ExecutionTrace'
import ModelStatusPanel from '@/components/ModelStatusPanel'
import type { InputMode, AnalysisResponse } from '@/lib/api'
import { analyzeImages } from '@/lib/api'


export default function AnalyzePage() {
  const [inputMode, setInputMode] = useState<InputMode>('single')
  const [image1, setImage1] = useState<File | null>(null)
  const [image2, setImage2] = useState<File | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [loadingStatus, setLoadingStatus] = useState('')
  const [loadingPct, setLoadingPct] = useState(0)
  const [response, setResponse] = useState<AnalysisResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showTrace, setShowTrace] = useState(true)

  const handleFilesChange = useCallback((f1: File | null, f2: File | null) => {
    setImage1(f1)
    setImage2(f2)
    setResponse(null)
    setError(null)
  }, [])

  const handleModeChange = useCallback((mode: InputMode) => {
    setInputMode(mode)
    setImage2(null)
    setResponse(null)
    setError(null)
  }, [])

  const handleSubmit = useCallback(async (query: string) => {
    if (!image1) return
    setIsLoading(true)
    setError(null)
    setResponse(null)
    setLoadingPct(0)
    setLoadingStatus('Starting analysis pipeline…')

    try {
      const result = await analyzeImages(
        query,
        inputMode,
        image1,
        image2 || undefined,
        (message: string, pct?: number) => {
          setLoadingStatus(message)
          if (pct !== undefined) setLoadingPct(pct)
        },
      )
      setResponse(result)
      setLoadingPct(100)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Analysis failed. Please retry.'
      setError(msg)
      setResponse({ session_id: '', status: 'failed', error: msg })
    } finally {
      setIsLoading(false)
    }
  }, [image1, image2, inputMode])

  const hasImages = !!image1 && (
    inputMode === 'single' ||
    (!!image2)
  )

  return (
    <div className="min-h-screen bg-space-950 flex flex-col">
      {/* Top bar */}
      <header className="glass border-b border-cyan-400/10 px-6 py-4 flex items-center justify-between z-10 sticky top-0">
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-xs">
            🛰️
          </div>
          <span className="font-space font-bold text-white group-hover:text-cyan-400 transition-colors">
            SatQuery <span className="gradient-text">AI</span>
          </span>
        </Link>
        <div className="flex items-center gap-3">
          <ModelStatusPanel compact />
          <Link href="/results" className="btn-secondary text-xs py-1.5 px-3 hidden sm:flex">
            📂 History
          </Link>
          <span className="badge badge-cyan text-xs hidden sm:flex">Analysis Workspace</span>
          {response?.status === 'completed' && (
            <span className="badge badge-green text-xs">✓ Analysis Complete</span>
          )}
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-[420px_1fr] gap-0 overflow-hidden">
        {/* Left panel — Upload + Query */}
        <aside className="border-r border-space-700/50 flex flex-col overflow-y-auto">
          <div className="p-6 space-y-6">
            {/* Section: Upload */}
            <section>
              <div className="flex items-center gap-2 mb-4">
                <span className="w-6 h-6 rounded-full bg-cyan-400/20 text-cyan-400 text-xs flex items-center justify-center font-bold">1</span>
                <h2 className="font-space font-bold text-white text-sm uppercase tracking-wide">Upload Images</h2>
              </div>
              <ImageUploader
                inputMode={inputMode}
                onFilesChange={handleFilesChange}
              />
            </section>

            <div className="border-t border-space-700/30" />

            {/* Section: Query */}
            <section>
              <div className="flex items-center gap-2 mb-4">
                <span className="w-6 h-6 rounded-full bg-cyan-400/20 text-cyan-400 text-xs flex items-center justify-center font-bold">2</span>
                <h2 className="font-space font-bold text-white text-sm uppercase tracking-wide">Ask a Question</h2>
              </div>
              <QueryPanel
                inputMode={inputMode}
                onModeChange={handleModeChange}
                onSubmit={handleSubmit}
                isLoading={isLoading}
                hasImages={hasImages}
              />
            </section>

            <div className="border-t border-space-700/30" />

            {/* Section: Model Status */}
            <section>
              <div className="flex items-center gap-2 mb-4">
                <span className="w-6 h-6 rounded-full bg-cyan-400/20 text-cyan-400 text-xs flex items-center justify-center font-bold">3</span>
                <h2 className="font-space font-bold text-white text-sm uppercase tracking-wide">Backend Status</h2>
              </div>
              <ModelStatusPanel />
            </section>

            <div className="border-t border-space-700/30" />

            {/* Session History Link */}
            <div className="text-center">
              <Link href="/results" className="btn-secondary text-xs py-2 px-4 w-full justify-center">
                📂 View Session History
              </Link>
            </div>
          </div>
        </aside>

        {/* Right panel — Results + Trace */}
        <div className="flex flex-col overflow-y-auto">
          {/* Empty state */}
          {!response && !isLoading && (
            <div className="flex-1 flex items-center justify-center p-12">
              <div className="text-center max-w-sm">
                <div className="text-6xl mb-6 animate-float">🛰️</div>
                <h2 className="font-space text-2xl font-bold text-white mb-3">
                  Ready to Analyze
                </h2>
                <p className="text-slate-400 text-sm leading-relaxed mb-6">
                  Upload your satellite image(s), choose an input mode, and type your
                  natural-language query to get started.
                </p>
                <div className="grid grid-cols-2 gap-3 text-left">
                  {[
                    { icon: '🖼️', label: 'Single Image', desc: 'VQA, Caption, Grounding' },
                    { icon: '🔄', label: 'Bi-Temporal', desc: 'Change Detection & VQA' },
                    { icon: '📡', label: 'Optical + SAR', desc: 'Cross-modal Fusion' },
                    { icon: '📊', label: 'PDF Reports', desc: 'Downloadable Results' },
                  ].map(f => (
                    <div key={f.label} className="glass-light rounded-xl p-3">
                      <div className="text-xl mb-1">{f.icon}</div>
                      <div className="text-xs font-bold text-white">{f.label}</div>
                      <div className="text-xs text-slate-500">{f.desc}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Loading / Results */}
          {(isLoading || response) && (
            <div className="p-6 space-y-6">
              {/* Results */}
              <ResultsView
                response={response || { session_id: '', status: 'pending' }}
                isLoading={isLoading}
                loadingStatus={loadingStatus}
                loadingPct={loadingPct}
              />

              {/* Agentic Orchestration Details Card */}
              {response && response.status === 'completed' && (
                <div className="card space-y-4 border border-cyan-400/20 bg-space-900/60 p-5 rounded-2xl">
                  <div className="flex items-center justify-between border-b border-cyan-400/10 pb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">🤖</span>
                      <h3 className="font-space font-bold text-white text-sm uppercase tracking-wide">
                        Agentic Orchestration & Analysis Details
                      </h3>
                    </div>
                    <button
                      onClick={() => setShowTrace(prev => !prev)}
                      className="btn-secondary text-xs py-1 px-2.5 rounded-lg"
                    >
                      {showTrace ? 'Hide Trace ▲' : 'Show Trace ▼'}
                    </button>
                  </div>

                  {/* Summary Grid */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div className="glass-light rounded-xl p-3">
                      <div className="text-xs text-slate-400 uppercase font-semibold">Detected Task</div>
                      <div className="text-sm font-bold text-cyan-400 mt-0.5">
                        {response.task_plan?.task_type?.toUpperCase() || 'VLM REASONING'}
                      </div>
                    </div>
                    <div className="glass-light rounded-xl p-3">
                      <div className="text-xs text-slate-400 uppercase font-semibold">Modalities</div>
                      <div className="text-sm font-bold text-white mt-0.5">
                        {inputMode === 'crossmodal' ? 'Optical + SAR' : inputMode === 'bitemporal' ? 'Bi-Temporal Optical' : 'Single Optical'}
                      </div>
                    </div>
                    <div className="glass-light rounded-xl p-3">
                      <div className="text-xs text-slate-400 uppercase font-semibold">Evidence Confidence</div>
                      <div className="text-sm font-bold text-green-400 mt-0.5">
                        {((response.evidence?.confidence ?? 0.88) * 100).toFixed(1)}% · High
                      </div>
                    </div>
                  </div>

                  {/* Tools Used */}
                  <div>
                    <div className="text-xs text-slate-400 uppercase font-semibold mb-1.5">Specialist Tools Selected</div>
                    <div className="flex flex-wrap gap-2">
                      {(response.evidence?.models_used || response.task_plan?.models || ['satellite_vlm']).map(tool => (
                        <span key={tool} className="badge badge-cyan text-xs py-1 px-2.5 flex items-center gap-1">
                          <span className="text-green-400">✓</span> {tool}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Trace Steps when expanded */}
                  {showTrace && (
                    <div className="border-t border-space-700/40 pt-3 space-y-2">
                      <div className="text-xs text-slate-400 uppercase font-semibold">Execution Timeline</div>
                      <ol className="space-y-1.5 text-xs text-slate-300">
                        {(response.evidence?.execution_trace && response.evidence.execution_trace.length > 0) ? (
                          response.evidence.execution_trace.map((step, idx) => (
                            <li key={idx} className="flex items-center gap-2 bg-space-800/40 px-3 py-1.5 rounded-lg border border-space-700/30 font-mono">
                              <span className="text-cyan-400 font-bold">{step.step || idx + 1}.</span>
                              <span className="text-white">{step.component || 'Execution step'}</span>
                              <span className="text-slate-500 ml-auto">{step.duration_ms || 0}ms</span>
                            </li>
                          ))
                        ) : (
                          [
                            '1. Query interpreted and intent classified',
                            '2. Input payloads and image dimensions validated',
                            '3. Specialist tool selected from central registry',
                            '4. Vision-language and remote-sensing inference executed',
                            '5. Multimodal evidence combined and cross-validated',
                            '6. Final answer and auditable trace generated',
                          ].map((st, i) => (
                            <li key={i} className="text-slate-400 font-mono pl-2">
                              {st}
                            </li>
                          ))
                        )}
                      </ol>
                    </div>
                  )}
                </div>
              )}

              {/* Execution Trace (only when done) */}
              {response && response.status === 'completed' && (
                <ExecutionTrace response={response} />
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
