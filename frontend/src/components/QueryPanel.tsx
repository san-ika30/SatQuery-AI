'use client'

import { useState, useRef, KeyboardEvent } from 'react'
import type { InputMode } from '@/lib/api'

const EXAMPLE_QUERIES: Record<InputMode, string[]> = {
  single: [
    'Describe the land-cover and major objects visible in this image.',
    'What type of vegetation is present in this scene?',
    'Identify and highlight the water bodies in this image.',
    'What is the dominant land use in this satellite image?',
    'Are there any urban structures visible? If yes, describe them.',
  ],
  bitemporal: [
    'What changed between these two dates, and where did the change occur?',
    'Has the built-up area increased, decreased, or remained unchanged?',
    'Describe the vegetation changes visible between the two images.',
    'Identify any flood or disaster-related changes in the scene.',
    'Quantify and describe the deforestation visible in these images.',
  ],
  crossmodal: [
    'Use the optical and SAR images together to identify built-up and water-covered regions.',
    'What complementary information does the SAR image provide about the scene?',
    'Identify flooded areas using both optical and SAR imagery.',
    'Detect built-up structures using the SAR backscatter and optical context.',
    'Describe the land cover using information from both image modalities.',
  ],
}

interface QueryPanelProps {
  inputMode: InputMode
  onModeChange: (mode: InputMode) => void
  onSubmit: (query: string) => void
  isLoading: boolean
  hasImages: boolean
}

export default function QueryPanel({
  inputMode,
  onModeChange,
  onSubmit,
  isLoading,
  hasImages,
}: QueryPanelProps) {
  const [query, setQuery] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleSubmit = () => {
    if (!query.trim() || isLoading || !hasImages) return
    onSubmit(query.trim())
  }

  const handleExampleClick = (q: string) => {
    setQuery(q)
    textareaRef.current?.focus()
  }

  const modeConfig: { id: InputMode; label: string; icon: string; desc: string }[] = [
    { id: 'single', label: 'Single Image', icon: '🖼️', desc: 'One optical, multispectral, or SAR image' },
    { id: 'bitemporal', label: 'Bi-Temporal', icon: '🔄', desc: 'Two images of the same area at different times' },
    { id: 'crossmodal', label: 'Optical + SAR', icon: '📡', desc: 'Co-registered optical and SAR image pair' },
  ]

  return (
    <div className="space-y-5">
      {/* Input Mode Selector */}
      <div>
        <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
          Input Mode
        </label>
        <div className="grid grid-cols-3 gap-2">
          {modeConfig.map(m => (
            <button
              key={m.id}
              id={`mode-${m.id}`}
              onClick={() => onModeChange(m.id)}
              className={`rounded-lg p-3 text-left transition-all duration-200 border ${
                inputMode === m.id
                  ? 'border-cyan-400/60 bg-cyan-400/10 text-white'
                  : 'border-space-700/60 bg-space-800/50 text-slate-400 hover:border-cyan-400/30 hover:text-slate-200'
              }`}
            >
              <div className="text-lg mb-1">{m.icon}</div>
              <div className="text-xs font-bold leading-tight">{m.label}</div>
            </button>
          ))}
        </div>
        <p className="text-xs text-slate-500 mt-2">
          {modeConfig.find(m => m.id === inputMode)?.desc}
        </p>
      </div>

      {/* Query Input */}
      <div>
        <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
          Natural Language Query
        </label>
        <div className={`relative rounded-xl border transition-all duration-200 ${
          query ? 'border-cyan-400/40' : 'border-space-600/50'
        } bg-space-800/60 focus-within:border-cyan-400/60 focus-within:shadow-[0_0_20px_rgba(0,212,255,0.1)]`}>
          <textarea
            ref={textareaRef}
            id="query-input"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your satellite image(s)..."
            rows={4}
            className="w-full bg-transparent px-4 pt-4 pb-10 text-sm text-white placeholder-slate-500 resize-none outline-none rounded-xl"
          />
          <div className="absolute bottom-3 right-3 flex items-center gap-2">
            <span className="text-xs text-slate-600">
              {query.length}/2000 · Ctrl+Enter to submit
            </span>
          </div>
        </div>
      </div>

      {/* Example Queries */}
      <div>
        <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
          Example Queries
        </label>
        <div className="space-y-2">
          {EXAMPLE_QUERIES[inputMode].slice(0, 3).map((q, i) => (
            <button
              key={i}
              id={`example-query-${i}`}
              onClick={() => handleExampleClick(q)}
              className="w-full text-left text-xs text-slate-400 hover:text-cyan-400 px-3 py-2 rounded-lg border border-transparent hover:border-cyan-400/20 hover:bg-cyan-400/5 transition-all duration-150 leading-relaxed"
            >
              <span className="text-cyan-400/60 mr-2">›</span>{q}
            </button>
          ))}
        </div>
      </div>

      {/* Submit */}
      <button
        id="analyze-button"
        onClick={handleSubmit}
        disabled={isLoading || !hasImages || !query.trim()}
        className="btn-primary w-full py-4 text-base"
      >
        {isLoading ? (
          <>
            <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
            </svg>
            Analyzing...
          </>
        ) : (
          <>🔍 Analyze with SatQuery AI</>
        )}
      </button>

      {!hasImages && (
        <p className="text-xs text-center text-amber-400/70">
          ⚠️ Upload at least one image to begin analysis
        </p>
      )}
    </div>
  )
}
