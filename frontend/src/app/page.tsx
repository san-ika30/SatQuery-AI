'use client'

import Link from 'next/link'
import { useState, useEffect } from 'react'

const TASKS = [
  { icon: '🛰️', label: 'Visual Question Answering', desc: 'Ask anything about a satellite image' },
  { icon: '📍', label: 'Region Grounding', desc: 'Locate and highlight specific features' },
  { icon: '🔄', label: 'Change Detection', desc: 'Detect what changed between two dates' },
  { icon: '📡', label: 'Optical–SAR Fusion', desc: 'Joint analysis of optical and radar imagery' },
]

const QUERIES = [
  '"Describe the land-cover and major objects visible in this image."',
  '"What changed between these two dates, and where did the change occur?"',
  '"Highlight the water body referred to in the query."',
  '"Use the optical and SAR images together to identify built-up regions."',
  '"Has the built-up area increased, decreased, or remained unchanged?"',
]

export default function HomePage() {
  const [queryIdx, setQueryIdx] = useState(0)
  const [displayed, setDisplayed] = useState('')
  const [charIdx, setCharIdx] = useState(0)

  // Typewriter effect
  useEffect(() => {
    const target = QUERIES[queryIdx]
    if (charIdx < target.length) {
      const t = setTimeout(() => {
        setDisplayed(target.slice(0, charIdx + 1))
        setCharIdx(c => c + 1)
      }, 28)
      return () => clearTimeout(t)
    } else {
      const t = setTimeout(() => {
        setQueryIdx(i => (i + 1) % QUERIES.length)
        setCharIdx(0)
        setDisplayed('')
      }, 3200)
      return () => clearTimeout(t)
    }
  }, [charIdx, queryIdx])

  return (
    <main className="relative min-h-screen overflow-hidden">
      {/* Background */}
      <div className="stars-bg" />
      <div
        className="fixed inset-0 pointer-events-none z-0"
        style={{
          backgroundImage: `
            radial-gradient(1px 1px at 20% 30%, rgba(0,212,255,0.6) 0%, transparent 0%),
            radial-gradient(1px 1px at 80% 15%, rgba(255,183,3,0.4) 0%, transparent 0%),
            radial-gradient(1px 1px at 60% 70%, rgba(0,212,255,0.3) 0%, transparent 0%),
            radial-gradient(1px 1px at 10% 80%, rgba(255,183,3,0.2) 0%, transparent 0%),
            radial-gradient(2px 2px at 45% 50%, rgba(0,212,255,0.4) 0%, transparent 0%),
            radial-gradient(1px 1px at 90% 60%, rgba(255,183,3,0.3) 0%, transparent 0%)
          `,
        }}
      />

      {/* Navbar */}
      <nav className="relative z-10 flex items-center justify-between px-8 py-5 glass border-b border-cyan-400/10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-sm">
            🛰️
          </div>
          <span className="font-space font-bold text-xl text-white">SatQuery <span className="gradient-text">AI</span></span>
        </div>
        <div className="flex items-center gap-4">
          <span className="badge badge-cyan text-xs">ISRO SIH 2026</span>
          <Link href="/results" className="btn-secondary text-sm py-2 px-4 hidden sm:flex">
            📂 History
          </Link>
          <Link href="/analyze" className="btn-primary text-sm py-2 px-5">
            Launch App →
          </Link>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative z-10 flex flex-col items-center text-center px-6 pt-24 pb-16">
        {/* Satellite orbital decoration */}
        <div className="relative w-48 h-48 mb-12 animate-float">
          <div className="absolute inset-0 rounded-full border border-cyan-400/20" />
          <div className="absolute inset-4 rounded-full border border-cyan-400/15" />
          <div className="absolute inset-8 rounded-full bg-gradient-to-br from-space-800 to-space-900 border border-cyan-400/30 flex items-center justify-center text-5xl">
            🛰️
          </div>
          {/* Orbiting dot */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full">
            <div
              className="absolute top-1/2 left-1/2 w-3 h-3 -translate-x-1/2 -translate-y-1/2 animate-orbit"
              style={{ animationDuration: '6s' }}
            >
              <div className="w-3 h-3 rounded-full bg-amber-400 shadow-lg" style={{ boxShadow: '0 0 8px #ffb703' }} />
            </div>
            <div
              className="absolute top-1/2 left-1/2 w-2 h-2 -translate-x-1/2 -translate-y-1/2 animate-orbit"
              style={{ animationDuration: '10s', animationDirection: 'reverse' }}
            >
              <div className="w-2 h-2 rounded-full bg-cyan-400" style={{ boxShadow: '0 0 6px #00d4ff' }} />
            </div>
          </div>
        </div>

        <h1 className="font-space text-5xl md:text-7xl font-bold mb-6 leading-tight max-w-4xl">
          Ask Questions About{' '}
          <span className="gradient-text text-glow-cyan">Satellite Imagery</span>
        </h1>
        <p className="text-lg md:text-xl text-slate-400 max-w-2xl mb-8 leading-relaxed">
          An agentic vision-language AI that automatically selects specialist remote-sensing models,
          executes analysis pipelines, and returns evidence-grounded answers.
        </p>

        {/* Typewriter query demo */}
        <div className="glass rounded-xl px-6 py-4 mb-10 max-w-2xl w-full min-h-[64px] flex items-center">
          <span className="text-cyan-400 font-mono text-sm mr-2">Query:</span>
          <span className="text-slate-200 font-mono text-sm text-left flex-1">
            {displayed}
            <span className="inline-block w-0.5 h-4 bg-cyan-400 ml-0.5 animate-pulse" />
          </span>
        </div>

        <div className="flex flex-col sm:flex-row gap-4">
          <Link href="/analyze" className="btn-primary text-base px-8 py-3">
            🚀 Start Analyzing
          </Link>
          <a
            href="https://github.com/your-repo/satquery-ai"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-secondary text-base px-8 py-3"
          >
            📂 View on GitHub
          </a>
        </div>
      </section>

      {/* Feature Cards */}
      <section className="relative z-10 px-6 md:px-12 py-16 max-w-6xl mx-auto">
        <h2 className="font-space text-3xl font-bold text-center mb-3">
          Specialist <span className="gradient-text">AI Capabilities</span>
        </h2>
        <p className="text-slate-400 text-center mb-12">
          Automatically routed by an agentic controller — you just ask the question.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {TASKS.map((t) => (
            <div key={t.label} className="card group cursor-default">
              <div className="text-3xl mb-4">{t.icon}</div>
              <h3 className="text-base font-bold text-white mb-2 group-hover:text-cyan-400 transition-colors">
                {t.label}
              </h3>
              <p className="text-sm text-slate-400 leading-relaxed">{t.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Pipeline Steps */}
      <section className="relative z-10 px-6 md:px-12 py-16 max-w-5xl mx-auto">
        <h2 className="font-space text-3xl font-bold text-center mb-3">
          How <span className="gradient-text">It Works</span>
        </h2>
        <p className="text-slate-400 text-center mb-12">
          Six-step agentic pipeline from image upload to evidence-grounded answer.
        </p>
        <div className="flex flex-col md:flex-row items-start md:items-center gap-4 md:gap-0">
          {[
            { step: '01', title: 'Upload', desc: 'Single, bi-temporal, or optical+SAR pair', icon: '📁' },
            { step: '02', title: 'Preprocess', desc: 'GeoTIFF parsing, band extraction, normalization', icon: '⚙️' },
            { step: '03', title: 'Route', desc: 'Mistral-7B classifies task and selects models', icon: '🧠' },
            { step: '04', title: 'Execute', desc: 'Specialist models run VQA, grounding, change detection', icon: '🤖' },
            { step: '05', title: 'Integrate', desc: 'Evidence merged, confidence computed', icon: '🔗' },
            { step: '06', title: 'Report', desc: 'Visual overlays + downloadable PDF report', icon: '📊' },
          ].map((s, i, arr) => (
            <div key={s.step} className="flex md:flex-col items-center gap-3 md:gap-0 flex-1">
              <div className="glass rounded-xl p-4 text-center flex-shrink-0 md:w-full hover:border-cyan-400/30 transition-all duration-200">
                <div className="text-2xl mb-2">{s.icon}</div>
                <div className="text-xs font-mono text-cyan-400 mb-1">{s.step}</div>
                <div className="font-bold text-sm text-white mb-1">{s.title}</div>
                <div className="text-xs text-slate-400 leading-relaxed">{s.desc}</div>
              </div>
              {i < arr.length - 1 && (
                <div className="text-cyan-400/40 font-bold text-xl md:text-sm md:my-2 md:mx-auto">
                  <span className="md:hidden">↓</span>
                  <span className="hidden md:block">→</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Model Stack */}
      <section className="relative z-10 px-6 md:px-12 py-16 max-w-5xl mx-auto">
        <h2 className="font-space text-3xl font-bold text-center mb-12">
          Model <span className="gradient-text">Stack</span>
        </h2>
        <div className="glass rounded-2xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-cyan-400/10">
                <th className="text-left px-6 py-4 text-slate-400 font-medium">Task</th>
                <th className="text-left px-6 py-4 text-slate-400 font-medium">Model</th>
                <th className="text-left px-6 py-4 text-slate-400 font-medium">Source</th>
              </tr>
            </thead>
            <tbody>
              {[
                { task: 'VQA / Captioning', model: 'GeoChat-7B', source: 'MBZUAI / HF', primary: true },
                { task: 'Region Grounding', model: 'GroundingDINO-Tiny', source: 'IDEA Research / HF', primary: false },
                { task: 'Change Detection', model: 'ChangeFormer', source: 'wgcban / HF Space', primary: false },
                { task: 'Optical–SAR Fusion', model: 'Dual-Encoder (composite)', source: 'Custom', primary: false },
                { task: 'Task Routing', model: 'Mistral-7B-Instruct', source: 'Mistral AI / HF', primary: false },
              ].map((r) => (
                <tr key={r.task} className="border-b border-space-700/50 hover:bg-space-700/20 transition-colors">
                  <td className="px-6 py-4 text-white">{r.task}</td>
                  <td className="px-6 py-4">
                    <span className={`font-mono text-xs px-2 py-1 rounded ${r.primary ? 'bg-cyan-400/15 text-cyan-400' : 'bg-space-700 text-slate-300'}`}>
                      {r.model}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-slate-400">{r.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* CTA */}
      <section className="relative z-10 text-center px-6 py-20">
        <div className="glass rounded-2xl max-w-2xl mx-auto px-8 py-12 glow-cyan">
          <h2 className="font-space text-3xl font-bold text-white mb-4">
            Ready to Analyze Satellite Imagery?
          </h2>
          <p className="text-slate-400 mb-8">
            Upload a GeoTIFF or TIFF, type your question, and let SatQuery AI do the rest.
          </p>
          <Link href="/analyze" className="btn-primary text-base px-10 py-4">
            🛰️ Open Analysis Workspace
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-space-700/50 px-8 py-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-500">
        <div>© 2026 SatQuery AI — ISRO Smart India Hackathon</div>
        <div className="flex items-center gap-4">
          <span>Built with GeoChat · ChangeFormer · GroundingDINO</span>
          <span className="badge badge-cyan">SIH Problem ID 26167</span>
        </div>
      </footer>
    </main>
  )
}
