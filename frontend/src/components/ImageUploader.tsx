'use client'

import { useState, useCallback, useRef } from 'react'
import type { InputMode } from '@/lib/api'

interface UploadedFile {
  file: File
  preview: string | null
  isGeoTiff: boolean
}

interface ImageUploaderProps {
  inputMode: InputMode
  onFilesChange: (image1: File | null, image2: File | null) => void
}

const ACCEPTED_TYPES = ['.tif', '.tiff', '.geotiff', '.png', '.jpg', '.jpeg']
const MAX_SIZE_MB = 50

function isGeoTiff(file: File): boolean {
  return file.name.toLowerCase().endsWith('.tif') ||
         file.name.toLowerCase().endsWith('.tiff') ||
         file.name.toLowerCase().endsWith('.geotiff')
}

function UploadSlot({
  slot,
  label,
  sublabel,
  file,
  onFile,
}: {
  slot: number
  label: string
  sublabel: string
  file: UploadedFile | null
  onFile: (f: File) => void
}) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) onFile(f)
  }, [onFile])

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (f) onFile(f)
  }, [onFile])

  return (
    <div
      className={`upload-zone flex flex-col items-center justify-center p-6 min-h-[200px] relative transition-all ${dragging ? 'drag-over' : ''} ${file ? 'border-cyan-400/50 bg-cyan-400/5' : ''}`}
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      id={`upload-slot-${slot}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(',')}
        className="hidden"
        onChange={handleChange}
        id={`file-input-${slot}`}
      />

      {file ? (
        <div className="text-center w-full">
          <div className="flex items-center justify-center gap-2 mb-3">
            <span className="text-2xl">{file.isGeoTiff ? '🗺️' : '🖼️'}</span>
            <div className="text-left">
              <div className="text-sm font-semibold text-white truncate max-w-[200px]">
                {file.file.name}
              </div>
              <div className="text-xs text-slate-400">
                {(file.file.size / 1024 / 1024).toFixed(2)} MB
                {file.isGeoTiff && ' · GeoTIFF'}
              </div>
            </div>
          </div>
          {file.preview && (
            <img
              src={file.preview}
              alt="Preview"
              className="w-full max-h-[120px] object-contain rounded-lg opacity-80"
            />
          )}
          <div className="badge badge-green mt-3">✓ Ready</div>
        </div>
      ) : (
        <div className="text-center">
          <div className="text-4xl mb-3 opacity-60">📡</div>
          <div className="font-semibold text-sm text-white mb-1">{label}</div>
          <div className="text-xs text-slate-400 mb-3">{sublabel}</div>
          <div className="text-xs text-cyan-400/70">
            Drop file or click to browse
          </div>
          <div className="text-xs text-slate-500 mt-1">
            GeoTIFF, TIFF, PNG, JPEG · max {MAX_SIZE_MB}MB
          </div>
        </div>
      )}
    </div>
  )
}

export default function ImageUploader({ inputMode, onFilesChange }: ImageUploaderProps) {
  const [file1, setFile1] = useState<UploadedFile | null>(null)
  const [file2, setFile2] = useState<UploadedFile | null>(null)
  const [errors, setErrors] = useState<string[]>([])

  const validateAndSet = useCallback(
    (f: File, slot: 1 | 2) => {
      const errs: string[] = []
      const ext = '.' + f.name.split('.').pop()?.toLowerCase()
      if (!ACCEPTED_TYPES.includes(ext)) {
        errs.push(`Unsupported file type: ${ext}`)
      }
      if (f.size > MAX_SIZE_MB * 1024 * 1024) {
        errs.push(`File too large: ${(f.size / 1024 / 1024).toFixed(1)}MB (max ${MAX_SIZE_MB}MB)`)
      }
      setErrors(errs)
      if (errs.length > 0) return

      const uploaded: UploadedFile = {
        file: f,
        isGeoTiff: isGeoTiff(f),
        preview: null,
      }

      // Generate preview for raster images (PNG/JPG only)
      if (!isGeoTiff(f)) {
        const url = URL.createObjectURL(f)
        uploaded.preview = url
      }

      if (slot === 1) {
        setFile1(uploaded)
        onFilesChange(f, file2?.file ?? null)
      } else {
        setFile2(uploaded)
        onFilesChange(file1?.file ?? null, f)
      }
    },
    [file1, file2, onFilesChange],
  )

  const slotConfig = {
    single: [
      { slot: 1 as const, label: 'Upload Satellite Image', sublabel: 'Single optical, multispectral, or SAR image' },
    ],
    bitemporal: [
      { slot: 1 as const, label: 'Upload Image T₁ (Earlier)', sublabel: 'First temporal acquisition' },
      { slot: 2 as const, label: 'Upload Image T₂ (Later)', sublabel: 'Second temporal acquisition' },
    ],
    crossmodal: [
      { slot: 1 as const, label: 'Upload Optical / Multispectral', sublabel: 'Sentinel-2, Cartosat, etc.' },
      { slot: 2 as const, label: 'Upload SAR Image', sublabel: 'Sentinel-1, RISAT, etc.' },
    ],
  }

  const slots = slotConfig[inputMode]

  return (
    <div className="space-y-4">
      <div className={`grid gap-4 ${slots.length > 1 ? 'grid-cols-1 sm:grid-cols-2' : 'grid-cols-1'}`}>
        {slots.map(s => (
          <UploadSlot
            key={s.slot}
            slot={s.slot}
            label={s.label}
            sublabel={s.sublabel}
            file={s.slot === 1 ? file1 : file2}
            onFile={f => validateAndSet(f, s.slot)}
          />
        ))}
      </div>

      {errors.length > 0 && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 space-y-1">
          {errors.map(e => (
            <div key={e} className="text-sm text-red-400 flex items-center gap-2">
              <span>⚠️</span> {e}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
