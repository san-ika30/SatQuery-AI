import type { Metadata } from 'next'
import { Inter, Space_Grotesk } from 'next/font/google'
import './globals.css'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
})

const spaceGrotesk = Space_Grotesk({
  subsets: ['latin'],
  variable: '--font-space',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'SatQuery AI — Remote Sensing Vision-Language Assistant',
  description:
    'Agentic AI assistant for multimodal remote sensing image analysis. ' +
    'Ask natural-language questions about satellite imagery, detect changes, ' +
    'fuse optical and SAR data, and get evidence-grounded answers.',
  keywords: [
    'satellite image analysis', 'remote sensing AI', 'vision language model',
    'SAR fusion', 'change detection', 'VQA', 'GeoChat', 'ISRO', 'SIH 2026',
  ],
  openGraph: {
    title: 'SatQuery AI',
    description: 'Agentic vision-language assistant for remote sensing imagery',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${inter.variable} ${spaceGrotesk.variable}`}>
      <body className="bg-space-950 text-slate-100 antialiased">
        {children}
      </body>
    </html>
  )
}
