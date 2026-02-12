import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'NomadaLLM - Private AI SDK for Developers',
  description: 'AI that never leaves your device. 100% local, works offline.',
  keywords: ['LLM', 'SDK', 'AI', 'privacy', 'on-device', 'local', 'machine learning', 'offline'],
  authors: [{ name: 'Nomada Health' }],
  openGraph: {
    title: 'NomadaLLM - Private AI SDK for Developers',
    description: 'AI that never leaves your device. 100% local, works offline.',
    url: 'https://nomadallm.nomadahealth.com',
    siteName: 'NomadaLLM',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'NomadaLLM - Private AI SDK for Developers',
    description: 'AI that never leaves your device. 100% local, works offline.',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  )
}
