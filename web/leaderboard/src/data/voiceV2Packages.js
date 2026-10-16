import { useEffect, useState } from 'react'

// 2.0 packages listed under `voice_v2_submissions` in manifest.json. Each
// directory holds the three files `tau2 submit prepare` writes; see
// src/tau2/scripts/leaderboard/benchmark_package.py.
const SUBMISSIONS_BASE = import.meta.env.VITE_SUBMISSIONS_BASE_URL
  || `${import.meta.env.BASE_URL}submissions`
const NO_CACHE = { cache: 'no-cache' }
const FILES = {
  metadata: 'benchmark-metadata.json',
  submission: 'benchmark-submission.json',
  diagnostics: 'benchmark-diagnostics.json',
}

const providerLogo = { gemini: 'Google__G__logo.svg.png', openai: 'openai.svg', openai_live: 'openai.svg', xai: 'xai-logo.svg' }
export const toPercent = value => (value == null ? null : 100 * value)

async function loadPackage(dir) {
  const entries = await Promise.all(Object.entries(FILES).map(async ([key, file]) => {
    const response = await fetch(`${SUBMISSIONS_BASE}/${dir}/${file}`, NO_CACHE)
    if (!response.ok) {
      if (key === 'diagnostics') return [key, null]
      throw new Error(`${dir}/${file}: HTTP ${response.status}`)
    }
    return [key, await response.json()]
  }))
  const pkg = Object.fromEntries(entries)
  const { summary, frame } = pkg.submission
  if (frame.benchmark_version !== 'voice-v2' || !summary.complete) throw new Error(`${dir}: not a complete voice-v2 package`)
  return { dir, ...pkg }
}

// Loads every complete voice-v2 package in the manifest. `packages` is null
// while loading; `errors` names packages that were skipped.
export function useVoiceV2Packages() {
  const [packages, setPackages] = useState(null)
  const [errors, setErrors] = useState([])
  useEffect(() => {
    let cancelled = false
    const load = async () => {
      const response = await fetch(`${SUBMISSIONS_BASE}/manifest.json`, NO_CACHE)
      const manifest = await response.json()
      const results = await Promise.allSettled((manifest.voice_v2_submissions || []).map(loadPackage))
      if (cancelled) return
      setPackages(results.filter(result => result.status === 'fulfilled').map(result => result.value))
      setErrors(results.filter(result => result.status === 'rejected').map(result => result.reason.message))
    }
    load().catch(error => !cancelled && (setPackages([]), setErrors([error.message])))
    return () => { cancelled = true }
  }, [])
  return { packages, errors }
}

// Task-completion pass@1 (percent) for one language or 'all'; null if the
// package does not cover that language.
export const packageScore = (pkg, language) => toPercent(language === 'all' ? pkg.submission.summary.score : pkg.submission.summary.by_language[language]?.score)
export const packageLogo = pkg => providerLogo[pkg.metadata.agent_provider]
