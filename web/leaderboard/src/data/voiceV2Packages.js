import { useEffect, useState } from 'react'

// 2.0 packages listed under `voice_v2_submissions` in manifest.json, plus the
// separately scored banking and entity packages listed under
// `voice_v2_domain_submissions`. Each directory holds the files
// `tau2 submit prepare` writes; see
// src/tau2/scripts/leaderboard/benchmark_package.py.
const SUBMISSIONS_BASE = import.meta.env.VITE_SUBMISSIONS_BASE_URL
  || `${import.meta.env.BASE_URL}submissions`
const NO_CACHE = { cache: 'no-cache' }
const FILES = {
  metadata: 'benchmark-metadata.json',
  submission: 'benchmark-submission.json',
  diagnostics: 'benchmark-diagnostics.json',
}

// Domain benchmarks score domains outside the voice-v2 Overall. Their packages
// attach to the same agent's voice-v2 row; they never change its Overall.
export const DOMAIN_BENCHMARKS = { 'voice-v2-banking': 'banking_knowledge', 'voice-v2-entity': 'intake' }

const providerLogo = { gemini: 'Google__G__logo.svg.png', openai: 'openai.svg', openai_live: 'openai.svg', xai: 'xai-logo.svg' }
export const toPercent = value => (value == null ? null : 100 * value)

async function loadPackage(dir, accepts) {
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
  if (!accepts(frame.benchmark_version) || !summary.complete) throw new Error(`${dir}: not a complete ${frame.benchmark_version} package of this list`)
  return { dir, ...pkg }
}

// One submitted system: the same model, organization, agent and submission type.
const agentKey = ({ metadata }) => [metadata.model_name, metadata.model_organization, metadata.agent_provider, metadata.agent_model, metadata.submission_type || 'standard'].join('\u0000')

// Attach each domain package to the one voice-v2 package of the same agent.
// A domain package without one becomes its own row with no Overall. Rows are
// voice-v2-shaped: `submission` is the voice-v2 package (or null) and
// `extras` maps a domain to its separately scored package.
export function groupPackages(cores, domainPackages) {
  const errors = []
  // One row per agent: the first listed voice-v2 package wins.
  const byAgent = new Map()
  for (const pkg of cores) {
    const key = agentKey(pkg)
    if (byAgent.has(key)) errors.push(`${pkg.dir}: duplicate voice-v2 package for ${byAgent.get(key).dir}`)
    else byAgent.set(key, { ...pkg, extras: {} })
  }
  const rows = [...byAgent.values()]
  const standalone = new Map()
  for (const pkg of domainPackages) {
    const domain = DOMAIN_BENCHMARKS[pkg.submission.frame.benchmark_version]
    const key = agentKey(pkg)
    const core = byAgent.get(key)
    const row = core || standalone.get(key) || { dir: pkg.dir, metadata: pkg.metadata, submission: null, diagnostics: null, extras: {} }
    if (row.extras[domain]) {
      errors.push(`${pkg.dir}: duplicate ${domain} package for ${row.dir}`)
      continue
    }
    row.extras[domain] = pkg
    if (!core) standalone.set(key, row)
  }
  return { rows: [...rows, ...standalone.values()], errors }
}

// Loads every complete package in the manifest. `packages` is null while
// loading; `errors` names packages that were skipped.
export function useVoiceV2Packages() {
  const [packages, setPackages] = useState(null)
  const [errors, setErrors] = useState([])
  useEffect(() => {
    let cancelled = false
    const settle = results => [
      results.filter(result => result.status === 'fulfilled').map(result => result.value),
      results.filter(result => result.status === 'rejected').map(result => result.reason.message),
    ]
    const load = async () => {
      const response = await fetch(`${SUBMISSIONS_BASE}/manifest.json`, NO_CACHE)
      const manifest = await response.json()
      const [cores, coreErrors] = settle(await Promise.allSettled((manifest.voice_v2_submissions || []).map(dir => loadPackage(dir, version => version === 'voice-v2'))))
      const [extras, extraErrors] = settle(await Promise.allSettled((manifest.voice_v2_domain_submissions || []).map(dir => loadPackage(dir, version => version in DOMAIN_BENCHMARKS))))
      if (cancelled) return
      const grouped = groupPackages(cores, extras)
      setPackages(grouped.rows)
      setErrors([...coreErrors, ...extraErrors, ...grouped.errors])
    }
    load().catch(error => !cancelled && (setPackages([]), setErrors([error.message])))
    return () => { cancelled = true }
  }, [])
  return { packages, errors }
}

// Task-completion pass@1 (percent) for one language or 'all'; null if the
// package does not cover that language or has no voice-v2 Overall.
export const packageScore = (pkg, language) => pkg.submission && toPercent(language === 'all' ? pkg.submission.summary.score : pkg.submission.summary.by_language[language]?.score)

// Pass@1 (percent) for one domain and language or 'all', from the domain's own
// package when it has one, else from the voice-v2 package.
export function domainScore(pkg, domain, language = 'all') {
  const summary = pkg.extras?.[domain]?.submission.summary || pkg.submission?.summary
  if (!summary) return null
  if (language === 'all') return toPercent(summary.by_domain[domain]?.score)
  return toPercent(summary.cells.find(cell => cell.language === language && cell.domain === domain)?.score)
}

export const packageLogo = pkg => providerLogo[pkg.metadata.agent_provider]
