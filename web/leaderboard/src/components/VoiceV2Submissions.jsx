import { Fragment, useEffect, useState } from 'react'
import multilingual from '../data/multilingual/results.json'
import { percent } from '../data/multilingual/metrics'
import './MultilingualVoice.css'

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

const languageName = Object.fromEntries(multilingual.languages.map(item => [item.id, item.name]))
const domainName = { airline: 'Airline', retail: 'Retail', telecom: 'Telecom' }
const measureName = {
  naturalness: 'Naturalness',
  speech_fidelity: 'Speech fidelity',
  tool_use: 'Tool use',
  counting_units: 'Counting units',
  gender_agreement: 'Gender agreement',
  honorific_agreement: 'Honorific agreement',
  modal_particles: 'Modal particles',
  name_address_conventions: 'Names and addresses',
  register_formality: 'Register and formality',
}
const label = (names, key) => names[key] || key.replace(/_/g, ' ').replace(/^./, c => c.toUpperCase())
const providerLogo = { gemini: 'Google__G__logo.svg.png', openai: 'openai.svg', openai_live: 'openai.svg', xai: 'xai-logo.svg' }
const toPercent = value => (value == null ? null : 100 * value)
const heat = value => (value == null ? '#f1f5f9' : `rgba(25, 121, 91, ${0.04 + 0.55 * Math.max(0, Math.min(100, value)) / 100})`)

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

// Pool each judged measure's units over the cells of one language (or all).
function measureTable(diagnostics) {
  if (!diagnostics) return { measures: [], value: () => null, coverage: () => null }
  const measures = [...new Set(diagnostics.cells.flatMap(cell => cell.summary.map(row => row.measure)))]
  const pooled = (measure, language) => {
    const rows = diagnostics.cells
      .filter(cell => language === 'all' || cell.language === language)
      .flatMap(cell => cell.summary.filter(row => row.measure === measure && row.applicable_calls > 0))
    if (!rows.length) return null
    const sum = key => rows.reduce((total, row) => total + row[key], 0)
    return { passed: sum('pass_units'), failed: sum('fail_units'), judged: sum('judged_calls'), applicable: sum('applicable_calls') }
  }
  return {
    measures,
    value: (measure, language) => {
      const totals = pooled(measure, language)
      return totals && totals.passed + totals.failed ? 100 * totals.passed / (totals.passed + totals.failed) : null
    },
    coverage: pooled,
  }
}

function Heatmap({ title, rowLabel, rows, columns, note }) {
  return <div className="mv-factor-breakdown">
    <div className="mv-factor-heading"><h3>{title}</h3><span>{note}</span></div>
    <div className="mv-table-scroll" role="region" aria-label={title} tabIndex="0"><table className="mv-factor-heatmap mv-pass-heatmap" style={{ minWidth: `${180 + columns.length * 105}px` }}>
      <colgroup><col className="mv-factor-label-col" />{columns.map(column => <col key={column.key} />)}</colgroup>
      <thead><tr><th scope="col">{rowLabel}</th>{columns.map(column => <th scope="col" key={column.key} className={column.key === 'all' ? 'mv-overall' : ''}>{column.label}</th>)}</tr></thead>
      <tbody>{rows.map(row => <tr key={row.key} data-factor={row.key} className={row.key === 'all' ? 'mv-overall-row' : ''}><th scope="row">{row.label}</th>{columns.map(column => {
        const value = row.value(column.key)
        return <td key={column.key} data-language={column.key} className={column.key === 'all' ? 'mv-overall' : ''}><span className={`mv-heat-cell${value == null ? ' is-missing' : ''}`} style={{ background: heat(value) }} title={`${row.label} · ${column.label}: ${value == null ? 'Not applicable' : percent(value)}${row.title ? row.title(column.key) : ''}`}>{percent(value)}</span></td>
      })}</tr>)}</tbody>
    </table></div>
  </div>
}

function Breakdown({ pkg, languages }) {
  const { summary } = pkg.submission
  const domains = Object.keys(summary.by_domain)
  const cellScore = (language, domain) => toPercent(summary.cells.find(cell => cell.language === language && cell.domain === domain)?.score)
  const columns = [...languages.map(key => ({ key, label: label(languageName, key) })), { key: 'all', label: 'Overall' }]
  const taskRows = [
    ...domains.map(domain => ({ key: domain, label: label(domainName, domain), value: language => (language === 'all' ? toPercent(summary.by_domain[domain].score) : cellScore(language, domain)) })),
    { key: 'all', label: 'Overall', value: language => toPercent(language === 'all' ? summary.score : summary.by_language[language].score) },
  ]
  const judged = measureTable(pkg.diagnostics)
  const judgeRows = judged.measures.map(measure => ({
    key: measure,
    label: label(measureName, measure),
    value: language => judged.value(measure, language),
    title: language => {
      const coverage = judged.coverage(measure, language)
      return coverage ? ` · ${coverage.judged}/${coverage.applicable} applicable calls judged` : ''
    },
  }))
  const gaps = judged.measures.flatMap(measure => {
    const coverage = judged.coverage(measure, 'all')
    return coverage && coverage.judged < coverage.applicable ? [`${label(measureName, measure)} ${coverage.judged}/${coverage.applicable}`] : []
  })
  const { metadata } = pkg
  return <div className="mv-detail" id={`v2-breakdown-${pkg.dir}`}>
    <div className="mv-detail-heading"><span><strong>{metadata.model_name}</strong></span><div className="mv-breakdown-meta"><span>{summary.scored_trials} scored calls · {pkg.submission.frame.cells.length} cells</span></div></div>
    <Heatmap title="Task completion (pass@1)" rowLabel="Domain" rows={taskRows} columns={columns} note="Mean task reward · cells weighted equally" />
    {pkg.diagnostics && <Heatmap title="Judge measures" rowLabel="Measure" rows={judgeRows} columns={columns} note="Share of judged opportunities without a violation · diagnostics only, not in the score" />}
    <div className="mv-factor-notes">
      {gaps.length > 0 && <p className="mv-footnote">Judge coverage gaps (judged / applicable calls): {gaps.join(' · ')}. Calls without a usable verdict are excluded, never counted as failures.</p>}
      <p className="mv-footnote">“—” means the measure does not apply to that language. Judge catalog <code>{pkg.diagnostics?.catalog_revision}</code>.</p>
      <dl className="v2-meta">
        <dt>Agent</dt><dd>{metadata.agent_provider} · <code>{metadata.agent_model}</code></dd>
        <dt>Submitted by</dt><dd>{metadata.submitting_organization} · {metadata.submission_date}</dd>
        <dt>Benchmark</dt><dd>{summary.benchmark_version} · frame <code>{summary.frame_revision}</code></dd>
        {metadata.notes && <><dt>Notes</dt><dd>{metadata.notes}</dd></>}
        {metadata.references?.length > 0 && <><dt>References</dt><dd>{metadata.references.map(ref => <a key={ref.url} href={ref.url} target="_blank" rel="noopener noreferrer">{ref.title} ↗</a>)}</dd></>}
        <dt>Trajectories</dt><dd>{metadata.trajectories
          ? <code className="v2-command">aws s3 sync {metadata.trajectories}/trajectories/&lt;language&gt;/&lt;domain&gt;/ ./{pkg.dir} --no-sign-request</code>
          : 'Not yet uploaded'}</dd>
      </dl>
    </div>
  </div>
}

export default function VoiceV2Submissions({ submissionTypeFilters, showStandard = true, showCustom = false }) {
  const [packages, setPackages] = useState(null)
  const [errors, setErrors] = useState([])
  const [expanded, setExpanded] = useState(null)
  const [sort, setSort] = useState('all')

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

  if (packages === null) return <section className="mv"><div className="loading-spinner" /><p className="mv-footnote">Loading 2.0 submissions…</p></section>
  const languages = [...new Set(packages.flatMap(pkg => pkg.submission.frame.cells.map(cell => cell.language)))]
  const score = (pkg, language) => toPercent(language === 'all' ? pkg.submission.summary.score : pkg.submission.summary.by_language[language]?.score)
  const rows = packages
    .filter(pkg => (pkg.metadata.submission_type === 'custom' ? showCustom : showStandard))
    .sort((a, b) => (score(b, sort) ?? -1) - (score(a, sort) ?? -1))
  const columns = [{ key: 'all', label: 'Overall' }, ...languages.map(key => ({ key, label: label(languageName, key) }))]
  const colSpan = 4 + columns.length

  return <section className="mv v2-submissions" aria-label="τ-i18n 2.0 voice submissions">
    <p className="mv-footnote">Submissions to the released τ-i18n voice benchmark (<code>voice-v2</code>): six languages × three domains, 50 tasks per cell, one trial. Pass@1 is the mean task reward; Overall averages the 18 cells equally. Judge measures are reported per language as diagnostics and never enter the score. 1.x and English voice-v1 results stay on their own tabs.</p>
    <div className="mv-leaderboard-heading"><h2>2.0 Leaderboard</h2>{submissionTypeFilters}</div>
    <div className="mv-ranking-meta"><span>Pass@1 by language</span><span>Click a model for its language × domain and judge breakdown.</span></div>
    <div className="reliability-metrics mv-scores">
      <div className="metrics-table-container">
        <table className="reliability-table mv-leaderboard-table v2-table">
          <thead><tr>
            <th>Rank</th><th>Model</th><th>Organization</th>
            {columns.map(column => <th key={column.key} data-column={column.key} className="mv-metric-header" aria-sort={sort === column.key ? 'descending' : 'none'}><button onClick={() => setSort(column.key)}>{column.label}{sort === column.key && <span className="mv-sort-dot"> •</span>}</button></th>)}
            <th className="expand-header"><span className="mv-visually-hidden">Breakdown</span></th>
          </tr></thead>
          <tbody>{rows.map((pkg, index) => {
            const open = expanded === pkg.dir
            const toggle = () => setExpanded(open ? null : pkg.dir)
            const logo = providerLogo[pkg.metadata.agent_provider]
            return <Fragment key={pkg.dir}>
              <tr className={`model-row${open ? ' expanded' : ''}`} data-package={pkg.dir}>
                <td className="rank-cell"><span className={`rank-number ${['rank-gold', 'rank-silver', 'rank-bronze'][index] || ''}`}>#{index + 1}</span></td>
                <td className="model-info"><button className="mv-model-trigger" onClick={toggle} aria-expanded={open}><span className="model-name">{pkg.metadata.model_name}</span></button>{pkg.metadata.submission_type === 'custom' && <span className="legacy-badge">custom</span>}</td>
                <td className="organization-info"><div className="org-container">{logo && <div className="company-logo"><img src={`${import.meta.env.BASE_URL}${logo}`} alt="" className="logo-img" /></div>}<div className="org-text"><span className="org-name">{pkg.metadata.model_organization}</span></div></div></td>
                {columns.map(column => {
                  const value = score(pkg, column.key)
                  return <td key={column.key} className="metric-cell score-cell" data-language={column.key}>{value != null
                    ? (column.key === 'all'
                      ? <span className="score-bar-container"><span className="score-bar-track"><span className="score-bar-fill" style={{ width: `${value}%` }} /></span><span className="score-bar-value">{percent(value)}</span></span>
                      : <span className="mv-heat-cell" style={{ background: heat(value) }}>{percent(value)}</span>)
                    : <span className="no-data">—</span>}</td>
                })}
                <td className="expand-cell"><button className="mv-expand-trigger" aria-label={`${open ? 'Collapse' : 'Expand'} ${pkg.metadata.model_name} breakdown`} aria-expanded={open} onClick={toggle}><span className={`expand-caret${open ? ' open' : ''}`}>▶</span></button></td>
              </tr>
              {open && <tr className="domain-detail-row"><td colSpan={colSpan} className="domain-detail-cell"><Breakdown pkg={pkg} languages={languages} /></td></tr>}
            </Fragment>
          })}{!rows.length && <tr><td colSpan={colSpan}><div className="filter-empty-state"><h3>No Results</h3><p>{!showStandard && !showCustom ? 'Select Standard or Custom to view results.' : <>No 2.0 voice submissions yet. See the <a href="https://github.com/sierra-research/tau2-bench/blob/main/docs/leaderboard-submission.md">submission guide</a>.</>}</p></div></td></tr>}</tbody>
        </table>
      </div>
    </div>
    <p className="mv-footnote">Language columns average that language’s three domains equally.</p>
    {errors.length > 0 && <p className="mv-footnote" role="alert">Skipped: {errors.join('; ')}</p>}
  </section>
}
