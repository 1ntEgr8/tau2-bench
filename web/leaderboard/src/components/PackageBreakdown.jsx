import multilingual from '../data/multilingual/results.json'
import { percent } from '../data/multilingual/metrics'
import { toPercent } from '../data/voiceV2Packages'
import './MultilingualVoice.css'

// Breakdown of one submitted voice-v2 package, opened from its row in the
// τ-i18n leaderboard (MultilingualVoice).
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
const heat = value => (value == null ? '#f1f5f9' : `rgba(25, 121, 91, ${0.04 + 0.55 * Math.max(0, Math.min(100, value)) / 100})`)

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

export default function PackageBreakdown({ pkg, onClose }) {
  const { summary } = pkg.submission
  const languages = [...new Set(pkg.submission.frame.cells.map(cell => cell.language))]
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
    <div className="mv-detail-heading"><span><strong>{metadata.model_name}</strong></span><div className="mv-breakdown-meta"><span>Submitted package · {summary.scored_trials} scored calls · {pkg.submission.frame.cells.length} cells</span><button className="mv-close-breakdown" aria-label="Close breakdown" onClick={onClose}>×</button></div></div>
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

