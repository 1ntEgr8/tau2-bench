import { Fragment, useRef, useState } from 'react'
import data from '../data/multilingual/results.json'
import { summarize, frontier, percent, mean, heatScale } from '../data/multilingual/metrics'
import PackageBreakdown from './PackageBreakdown'
import { packageLogo, packageScore, toPercent, useVoiceV2Packages } from '../data/voiceV2Packages'
import './MultilingualVoice.css'

const metrics = [['task', 'Task completion'], ['interaction', 'Interaction'], ['experience', 'Experience']]
const componentNames = { nonresponse: 'Non-response', interruption: 'Interruptions', selectivity: 'Selectivity errors', monologue: 'Monologues', tool_use: 'Tool-use errors' }
const providers = { openai: ['OpenAI', 'openai.svg'], gemini: ['Google', 'Google__G__logo.svg.png'], xai: ['xAI', 'xai-logo.svg'] }
const shortName = row => `${row.name.replace('Gemini 3.1 Flash Live', 'Gemini 3.1').replace('GPT Realtime 2', 'GPT RT 2')} · ${row.effort}`

function Space({ rows, onSelect, selected, focused, onClearFocus }) {
  const [yaw, setYaw] = useState(32)
  const [tilt, setTilt] = useState(22)
  const [fit, setFit] = useState(true)
  const drag = useRef(null)
  const axes = [
    { key: 'task', label: 'Task completion' },
    { key: 'experience', label: 'Experience' },
    { key: 'interaction', label: 'Interaction' },
  ]
  const eligible = rows.filter(row => axes.every(axis => row[axis.key] != null))
  const ranges = axes.map(({ key }) => {
    const values = eligible.map(row => row[key])
    return fit && values.length ? [Math.max(0, Math.floor(Math.min(...values) / 5) * 5 - 5), Math.min(100, Math.ceil(Math.max(...values) / 5) * 5 + 5)] : [0, 100]
  })
  const coordinates = row => axes.map(({ key }, index) => (row[key] - ranges[index][0]) / (ranges[index][1] - ranges[index][0]))
  const radians = yaw * Math.PI / 180
  const elevation = tilt * Math.PI / 180
  const project = ([x, y, z]) => {
    const horizontal = (x - .5) * Math.cos(radians) + (z - .5) * Math.sin(radians)
    const depth = -(x - .5) * Math.sin(radians) + (z - .5) * Math.cos(radians)
    return [385 + horizontal * 350, 285 - ((y - .5) * Math.cos(elevation) + depth * Math.sin(elevation)) * 350]
  }
  const line = (from, to, props = {}) => {
    const a = project(from), b = project(to)
    return <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} {...props} />
  }
  const polygon = points => points.map(point => project(point).join(',')).join(' ')
  const best = frontier(rows, axes.map(axis => axis.key))
  const active = rows.find(row => row.id === selected)
  const vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
  const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
  const reset = () => { setYaw(32); setTilt(22) }
  const move = event => {
    if (!drag.current) return
    setYaw(Math.max(12, Math.min(78, drag.current.yaw + (event.clientX - drag.current.x) * .3)))
    setTilt(Math.max(8, Math.min(45, drag.current.tilt - (event.clientY - drag.current.y) * .2)))
  }
  return <div className="mv-cube-layout">
    <div className="mv-cube-stage">
      <div className="mv-cube-toolbar"><div className="mv-range-switch" aria-label="Axis scale"><button aria-pressed={fit} onClick={() => setFit(true)}>Fit results</button><button aria-pressed={!fit} onClick={() => setFit(false)}>Full scale · 0–100</button></div><div className="mv-cube-actions">{focused && <button className="mv-reset" onClick={onClearFocus}>Show all models</button>}<button className="mv-reset" onClick={reset}>↺ Reset view</button></div></div>
      {eligible.length ? <>
        <svg className="mv-cube" viewBox="0 0 800 640" role="group" aria-label="Interactive 3D metric cube" onPointerDown={event => {
          if (event.button !== 0 || event.target.closest('[role="button"]')) return
          drag.current = { x: event.clientX, y: event.clientY, yaw, tilt }
          event.currentTarget.setPointerCapture(event.pointerId)
        }} onPointerMove={move} onPointerUp={() => { drag.current = null }} onPointerCancel={() => { drag.current = null }} onLostPointerCapture={() => { drag.current = null }}>
          <title>Task completion, experience, and interaction. Drag left or right to rotate, and up or down to tilt.</title>
          <polygon points={polygon([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]])} fill="#eeeeee" fillOpacity=".85" />
          <polygon points={polygon([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])} fill="#f5f5f5" fillOpacity=".9" />
          <polygon points={polygon([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]])} fill="#f8f8f8" fillOpacity=".8" />
          {[0, .25, .5, .75, 1].map(value => <g key={value} stroke="#d6d6d6" strokeWidth=".8">
            {line([value, 0, 0], [value, 0, 1])}{line([0, 0, value], [1, 0, value])}
            {line([value, 0, 1], [value, 1, 1])}{line([0, value, 1], [1, value, 1])}
            {line([0, value, 0], [0, value, 1])}{line([0, 0, value], [0, 1, value])}
          </g>)}
          {edges.map(([a, b]) => <g key={`${a}-${b}`}>{line(vertices[a], vertices[b], { stroke: '#adadad', strokeWidth: 1.1, opacity: .7 })}</g>)}
          {line([0, 0, 0], [1, 0, 0], { stroke: '#747474', strokeWidth: 2.5 })}
          {line([0, 0, 0], [0, 1, 0], { stroke: '#747474', strokeWidth: 2.5 })}
          {line([1, 0, 0], [1, 0, 1], { stroke: '#747474', strokeWidth: 2.5 })}
          {[0, .25, .5, .75, 1].map(value => <g key={value}>{axes.map((axis, index) => {
            const location = index === 0 ? [value, 0, 0] : index === 1 ? [0, value, 0] : [1, 0, value]
            const [x, y] = project(location)
            const tick = ranges[index][0] + value * (ranges[index][1] - ranges[index][0])
            return <text key={axis.key} x={x + (index === 1 ? -15 : index === 2 ? 17 : 0)} y={y + (index === 0 ? 22 : 4)} textAnchor={index === 1 ? 'end' : index === 2 ? 'start' : 'middle'} fill="#737373" className="mv-cube-tick">{Number(tick.toFixed(1))}</text>
          })}</g>)}
          {axes.map((axis, index) => {
            const location = index === 0 ? [.5, 0, 0] : index === 1 ? [0, 1, 0] : [1, 0, .5]
            const [x, y] = project(location)
            return <text key={axis.key} x={x + (index === 1 ? -8 : index === 2 ? 34 : 0)} y={index === 0 ? Math.max(...vertices.map(vertex => project(vertex)[1])) + 48 : y + (index === 1 ? -22 : 36)} textAnchor={index === 1 ? 'end' : index === 2 ? 'start' : 'middle'} fill="#505050" className="mv-cube-axis">{axis.label} (%) ↑</text>
          })}
          {active?.experience != null && <g stroke={active.color} strokeWidth="1.6" strokeDasharray="4 5" opacity=".6">
            {line(coordinates(active), [coordinates(active)[0], 0, coordinates(active)[2]])}
            {line([coordinates(active)[0], 0, coordinates(active)[2]], [coordinates(active)[0], 0, 0])}
            {line([coordinates(active)[0], 0, coordinates(active)[2]], [1, 0, coordinates(active)[2]])}
            {line(coordinates(active), [0, coordinates(active)[1], coordinates(active)[2]])}
            {line([0, coordinates(active)[1], coordinates(active)[2]], [0, coordinates(active)[1], 0])}
          </g>}
          {[...eligible].sort((a, b) => {
            if (focused && a.id === selected) return 1
            if (focused && b.id === selected) return -1
            const depth = row => { const [x, y, z] = coordinates(row); return (-x * Math.sin(radians) + z * Math.cos(radians)) * Math.cos(elevation) - y * Math.sin(elevation) }
            return depth(b) - depth(a)
          }).map(row => {
            const [x, y] = project(coordinates(row)), index = rows.findIndex(item => item.id === row.id)
            const isBest = best.some(item => item.id === row.id)
            const muted = focused && selected !== row.id
            const pointColor = muted ? '#c5c9c7' : row.color
            return <g key={row.id} className={`mv-cube-point${muted ? ' is-muted' : ''}${focused && selected === row.id ? ' is-focused' : ''}`} role="button" tabIndex="0" aria-pressed={focused && selected === row.id} aria-label={`${shortName(row)}, task ${percent(row.task)}, experience ${percent(row.experience)}, interaction ${percent(row.interaction)}`} onClick={() => onSelect(row.id)} onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(row.id) } }}>
              <title>{shortName(row)} · task {percent(row.task)} · experience {percent(row.experience)} · interaction {percent(row.interaction)}{isBest ? ' · Pareto frontier' : ''}</title>
              <circle cx={x} cy={y} r={focused && selected === row.id ? 22 : 19} fill="none" stroke={pointColor} strokeWidth="1.5" opacity=".65" />
              <circle cx={x} cy={y} r={focused && selected === row.id ? 16 : 13} fill={pointColor} stroke="white" strokeWidth="2.5" />
              <text x={x} y={y + 4.5} textAnchor="middle" fill={muted ? '#6e7671' : row.effort === 'minimal' ? '#18333d' : 'white'} fontSize="12" fontWeight="700">{index + 1}</text>
            </g>
          })}
        </svg>
      </> : <div className="mv-cube-empty"><span>3D comparison unavailable</span><h3>Experience was not evaluated for English.</h3><p>Explore its measured task completion and interaction scores below, or choose a localized language to see all three dimensions.</p></div>}
    </div>
    <aside className="mv-cube-sidebar"><span className="mv-kicker">SELECT A MODEL</span><div className="mv-cube-models">{rows.map((row, index) => <button key={row.id} className={focused && selected !== row.id ? 'is-muted' : ''} aria-pressed={focused && selected === row.id} onClick={() => onSelect(row.id)}><span className="mv-model-number" style={{ background: focused && selected !== row.id ? '#c5c9c7' : row.color, color: row.effort === 'minimal' ? '#18333d' : 'white' }}>{index + 1}</span><span>{row.name}<small>{row.effort}</small></span>{best.some(item => item.id === row.id) && <span className="mv-frontier-tag" title="On the observed 3D Pareto frontier">Pareto</span>}</button>)}</div><div className="mv-selected-metrics"><span className="mv-kicker">{shortName(active)}</span>{axes.map(axis => <div key={axis.key}><span>{axis.label}</span><strong>{percent(active[axis.key])}</strong></div>)}</div></aside>
  </div>
}

// Only task completion is split by domain; Interaction and Experience are pooled
// over the three core domains in the paper export.
const domainOptions = [
  { key: 'all', label: '📊 Overall' },
  { key: 'retail', label: '🛍️ Retail' },
  { key: 'airline', label: '✈️ Airline' },
  { key: 'telecom', label: '📱 Telecom' },
  { key: 'banking_knowledge', label: '🏦 Banking' },
  { key: 'intake', label: '📋 Intake' },
]
const forDomain = (row, domain) => domain === 'all' ? row : { ...row, task: row.domains[domain] ?? null }

function DomainToggle({ domain, setDomain }) {
  const index = domainOptions.findIndex(option => option.key === domain)
  return <div className="domain-toggle-switch mv-domain-toggle">
    <div className="toggle-container domain-toggle-container" style={{ '--domain-count': domainOptions.length }}>
      {domainOptions.map(option => <button key={option.key} className={`toggle-option domain-toggle-option ${domain === option.key ? 'active' : ''}`} aria-pressed={domain === option.key} onClick={() => setDomain(option.key)}>{option.label}</button>)}
      <div className="toggle-slider domain-toggle-slider" style={{ width: `calc((100% - 8px) / ${domainOptions.length})`, transform: `translateX(${index * 100}%)` }} />
    </div>
  </div>
}

function LanguageMap({ metric, setMetric, domain }) {
  const columns = [...data.languages, { id: 'all', name: 'Overall' }]
  const rows = data.models.map(model => ({ ...model, values: Object.fromEntries(columns.map(language => [language.id, forDomain(summarize(model, language.id), domain)[metric]])) }))
  rows.push({ id: 'overall', name: 'Overall', values: Object.fromEntries(columns.map(language => [language.id, mean(rows.map(row => row.values[language.id]))])) })
  const metricLabel = metrics.find(([key]) => key === metric)[1]
  const heat = heatScale(rows.flatMap(row => columns.map(language => row.values[language.id])))
  return <div className="mv-map">
    <div className="mv-map-heading"><div className="mv-range-switch mv-map-metrics" role="group" aria-label="Language map metric">{metrics.map(([key, label]) => <button key={key} type="button" aria-pressed={metric === key} onClick={() => setMetric(key)}>{label}</button>)}</div></div>
    <div className="mv-table-scroll"><table className="mv-heatmap" aria-label={`${metricLabel} by model and language`}>
      <thead><tr><th scope="col">Model / reasoning</th>{columns.map(language => <th scope="col" key={language.id} className={language.id === 'all' ? 'mv-overall' : ''}>{language.name}</th>)}</tr></thead>
      <tbody>{rows.map(row => <tr key={row.id} data-model={row.id} className={row.id === 'overall' ? 'mv-overall-row' : ''}><th scope="row">{row.color && <i style={{ background: row.color }} />}{row.name}{row.effort && <small>{row.effort}</small>}</th>{columns.map(language => {
        const value = row.values[language.id]
        return <td key={language.id} data-language={language.id} className={language.id === 'all' ? 'mv-overall' : ''}><span className="mv-heat-cell" style={{ background: heat(value) }} title={`${row.name}${row.effort ? ` · ${row.effort}` : ''} · ${language.name} · ${metricLabel}: ${percent(value)}`}>{percent(value)}</span></td>
      })}</tr>)}</tbody>
    </table></div>
    <p className="mv-footnote">Overall column: mean across evaluated languages. Overall row: mean across model configurations. “—” means not evaluated. Shading is relative to the lowest and highest values in the table.</p>
  </div>
}

const taskFactors = [
  ...['airline', 'retail', 'telecom'].map(key => ({ key, label: key.charAt(0).toUpperCase() + key.slice(1), value: cell => cell.domains[key] })),
  { key: 'overall', label: 'Overall', value: cell => cell.task },
  { key: 'banking_knowledge', label: 'Banking', value: cell => cell.domains.banking_knowledge ?? null, extra: true },
  { key: 'intake', label: 'Intake', value: cell => cell.domains.intake ?? null, extra: true },
]
const interactionFactors = [
  ...Object.entries(componentNames).map(([key, label]) => ({ key, label, value: cell => cell.interaction_components[key] })),
  { key: 'overall', label: 'Overall', value: cell => cell.interaction_failure == null ? null : 100 - cell.interaction_failure },
  { key: 'latency', label: 'Latency (s) ↓', value: cell => cell.latency_seconds },
]
const experienceFactors = [
  { key: 'naturalness', label: 'Naturalness', value: cell => cell.fluency_failure == null ? null : 100 - cell.fluency_failure },
  { key: 'fidelity', label: 'Speech fidelity', value: cell => cell.speech_fidelity_failure == null ? null : 100 - cell.speech_fidelity_failure },
  { key: 'overall', label: 'Overall', value: cell => cell.experience },
]

function FactorHeatmap({ cells, metric }) {
  const failure = metric === 'interaction'
  const task = metric === 'task'
  const factors = task ? taskFactors : failure ? interactionFactors : experienceFactors
  const title = `${metrics.find(([key]) => key === metric)[1]} by language`
  const columns = cells.map(cell => ({ key: cell.language, label: data.languages.find(item => item.id === cell.language).name, value: factor => factor.value(cell) }))
  columns.push({ key: 'overall', label: 'Overall', value: factor => mean(cells.map(cell => factor.value(cell))) })
  // Interaction factors are failure rates, so shade on 100 - value to keep darker = better.
  const goodness = (factor, value) => value == null || factor.key === 'latency' ? null : failure && factor.key !== 'overall' ? 100 - value : value
  const heat = heatScale(factors.flatMap(factor => columns.map(column => goodness(factor, column.value(factor)))))
  return <div className="mv-factor-breakdown">
    <div className="mv-factor-heading"><h3>{title}</h3><span>{failure ? 'Factors: failure rate ↓ · Overall: pass score ↑' : 'Pass rate · higher is better'}</span></div>
    <div className="mv-table-scroll" role="region" aria-label={title} tabIndex="0"><table className={`mv-factor-heatmap ${task ? 'mv-task-heatmap' : failure ? 'mv-failure-heatmap' : 'mv-pass-heatmap'}`} style={{ minWidth: `${180 + columns.length * 105}px` }}>
      <colgroup><col className="mv-factor-label-col" />{columns.map(column => <col key={column.key} />)}</colgroup>
      <thead><tr><th scope="col">{task ? 'Domain' : 'Factor'}</th>{columns.map(column => <th scope="col" key={column.key} className={column.key === 'overall' ? 'mv-overall' : ''}>{column.label}</th>)}</tr></thead>
      <tbody>{factors.map(factor => <tr key={factor.key} data-factor={factor.key} className={factor.key === 'overall' ? 'mv-overall-row' : factor.key === 'latency' ? 'mv-latency-row' : factor.extra ? 'mv-extra-row' : ''}><th scope="row">{factor.label}</th>{columns.map(column => {
        const value = column.value(factor)
        const latency = factor.key === 'latency'
        const displayValue = latency && value != null ? `${value.toFixed(2)} s` : percent(value)
        const background = value == null ? '#f1f5f9' : latency ? '#f3f4f3' : heat(goodness(factor, value))
        return <td key={column.key} data-language={column.key} className={column.key === 'overall' ? 'mv-overall' : ''}><span className={`mv-heat-cell${value == null ? ' is-missing' : ''}`} style={{ background }} title={`${factor.label} · ${column.label}: ${value == null ? 'Not evaluated' : displayValue}${latency ? ' · Mean response and yield latency; excluded from the Interaction score' : ''}`}>{displayValue}</span></td>
      })}</tr>)}</tbody>
    </table></div>
    <div className="mv-heatmap-legend"><span>Lowest</span><span className="mv-heatmap-ramp" /><span>Highest in table</span><span>— Not evaluated</span></div>
  </div>
}

function MetricBreakdown({ active, language, metric, onMetricChange, onClose }) {
  const cells = data.cells.filter(cell => cell.model === active.id && (language === 'all' || cell.language === language))
  return <div className="mv-detail" id="voice-metric-breakdown">
    <div className="mv-detail-heading"><span><i style={{ background: active.color }} /><strong>{shortName(active)}</strong></span><div className="mv-breakdown-meta"><span>{active.calls} core calls · {active.utterances.toLocaleString()} aligned utterances</span><button className="mv-close-breakdown" aria-label="Close breakdown" onClick={onClose}>×</button></div></div>
    <div className="mv-breakdown-tabs" aria-label="Metric breakdown">{metrics.map(([key, label]) => <button key={key} aria-pressed={metric === key} onClick={() => onMetricChange(key)}>{label}<strong>{percent(active[key])}</strong></button>)}</div>
    <FactorHeatmap cells={cells} metric={metric} />
    <div className="mv-factor-notes">
      {metric === 'task' && <p className="mv-footnote">Overall averages Airline, Retail, and Telecom equally. Each domain–language pair contains 50 calls. Banking and Intake are not part of the multilingual cohort and are not included in Overall.</p>}
      {metric === 'interaction' && <p className="mv-footnote">Overall is the Interaction pass score: 100 minus the equal-weight mean of the five failure rates. Darker green indicates fewer factor failures and a higher Overall score.</p>}
      {metric === 'experience' && <p className="mv-footnote">Overall is the share of aligned utterances passing both the naturalness and speech-fidelity checks, not their average. {language === 'all' && 'Scores cover five languages; English is not evaluated.'}{language === 'en' && 'Experience, naturalness, and aligned speech fidelity were not evaluated for English in this cohort.'}</p>}
      <p className="mv-footnote">The Overall column averages evaluated languages in the selected scope equally. The intersection of the Overall row and column matches the headline {metrics.find(([key]) => key === metric)[1]} score.</p>
    </div>

  </div>
}

function PackageRow({ row, index, open, onToggle }) {
  const { pkg } = row
  const logo = packageLogo(pkg)
  return <Fragment>
    <tr className={`model-row mv-package-row${open ? ' expanded' : ''}`} data-package={pkg.dir}>
      <td className="rank-cell"><span className={`rank-number ${['rank-gold', 'rank-silver', 'rank-bronze'][index] || ''}`}>#{index + 1}</span></td>
      <td className="model-info"><button className="mv-model-trigger" onClick={onToggle} aria-expanded={open}><span className="model-name">{row.name}</span></button><span className="legacy-badge">{row.submissionType === 'custom' ? 'custom submission' : 'submission'}</span></td>
      <td className="organization-info"><div className="org-container">{logo && <div className="company-logo"><img src={`${import.meta.env.BASE_URL}${logo}`} alt="" className="logo-img" /></div>}<div className="org-text"><span className="org-name">{pkg.metadata.model_organization}</span></div></div></td>
      <td className="reasoning-info">—</td>
      {metrics.map(([key]) => <td key={key} className="metric-cell score-cell" data-metric={key}>{row[key] != null
        ? <span className="score-bar-container"><span className="score-bar-track"><span className="score-bar-fill" style={{ width: `${row[key]}%` }} /></span><span className="score-bar-value">{percent(row[key])}</span></span>
        : <span className="no-data">—</span>}</td>)}
      <td className="expand-cell"><button className="mv-expand-trigger" aria-label={`${open ? 'Collapse' : 'Expand'} ${row.name} breakdown`} aria-expanded={open} onClick={onToggle}><span className={`expand-caret${open ? ' open' : ''}`}>▶</span></button></td>
    </tr>
    {open && <tr className="domain-detail-row"><td colSpan={8} className="domain-detail-cell"><PackageBreakdown pkg={pkg} onClose={onToggle} /></td></tr>}
  </Fragment>
}

export default function MultilingualVoice({ submissionTypeFilters, showStandard = true, showCustom = false }) {
  const language = 'all'
  const [domain, setDomain] = useState('all')
  const domainLabel = domainOptions.find(option => option.key === domain).label.replace(/^\S+ /, '')
  const [view, setView] = useState(() => {
    const requested = new URLSearchParams(window.location.search).get('view')
    return requested === 'languages' ? 'languages' : 'space'
  })
  const [mapMetric, setMapMetric] = useState('task')
  const [sort, setSort] = useState('task')
  const [breakdown, setBreakdown] = useState(null)
  const openBreakdown = (id, metric) => {
    setOpenPackage(null)
    setSelected(id)
    setBreakdown(metric)
    requestAnimationFrame(() => document.getElementById('voice-metric-breakdown')?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
  }
  const [selected, setSelected] = useState('openai_xhigh')
  const [focused, setFocused] = useState(false)
  const selectModel = id => {
    setFocused(!(focused && selected === id))
    setSelected(id)
  }
  const rows = data.models.map(model => forDomain(summarize(model, language), domain))
  // Submitted voice-v2 packages join the paper cohort as rows scored on task
  // completion only; Interaction and Experience are paper-cohort measures.
  const { packages, errors: packageErrors } = useVoiceV2Packages()
  const [openPackage, setOpenPackage] = useState(null)
  const packageRows = (packages || []).map(pkg => ({ id: `package:${pkg.dir}`, pkg, name: pkg.metadata.model_name, submissionType: pkg.metadata.submission_type, task: domain === 'all' ? packageScore(pkg, language) : toPercent(pkg.submission.summary.by_domain[domain]?.score), interaction: null, experience: null }))
  const leaderboardRows = [...rows, ...packageRows].filter(row => (row.submissionType || 'standard') === 'custom' ? showCustom : showStandard)
  const active = rows.find(row => row.id === selected)
  return <section className="mv" aria-label="Multilingual voice benchmark">
    <div className="mv-toolbar"><div className="mv-view-tabs" aria-label="Chart views">{[['space', '01', '3D explorer'], ['languages', '02', 'Language map']].map(([id, number, label]) => <button key={id} aria-pressed={view === id} className={view === id ? 'active' : ''} onClick={() => setView(id)}><span>{number}</span>{label}</button>)}</div></div>
    <div className="mv-main-panel">
      {view === 'space' && <div className="mv-panel-heading"><h2>Three dimensions of a good conversation.</h2><span className="mv-core-note">{domain === 'all' ? 'Core domains only · Airline, Retail, Telecom' : `Task completion: ${domainLabel} · other axes: all core domains`}</span></div>}
      {!rows.some(row => row.task != null) ? <div className="mv-cube-empty"><span>Not yet evaluated</span><h3>No multilingual {domainLabel} results yet.</h3><p>The paper cohort covers Airline, Retail, and Telecom. Submitted packages that include {domainLabel} are ranked in the leaderboard below.</p></div> : view === 'languages' ? <LanguageMap metric={mapMetric} setMetric={setMapMetric} domain={domain} /> : <Space rows={rows} onSelect={selectModel} selected={selected} focused={focused} onClearFocus={() => setFocused(false)} />}
      <div className="mv-panel-footer"><span><strong>4,500</strong> calls</span><span><strong>3</strong> core domains</span></div>
    </div>
    <div className="mv-leaderboard-heading"><h2>Leaderboard</h2><div className="mv-leaderboard-controls"><DomainToggle domain={domain} setDomain={setDomain} />{submissionTypeFilters}</div></div>
    <div className="mv-ranking-meta"><span>All languages{domain !== 'all' && ` · ${domainLabel}`}</span><span>Click a score to see its breakdown.</span></div>
    <div className="reliability-metrics mv-scores">
      <div className="metrics-table-container">
        <table className="reliability-table mv-leaderboard-table">
          <thead><tr>
            <th>Rank</th><th>Model</th><th>Provider</th><th>Reasoning</th>
            {metrics.map(([key, label]) => <th key={key} className="mv-metric-header" aria-sort={sort === key ? 'descending' : 'none'}><button onClick={() => setSort(key)}>{label} ↑{sort === key && <span className="mv-sort-dot"> •</span>}</button>{domain !== 'all' && key !== 'task' && <small className="mv-pooled-note" title="Interaction and Experience are not split by domain">all domains</small>}</th>)}
            <th className="expand-header"><span className="mv-visually-hidden">Breakdown</span></th>
          </tr></thead>
          <tbody>{[...leaderboardRows].sort((a, b) => a[sort] == null ? b[sort] == null ? 0 : 1 : b[sort] == null ? -1 : b[sort] - a[sort]).map((row, index) => {
            if (row.pkg) return <PackageRow key={row.id} row={row} index={index} open={openPackage === row.pkg.dir} onToggle={() => { setBreakdown(null); setOpenPackage(openPackage === row.pkg.dir ? null : row.pkg.dir) }} />
            const expanded = !!breakdown && selected === row.id
            const [provider, logo] = providers[row.id.split('_')[0]]
            return <Fragment key={row.id}>
              <tr className={`model-row${expanded ? ' expanded' : ''}`}>
                <td className="rank-cell"><span className={`rank-number ${['rank-gold', 'rank-silver', 'rank-bronze'][index] || ''}`}>#{index + 1}</span></td>
                <td className="model-info"><button className="mv-model-trigger" onClick={() => expanded ? setBreakdown(null) : openBreakdown(row.id, 'task')} aria-expanded={expanded} aria-controls="voice-metric-breakdown"><span className="model-name">{row.name}</span></button></td>
                <td className="organization-info"><div className="org-container"><div className="company-logo"><img src={`${import.meta.env.BASE_URL}${logo}`} alt="" className="logo-img" /></div><div className="org-text"><span className="org-name">{provider}</span></div></div></td>
                <td className="reasoning-info">{row.effort}</td>
                {metrics.map(([key, label]) => <td key={key} className="metric-cell score-cell"><button className="score-bar-container mv-score-button" aria-label={`${shortName(row)}: ${label} breakdown`} aria-controls="voice-metric-breakdown" aria-expanded={expanded && breakdown === key} onClick={() => openBreakdown(row.id, key)}>
                  {row[key] != null ? <><span className="score-bar-track"><span className="score-bar-fill" style={{ width: `${row[key]}%` }} /></span><span className="score-bar-value">{percent(row[key])}</span></> : <span className="no-data">—</span>}
                </button></td>)}
                <td className="expand-cell"><button className="mv-expand-trigger" aria-label={`${expanded ? 'Collapse' : 'Expand'} ${shortName(row)} breakdown`} aria-expanded={expanded} aria-controls="voice-metric-breakdown" onClick={() => expanded ? setBreakdown(null) : openBreakdown(row.id, 'task')}><span className={`expand-caret${expanded ? ' open' : ''}`}>▶</span></button></td>
              </tr>
              {expanded && <tr className="domain-detail-row"><td colSpan={8} className="domain-detail-cell"><MetricBreakdown active={active} language={language} metric={breakdown} onMetricChange={setBreakdown} onClose={() => setBreakdown(null)} /></td></tr>}
            </Fragment>
          })}{!leaderboardRows.length && <tr><td colSpan={8}><div className="filter-empty-state"><h3>No Results</h3><p>{!showStandard && !showCustom ? 'Select Standard or Custom to view results.' : 'No custom multilingual submissions yet.'}</p></div></td></tr>}</tbody>
        </table>
      </div>
    </div>
    {packageErrors.length > 0 && <p className="mv-footnote" role="alert">Skipped submissions: {packageErrors.join('; ')}</p>}
    <nav className="mv-paper-links" aria-label="Research papers"><span>Papers</span>{[
      ['τ-Multilingual', '2609.35820', 'τ-Multilingual: Benchmarking Voice Agents Across Languages'],
      ['τ-Elicitation', '2609.13602', 'τ-Elicitation: Benchmarking multi-turn entity extraction in voice agents'],
      ['τ-Voice', '2603.13686', 'τ-Voice: Benchmarking Full-Duplex Voice Agents on Real-World Domains'],
    ].map(([label, id, title]) => <a key={id} href={`https://arxiv.org/abs/${id}`} title={title} target="_blank" rel="noopener noreferrer">{label} ↗</a>)}</nav>
  </section>
}
