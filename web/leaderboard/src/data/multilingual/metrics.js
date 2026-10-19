import data from './results.json'

export const mean = values => {
  const present = values.filter(value => value !== null && value !== undefined)
  return present.length ? present.reduce((sum, value) => sum + value, 0) / present.length : null
}
export const percent = value => value == null ? '—' : `${value.toFixed(1)}%`
export const summarize = (model, language = 'all') => {
  const cells = data.cells.filter(cell => cell.model === model.id && (language === 'all' || (language === 'multi' ? cell.language !== 'en' : cell.language === language)))
  const average = key => mean(cells.map(cell => cell[key]))
  const fluencyFailure = average('fluency_failure')
  const fidelityFailure = average('speech_fidelity_failure')
  const interactionFailure = average('interaction_failure')
  return {
    ...model,
    task: average('task'),
    naturalness: fluencyFailure == null ? null : 100 - fluencyFailure,
    fidelity: fidelityFailure == null ? null : 100 - fidelityFailure,
    experience: average('experience'),
    naturalnessParts: {
      fluentAndFaithful: average('experience'),
      fluencyOnly: fluencyFailure == null ? null : fluencyFailure - average('overlapping_failure'),
      fidelityOnly: fidelityFailure == null ? null : fidelityFailure - average('overlapping_failure'),
      both: average('overlapping_failure'),
    },
    interaction: interactionFailure == null ? null : 100 - interactionFailure,
    latency: average('latency_seconds'),
    components: Object.fromEntries(['nonresponse', 'interruption', 'selectivity', 'monologue', 'tool_use'].map(key => [key, mean(cells.map(cell => cell.interaction_components[key]))])),
    domains: Object.fromEntries(['airline', 'retail', 'telecom'].map(key => [key, mean(cells.map(cell => cell.domains[key]))])),
    languages: cells.length,
    experienceLanguages: cells.filter(cell => cell.experience != null).length,
    calls: cells.reduce((sum, cell) => sum + cell.n_calls, 0),
    utterances: cells.reduce((sum, cell) => sum + cell.denominators.experience_utterances, 0),
  }
}

// A point is dominated only if another is at least as good on every axis,
// and strictly better on at least one. Missing measurements cannot compete.
export const frontier = (rows, axes) => rows.filter(row => axes.every(axis => row[axis] != null) && !rows.some(other =>
  axes.every(axis => other[axis] != null && other[axis] >= row[axis]) && axes.some(axis => other[axis] > row[axis])
))

// Heatmap shading relative to the values on screen, so small spreads stay
// visible: the lowest value gets the lightest green and the highest the darkest.
export const heatScale = values => {
  const present = values.filter(value => value != null)
  const low = Math.min(...present)
  const high = Math.max(...present)
  return value => {
    if (value == null) return '#f1f5f9'
    const intensity = high > low ? (value - low) / (high - low) : 0.5
    return `rgba(25, 121, 91, ${0.06 + 0.54 * intensity})`
  }
}
