// Frozen multilingual cohort and browser regression checks.
import assert from 'node:assert/strict'
import { test, expect } from '@playwright/test'
import { readFile } from 'node:fs/promises'
// Use a data URL to test the pure module without Node's JSON import-attribute requirements.
const source = await readFile(new URL('../src/data/multilingual/metrics.js', import.meta.url), 'utf8')
const data = JSON.parse(await readFile(new URL('../src/data/multilingual/results.json', import.meta.url), 'utf8'))
const { summarize, frontier } = await import(`data:text/javascript;base64,${Buffer.from(source.replace("import data from './results.json'", `const data = ${JSON.stringify(data)}`)).toString('base64')}`)
test('multilingual cohort aggregates and Pareto frontier', () => {
assert.equal(data.cells.length, 30)
assert.equal(summarize({ id: 'missing-model' }).interaction, null)
for (const model of data.models) {
  const row = summarize(model)
  for (const scope of ['all', 'multi', ...data.languages.map(language => language.id)]) {
    const scoped = summarize(model, scope)
    if (scope === 'en') {
      assert.ok(Object.values(scoped.naturalnessParts).every(value => value === null))
    } else {
      assert.ok(Object.values(scoped.naturalnessParts).every(value => value >= 0 && value <= 100))
      assert.ok(Math.abs(Object.values(scoped.naturalnessParts).reduce((a, b) => a + b) - 100) < 1e-8)
      assert.ok(Math.abs(scoped.naturalnessParts.fluentAndFaithful + scoped.naturalnessParts.fidelityOnly - scoped.naturalness) < 1e-8)
    }
  }
  assert.equal(row.calls, 900)
  assert.equal(row.languages, 6)
  assert.equal(row.experienceLanguages, 5)
  assert.equal(summarize(model, 'multi').calls, 750)
  assert.equal(row.experience, summarize(model, 'multi').experience)
  const sixLanguageTask = data.languages.reduce((sum, language) => sum + summarize(model, language.id).task, 0) / 6
  assert.ok(Math.abs(row.task - sixLanguageTask) < 1e-8)
  assert.equal(summarize(model, 'en').naturalness, null)
  assert.equal(summarize(model, 'en').experience, null)
  assert.ok(Math.abs(row.task - Object.values(row.domains).reduce((a, b) => a + b) / 3) < 1e-8)
}
const points = [{ id: 'a', x: 1, y: 3, z: 1 }, { id: 'b', x: 2, y: 4, z: 0 }, { id: 'c', x: null, y: 5, z: 5 }, { id: 'tie', x: 2, y: 4, z: 0 }]
assert.deepEqual(frontier(points, ['x', 'y']).map(p => p.id), ['b', 'tie'])
assert.deepEqual(frontier(points, ['x', 'y', 'z']).map(p => p.id), ['a', 'b', 'tie'])

})

test('multilingual explorer, breakdowns, filters and navigation', async ({ page }, testInfo) => {
  test.setTimeout(90000)
  await page.setViewportSize({ width: 1440, height: 1100 })
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/leaderboard?benchmark=voice', { waitUntil: 'networkidle' })
  await page.getByRole('heading', { name: /Three dimensions of a good conversation/ }).waitFor()
  assert.equal(await page.locator('.mv-scope select').count(), 0)
  assert.equal(await page.locator('.mv-scores .model-row').count(), 5)
  assert.equal(await page.locator('.mv-scores > .metrics-table-container > table > thead th').count(), 8)
  assert.equal(await page.locator('.mv-score-button').count(), 15)
  assert.equal(await page.locator('.mv-detail').count(), 0)
  assert.equal(await page.getByText('THE FULL PICTURE', { exact: true }).count(), 0)
  assert.equal(await page.locator('.mv-diagnostics').count(), 0)
  assert.equal(await page.locator('.mv-view-tabs button').count(), 2)
  assert.equal(await page.locator('.mv-view-tabs button').first().getAttribute('aria-pressed'), 'true')
  assert.equal(await page.locator('.mv-cube-point circle[fill="none"]').count(), 5)
  assert.equal(await page.locator('.mv-pareto-edge').count(), 0)
  await page.locator('.mv-main-panel').screenshot({ path: testInfo.outputPath('tau-multilingual-cube.png') })
  await page.locator('.mv-scores').screenshot({ path: testInfo.outputPath('tau-multilingual-scores.png') })
  await page.getByRole('button', { name: 'Full scale · 0–100' }).click()
  assert.equal(await page.getByRole('button', { name: 'Full scale · 0–100' }).getAttribute('aria-pressed'), 'true')
  await page.getByRole('button', { name: 'Fit results', exact: true }).click()
  assert.equal(await page.getByRole('slider').count(), 0)
  const cube = await page.locator('.mv-cube').boundingBox()
  const geometry = () => page.locator('.mv-cube polygon').first().getAttribute('points')
  const initialGeometry = await geometry()
  const dragCube = async (dx, dy) => {
    await page.mouse.move(cube.x + 30, cube.y + 30)
    await page.mouse.down()
    await page.mouse.move(cube.x + 30 + dx, cube.y + 30 + dy, { steps: 5 })
    await page.mouse.up()
  }
  await dragCube(70, 0)
  assert.notEqual(await geometry(), initialGeometry)
  await page.getByRole('button', { name: 'Reset view' }).click()
  assert.equal(await geometry(), initialGeometry)
  await dragCube(0, 60)
  assert.notEqual(await geometry(), initialGeometry)
  await page.getByRole('button', { name: 'Reset view' }).click()
  assert.equal(await geometry(), initialGeometry)
  await page.getByRole('button', { name: /^GPT RT 2 · xhigh, task/ }).click()
  assert.equal(await page.locator('.mv-cube-point.is-muted').count(), 4)
  assert.equal(await page.locator('.mv-cube-point.is-focused').count(), 1)
  await page.getByRole('button', { name: /^GPT RT 2 · xhigh, task/ }).click()
  assert.equal(await page.locator('.mv-cube-point.is-muted').count(), 0)
  await page.getByRole('button', { name: /^GPT RT 2 · xhigh, task/ }).press('Enter')
  await page.getByRole('button', { name: 'Show all models', exact: true }).click()
  assert.equal(await page.locator('.mv-cube-point.is-muted').count(), 0)

  await page.getByRole('button', { name: 'GPT RT 2 · xhigh: Experience breakdown', exact: true }).click()
  await page.locator('.mv-pass-heatmap').waitFor()
  assert.equal(await page.locator('.mv-breakdown-tabs button').count(), 3)
  assert.equal(await page.locator('.model-row.expanded + .domain-detail-row .mv-detail').count(), 1)
  assert.equal(await page.locator('.mv-pass-heatmap tbody tr').count(), 3)
  assert.equal(await page.locator('.mv-pass-heatmap thead th').count(), 8)
  assert.equal(await page.locator('.mv-pass-heatmap .is-missing').count(), 3)
  for (const cell of data.cells.filter(cell => cell.model === 'openai_xhigh')) {
    const expected = { naturalness: cell.fluency_failure == null ? null : 100 - cell.fluency_failure, fidelity: cell.speech_fidelity_failure == null ? null : 100 - cell.speech_fidelity_failure, overall: cell.experience }
    for (const [factor, value] of Object.entries(expected)) {
      assert.equal(await page.locator(`.mv-pass-heatmap [data-factor="${factor}"] [data-language="${cell.language}"]`).innerText(), value == null ? '—' : `${value.toFixed(1)}%`)
    }
  }
  const experienceTotals = summarize(data.models.find(model => model.id === 'openai_xhigh'))
  for (const [factor, value] of Object.entries({ naturalness: experienceTotals.naturalness, fidelity: experienceTotals.fidelity, overall: experienceTotals.experience })) {
    assert.equal(await page.locator(`.mv-pass-heatmap [data-factor="${factor}"] [data-language="overall"]`).innerText(), `${value.toFixed(1)}%`)
  }
  assert.equal(await page.locator('.mv-pass-heatmap .mv-overall-row th').innerText(), 'Overall')
  await page.locator('.mv-detail').screenshot({ path: testInfo.outputPath('tau-multilingual-experience.png') })
  await page.locator('.mv-breakdown-tabs').getByRole('button', { name: /Interaction/ }).click()
  assert.equal(await page.locator('.mv-failure-heatmap tbody tr').count(), 7)
  assert.equal(await page.locator('.mv-failure-heatmap tbody td').count(), 49)
  for (const cell of data.cells.filter(cell => cell.model === 'openai_xhigh')) {
    for (const [factor, value] of Object.entries({ ...cell.interaction_components, overall: 100 - cell.interaction_failure })) {
      assert.equal(await page.locator(`.mv-failure-heatmap [data-factor="${factor}"] [data-language="${cell.language}"]`).innerText(), `${value.toFixed(1)}%`)
    }
    assert.equal(await page.locator(`.mv-latency-row [data-language="${cell.language}"]`).innerText(), `${cell.latency_seconds.toFixed(2)} s`)
  }
  const interactionTotals = summarize(data.models.find(model => model.id === 'openai_xhigh'))
  assert.equal(await page.locator('.mv-latency-row [data-language="overall"]').innerText(), `${interactionTotals.latency.toFixed(2)} s`)
  for (const [factor, value] of Object.entries({ ...interactionTotals.components, overall: interactionTotals.interaction })) {
    assert.equal(await page.locator(`.mv-failure-heatmap [data-factor="${factor}"] [data-language="overall"]`).innerText(), `${value.toFixed(1)}%`)
  }
  await page.locator('.mv-detail').screenshot({ path: testInfo.outputPath('tau-multilingual-interaction.png') })
  await page.locator('.mv-breakdown-tabs').getByRole('button', { name: /Task completion/ }).click()
  assert.equal(await page.locator('.mv-task-heatmap tbody tr').count(), 6)
  assert.equal(await page.locator('.mv-task-heatmap tbody tr.mv-extra-row').count(), 2)
  assert.equal(await page.locator('.mv-task-heatmap thead th').count(), 8)
  const selectedModel = data.models.find(model => model.id === 'openai_xhigh')
  for (const cell of data.cells.filter(cell => cell.model === selectedModel.id)) {
    for (const [factor, value] of Object.entries({ ...cell.domains, overall: cell.task })) {
      assert.equal(await page.locator(`.mv-task-heatmap [data-factor="${factor}"] [data-language="${cell.language}"]`).innerText(), `${value.toFixed(1)}%`)
    }
  }
  const totals = summarize(selectedModel)
  for (const [factor, value] of Object.entries({ ...totals.domains, overall: totals.task })) {
    assert.equal(await page.locator(`.mv-task-heatmap [data-factor="${factor}"] [data-language="overall"]`).innerText(), `${value.toFixed(1)}%`)
  }
  await page.locator('.mv-detail').screenshot({ path: testInfo.outputPath('tau-multilingual-task.png') })
  assert.equal(await page.locator('.mv-supplemental-row').count(), 0)
  await page.getByRole('button', { name: 'Close breakdown', exact: true }).click()
  assert.equal(await page.locator('.mv-detail').count(), 0)
  await page.getByRole('button', { name: 'Grok Voice · default: Interaction breakdown', exact: true }).click()
  assert.ok((await page.locator('.mv-detail-heading').innerText()).includes('Grok Voice'))
  await page.getByRole('button', { name: 'Close breakdown', exact: true }).click()

  await page.getByRole('button', { name: 'Language map', exact: true }).click()
  assert.equal(await page.locator('.mv-heatmap tbody td').count(), 42)
  assert.equal(await page.locator('.mv-heatmap thead th').count(), 8)
  const metricButtons = page.locator('.mv-map-metrics')
  for (const [key, label] of [['interaction', 'Interaction'], ['experience', 'Experience'], ['task', 'Task completion'], ['experience', 'Experience']]) {
    await metricButtons.getByRole('button', { name: label, exact: true }).click()
    assert.equal(await metricButtons.getByRole('button', { name: label, exact: true }).getAttribute('aria-pressed'), 'true')
    for (const model of data.models) {
      for (const language of [...data.languages.map(l => l.id), 'all']) {
        const value = summarize(model, language)[key]
        assert.equal(await page.locator(`.mv-heatmap [data-model="${model.id}"] [data-language="${language}"]`).innerText(), value == null ? '—' : `${value.toFixed(1)}%`)
      }
    }
    for (const language of [...data.languages.map(l => l.id), 'all']) {
      const values = data.models.map(model => summarize(model, language)[key]).filter(value => value != null)
      const expected = values.length ? `${(values.reduce((a,b) => a+b,0) / values.length).toFixed(1)}%` : '—'
      assert.equal(await page.locator(`.mv-heatmap [data-model="overall"] [data-language="${language}"]`).innerText(), expected)
    }
  }
  await page.locator('.mv-main-panel').screenshot({ path: testInfo.outputPath('tau-multilingual-language-map.png') })
  for (const text of ['SIX-LANGUAGE COMPARISON', 'Look beyond the leaderboard.', 'LANGUAGE COVERAGE', 'ALL LANGUAGES', 'Metric definitions, coverage & source data', 'Trial 0 · Task & interaction: 6 languages · Experience: 5', 'papers/tau-multilingual/reproduction/audit.json', 'papers/tau-multilingual/reproduction/experience.json']) {
    assert.equal(await page.getByText(text, { exact: true }).count(), 0)
  }
  assert.equal(await page.locator('.mv-methodology').count(), 0)
  assert.equal(await page.locator('.mv-paper-links a[href^="https://arxiv.org/abs/"]').count(), 2)

  await page.locator('.mv-map-metrics').getByRole('button', { name: 'Experience', exact: true }).click()
  assert.equal(await page.locator('.mv-heatmap tbody .mv-heat-cell').filter({ hasText: '—' }).count(), 6)
  assert.equal(await page.locator('.mv-heatmap tbody button').count(), 0)
  await page.getByRole('button', { name: 'GPT RT 2 · xhigh: Experience breakdown', exact: true }).click()
  assert.equal(await page.locator('.mv-pass-heatmap thead th').count(), 8)
  assert.equal(await page.locator('.mv-pass-heatmap [data-language="es"]').count(), 3)
  assert.equal(await page.locator('.mv-pass-heatmap [data-language="en"] .is-missing').count(), 3)
  await page.locator('.mv-breakdown-tabs').getByRole('button', { name: /Task completion/ }).click()
  assert.equal(await page.locator('.mv-task-heatmap thead th').count(), 8)
  for (const factor of ['banking_knowledge', 'intake']) {
    const row = page.locator(`.mv-task-heatmap tr.mv-extra-row[data-factor="${factor}"]`)
    assert.equal(await row.locator('.is-missing').count(), 7)
  }
  await page.getByRole('button', { name: 'Experience ↑' }).click()
  assert.ok((await page.locator('.mv-scores .model-row').first().innerText()).includes('xhigh'))
  await page.getByRole('button', { name: 'English only', exact: true }).click()
  await page.getByRole('heading', { name: 'τ³-Voice Leaderboard' }).waitFor()
  await page.getByRole('button', { name: 'Multilingual', exact: true }).click()
  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('button', { name: 'GPT RT 2 · xhigh: Experience breakdown', exact: true }).click()
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
  await page.locator('.mv-detail').screenshot({ path: testInfo.outputPath('tau-multilingual-mobile.png') })
  await page.locator('.mv-breakdown-tabs').getByRole('button', { name: /Task completion/ }).click()
  assert.equal(await page.locator('.mv-supplemental-row').count(), 0)
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))
  await page.locator('.mv-detail').screenshot({ path: testInfo.outputPath('tau-multilingual-task-mobile.png') })

  await page.goto('/')
  await page.locator('.preview-table-wrapper').filter({ hasText: 'All languages' }).waitFor()
  await page.locator('.preview-table-wrapper').filter({ hasText: 'Multilingual' }).click()
  await page.getByRole('heading', { name: /Three dimensions of a good conversation/ }).waitFor()
  assert.deepEqual(errors, [])
  console.log('PASS: six-language task/interaction averages, five-language Experience coverage, missing data, three headline metrics, factor-by-language heatmap values and missing data, clickable/tabbed breakdowns, cube gestures/focus, mobile layout, homepage navigation, browser errors.')
})

test('multilingual submission filters and English progress remain available', async ({ page }) => {
  await page.goto('/leaderboard?benchmark=voice')
  await expect(page.getByText(/Multilingual voice benchmark within τ³-Voice/)).toBeVisible()
  const filters = page.locator('.mv-leaderboard-heading')
  await expect(filters.getByRole('heading', { name: 'Leaderboard', exact: true })).toBeVisible()
  await expect(page.locator('.mv-scores .model-row')).toHaveCount(5)
  const toggle = name => filters.locator('.checkbox-container').filter({ hasText: new RegExp(`^${name}$`) }).click()
  await toggle('Standard')
  await expect(page.locator('.mv-scores .model-row')).toHaveCount(0)
  await expect(page.getByText('Select Standard or Custom to view results.')).toBeVisible()
  await toggle('Custom')
  await expect(page.getByText('No custom multilingual submissions yet.')).toBeVisible()
  await toggle('Standard')
  await expect(page.locator('.mv-scores .model-row')).toHaveCount(5)
  await page.getByRole('button', { name: 'English only', exact: true }).click()
  await expect(page.locator('.mv')).toHaveCount(0)
  await expect(page.locator('.domain-toggle-switch')).toBeVisible()
  await page.goto('/progress?benchmark=voice')
  await expect(page.getByRole('button', { name: 'English only', exact: true })).toHaveAttribute('aria-pressed', 'true')
  await expect(page.locator('.mv')).toHaveCount(0)
})
