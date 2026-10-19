// voice-v2-banking and voice-v2-entity packages listed under
// manifest.voice_v2_domain_submissions fill the Banking and Intake rows of the
// same agent's voice-v2 row without changing its Overall. Packages are
// synthetic and served through request interception, so the committed
// manifest stays untouched.
import { test, expect } from '@playwright/test'

const languages = ['en', 'es', 'hi', 'ko', 'pt', 'zh']
const mean = values => values.reduce((a, b) => a + b, 0) / values.length
const coverage = group => ({ expected_trials: group.length, scored_trials: group.length, error_trials: 0, unscored_trials: 0, missing_trials: 0, complete: true, score: mean(group.map(cell => cell.score)), observed_score: mean(group.map(cell => cell.score)) })
const metadata = (model, provider) => ({
  schema_version: 'benchmark-metadata-v1', model_name: model, model_organization: 'Fixture Labs', submitting_organization: 'Fixture Labs',
  submission_date: '2026-10-20', contact_info: { email: 'a@example.org' }, submission_type: 'standard', references: [], notes: null,
  agent_provider: provider, agent_model: model, evidence: 'matrix', trajectories: null,
})
const submission = (version, domains, score) => {
  const cells = languages.flatMap((language, l) => domains.map((domain, d) => ({ language, domain, score: score(l, d) })))
  return {
    frame: { benchmark_version: version, revision: `${version}-rev`, cells: cells.map(({ language, domain }) => ({ language, domain })) },
    summary: {
      benchmark_version: version, frame_revision: `${version}-rev`, complete: true, scored_trials: cells.length, score: mean(cells.map(cell => cell.score)),
      cells: cells.map(cell => ({ ...cell, complete: true })),
      by_language: Object.fromEntries(languages.map(language => [language, coverage(cells.filter(cell => cell.language === language))])),
      by_domain: Object.fromEntries(domains.map(domain => [domain, coverage(cells.filter(cell => cell.domain === domain))])),
    },
  }
}
const core = submission('voice-v2', ['airline', 'retail', 'telecom'], (l, d) => (40 + 5 * l + 10 * d) / 100)
const packages = {
  fixture_core: { 'benchmark-metadata.json': metadata('fixture-live', 'gemini'), 'benchmark-submission.json': core },
  // Same agent as fixture_core: fills its Banking row.
  fixture_banking: { 'benchmark-metadata.json': metadata('fixture-live', 'gemini'), 'benchmark-submission.json': submission('voice-v2-banking', ['banking_knowledge'], l => (10 + l) / 100) },
  // No voice-v2 package for this agent: shown only under Intake.
  fixture_entity: { 'benchmark-metadata.json': metadata('fixture-entity-only', 'xai'), 'benchmark-submission.json': submission('voice-v2-entity', ['intake'], l => (60 + 2 * l) / 100) },
}
const percent = value => `${(100 * value).toFixed(1)}%`

test.beforeEach(async ({ page }) => {
  await page.route('**/submissions/manifest.json', async route => {
    const manifest = await (await route.fetch()).json()
    await route.fulfill({ json: { ...manifest, voice_v2_submissions: ['fixture_core'], voice_v2_domain_submissions: ['fixture_banking', 'fixture_entity'] } })
  })
  await page.route('**/submissions/fixture_*/*', route => {
    const [dir, file] = route.request().url().split('/').slice(-2)
    const json = packages[dir][file]
    return json ? route.fulfill({ json }) : route.fulfill({ status: 404, body: '' })
  })
})

const selectDomain = (page, name) => page.locator('.mv-domain-toggle button', { hasText: name }).click()

test('domain packages never change Overall or add Overall rows', async ({ page }) => {
  await page.goto('/leaderboard?benchmark=voice')
  const table = page.locator('.mv-leaderboard-table')
  await expect(table.locator('tr.mv-package-row')).toHaveCount(1)
  await expect(table.locator('tr[data-package="fixture_core"] td[data-metric="task"]')).toContainText(percent(core.summary.score))
})

test('banking fills the matching voice-v2 row', async ({ page }) => {
  await page.goto('/leaderboard?benchmark=voice')
  await selectDomain(page, 'Banking')
  const row = page.locator('.mv-leaderboard-table tr[data-package="fixture_core"]')
  await expect(row.locator('td[data-metric="task"]')).toContainText(percent(packages.fixture_banking['benchmark-submission.json'].summary.score))
  await row.locator('.mv-model-trigger').click()
  const detail = page.locator('.mv-leaderboard-table .mv-detail')
  await expect(detail.locator('tr[data-factor="banking_knowledge"] td[data-language="ko"]')).toHaveText('13.0%')
  await expect(detail.locator('tr[data-factor="intake"] td[data-language="ko"]')).toHaveText('—')
  await expect(detail.locator('tr[data-factor="all"] td[data-language="all"]')).toHaveText(percent(core.summary.score))
  await expect(detail).toContainText('voice-v2-banking · frame')
  await expect(detail).toContainText('Submitted 2 packages')
})

test('an entity package without voice-v2 appears only under Intake', async ({ page }) => {
  await page.goto('/leaderboard?benchmark=voice')
  const table = page.locator('.mv-leaderboard-table')
  await expect(table.locator('tr[data-package="fixture_entity"]')).toHaveCount(0)
  await selectDomain(page, 'Intake')
  const row = table.locator('tr[data-package="fixture_entity"]')
  await expect(row.locator('td[data-metric="task"]')).toContainText(percent(packages.fixture_entity['benchmark-submission.json'].summary.score))
  await expect(table.locator('tr[data-package="fixture_core"] td[data-metric="task"]')).toHaveText('—')
  await row.locator('.mv-model-trigger').click()
  const detail = page.locator('.mv-leaderboard-table .mv-detail')
  await expect(detail.locator('tr[data-factor="intake"] td[data-language="en"]')).toHaveText('60.0%')
  await expect(detail.locator('tr[data-factor="all"] td[data-language="all"]')).toHaveText('—')
  await selectDomain(page, 'Airline')
  await expect(row).toHaveCount(0)
})
