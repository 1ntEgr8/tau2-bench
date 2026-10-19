// 2.0 (voice-v2) packages listed under manifest.voice_v2_submissions appear as
// rows of the multilingual leaderboard, next to the paper cohort.
// The package is synthetic and served through request interception, so the
// committed manifest stays untouched.
import { test, expect } from '@playwright/test'

const languages = ['en', 'es', 'hi', 'ko', 'pt', 'zh']
const domains = ['airline', 'retail', 'telecom']
const cells = languages.flatMap((language, l) => domains.map((domain, d) => ({ language, domain, score: (40 + 5 * l + 10 * d) / 100 })))
const mean = values => values.reduce((a, b) => a + b, 0) / values.length
const coverage = group => ({ expected_trials: 50 * group.length, scored_trials: 50 * group.length, error_trials: 0, unscored_trials: 0, missing_trials: 0, complete: true, score: mean(group.map(cell => cell.score)), observed_score: mean(group.map(cell => cell.score)) })
const pkg = {
  metadata: {
    schema_version: 'benchmark-metadata-v1', model_name: 'fixture-live', model_organization: 'Fixture Labs', submitting_organization: 'Fixture Labs',
    submission_date: '2026-10-20', contact_info: { email: 'a@example.org' }, submission_type: 'standard', references: [], notes: null,
    agent_provider: 'gemini', agent_model: 'fixture-live', evidence: 'matrix', trajectories: 's3://sierra-tau-bench-public/submissions/fixture',
  },
  submission: {
    frame: { benchmark_version: 'voice-v2', revision: 'fixture-rev', cells: cells.map(({ language, domain }) => ({ language, domain })) },
    summary: {
      benchmark_version: 'voice-v2', frame_revision: 'fixture-rev', complete: true, scored_trials: 900, score: mean(cells.map(cell => cell.score)),
      cells: cells.map(cell => ({ ...cell, complete: true })),
      by_language: Object.fromEntries(languages.map(language => [language, coverage(cells.filter(cell => cell.language === language))])),
      by_domain: Object.fromEntries(domains.map(domain => [domain, coverage(cells.filter(cell => cell.domain === domain))])),
    },
  },
  diagnostics: {
    catalog_revision: 'fixture-catalog',
    cells: cells.map(({ language, domain }) => ({
      language, domain,
      summary: [
        { measure: 'naturalness', applicable_calls: 50, judged_calls: language === 'zh' ? 49 : 50, pass_units: 75, fail_units: 25 },
        ...(language === 'ko' ? [{ measure: 'honorific_agreement', applicable_calls: 50, judged_calls: 50, pass_units: 9, fail_units: 1 }] : []),
      ],
    })),
  },
}
const files = { 'benchmark-metadata.json': pkg.metadata, 'benchmark-submission.json': pkg.submission, 'benchmark-diagnostics.json': pkg.diagnostics }

test.beforeEach(async ({ page }) => {
  await page.route('**/submissions/manifest.json', async route => {
    const manifest = await (await route.fetch()).json()
    await route.fulfill({ json: { ...manifest, voice_v2_submissions: ['fixture_v2'] } })
  })
  await page.route('**/submissions/fixture_v2/*', route => route.fulfill({ json: files[route.request().url().split('/').pop()] }))
})

test('multilingual leaderboard ranks packages with the paper cohort', async ({ page }) => {
  await page.goto('/leaderboard?benchmark=voice')
  const table = page.locator('.mv-leaderboard-table')
  await expect(table.locator('tr.model-row:not(.mv-package-row)')).toHaveCount(5)
  const row = table.locator('tr[data-package="fixture_v2"]')
  await expect(row.locator('.model-name')).toHaveText('fixture-live')
  await expect(row).toContainText('submission')
  await expect(row.locator('td[data-metric="task"]')).toContainText(`${(100 * pkg.submission.summary.score).toFixed(1)}%`)
  await expect(row.locator('td[data-metric="interaction"]')).toHaveText('—')
  await row.locator('.mv-model-trigger').click()
  const detail = page.locator('.mv-leaderboard-table .mv-detail')
  await expect(detail.locator('tr[data-factor="telecom"] td[data-language="en"]')).toHaveText('60.0%')
  await expect(detail.locator('tr[data-factor="all"] td[data-language="ko"]')).toHaveText('65.0%')
  await expect(detail.locator('tr.mv-extra-row')).toHaveCount(2)
  await expect(detail.locator('tr[data-factor="honorific_agreement"] td[data-language="ko"]')).toHaveText('90.0%')
  await expect(detail.locator('tr[data-factor="honorific_agreement"] td[data-language="en"]')).toHaveText('—')
  await expect(detail).toContainText('Naturalness 897/900')
  await expect(detail).toContainText('aws s3 sync s3://sierra-tau-bench-public/submissions/fixture/trajectories/')
})

test('1.x English voice tab is unchanged by 2.0 packages', async ({ page }) => {
  await page.goto('/leaderboard?benchmark=voice')
  await page.getByRole('button', { name: 'English only' }).click()
  await expect(page.locator('.reliability-table.voice-table tbody tr.model-row').first()).toBeVisible()
  await expect(page.locator('tr[data-package]')).toHaveCount(0)
})
