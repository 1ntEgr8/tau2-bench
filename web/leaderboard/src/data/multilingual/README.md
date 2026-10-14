# τ-i18n Voice paper cohort

τ-i18n is the current name for τ-Multilingual (the legacy “multilingual”
benchmark). The related entity-capture benchmark is τ-entity, formerly
“elicitation”; it is not included in this cohort. Internal identifiers and
historical provenance keep their original names.

The Voice leaderboard defaults to this fixed paper cohort. The English-only
leaderboard and progress view retain the existing submission-backed behavior.
This view does not yet ingest Voice V2 submissions; do not present the static
cohort as an automatically updated submission leaderboard.

`results.json` contains 30 language/configuration summaries from commit
`4c7d1463bf3efe0fb85f9406fda4d0d00add1df8` on
`codex/tau-multilingual-public-release`. Its source metadata records the original
artifact paths and SHA-256 hashes. Task and domain scores came from `audit.json`;
the other measurements came from `experience.json`, using the same trial-0 slice.
The imported JSON is unchanged from the website design session.

- Task completion and Interaction macro-average six languages: 900 calls per
  configuration, 4,500 total, with 50 calls per domain/language/configuration.
- Experience, Naturalness, and aligned Speech fidelity macro-average the five
  evaluated localized languages. English is missing, never zero.
- Experience is the share of aligned utterances passing both checks, not the
  average of Naturalness and Speech fidelity.
- Interaction is 100 minus the equal-weight mean of five call-level failure
  rates. These differ from legacy English event-level rates.
- Latency is shown separately in seconds and does not enter Interaction.
- The 3D frontier compares Task completion, Interaction, and Experience with
  higher being better; missing measurements cannot compete.

The homepage Voice card uses the same six-language task averages. Supplemental
banking/elicitation results, raw trajectories, audio, and recovery artifacts are
not included in this integration. Historical judging and infrastructure-outcome checks are complete, as confirmed
by the release owner on 2026-10-14.

Run `npm run build && npm run prerender && npm run test:e2e` from
`web/leaderboard`. `e2e/multilingual.spec.js` checks the frozen aggregates,
missing-data behavior, Pareto comparisons, chart interaction, every breakdown,
latency, filters, mobile layout, and navigation against the built site. Browser
screenshots go to Playwright's ignored test output directory.
