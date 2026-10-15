# Voice V2 publication package

Voice V2 uses a separate, versioned submission contract. The existing flat
`submission.json` and English Voice V1 leaderboard remain unchanged. A Voice V2
package has all six languages (`en`, `es`, `hi`, `ko`, `pt`, `zh`) and all three
domains (`airline`, `retail`, `telecom`), with 50 selected task/trial keys per
cell. The contract is generated at
`web/leaderboard/public/submissions/schema_voice_v2.json`.

> Terms such as *pack*, *cell*, *binding revision* and *pin* are defined in the
> [glossary](glossary.md).

The preparation command consumes **saved** trajectories, interaction-event
facts, and judge verdicts. It does not run simulations or judges:

```bash
tau2 submit prepare /path/to/18-cell-source \
  --historical-voice-v2 \
  --provider gemini \
  --interaction-facts /path/to/interaction_call_outcomes.csv \
  --judges-root /path/to/consolidated/judge-verdicts \
  --metadata-file /path/to/publication-metadata.json \
  --output /path/to/new-voice-v2-package

tau2 submit validate /path/to/new-voice-v2-package
```

The metadata file supplies the publication fields missing from a local draft,
particularly `submission_date` and `contact_info.email`. It can override draft
metadata, but it cannot override results: all 18 cells are rederived from the
saved inputs. For example:

```json
{
  "submission_date": "2026-10-09",
  "contact_info": {"email": "submitter@example.com"},
  "methodology": {"notes": "Describe the evaluated agent and simulator here."}
}
```

`submission.json` contains the canonical `BenchmarkSubmission` task/trial
observations and recomputable summary, nested language/domain projections,
coverage and outcome counts for each measure, and the hash of
`provenance.json`. The package also contains `trajectories/<language>/<domain>`
(including results, simulation files, and audio), `judges/<provider>/<language>`
(the saved per-call verdicts), and `interaction_facts.csv`. The provenance
manifest hashes every result, simulation, verdict, and fact input used to
derive the score. If the source contains a retry-correction manifest,
preparation preserves and hashes it as `source_correction.json`.
When the fact table comes with an upstream provenance JSON and README,
preparation preserves and hashes both. The fact table's underlying
nonresponse, interruption, selectivity, and monologue inputs came from
`extract_interaction_quality_metrics` and the registered quality checkers;
tool-misuse inputs came from the saved `tool_use` judge factors. The package
provenance records this derivation explicitly.

The `--historical-voice-v2` adapter is only for the frozen 2026-10-02 release
exports. It checks each saved run's explicit `language_tasks` canonical ID map,
binding revision, selected tasks/trials, result index, and simulation reward
against the published Voice V2 frame. It preserves the recorded condition
fingerprint and hashes each `results.json`; it does not invent condition fields
that were added after those calls ran. New runs use the strict
`tau2 submit prepare <matrix-root>` reader (described in the [submission guide](leaderboard-submission.md#20-benchmarks-text-v1-and-voice-v2)), which remains unchanged
and requires the current complete recorded conditions. The historical adapter
must never be used to qualify an otherwise incomplete new run.

For each cell, `selected = scored + infrastructure` and `scored = pass + fail`.
Pass@1 is `100 × pass / scored`. Infrastructure calls have no benchmark reward
and are not counted as task failures. Each call-level measure reports `pass`,
`fail`, `unscored`, `excluded`, and `pass_rate`; the rate denominator is only
`pass + fail`. Missing judge input cannot become a failure.

Interaction uses `voice-v2-interaction-call-v1`: the five components are
nonresponse, interruption, selectivity, monologue, and tool misuse. An
interruption component fails after **more than three** detected events;
selectivity fails after **more than three** combined backchannel, vocal-tic,
or non-directed-speech errors. The other three components retain their saved
binary results. A call fails Interaction when at least two components fail.
When missing components could change that decision, the call is unscored.
The 12 documented Korean unresponsive-period calls are excluded.

Experience uses `voice-v2-experience-call-v1`: pair the naturalness and
speech-fidelity judgments for each eligible agent utterance. A call fails
when **more than 25%** of eligible paired utterances have either issue. Calls
with missing or incomplete judge coverage are unscored. The pairing follows
the recorded Live segment IDs, or the frozen non-Live grouping of agent
speech turns and audio tick spans. The two measures are separate from task
completion.

`tau2 submit validate` rehashes the inputs, verifies the complete benchmark
frame and unique task/trial keys, checks every simulation against its result
index, rejects extra or missing verdict IDs, and recomputes all scores. The
existing maintainer upload tool validates Voice V2 locally but intentionally
blocks upload through the flat-domain S3 path. Publishing to the Voice V2
leaderboard requires a separate nested-path uploader and display/ranking work.
Preparation and local validation do not upload the package or imply that the
V2 leaderboard UI is ready.

The 2026-10-02 release's Gemini local draft export has a versioned correction
at `data/simulations/release_runs/2026-10-02/leaderboard-v2-corrected-2026-10-09`.
It replaces two omitted scored retries in the new copy; the original draft is
preserved. Its `correction.json` records old and new simulation IDs, selected
task/trial keys, rewards, and source/output hashes. Across all three providers,
the corrected release has 2,700 selected calls, 2,689 scored calls, and 11
infrastructure calls. Golden recomputation matched every saved call-level
Interaction and Experience outcome: Interaction has 912 passes, 1,743 fails,
22 unscored, and 12 excluded; Experience has 1,019 passes, 1,500 fails, and
170 unscored. The adapter was exercised offline across all three 900-call
exports. A self-contained, copy-on-write Gemini package also passed the full
rehash and rederivation validator with 900 scored calls and 1,822 hashed input
files. The historical release adapter verifies recorded fingerprints but does
not claim that pre-release calls recorded later-added condition fields.
