# Selected-entity intake localization

The intake extension uses the existing database-localization maps and publication
workflow. It maps the frozen 200 tasks (clinic, auto and hotel), their used context,
and their selected stimulus dependencies. It never redraws a localized bank or
requires 40 entries per tier. The canonical English task files stay unchanged.

Complete draft maps for Spanish, Portuguese, Hindi, Korean and Chinese are checked
in at `data/tau2/multilingual/<lang>/localization/intake.json`. Each contains all
2,007 selected inventory rows, including 39 reused retail callee identities with
review lineage. No new locale content is approved by this change. Do not bind new pack domains or run bulk paid
voice evaluations until their exact maps, speech stimuli and native conventions
have human review.

## Inventory and authoring

Use the ordinary factory verbs; the inventory adapter belongs to intake:

```bash
tau2 factory inventory-database --domain intake --output intake-source.json
tau2 factory draft-database-localization --request intake-request.json \
  --output intake-map.json --checkpoints intake-checkpoints
```

The draft request follows `DatabaseDraftCommand`, including the source file,
language/region settings, revision, explicit model arguments and references to
reviewed maps from other domains. `reuse_source` binds a digest-approved retail map
for incidental callee identities, with explicit entity lineage. Those donor values
stay local and are excluded from provider context. Measurement slots retain their
own family/tier constraints and need a new capture-hazard review; the donor
allocator does not assign measured values. Authoring uses the existing fixed versioned
prompt and `generate()` seam. The recorded `intake-authoring-v3` addition explicitly
allows localized spoken greetings/openers and selected TTS distortion fields while
keeping simulator instructions in English; other constraints live in versioned
source entry context. Written answers are generated before their task-scoped
decoys/pronunciation/style dependencies, which receive the exact completed mapping.
The request separates required output IDs from read-only reference context.
Reuse existing reviewed identities and contact values only where
the new task's spelling hazard, difficulty and context fit. Reuse is supporting
review evidence, not approval for a new use.

`intake_localization` is a typed dependency document inside the same source/map.
It contains a shared caller greeting, callback opener template, written-clarification
conventions, spoken digit names and exactly 200 selected stimulus records. These
records include only used decoys, pronunciation distortions and applicable style
or omission instructions. When a source entry has no decoy, the deterministic
inventory uses one other selected value from the same family/tier; it never loads
unused entries into the localized review surface.

A source extraction currently contains 200 measured slots, 272 decoy references,
69 distortion-bearing tokens, 237 spelling-style references and 50 applicable
omission references. Context and repeated roles also require review. These counts
are not the number of unique spreadsheet rows and do not include whole banks.

The raw source inventory is source evidence, not an approved locale or generated
candidate. Locale suggestions must come from reference reuse or the factory model
workflow. Native and romanized forms are views of one entity; romanization is not
an accepted answer unless the required protocol itself uses Latin characters.

The candidate maps pin canonical source revision
`2e51424f7e58424b34b04a56bb1a2bb1360cc80a`. The generation used `gpt-5.5`
with explicit `reasoning_effort=low`, batches of eight and concurrency three.
The final eight Chinese name-decoy repairs use singleton batches and explicit
`reasoning_effort=medium` to resolve pinyin collisions.
Versioned selected-entry revisions repaired compiler failures without redrawing
successful entries. Maps retain model/prompt/input digests and parent-map lineage.
The `intake-candidate-repair-v3` guidance preserves ASCII phone/code submissions,
ISO dates, exact AM/PM clock values, noncollapsing decoys and the English machine
annotation around localized spoken dates. Clock/code/phone repairs include their
linked decoys and clarification instructions. Generation validates pronunciation
shape, literal written-token matching and no-op distortions before checkpointing.

Only public synthetic intake fixture content and new generated measurement forms
are supplied to the provider. All retained banks, tasks and task manifest match
public `tau-elicitation` commit `5e1fcb00d` byte-for-byte; reused reviewed retail
identities stay local. No locale approval artifact or published bundle is added.

Review the maps through `tau2 factory export-localization-spreadsheet`, together with the
banking maps, to obtain one language tab with stable domain/entity identifiers.
All new intake decisions must start pending, including reused identities in their
new context. Apply corrections through the ordinary map revision and review
workflow; never hand-edit generated candidates or infer approval from compilation.

## Compilation and publication

The compiler applies exact map occurrences to the selected tasks, gold actions,
private lookup payloads and selected stimulus records. It then deterministically
renders caller identity/context and callback openers from the mapped records and
reviewed shared template. It validates the complete ten-family × two-tier × ten-slot
frame and replays every gold submission through the real scaffolded toolkit.
Caller/gold disagreement, collapsed decoys, missing pronunciation tokens and invalid
VIN/member-ID checksums fail compilation. Medication strengths/units and protocol
date/time values must remain unchanged.

Native name fields use the explicit `native_name`/`native_name_list` folds:
Unicode NFC, casing and whitespace normalization, preserving vowel marks,
accents, punctuation and written identity. This is deliberately stricter than the
English name fold. Dates, times, phones, emails and codes retain their declared
protocol representations. Native-speaker review must assess acceptable written
variants and likely false negatives before content publication.

Publication goes through the language factory (`add-domain` with the map in
`reuse` mode, then the whole-draft content review and `approve`). The resulting `PublishedIntakeBundle` includes fresh typed agent/caller DBs,
strategy policies, tasks, metadata and the selected stimulus inventory. Pack
bindings must point to this published bundle. The new adapter lives in the intake
package so unrelated regional scenario fingerprints remain unchanged.

Agent-directed and scaffolded tasks retain their separate tool/caller contracts.
Only the domain-owned greeting and reviewed clarification clause vary; no mandatory
spelling strategy is added to the agent-directed agent. Prompt preview uses the
same instruction override as execution. Native callers receive a minimal outbound
frame plus their language persona, rather than the unrelated shared voice task
instructions.

## Stimuli and measurement

Native calls use `intake-selected-stimuli-v1`, distinct from the historical English
bank sampler. Self-correction samples only reviewed selected decoys. Wrong-slot
errors use localized filled fields. Mispronunciation uses the selected token's exact
reviewed TTS distortion. Spelling-style and omission instructions are authored in
the source map; an empty string explicitly marks an inapplicable phenomenon.
Restart stimuli mutate a Unicode grapheme in the exact written value only when
there are enough units. Inapplicable probability mass remains clean; no fallback
uses unused English bank entries.

### Caller clock expressions

`intake-local-time-display-v1` projects time values at the caller boundary while
keeping the maps, records and submitted answers in their required H:MM AM/PM
format. English strategy loaders apply the same projection after reading the raw
frozen tasks; source inventory and frozen files remain unchanged. Localized
compilation applies it to caller lookups, decoys and omission examples, and
records each stimulus's time-display format for runtime complications.

For an idiomatic time, a private lookup contains both forms, for example:
`quarter past two (the real time this refers to, from your records: 2:15 PM)`.
The caller instruction says to initially use only the expression, without its
day period, and to give the complete time when asked for clarification or its
written form. Read-back confirmation uses the complete time. This initial
omission is part of the idiomatic payload, independent of whether an additional
lazy-omission complication is sampled. Times without a quarter/half expression
use complete local notation unless the existing omission complication applies.

Hindi uses सवा/पौने and half-hour forms including the exceptions डेढ़ (1:30)
and ढाई (2:30). English uses quarter past/to and half past. Spanish uses
y cuarto/y media/menos cuarto; Brazilian Portuguese uses e quinze/e meia/quinze
para. Mandarin uses 一刻/半/三刻. Korean uses 반 and 15분 전 rather than a
literal translation of English “quarter.” Their complete written anchors retain
the original instant, including noon/midnight and the hour wrap in “to” forms.
Romanized forms express the same values in Latin script for review only.

Self-correction and wrong-slot examples keep a complete time, so a period-flip
decoy does not collapse into the same bare expression as the answer. Written
restarts use the complete written anchor. The time annotation is excluded from
spell-event matching; detector version 1.2.0 records this extension. English
complication catalog 2.5.0 retains the 2.4.0 draw stream, preserving existing
kind assignments while recording the changed caller surfaces.

The formatting rules follow [University of Texas Hindi grammar](https://hindi.la.utexas.edu/wp-content/uploads/sites/8/2022/03/Skeleton_Grammar.pdf),
[RAE Spanish time guidance](https://www.rae.es/dpd/hora),
[Brazilian Cooperation Agency Portuguese materials](https://abc.gov.br/api/publicacaoarquivo/208),
[Lanzhou University Mandarin lessons](https://chinese.lzu.edu.cn/chinese/upload/files/20201116/d062946b33b94a7089d1cf0d0ff916a9.pdf),
and [National Institute of Korean Language midnight guidance](https://www.korean.go.kr/front/mcfaq/mcfaqView.do?mcfaq_seq=6073&mn_id=217).
Native review must still assess regional frequency/register (especially Mandarin
三刻 and Korean minutes-before forms), clarification behavior and spoken delivery.
No new tasks or difficulty reassignment are introduced.

All affected prompt/scorer/catalog/detector versions and scenario fingerprints are
recorded on results. English source runs keep their existing sampler and score
rules. Localized calls require a reviewed regional selected-stimulus bundle; they
fail rather than silently running without it.

Native effort detection counts explicitly comma-separated Unicode graphemes or
reviewed spoken digit names and tracks faltered prefixes only alongside a correct
redelivery. Character explanations and unseparated clarifications can be missed.
The recorded detection version distinguishes these units from historical English
letter counts; raw cross-script counts are not validated measures of equal effort.

## Validation and remaining human gates

Automated coverage includes byte-identical English task reconstruction; all 200
gold replays; source/map fidelity; caller/gold agreement; Unicode identity-positive
and wrong-value-negative cases; code checksums; selected-only runtime access;
complication feasibility; strategy adaptation; exact publication approvals; fresh
DB isolation; and existing-domain inventory/compile/runtime regressions.

Manual offline review:

1. Inventory from the source revision and draft into a fresh output directory.
2. Inspect original, suggested, decision, replacement and reason on each language
   tab of the shared entity review sheet; retain stable domain/entity IDs.
3. Reject stale source/candidate revisions and pending/disapproved publication.
4. Ingest revisions, recompile, replay gold actions, and inspect both strategies'
   prompt previews. The caller's private value must not appear in agent prompts.
5. After native review, run a recorded explicit `--benchmark none` pilot with
   supported reasoning effort pinned and coverage of all 20 family/tier cells,
   plus applicable complication/clarification paths. Review actual TTS audio.

Native review is still required for every used entity/decoy in context, tier
assignment, pronunciation distortion, digit/character convention, written-form
rule and changed prompt. Medication terminology needs a bilingual subject-aware
reviewer. Human review and live audio evidence cannot be replaced by offline gold
replay. These gates leave the PR in draft; no default language binding or official
headline manifest is changed here.

## Country-specific candidate audit and repair

`tau2 factory audit-intake-candidates --request REQUEST.json --output AUDIT.json
--checkpoints DIRECTORY` reviews the selected primary entities and their linked
pronunciation/decoy candidates. The typed request names the map, revision, model,
explicit model arguments, batch size and concurrency. An optional `entry_ids` list records an explicit bounded follow-up audit scope.
Country/language settings
come from the source-checked map; the engine is reusable across locales. Fixed
versioned prompts distinguish actual local identities from foreign-name
transliterations and literal property/company calques, while preserving legitimate
industry borrowings and pharmaceutical identities. Reviewed donor forms remain
local; their task roles are checked and recorded independently.

The audit records keep/revise decisions without claiming human approval and emits
a normal `revise-database-localization` request for only the flagged entries and
all necessary dependent stimulus records. A clean audit writes evidence without
creating an empty revision request. Explicit source-review corrections can reuse
an exact prior audit with `prior_audit_file`, `required_revision_ids` and
`required_revision_reason`; these corrections have separate provenance and do not
impersonate model findings. Fixed country checks reject malformed Indian phone
shapes. License plates use local orthography, including Korean and Chinese plate letters, while VINs,
member-ID checksums, dates/times and medication strengths remain invariant.

`tau2 factory export-localization-entities-csv --map MAP.json --output DIRECTORY`
produces the short seven-column entity-only review form. It includes selected decoy
and pronunciation examples, with source-bound lineage in a separate sidecar, and
excludes simulator scripts and protocol instructions. That entity form is not an
approval of the full map: linked instructions, speech difficulty and actual TTS
behavior still need their separate review gates.

Audit records identify their exact parent map and source digest. A targeted
follow-up covers only the repaired groups, not a fresh full-map audit. A model's
`keep` decision is not human approval. Historical Hindi follow-up and repair
lineage are retained as regression fixtures under
`tests/test_multilingual/fixtures/intake_review/hi/`; they remain pending review
and must not be reused as evidence for newer map revisions.

## Plain pronunciation references

Hindi pronunciation references now equal their existing correct token in each
representation. They contain no stress capitalization, syllable respellings or
hyphenated phonetic guides. Native Hindi token/reference/distorted fields reject
Latin letters; actual distortions retain their exact native orthography. ASCII
phone, email and code protocols are unaffected. Romanized metadata stays separate
and its correct reference likewise uses the plain romanized token.

`tau2 factory repair-intake-pronunciation-references --map INPUT.json --revision
REVISION --output NEW.json --record REPAIR.json` performs the deterministic,
source-checked reference repair without an LLM call. The emitted repair record binds the previous and resulting map digests
and lists the exact reference-only changes. Earlier model audit records remain
bound to their original map; this mechanical repair does not claim a new model or
native approval. All 69 tokens and actual distortions are unchanged in both forms.
The entity CSV presents correct native tokens and actual native errors; redundant
correct-reference rows, including source stress guides, are omitted.

### Local date display and review difficulty

Compilation separates caller presentation from the agent submission contract.
`intake-local-date-display-v1` formats caller lookups and relative-date anchors in
local order while retaining exact ISO record/golden values. It also projects date
decoys, written-clarification restarts, wrong-slot dates and existing omit-year
complications; it does not alter the 200 frozen task assignments. Hindi uses
DD-MM-YYYY; Spanish and Brazilian Portuguese use DD/MM/YYYY; Mandarin and Korean
use year-month-day in their ordinary written scripts (numeric YYYY-MM-DD in romanized metadata). Existing tool documentation
already states YYYY-MM-DD, so no extra agent prompt injection is added.

The entity CSV uses the same date projection and records it in row lineage. Its
Difficulty column describes measured Easy/Hard assignments; Context only labels
incidental values instead of implying they are independently scored. Identical
banking reference pairs are consolidated with every source binding retained in
manifest reference_occurrences.

### Complete Hard-difficulty review

The locality review above does not establish capture difficulty. The separate
difficulty pipeline enumerates all 100 Hard slots in each of the five locales,
including dates, times, identifiers and coined names (company, employer and
aftermarket-brand names). Each case carries all ten same-family Easy controls,
its caller lookup, complete known answer, selected decoys and speech dependencies.
Reused callback identities and unrelated narrative context stay out of model
payloads. Each artifact pins the map, source, display versions and evidence hashes.

```bash
tau2 factory inventory-intake-difficulty --map intake.json --output inventory.json
tau2 factory audit-intake-difficulty --request audit-request.json \
  --output audit.json --checkpoints audit-checkpoints
tau2 factory plan-intake-difficulty --map intake.json --audit audit.json \
  --output plan.json
tau2 factory revise-intake-difficulty --request revision-request.json \
  --output revised-intake.json --checkpoints revision-checkpoints
tau2 factory report-intake-difficulty \
  --plan hi-plan.json --plan es-plan.json --plan pt-plan.json \
  --plan zh-plan.json --plan ko-plan.json --output difficulty-summary.json
```

`DifficultyAuditRequest` records `map_file`, `revision`, `model`, `model_args`,
`batch_size` and `max_concurrency`. `DifficultyRevisionRequest` adds `audit_file`.
Requests are JSON generated from those models. Authoring uses fixed versioned
prompts through `generate()`. The plan selects exact entries, closes changed
primary dependencies and leaves Easy controls and reused identities unchanged.
An omission-only finding does not require regenerating a good primary name.
Immutable or shared-scope conflicts remain explicit blockers in the plan.

Hard time repairs use typed `intake_caller_surfaces` instead of changing canonical
time values. Each authored initial phrase is bound to one task and its exact
H:MM AM/PM answer. Compilation appends the complete local time privately; the
caller initially uses the bare phrase and supplies the known day period when
asked. Relative-date repairs likewise retain their exact canonical-date anchor
and the pinned source clock. The original relative wording is not immutable.

Entity and time-expression revisions are separate validated stages. Their merge
checks that neither stage changed the other's fields, unselected entries, source
or settings, and records both input digests. Re-audit the resulting exact map
with a new audit request and output path before treating any finding as resolved.
The initial audit remains evidence of its original inputs. A temporal source
blocker may receive a conservative repair attempt, but authoring alone cannot
clear that blocker or establish Hard difficulty.

If an entity stage fails compilation, retain its typed failed-draft evidence and
repair the exact offending entries through `revise-database-localization` with
`previous_kind="failed"`. `merge-intake-difficulty --request merge-request.json
--output revised-intake.json --record merge.json` accepts validated stage outputs
through `DifficultyMergeRequest`. It rechecks the original audit plan's allowed
scope before joining a repaired entity-stage descendant with the caller surfaces.
It does not replace the failed evidence or relax the final audit requirement.

Plans export a separate CSV ledger containing every Hard task, hazards, Easy
comparisons, repair scope, model disposition and blockers. The ordinary entity
CSVs retain their seven review columns with blank human decisions. Neither a
compiled candidate nor a model `keep` decision grants native approval. The
machine gate stays closed for any unresolved revision, source blocker or audit
error; naturalness, pronunciation, local conventions and actual relative
difficulty still require native-speaker and listening review.

Time-authoring v2 records `feasible` or `source_limited` in its typed generation
checkpoint. When a fixed answer cannot naturally be harder than the Easy controls,
the generator must use ordinary clock speech and retain the limitation instead
of inventing a long countdown. Supplemental review feedback must name the exact
task and map digest; it does not masquerade as an independent audit decision.
Resuming a completed time stage requires its exact validated checkpoints and a
reconstructed map identical to the saved output. Missing evidence or changed
speech/metadata fails locally without issuing another provider call.

Read model findings against the actual complication sampler. A call samples one
complication kind, and the spelling-style branch selects one nonempty style from
the array. Those entries are alternatives, not simultaneous instructions. Likewise,
lazy omission applies only when selected; a complete default caller value does not
mean the caller has lost day-period knowledge. Preserve raw model findings and
record any source-grounded disagreement separately. Clinical decoy identities,
strengths and formulations cannot be replaced merely to make a judge label a case
Hard. These limitations keep the readiness gate closed pending review.

### Independent Easy review

Easy controls in a Hard comparison are not independent Easy approvals. Audit each
of the 100 selected Easy cases per language with the separate factory stage:

```bash
tau2 factory audit-intake-easy --request easy-request.json \
  --output easy-audit.json --checkpoints easy-checkpoints
tau2 factory inspect-intake-easy --request easy-request.json \
  --output easy-inspection.json --checkpoints easy-checkpoints
tau2 factory plan-intake-easy --map intake.json --audit easy-audit.json \
  --output easy-plan.json
tau2 factory revise-intake-easy --request easy-revision-request.json \
  --output revised-intake.json --checkpoints easy-revision-checkpoints
tau2 factory report-intake-easy \
  --report hi-easy-audit.json --report es-easy-audit.json \
  --report pt-easy-audit.json --report zh-easy-audit.json \
  --report ko-easy-audit.json --output easy-summary.json
```

`EasyAuditRequest` records the same explicit map, revision, model and batching
settings as the Hard audit. `EasyRevisionRequest` also requires the exact
`audit_file`. The audit emits one decision or explicit error for every Easy task,
a separate CSV ledger and a typed manifest. The five-language report requires
all 500 unique Easy cases from the same source frame. Neither report approves
content for native use.

The Easy prompt checks local plausibility, straightforward capture, known-answer
recoverability and every linked speech/decoy dependency. It distinguishes
ordinary consumer number reading from specialist code words, calendar years from
quantities, and hour/minute and postal-code conventions. Complication sampling
does not disable always-active lookup guidance: a style can conflict with the
initial bare-time instruction even when the complete time is privately known.
In this outbound domain the agent is the caller and the user simulator is the
callee; references to asking or correcting "the caller" are therefore intentional.

The Easy repair plan selects only audited Easy entries and closes primary changes
over their dependent content. Shared owners, protected Hard entries and immutable
source constraints remain explicit blockers. Repairs preserve canonical answers,
clinical identities and doses, source-relative phone errors, and existing Hard
caller surfaces. Use fixed versioned generation, re-audit the resulting map, and
refresh Hard audits when their Easy comparisons change. Keep previous review
exports intact and generate new entity CSVs with blank human feedback fields.

Source review can identify a concrete defect missed by a model `keep`.
`EasyRevisionRequest.review_feedback` accepts typed `EasyReviewFeedback` records
bound to the exact map and Easy task, with precise repair targets and factual
constraints. The planner also accepts their JSON array via `--feedback`. These
records remain separate from the unchanged audit verdicts and pass the same
scope and source-invariant checks; they do not create native approval or override
a source blocker to permit a different clinical or protocol identity.
