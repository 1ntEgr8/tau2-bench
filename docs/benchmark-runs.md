# Running and submitting released benchmarks

A released benchmark is a manifest under `data/tau2/benchmarks/`. Each manifest
declares one call mode (`voice` or `text`); otherwise both share one schema,
resolver, controller, viewer and submission path. Two ship today:

| Manifest | Mode | Selected by |
|---|---|---|
| `voice.json` | voice | `tau2 run --audio-native` (default `--benchmark voice`) |
| `text.json` | text | `tau2 run --benchmark text` (text runs default to `--benchmark none`) |

Both cover the same 18 cells and the same frozen task subsets. Their binding
revisions differ because the call mode is part of each binding: text and voice
calls resolve different persona and scaffold content from the same pack.

`tau2 run --audio-native` runs the current released voice benchmark. Its manifest,
`data/tau2/benchmarks/voice.json`, pins six languages (English, Spanish, Portuguese,
Hindi, Chinese and Korean), three domains and the frozen `airline_50`, `retail_50`
and `telecom_50` task subsets. That is **900 calls at one trial**. English uses
its language pack, just like every other language.

Supported languages and released benchmark membership are separate. Installing
or contributing another pack makes it available for custom runs, without changing
the default matrix or its overall score. Including another language in the official
score requires an explicitly published benchmark revision and results on the
expanded population. Results retain their benchmark generation, manifest revision,
exact task/pack hashes, effective model settings and code commit.

## Run the released benchmark

The examples use GPT-Realtime-2 with minimal reasoning for the agent. Select both
explicitly with `--audio-native-model gpt-realtime-2 --reasoning-effort minimal`.
The caller remains GPT-5.5 with low reasoning.

```bash
# Preview all 900 calls, without provider requests or output writes.
tau2 run --audio-native --audio-native-model gpt-realtime-2 --reasoning-effort minimal --save-to voice-run --dry-run
# Remove --dry-run to execute.

# A domain across all released languages: 300 calls.
tau2 run --audio-native --domain retail --audio-native-model gpt-realtime-2 --reasoning-effort minimal --save-to retail-run

# English across all released domains: 150 calls.
tau2 run --audio-native --language en --audio-native-model gpt-realtime-2 --reasoning-effort minimal --save-to english-run

# Two frozen retail tasks per language: 12 calls.
tau2 run --audio-native --domain retail --num-tasks 2 --audio-native-model gpt-realtime-2 --reasoning-effort minimal --save-to sample-run
```

Repeated `--language` and `--domain` flags select a slice of the released matrix.
`--num-tasks N` takes N tasks in frozen-subset order in each selected domain; every
language receives the same canonical tasks. This does not take a prefix of the
original domain task file. `--task-ids` selects explicit **canonical** IDs and
requires exactly one domain. Choose either a task limit or explicit IDs.
Alternative task sets/splits and unreleased languages/domains require custom mode.

Omitted settings resolve through `config.py`: GPT-5.5 with low reasoning for the
user simulator, 40 simulated minutes, Marin for OpenAI Realtime, and a separately
pinned semantic communication evaluator. Explicit settings take precedence.
Other providers retain their model and voice configuration. For example:

```bash
tau2 run --audio-native --domain retail --language en --num-tasks 2 \
  --user-llm openai/gpt-5.5 --user-llm-args '{"reasoning_effort":"low"}' \
  --voice cedar --max-steps-seconds 120 --audio-native-model gpt-realtime-2 --reasoning-effort minimal --dry-run
```

The command prints languages, domains, call count, trials and worker allocation
before execution. One existing controller schedules the complete matrix.
Omitting `--workers` (the CLI default) selects **one worker** for a released
benchmark, and `--workers 0` is overridden to one worker with a warning; custom
runs without `--workers` (or with `--workers 0`) retain in-process execution. With `--workers 3
--max-concurrency 10`, up to 30 calls share the worker fleet. `--global-limit` caps
the fleet, and `--provider-limit openai=20` applies the existing provider cap.

`--save-to voice-run` writes under `data/simulations/voice-run`; an absolute path
selects another root. `--output` is another spelling of this existing path option.
The root records the full frame, cumulative selected tasks and execution settings;
ordinary results live under cells such as `en-retail/`.

Repeating a command resumes completed calls automatically. Increasing the task
limit, adding task IDs or adding selected cells extends the same saved benchmark.
The recorded execution selection becomes their union. Changed release data or
model/evaluation conditions require a new output root. Trial count stays fixed
within a saved matrix; use a new root for a different count.

## Run the released text benchmark

```bash
# Preview all 900 text conversations.
tau2 run --benchmark text --agent-llm openai/gpt-5.5 \
  --agent-llm-args '{"reasoning_effort":"low"}' --save-to text-run --dry-run
# One domain across all released languages: 300 conversations.
tau2 run --benchmark text --domain retail --agent-llm openai/gpt-5.5 \
  --agent-llm-args '{"reasoning_effort":"low"}' --save-to text-retail
```

Selection (`--language`, `--domain`, `--num-tasks`, `--task-ids`), caller defaults,
worker allocation, resume and submission behave exactly as for voice. A manifest
runs only in its own mode: `--benchmark text --audio-native` and `--benchmark voice`
without `--audio-native` are rejected. Without `--benchmark`, a text run is an
ordinary custom single-domain run.

## Regenerate manifest pins

Subset digests and binding revisions are derived, never authored. Regenerate them
from the current checkout after any pack, map, task or prompt change:

```bash
tau2 benchmarks list                  # installed manifests, modes and sizes
tau2 benchmarks pin voice             # verify voice.json pins (no change allowed)
tau2 benchmarks pin voice --revision regional-release-data-v17   # repin + relabel
tau2 benchmarks pin text --revision regional-release-data-v17
# Derive a new manifest with another's cells and subsets in a different mode:
tau2 benchmarks pin text --from voice --mode text \
  --revision regional-release-data-v17 --benchmark-version text-v1
```

`pin` is idempotent: rerunning it on current data leaves the manifest byte-identical.
A changed pin means released results on the old revision no longer validate, so
`pin` refuses to write changed pins under the manifest's existing revision: pass a
new `--revision` whenever any subset digest or binding revision changes.

## Custom runs and Python callers

```bash
# A shipped language on a domain or task count outside the frozen matrix:
tau2 run --audio-native --benchmark none --domain retail --language es --num-tasks 2
# A domain outside the released matrix:
tau2 run --audio-native --benchmark none --domain intake --num-tasks 2
```

Custom mode keeps ordinary single-domain task selection and defaults. Every run,
text or voice and English included, defaults to semantic communication scoring.
The resolved judge model, arguments and prompt version are saved with the run and
reused by `tau2 evaluate-trajs`. Override the judge with
`--communicate-judge '{"model":"...","model_args":{"reasoning_effort":"low"}}'`;
select `--communicate-scorer substring` only for custom substring-scoring
experiments. Python callers set `communicate_scoring=CommunicateScoring(...)` on
the run config.

Spoken names and airline identifiers tolerate case and accents; phone lookups
tolerate punctuation. Ambiguous lookups fail, verification fields such as ZIP codes
and birth dates remain exact, and mutations store canonical identifiers. The domain
tools revision is recorded in results and benchmark condition fingerprints. Start a
new output root for this revision; older results must not be resumed into it.
Saved voice results from main remain readable: the legacy `cascaded_config_name`
field is translated to a preset request without adding missing run provenance.
Rescoring language/benchmark results uses embedded tasks and rejects `--fresh-tasks`
to preserve their pinned definitions.

Provider capability mappings live independently of released benchmark membership in
`tau2.voice.audio_native.languages`. For example, an installed French pack can use
LiveKit Nova-3/ElevenLabs, Gemini or xAI without entering the official matrix. Regional
aliases are explicitly mapped; unsupported hints fail instead of silently selecting
English or automatic detection. Gemini's input-transcription hint follows the run
language; explicitly supplied provider hints (including an empty list for automatic
detection) take precedence.

Python callers supply explicit LiveKit pipelines with
`CascadedConfigRequest(config=...)`. The resolved `cascaded_config` records effective
settings; conflicting values are rejected before language defaults are applied.

Text runs without an explicit language keep their existing behavior. Unreleased selections never silently fall back
to English or change the official benchmark population.

Python benchmark callers use the same resolver as the CLI. Pass a `TextRunConfig`
template with `preset="text"`; the template's mode must match the manifest's.

```python
from pathlib import Path
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import VoiceRunConfig
from tau2.runner.presets import (
    BenchmarkRequest, ExecutionSettings, execute_plan, resolve_benchmark,
)

output = Path("voice-run")
plan = resolve_benchmark(BenchmarkRequest(
    output=output,
    template=VoiceRunConfig(audio_native_config=AudioNativeConfig(reasoning_effort="low")),
    domains=["retail"],
))
execute_plan(plan, output, ExecutionSettings(workers=1), dry_run=True)
```

`VoiceRunConfig` and `run_domain()` still describe one concrete language/domain
cell. Only the benchmark request expands into the matrix.

## Coverage and scores

```bash
tau2 view --file data/simulations/voice-run
```

The viewer shows the official score and coverage, language/domain cells, and the
recorded requested scope before individual calls. Task success averages trials
within a task, tasks within a cell, then cells equally. At one trial this is
pass@1. Language and domain breakdowns use the same equal-cell rule.

A scored zero is a failed task. Infrastructure errors, unevaluated trials and
missing trials remain separate counts. An incomplete matrix has no official
score; `observed_score` is explicitly partial. A completed sample or slice can
have a **requested-scope** score while the official benchmark remains incomplete.

Post-hoc diagnostic judges remain opt-in through `tau2 judges`. They accept the
matrix root, preserve each language's applicability, and write separate diagnostic
artifacts per cell. A transport revision invalidates cached verdicts produced before
audio forwarding and Responses routing were restored, including custom judge catalogs.
They do not alter task success or run during viewing,
submission preparation or validation.

## Submit saved results

```bash
tau2 submit prepare data/simulations/voice-run --output ./voice-submission
tau2 submit validate ./voice-submission --results data/simulations/voice-run
```

Commands recognize matrix provenance automatically. Preparation requires complete
coverage and writes `benchmark-submission.json`. Keep the original saved matrix
and pinned release data; the export does not copy audio or trajectories. Validation
reads those sources, checks settings and artifact provenance, and recomputes scores.
It rejects changed membership, incompatible conditions, duplicate trials, missing
evidence and summaries that disagree with their source results. Python readers use
`tau2.metrics.benchmark.load_submission(path)` to validate retained observations
and recompute both summaries; constructing `BenchmarkSubmission` validates its
format and membership only.

## Review prompts offline

```bash
tau2 languages render-prompts --benchmark voice --language en --domain retail --output ./prompt-review
```

Benchmark prompt review uses the same resolved defaults and frozen task population
as execution. It renders voice prompts, so it accepts voice manifests only. Optional `--config` accepts an ordinary `VoiceRunConfig` JSON template.
Omit `--benchmark` to review installed packs with ordinary standalone voice defaults.
