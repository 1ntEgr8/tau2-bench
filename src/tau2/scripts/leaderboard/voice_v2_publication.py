"""Prepare and verify Voice V2 packages entirely from saved results and verdicts."""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

from tau2.data_model.benchmark import (
    BenchmarkFrame,
    BenchmarkSubmission,
    TaskTrial,
    TrialStatus,
)
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import SimulationRun
from tau2.metrics.benchmark import (
    summarize,
)
from tau2.metrics.benchmark import (
    validate_submission as validate_benchmark_submission,
)
from tau2.runner.presets import load_frame
from tau2.scripts.leaderboard.submission import Submission
from tau2.scripts.leaderboard.voice_v2_submission import (
    DOMAINS,
    EXPERIENCE_VERSION,
    INTERACTION_VERSION,
    LANGUAGES,
    CallOutcome,
    InteractionFacts,
    VoiceV2Benchmark,
    VoiceV2Cell,
    VoiceV2Submission,
    coverage,
    experience_outcome,
    file_sha256,
    interaction_outcome,
)


class InteractionFactsRow(BaseModel):
    """One row of the saved interaction-event fact table."""

    provider: Annotated[str, Field(description="Provider arm")]
    language: Annotated[str, Field(description="Language code")]
    domain: Annotated[str, Field(description="Domain name")]
    simulation_id: Annotated[str, Field(description="Scored simulation ID")]
    had_unresponsive_period: Annotated[
        bool, Field(description="Saved unresponsive-period detector result")
    ]
    nonresponse_failure: Annotated[
        int | None, Field(description="Nonresponse failure flag")
    ]
    interruption_count: Annotated[
        int, Field(ge=0, description="Detected agent interruption events")
    ]
    selectivity_error_count: Annotated[
        int, Field(ge=0, description="Total selectivity errors")
    ]
    selectivity_failure: Annotated[
        int | None, Field(description="Saved selectivity availability and flag")
    ]
    monologue_failure: Annotated[int | None, Field(description="Monologue flag")]
    tool_misuse_failure: Annotated[int | None, Field(description="Tool-use flag")]

    @model_validator(mode="after")
    def _check_flags(self) -> InteractionFactsRow:
        for value in (
            self.nonresponse_failure,
            self.selectivity_failure,
            self.monologue_failure,
            self.tool_misuse_failure,
        ):
            if value not in (None, 0, 1):
                raise ValueError("failure flags must be 0, 1, or missing")
        return self

    def facts(self) -> InteractionFacts:
        """Convert saved detector fields into versioned Interaction inputs."""
        return InteractionFacts(
            nonresponse=_flag(self.nonresponse_failure),
            interruption_count=self.interruption_count,
            selectivity_error_count=self.selectivity_error_count,
            selectivity_available=self.selectivity_failure is not None,
            monologue=_flag(self.monologue_failure),
            tool_misuse=_flag(self.tool_misuse_failure),
            korean_unresponsive_period=(
                self.language == "ko" and self.had_unresponsive_period
            ),
        )


def _flag(value: int | None) -> bool | None:
    return bool(value) if value is not None else None


class SourceFile(BaseModel):
    """Hash of an immutable package input relative to the package root."""

    path: Annotated[str, Field(description="Relative package path")]
    sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Content SHA-256")
    ]


class VoiceV2Provenance(BaseModel):
    """Complete source inventory needed to reproduce the V2 summary."""

    format: Annotated[
        Literal["voice-v2-provenance-v1"], Field(description="Manifest version")
    ]
    frame_revision: Annotated[str, Field(description="Recorded benchmark frame")]
    provider: Annotated[str, Field(description="Provider arm")]
    interaction_version: Annotated[
        Literal["voice-v2-interaction-call-v1"],
        Field(description="Interaction derivation version"),
    ]
    experience_version: Annotated[
        Literal["voice-v2-experience-call-v1"],
        Field(description="Experience derivation version"),
    ]
    interaction_facts_derivation: Annotated[
        Literal[
            "extract_interaction_quality_metrics plus CHECKER_REGISTRY for "
            "nonresponse, interruption, selectivity and monologue; saved tool_use "
            "judge factors for tool misuse"
        ],
        Field(description="Origin of the saved per-call event facts"),
    ]
    files: Annotated[
        list[SourceFile], Field(description="All trajectory, judge, and fact inputs")
    ]

    @model_validator(mode="after")
    def _unique_files(self) -> VoiceV2Provenance:
        paths = [entry.path for entry in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate provenance file paths")
        return self


def _safe_path(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"path escapes submission root: {relative}")
    return path


def _load_facts(
    path: Path, provider: str
) -> dict[tuple[str, str, str], InteractionFacts]:
    """Load the saved event facts, rejecting duplicates and wrong provider arms."""
    rows: dict[tuple[str, str, str], InteractionFacts] = {}
    with path.open(newline="") as stream:
        for raw in csv.DictReader(stream):
            row = InteractionFactsRow.model_validate(
                {key: value if value != "" else None for key, value in raw.items()}
            )
            if row.provider != provider:
                continue
            key = (row.language, row.domain, row.simulation_id)
            if key in rows:
                raise ValueError(f"duplicate interaction fact: {key}")
            rows[key] = row.facts()
    return rows


def _load_judgments(path: Path, simulation_id: str) -> dict[str, dict]:
    """Index recorded judge measures and reject duplicate or foreign verdicts."""
    measures: dict[str, dict] = {}
    for result in json.loads(path.read_text()):
        identity = result["identity"]
        if identity["input"]["simulation_id"] != simulation_id:
            raise ValueError(f"verdict simulation ID mismatch: {path}")
        name = identity["measure"]
        if name in measures:
            raise ValueError(f"duplicate judge measure {name}: {path}")
        measures[name] = result
    return measures


def _experience_for_call(simulation_path: Path, verdict_path: Path) -> CallOutcome:
    """Align the two judges by the saved agent-turn/tick-span mapping.

    Naturalness unit IDs follow segment IDs or turn indexes. Speech-fidelity
    unit IDs follow segment IDs or tick-span indexes. They cannot be joined by
    ordinal position; some calls have excluded control utterances in only one
    measure.
    """
    simulation_id = simulation_path.stem
    measures = _load_judgments(verdict_path, simulation_id)
    naturalness = measures.get("naturalness")
    fidelity = measures.get("speech_fidelity")
    if not naturalness or not fidelity:
        return CallOutcome.UNSCORED
    if naturalness.get("error") or fidelity.get("error"):
        return CallOutcome.UNSCORED
    nat_verdict = naturalness.get("verdict") or {}
    fid_verdict = fidelity.get("verdict") or {}
    if nat_verdict.get("unit_errors") or fid_verdict.get("unit_errors"):
        return CallOutcome.UNSCORED
    nat = {
        unit["unit_id"]: unit
        for unit in nat_verdict.get("units", [])
        if unit["opportunity"]
    }
    fid = {
        unit["unit_id"]: unit
        for unit in fid_verdict.get("units", [])
        if unit["opportunity"]
    }
    simulation = SimulationRun.model_validate_json(simulation_path.read_text())
    if nat and all(unit_id.startswith("live-segment:") for unit_id in nat):
        paired = {unit_id: unit_id for unit_id in nat}
    else:
        paired = _legacy_utterance_alignment(simulation)
    paired_units = [
        (nat[n], fid[f]) for n, f in paired.items() if n in nat and f in fid
    ]
    return experience_outcome(
        len(nat),
        [unit["violation"] for unit, _ in paired_units],
        [unit["violation"] for _, unit in paired_units],
    )


def _legacy_utterance_alignment(simulation: SimulationRun) -> dict[str, str]:
    """Reconstruct the frozen judge's non-Live turn-to-audio-span indexes.

    The frozen scorer numbered every non-tool agent audio group, including
    silent chunks, for fidelity. Naturalness numbered only speech-bearing
    groups with text. Intersecting utterance IDs join consecutive chunks.
    """
    groups: list[list] = []
    current_ids: set[str] = set()
    for tick in simulation.ticks or []:
        chunk = tick.agent_chunk
        if chunk is None or chunk.is_tool_call():
            continue
        ids = set(chunk.utterance_ids or [])
        if groups and current_ids and ids and not current_ids.isdisjoint(ids):
            groups[-1].append(chunk)
            current_ids.update(ids)
        else:
            groups.append([chunk])
            current_ids = ids
    paired: dict[str, str] = {}
    turn_index = 0
    for span_index, chunks in enumerate(groups):
        merged = (
            chunks[0] if len(chunks) == 1 else AssistantMessage.merge_chunks(chunks)
        )
        if merged.contains_speech is False or not (
            merged.audio_script_gold or merged.content
        ):
            continue
        paired[str(turn_index)] = str(span_index)
        turn_index += 1
    return paired


def _load_cell(root: Path, language: str, domain: str) -> tuple[dict, list[dict]]:
    results_path = root / "trajectories" / language / domain / "results.json"
    data = json.loads(results_path.read_text())
    return data, data["simulation_index"]


def _validate_simulation_files(
    root: Path, simulations: list[dict], language: str, domain: str
) -> None:
    """Check the directory payloads against the canonical result index."""
    sim_dir = root / "trajectories" / language / domain / "simulations"
    indexed_ids = {entry["id"] for entry in simulations}
    file_ids = {path.stem for path in sim_dir.glob("*.json")}
    if indexed_ids != file_ids:
        raise ValueError(f"{language}/{domain}: simulation files differ from index")
    for entry in simulations:
        payload = json.loads((sim_dir / f"{entry['id']}.json").read_text())
        if any(
            payload.get(field) != entry[field]
            for field in ("id", "task_id", "trial", "termination_reason")
        ):
            raise ValueError(
                f"{language}/{domain}: simulation/index identity mismatch: {entry['id']}"
            )
        reward_info = payload.get("reward_info") or {}
        if reward_info.get("reward") != entry["reward"]:
            raise ValueError(
                f"{language}/{domain}: simulation/index reward mismatch: {entry['id']}"
            )


def _validate_cell_frame(
    data: dict,
    simulations: list[dict],
    language: str,
    domain: str,
    official: BenchmarkFrame,
) -> dict[str, str]:
    """Adapt the 2026-10-02 release metadata without inventing current-run fields."""
    info = data["info"]
    frame = info.get("benchmark") or {}
    cell = next(
        item
        for item in official.cells
        if (item.language, item.domain) == (language, domain)
    )
    if frame.get("benchmark_version") != official.benchmark_version:
        raise ValueError(f"{language}/{domain}: trajectory is not voice-v2")
    if frame.get("frame_revision") != official.revision:
        raise ValueError(f"{language}/{domain}: frame revision mismatch")
    if frame.get("binding_revision") != cell.binding_revision:
        raise ValueError(f"{language}/{domain}: frozen binding mismatch")
    if frame.get("canonical_task_ids") != cell.canonical_task_ids:
        raise ValueError(f"{language}/{domain}: canonical task frame mismatch")
    if frame.get("language") != language or frame.get("domain") != domain:
        raise ValueError(f"{language}/{domain}: benchmark frame identity mismatch")
    if info.get("language") != language:
        raise ValueError(f"{language}/{domain}: recorded language mismatch")
    if info["environment_info"]["domain_name"] != domain:
        raise ValueError(f"{language}/{domain}: recorded domain mismatch")
    selected = set(frame["selected_task_ids"])
    language_tasks = info["language_tasks"]
    mapped = {
        str(task_id): details["canonical_task_id"]
        for task_id, details in language_tasks.items()
    }
    task_ids = [mapped[str(task["id"])] for task in data["tasks"]]
    if (
        frame["selected_task_ids"] != cell.canonical_task_ids
        or len(task_ids) != len(cell.canonical_task_ids)
        or len(set(task_ids)) != len(task_ids)
        or set(task_ids) != selected
    ):
        raise ValueError(f"{language}/{domain}: selected 50-task frame mismatch")
    if frame["trials"] != official.trials or info["num_trials"] != 1:
        raise ValueError(f"{language}/{domain}: V2 publication requires one trial")
    expected = {(task_id, 0) for task_id in selected}
    actual = [(mapped[str(sim["task_id"])], sim["trial"]) for sim in simulations]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"{language}/{domain}: missing or duplicate task/trial keys")
    if any(
        sim["reward"] is None and sim["termination_reason"] != "infrastructure_error"
        for sim in simulations
    ):
        raise ValueError(
            f"{language}/{domain}: unscored call lacks infrastructure status"
        )
    return mapped


def _source_files(root: Path, provider: str) -> list[SourceFile]:
    paths = [root / "interaction_facts.csv"]
    for name in ("interaction_facts_upstream.json", "interaction_facts_method.md"):
        if (root / name).is_file():
            paths.append(root / name)
    if (root / "source_correction.json").is_file():
        paths.append(root / "source_correction.json")
    for language in LANGUAGES:
        for domain in DOMAINS:
            base = root / "trajectories" / language / domain
            paths.append(base / "results.json")
            paths.extend(sorted((base / "simulations").glob("*.json")))
        paths.extend(sorted((root / "judges" / provider / language).glob("*.json")))
    return [
        SourceFile(path=str(path.relative_to(root)), sha256=file_sha256(path))
        for path in sorted(set(paths))
    ]


def _validate_source_correction(root: Path) -> None:
    path = root / "source_correction.json"
    if not path.is_file():
        return
    correction = json.loads(path.read_text())
    if correction.get("format") != "voice-v2-gemini-retry-correction-v1":
        raise ValueError("unknown source correction format")
    for item in correction["corrections"]:
        cell = root / "trajectories" / item["language"] / item["domain"]
        simulation = cell / "simulations" / f"{item['new_index']['id']}.json"
        if file_sha256(cell / "results.json") != item["corrected_results_sha256"]:
            raise ValueError("corrected result differs from source correction")
        if file_sha256(simulation) != item["corrected_simulation_sha256"]:
            raise ValueError("corrected simulation differs from source correction")


def build_voice_v2_submission(
    package_root: Path,
    provider: str,
    metadata: Submission,
) -> tuple[VoiceV2Submission, VoiceV2Provenance]:
    """Derive V2 scores and coverage from a self-contained offline package."""
    _validate_source_correction(package_root)
    upstream_path = package_root / "interaction_facts_upstream.json"
    if upstream_path.is_file():
        upstream = json.loads(upstream_path.read_text())
        source_hash = file_sha256(package_root / "interaction_facts.csv")
        if source_hash not in (upstream.get("output_sha256") or {}).values():
            raise ValueError("interaction fact table has no upstream provenance match")
    facts = _load_facts(package_root / "interaction_facts.csv", provider)
    official = load_frame("voice", num_trials=1)
    used_facts: set[tuple[str, str, str]] = set()
    results: dict[str, dict[str, VoiceV2Cell]] = defaultdict(dict)
    trajectory_files: dict[str, dict[str, str]] = defaultdict(dict)
    trials: list[TaskTrial] = []
    sources: dict[str, str] = {}
    expected_judges: dict[str, set[str]] = defaultdict(set)
    agent_models: set[str] = set()
    user_models: set[str] = set()
    for language in LANGUAGES:
        for domain in DOMAINS:
            data, simulations = _load_cell(package_root, language, domain)
            info = data["info"]
            agent_models.add(info["agent_info"]["llm"])
            user_models.add(info["user_info"]["llm"])
            audio_config = info["benchmark"]["conditions"]["audio_native_config"]
            if (
                audio_config["provider"]
                != {
                    "gemini": "gemini",
                    "gpt_live": "openai_live",
                    "grok": "xai",
                }[provider]
            ):
                raise ValueError(f"{language}/{domain}: provider arm mismatch")
            if audio_config["model"] != metadata.model_name:
                raise ValueError(f"{language}/{domain}: model metadata mismatch")
            task_mapping = _validate_cell_frame(
                data, simulations, language, domain, official
            )
            benchmark = info["benchmark"]
            sources[f"{language}/{domain}"] = file_sha256(
                package_root / "trajectories" / language / domain / "results.json"
            )
            trials.extend(
                TaskTrial(
                    language=language,
                    domain=domain,
                    canonical_task_id=task_mapping[str(sim["task_id"])],
                    trial=sim["trial"],
                    condition_fingerprint=benchmark["condition_fingerprint"],
                    binding_revision=benchmark["binding_revision"],
                    status=(
                        TrialStatus.SCORED
                        if sim["reward"] is not None
                        else TrialStatus.ERROR
                    ),
                    reward=sim["reward"],
                )
                for sim in simulations
            )
            _validate_simulation_files(package_root, simulations, language, domain)
            scored = [sim for sim in simulations if sim["reward"] is not None]
            passed = sum(sim["reward"] == 1 for sim in scored)
            interaction: list[CallOutcome] = []
            experience: list[CallOutcome] = []
            for sim in scored:
                key = (language, domain, sim["id"])
                if key not in facts:
                    raise ValueError(f"missing saved interaction facts: {key}")
                used_facts.add(key)
                expected_judges[language].add(sim["id"])
                interaction.append(interaction_outcome(facts[key]))
                experience.append(
                    _experience_for_call(
                        package_root
                        / "trajectories"
                        / language
                        / domain
                        / "simulations"
                        / f"{sim['id']}.json",
                        package_root
                        / "judges"
                        / provider
                        / language
                        / f"{sim['id']}.json",
                    )
                )
            results[language][domain] = VoiceV2Cell(
                selected=len(simulations),
                scored=len(scored),
                **{
                    "pass": passed,
                    "fail": len(scored) - passed,
                },
                infrastructure=len(simulations) - len(scored),
                pass_1=100 * passed / len(scored) if scored else None,
                interaction=coverage(interaction, INTERACTION_VERSION),
                experience=coverage(experience, EXPERIENCE_VERSION),
            )
            trajectory_files[language][domain] = f"{language}/{domain}"
    canonical = BenchmarkSubmission(
        frame=official,
        trials=trials,
        summary=summarize(official, trials),
        sources=sources,
    )
    validate_benchmark_submission(canonical)
    if len(agent_models) != 1 or len(user_models) != 1:
        raise ValueError("Voice V2 submission mixes agent or caller models")
    if set(facts) != used_facts:
        raise ValueError("interaction fact table has extra or unscored calls")
    for language in LANGUAGES:
        actual_judges = {
            path.stem
            for path in (package_root / "judges" / provider / language).glob("*.json")
        }
        if actual_judges != expected_judges[language]:
            raise ValueError(f"{language}: judge IDs differ from scored calls")
    provenance = VoiceV2Provenance(
        format="voice-v2-provenance-v1",
        frame_revision=official.revision,
        provider=provider,
        interaction_version=INTERACTION_VERSION,
        experience_version=EXPERIENCE_VERSION,
        interaction_facts_derivation=(
            "extract_interaction_quality_metrics plus CHECKER_REGISTRY for "
            "nonresponse, interruption, selectivity and monologue; saved tool_use "
            "judge factors for tool misuse"
        ),
        files=_source_files(package_root, provider),
    )
    provenance_bytes = (
        provenance.model_dump_json(indent=2, exclude_none=True) + "\n"
    ).encode()
    submission = VoiceV2Submission.model_validate(
        {
            **metadata.model_dump(exclude={"results", "trajectory_files"}),
            "format": "voice-v2",
            "trajectories_available": True,
            "benchmark": VoiceV2Benchmark(
                version="voice-v2",
                frame_revision=official.revision,
                languages=list(LANGUAGES),
                domains=list(DOMAINS),
                task_trials_per_cell=50,
            ).model_dump(),
            "benchmark_submission": canonical.model_dump(mode="json"),
            "results": {
                language: {
                    domain: cell.model_dump(by_alias=True)
                    for domain, cell in domains.items()
                }
                for language, domains in results.items()
            },
            "trajectory_files": dict(trajectory_files),
            "provenance_file": "provenance.json",
            "provenance_sha256": sha256(provenance_bytes).hexdigest(),
        }
    )
    return submission, provenance


def validate_voice_v2_package(root: Path) -> VoiceV2Submission:
    """Rehash every source and recompute every published V2 score offline."""
    submission = VoiceV2Submission.model_validate_json(
        (root / "submission.json").read_text()
    )
    provenance_path = _safe_path(root, submission.provenance_file)
    if file_sha256(provenance_path) != submission.provenance_sha256:
        raise ValueError("provenance manifest SHA-256 mismatch")
    provenance = VoiceV2Provenance.model_validate_json(provenance_path.read_text())
    if provenance.frame_revision != submission.benchmark.frame_revision:
        raise ValueError("provenance frame revision mismatch")
    for source in provenance.files:
        path = _safe_path(root, source.path)
        if not path.is_file() or file_sha256(path) != source.sha256:
            raise ValueError(f"source SHA-256 mismatch: {source.path}")
    derived, expected_provenance = build_voice_v2_submission(
        root, provenance.provider, submission
    )
    if expected_provenance != provenance:
        raise ValueError("source inventory mismatch")
    if derived != submission:
        raise ValueError("published Voice V2 fields do not match saved inputs")
    return submission


def prepare_voice_v2_package(
    source_root: Path,
    destination_root: Path,
    provider: str,
    interaction_facts: Path,
    judges_root: Path,
    metadata_file: Path,
) -> VoiceV2Submission:
    """Copy saved inputs, derive V2 results, and write a verified package.

    ``source_root`` is a pre-exported, 18-cell voice trajectory package with
    metadata in submission.json. No simulation or judge is executed here.
    """
    source_path = source_root.resolve()
    destination_path = destination_root.resolve()
    if source_path == destination_path or destination_path.is_relative_to(source_path):
        raise ValueError("destination must be outside the source package")
    if destination_root.exists():
        raise FileExistsError(f"destination already exists: {destination_root}")
    source_metadata = json.loads((source_root / "submission.json").read_text())
    supplied_metadata = json.loads(metadata_file.read_text())
    metadata = {**source_metadata, **supplied_metadata}
    missing = [
        field
        for field in ("submission_date", "contact_info")
        if not metadata.get(field)
    ]
    if missing:
        raise ValueError(
            "Voice V2 publication metadata requires: " + ", ".join(missing)
        )
    if not isinstance(metadata["contact_info"], dict) or not metadata[
        "contact_info"
    ].get("email"):
        raise ValueError("Voice V2 publication metadata requires contact_info.email")
    upstream_path = interaction_facts.parent / "provenance.json"
    if upstream_path.is_file():
        upstream = json.loads(upstream_path.read_text())
        recorded_hash = (upstream.get("output_sha256") or {}).get(
            interaction_facts.name
        )
        if recorded_hash != file_sha256(interaction_facts):
            raise ValueError("interaction fact table differs from upstream provenance")
    draft = Submission.model_validate(
        {
            **{key: value for key, value in metadata.items() if key != "format"},
            "results": {},
            "trajectory_files": {},
        }
    )
    if draft.modality != "voice":
        raise ValueError("Voice V2 requires voice modality")
    destination_root.mkdir(parents=True)
    try:
        for language in LANGUAGES:
            for domain in DOMAINS:
                shutil.copytree(
                    source_root / "trajectories" / language / domain,
                    destination_root / "trajectories" / language / domain,
                )
        if (source_root / "correction.json").is_file():
            shutil.copy2(
                source_root / "correction.json",
                destination_root / "source_correction.json",
            )
        shutil.copy2(interaction_facts, destination_root / "interaction_facts.csv")
        for source_name, destination_name in (
            ("provenance.json", "interaction_facts_upstream.json"),
            ("README.md", "interaction_facts_method.md"),
        ):
            source = interaction_facts.parent / source_name
            if source.is_file():
                shutil.copy2(source, destination_root / destination_name)
        facts = _load_facts(destination_root / "interaction_facts.csv", provider)
        for language, _, simulation_id in facts:
            source = judges_root / provider / language / f"{simulation_id}.json"
            target = (
                destination_root
                / "judges"
                / provider
                / language
                / f"{simulation_id}.json"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        submission, provenance = build_voice_v2_submission(
            destination_root, provider, draft
        )
        (destination_root / "provenance.json").write_text(
            provenance.model_dump_json(indent=2, exclude_none=True) + "\n"
        )
        (destination_root / "submission.json").write_text(
            submission.model_dump_json(indent=2, by_alias=True) + "\n"
        )
        validate_voice_v2_package(destination_root)
    except Exception:
        shutil.rmtree(destination_root)
        raise
    return submission
