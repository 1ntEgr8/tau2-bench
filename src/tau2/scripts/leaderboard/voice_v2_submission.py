"""Versioned Voice V2 submission contract and offline score derivation."""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.data_model.benchmark import BenchmarkSubmission
from tau2.scripts.leaderboard.submission import ContactInfo, Submission

LANGUAGES = ("en", "es", "hi", "ko", "pt", "zh")
DOMAINS = ("airline", "retail", "telecom")
INTERACTION_VERSION = "voice-v2-interaction-call-v1"
EXPERIENCE_VERSION = "voice-v2-experience-call-v1"


class CallOutcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNSCORED = "unscored"
    EXCLUDED = "excluded"


class MeasureCoverage(BaseModel):
    """Coverage and outcome of one call-level measure."""

    version: Annotated[str, Field(description="Version of the derivation rule")]
    passed: Annotated[int, Field(ge=0, description="Scored calls that passed")]
    failed: Annotated[int, Field(ge=0, description="Scored calls that failed")]
    unscored: Annotated[
        int, Field(ge=0, description="Scored calls with insufficient inputs")
    ]
    excluded: Annotated[
        int, Field(ge=0, description="Scored calls excluded by a documented rule")
    ] = 0
    pass_rate: Annotated[
        float | None,
        Field(ge=0, le=100, description="100 × passed / (passed + failed)"),
    ]

    @model_validator(mode="after")
    def _check_rate(self) -> MeasureCoverage:
        denominator = self.passed + self.failed
        expected = 100 * self.passed / denominator if denominator else None
        if expected is None and self.pass_rate is not None:
            raise ValueError("pass_rate must be null without scored calls")
        if expected is not None and (
            self.pass_rate is None or abs(self.pass_rate - expected) > 0.0001
        ):
            raise ValueError("pass_rate does not match passed / determinate calls")
        return self


class VoiceV2Cell(BaseModel):
    """One language and domain, with infrastructure separated from task failure."""

    selected: Annotated[int, Field(ge=1, description="Selected task/trial keys")]
    scored: Annotated[int, Field(ge=0, description="Calls with a benchmark reward")]
    passed: Annotated[int, Field(ge=0, alias="pass", description="Reward-one calls")]
    failed: Annotated[int, Field(ge=0, alias="fail", description="Scored nonpasses")]
    infrastructure: Annotated[
        int, Field(ge=0, description="Calls without a benchmark reward")
    ]
    pass_1: Annotated[
        float | None,
        Field(ge=0, le=100, description="100 × passed / scored; null if none scored"),
    ]
    interaction: Annotated[
        MeasureCoverage, Field(description="Call-level Interaction coverage")
    ]
    experience: Annotated[
        MeasureCoverage, Field(description="Call-level Experience coverage")
    ]

    @model_validator(mode="after")
    def _check_counts(self) -> VoiceV2Cell:
        if self.selected != self.scored + self.infrastructure:
            raise ValueError("selected != scored + infrastructure")
        if self.scored != self.passed + self.failed:
            raise ValueError("scored != pass + fail")
        expected = 100 * self.passed / self.scored if self.scored else None
        if expected is None and self.pass_1 is not None:
            raise ValueError("pass_1 must be null without scored calls")
        if expected is not None and (
            self.pass_1 is None or abs(self.pass_1 - expected) > 0.0001
        ):
            raise ValueError("pass_1 does not match pass / scored")
        for measure in (self.interaction, self.experience):
            total = (
                measure.passed + measure.failed + measure.unscored + measure.excluded
            )
            if total != self.scored:
                raise ValueError("measure coverage does not equal scored calls")
        if self.interaction.version != INTERACTION_VERSION:
            raise ValueError("unexpected Interaction version")
        if self.experience.version != EXPERIENCE_VERSION:
            raise ValueError("unexpected Experience version")
        return self


class VoiceV2Benchmark(BaseModel):
    """Frozen frame that defines what a Voice V2 score measures."""

    version: Annotated[Literal["voice-v2"], Field(description="Benchmark version")]
    frame_revision: Annotated[str, Field(description="Frozen task-frame revision")]
    languages: Annotated[
        list[Literal["en", "es", "hi", "ko", "pt", "zh"]],
        Field(description="Languages included in this submission"),
    ]
    domains: Annotated[
        list[Literal["airline", "retail", "telecom"]],
        Field(description="Domains included in this submission"),
    ]
    task_trials_per_cell: Annotated[
        int, Field(ge=1, description="Frozen selected task/trial count per cell")
    ]


class VoiceV2ContactInfo(ContactInfo):
    """Submission contact with the email required by the V2 contract."""

    email: Annotated[str, Field(min_length=1, description="Submitter contact email")]


class VoiceV2Submission(Submission):
    """Strict V2 contract; legacy flat submissions continue using Submission."""

    model_config = ConfigDict(extra="forbid")
    format: Annotated[Literal["voice-v2"], Field(description="Schema discriminator")]
    modality: Literal["voice"] = "voice"
    trajectories_available: Literal[True] = True
    contact_info: VoiceV2ContactInfo
    interaction_metrics: None = None
    benchmark: Annotated[
        VoiceV2Benchmark, Field(description="Frozen multilingual benchmark frame")
    ]
    benchmark_submission: Annotated[
        BenchmarkSubmission,
        Field(description="Canonical task-trial observations and recomputable score"),
    ]
    results: Annotated[
        dict[str, dict[str, VoiceV2Cell]],
        Field(description="Results by language, then domain"),
    ]
    trajectory_files: Annotated[
        dict[str, dict[str, str]],
        Field(description="Relative trajectory directory by language and domain"),
    ]
    provenance_file: Annotated[
        str, Field(description="Relative path to the source-hash manifest")
    ]
    provenance_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256 of provenance file")
    ]

    @model_validator(mode="after")
    def _check_frame(self) -> VoiceV2Submission:
        canonical = self.benchmark_submission
        if (
            canonical.frame.benchmark_version != "voice-v2"
            or canonical.frame.revision != self.benchmark.frame_revision
            or canonical.frame.trials != [0]
        ):
            raise ValueError("Voice V2 projection differs from canonical task frame")
        if len(canonical.frame.cells) != len(LANGUAGES) * len(DOMAINS):
            raise ValueError("Voice V2 canonical frame requires 18 cells")
        if len(self.benchmark.languages) != len(LANGUAGES) or set(
            self.benchmark.languages
        ) != set(LANGUAGES):
            raise ValueError("Voice V2 requires all six language cells")
        if len(self.benchmark.domains) != len(DOMAINS) or set(
            self.benchmark.domains
        ) != set(DOMAINS):
            raise ValueError("Voice V2 requires all three domains")
        if set(self.results) != set(LANGUAGES):
            raise ValueError("Voice V2 requires results for all six languages")
        if set(self.trajectory_files) != set(LANGUAGES):
            raise ValueError("Voice V2 requires trajectories for all six languages")
        for language in LANGUAGES:
            if set(self.results[language]) != set(DOMAINS):
                raise ValueError(f"{language}: results require all three domains")
            if set(self.trajectory_files[language]) != set(DOMAINS):
                raise ValueError(f"{language}: trajectories require all three domains")
            for domain in DOMAINS:
                cell = self.results[language][domain]
                canonical_cell = next(
                    (
                        item
                        for item in canonical.summary.cells
                        if (item.language, item.domain) == (language, domain)
                    ),
                    None,
                )
                if (
                    canonical_cell is None
                    or cell.selected != self.benchmark.task_trials_per_cell
                ):
                    raise ValueError(f"{language}/{domain}: incomplete benchmark frame")
                if (
                    canonical_cell.expected_trials != cell.selected
                    or canonical_cell.scored_trials != cell.scored
                    or canonical_cell.error_trials != cell.infrastructure
                    or canonical_cell.unscored_trials
                    or canonical_cell.missing_trials
                ):
                    raise ValueError(
                        f"{language}/{domain}: task projection differs from canonical score"
                    )
                canonical_passes = sum(
                    trial.reward == 1
                    for trial in canonical.trials
                    if trial.language == language
                    and trial.domain == domain
                    and trial.reward is not None
                )
                if canonical_passes != cell.passed:
                    raise ValueError(
                        f"{language}/{domain}: task success projection differs"
                    )
        return self


class InteractionFacts(BaseModel):
    """Saved facts for one scored call, before call-level thresholding."""

    nonresponse: Annotated[
        bool | None, Field(description="Saved nonresponse component failure")
    ]
    interruption_count: Annotated[
        int | None, Field(ge=0, description="Detected agent interruption events")
    ]
    selectivity_error_count: Annotated[
        int | None,
        Field(ge=0, description="Sum of backchannel, tic, and non-directed errors"),
    ]
    selectivity_available: Annotated[
        bool, Field(description="Whether a selectivity opportunity existed")
    ]
    monologue: Annotated[
        bool | None, Field(description="Saved over-45-second monologue failure")
    ]
    tool_misuse: Annotated[
        bool | None, Field(description="Saved validated tool-misuse failure")
    ]
    korean_unresponsive_period: Annotated[
        bool, Field(description="Known Korean unresponsive-period exclusion")
    ] = False


def interaction_outcome(facts: InteractionFacts) -> CallOutcome:
    """Apply the accepted >3-event, two-of-five Interaction rule."""
    if facts.korean_unresponsive_period:
        return CallOutcome.EXCLUDED
    flags = (
        facts.nonresponse,
        facts.interruption_count > 3 if facts.interruption_count is not None else None,
        facts.selectivity_error_count > 3
        if facts.selectivity_available and facts.selectivity_error_count is not None
        else None,
        facts.monologue,
        facts.tool_misuse,
    )
    known_failed = sum(flag is True for flag in flags)
    unknown = sum(flag is None for flag in flags)
    if known_failed >= 2:
        return CallOutcome.FAIL
    if known_failed + unknown < 2:
        return CallOutcome.PASS
    return CallOutcome.UNSCORED


def experience_outcome(
    eligible_utterances: int,
    naturalness: list[bool | None],
    fidelity: list[bool | None],
) -> CallOutcome:
    """Fail only when >25% of complete paired eligible utterances have issues."""
    if (
        eligible_utterances == 0
        or len(naturalness) != eligible_utterances
        or len(fidelity) != eligible_utterances
        or any(value is None for value in (*naturalness, *fidelity))
    ):
        return CallOutcome.UNSCORED
    issues = sum(a or b for a, b in zip(naturalness, fidelity))
    return CallOutcome.FAIL if 4 * issues > eligible_utterances else CallOutcome.PASS


def coverage(outcomes: list[CallOutcome], version: str) -> MeasureCoverage:
    """Summarize call outcomes without treating missing inputs as failures."""
    counts = Counter(outcomes)
    denominator = counts[CallOutcome.PASS] + counts[CallOutcome.FAIL]
    return MeasureCoverage(
        version=version,
        passed=counts[CallOutcome.PASS],
        failed=counts[CallOutcome.FAIL],
        unscored=counts[CallOutcome.UNSCORED],
        excluded=counts[CallOutcome.EXCLUDED],
        pass_rate=100 * counts[CallOutcome.PASS] / denominator if denominator else None,
    )


def file_sha256(path: Path) -> str:
    """Hash a saved input without modifying it."""
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
