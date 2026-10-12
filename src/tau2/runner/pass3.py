"""Two explicit Pass^3 designs: repeated trials and crossed voice conditions."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tau2.data_model.simulation import Results


class CrossedCondition(BaseModel):
    """One versioned realization of the established R/C/S voice cross."""

    model_config = ConfigDict(frozen=True)

    label: str
    seed: int
    channel_effects_mode: Literal["light", "regular", "heavy"]
    speech_effects_mode: Literal["light", "regular", "heavy"]


# The agent-directed intake paper used fresh paired seeds across systems. The
# effect intensities match the established robustness catalog v1.2.0; the
# different seed trio is a new, explicit V2 catalog, never a silent default.
CROSSED_PASS3_VERSION = "voice-v2-rcs-v2"
CROSSED_CONDITIONS = (
    CrossedCondition(
        label="regular",
        seed=9401,
        channel_effects_mode="regular",
        speech_effects_mode="regular",
    ),
    CrossedCondition(
        label="channel_heavy",
        seed=9402,
        channel_effects_mode="heavy",
        speech_effects_mode="light",
    ),
    CrossedCondition(
        label="speech_heavy",
        seed=9403,
        channel_effects_mode="light",
        speech_effects_mode="heavy",
    ),
)


class TaskCrossedPass3(BaseModel):
    """One task's success vector in R/C/S order."""

    task_id: str
    passes: tuple[bool, bool, bool]
    passed_all: bool


class CrossedPass3Report(BaseModel):
    """Provenance-bearing, reproducible aggregate of three aligned runs."""

    version: str = CROSSED_PASS3_VERSION
    domain: str
    language: str | None
    conditions: tuple[CrossedCondition, CrossedCondition, CrossedCondition]
    source_paths: dict[str, str]
    git_commit: str
    tasks: list[TaskCrossedPass3]
    passed_all: int = Field(ge=0)
    total_tasks: int = Field(gt=0)
    pass3: float = Field(ge=0, le=1)


def summarize_crossed_pass3(
    results: dict[str, Results], source_paths: dict[str, str]
) -> CrossedPass3Report:
    """Score one result per task in each pinned condition; refuse partial grids."""
    expected = {condition.label for condition in CROSSED_CONDITIONS}
    if set(results) != expected or set(source_paths) != expected:
        raise ValueError(f"Crossed Pass^3 needs exactly {sorted(expected)}")
    first = results[CROSSED_CONDITIONS[0].label]
    domain = first.info.environment_info.domain_name
    language = first.info.language
    stable_info_fields = (
        "git_commit",
        "localize_db",
        "language_artifacts",
        "semantic_communicate",
        "elicitation",
        "agent_info",
        "user_info",
        "audio_native_config",
        "retrieval_config",
        "retrieval_config_kwargs",
        "seed",
        "language_tasks",
    )
    rows = []
    for condition in CROSSED_CONDITIONS:
        result = results[condition.label]
        if result.info.environment_info.domain_name != domain:
            raise ValueError("Crossed Pass^3 domains differ")
        if result.info.language != language:
            raise ValueError("Crossed Pass^3 languages differ")
        if (
            any(
                getattr(result.info, field) != getattr(first.info, field)
                for field in stable_info_fields
            )
            or result.tasks != first.tasks
        ):
            raise ValueError("Crossed Pass^3 run settings or task frames differ")
        if result.info.voice_sampling_seed != condition.seed:
            raise ValueError("Crossed Pass^3 voice seed does not match catalog")
        if (
            result.info.pass3_condition != condition.label
            or result.info.num_trials != 1
            or result.info.channel_effects_mode != condition.channel_effects_mode
            or result.info.speech_effects_mode != condition.speech_effects_mode
        ):
            raise ValueError("Crossed Pass^3 effects do not match catalog")
        index = result.simulation_index or result._build_simulation_index()
        if any(entry.trial != 0 for entry in index):
            raise ValueError("Crossed Pass^3 requires one trial per task")
        rewards = {str(entry.task_id): entry.reward for entry in index}
        if not rewards or len(rewards) != len(index):
            raise ValueError("Crossed Pass^3 needs unique completed task results")
        expected_tasks = {str(task.id) for task in result.tasks}
        if len(expected_tasks) != len(result.tasks) or set(rewards) != expected_tasks:
            raise ValueError("Crossed Pass^3 has missing or unexpected task results")
        if any(reward is None for reward in rewards.values()):
            raise ValueError("Crossed Pass^3 has unscored task results")
        rows.append(rewards)
    task_ids = set(rows[0])
    if any(set(row) != task_ids for row in rows[1:]):
        raise ValueError("Crossed Pass^3 conditions have different task sets")
    tasks = [
        TaskCrossedPass3(
            task_id=task_id,
            passes=tuple(row[task_id] == 1.0 for row in rows),
            passed_all=all(row[task_id] == 1.0 for row in rows),
        )
        for task_id in sorted(task_ids)
    ]
    passed_all = sum(task.passed_all for task in tasks)
    return CrossedPass3Report(
        domain=domain,
        language=language,
        conditions=CROSSED_CONDITIONS,
        source_paths=source_paths,
        git_commit=first.info.git_commit,
        tasks=tasks,
        passed_all=passed_all,
        total_tasks=len(tasks),
        pass3=passed_all / len(tasks),
    )
