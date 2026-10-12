"""Exhaustive, privacy-scoped evidence for the frozen intake difficulty frame."""

from collections import Counter
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from tau2.data_model.tasks import Task
from tau2.domains.intake.date_display import DATE_DISPLAY_VERSION
from tau2.domains.intake.localization import validate_intake_tasks
from tau2.domains.intake.localized import IntakeLocalizationData, SelectedStimulus
from tau2.domains.intake.tasks.generator import CANONICAL_BANKS
from tau2.domains.intake.time_display import TIME_CALLER_GUIDANCE, TIME_DISPLAY_VERSION
from tau2.domains.intake.utils import get_now
from tau2.multilingual.factory.localization_compile import (
    _compile_documents,
    validate_localization_source,
)
from tau2.multilingual.localization import (
    LocalizationMap,
    LocalizationModel,
    RegionalSettings,
    SourceDocument,
)
from tau2.utils.artifacts import content_digest

INVENTORY_VERSION = "intake-difficulty-inventory-v1"
Family = Literal[
    "person_names",
    "codes",
    "phones",
    "dates",
    "times",
    "addresses",
    "properties",
    "emails",
    "medications",
    "coined",
]
Digest = Annotated[
    str, Field(pattern=r"^[0-9a-f]{64}$", description="SHA-256 of typed content.")
]


class DifficultyGuidance(LocalizationModel):
    """Only shared capture guidance; callback identities and narrative stay local."""

    pinned_clock: Annotated[
        str,
        Field(
            description="Exact source get_now clock for interpreting relative dates; no caller identity or context."
        ),
    ]
    clarification_guidance: Annotated[
        str,
        Field(
            description="Shared locale spelling and clarification instructions, without task-owner context."
        ),
    ]
    time_guidance: Annotated[
        str,
        Field(
            description="Time-specific caller knowledge and initial-utterance instructions, or empty for other families."
        ),
    ]


class DifficultyComparator(LocalizationModel):
    """One measured slot and its exact safe compiled capture dependencies."""

    canonical_task_id: Annotated[
        str, Field(description="Exact frozen source task identity.")
    ]
    family: Annotated[Family, Field(description="Measured entity bank family.")]
    tier: Annotated[
        Literal["easy"], Field(description="Easy source comparator assignment.")
    ]
    field_name: Annotated[
        str,
        Field(
            description="Measured protocol field, independent of callback-owner fields."
        ),
    ]
    measured_entry_id: Annotated[
        str,
        Field(
            description="Unique entry owning this selected written-stimulus occurrence."
        ),
    ]
    caller_entry_ids: Annotated[
        list[str],
        Field(
            min_length=1,
            description="Exact map-entry owners of the measured set_entities field, separate from the canonical written answer.",
        ),
    ]
    entry_sha256: Annotated[
        dict[str, Digest],
        Field(
            description="Full typed entry digests for all measured, caller and dependency IDs; no donor content."
        ),
    ]
    dependency_entry_ids: Annotated[
        list[str],
        Field(
            description="Exact entry owners of this stimulus's decoy, speech, omission, spelling and measured caller occurrences."
        ),
    ]
    source_value: Annotated[
        str, Field(description="Canonical written measured answer.")
    ]
    native_value: Annotated[
        str, Field(description="Compiled native written measured answer.")
    ]
    romanized_value: Annotated[
        str, Field(description="Compiled romanized written measured answer.")
    ]
    source_caller_payload: Annotated[
        str,
        Field(
            description="Canonical caller lookup for the measured field only, including private exact-value annotation when present."
        ),
    ]
    native_caller_payload: Annotated[
        str,
        Field(
            description="Compiled native caller lookup, preserving local dates/times and private exact-value annotation."
        ),
    ]
    romanized_caller_payload: Annotated[
        str,
        Field(
            description="Compiled romanized caller lookup for the measured field only."
        ),
    ]
    source_dependencies: Annotated[
        SelectedStimulus,
        Field(
            description="Selected canonical decoys, pronunciation and omission dependencies, never the whole bank."
        ),
    ]
    native_dependencies: Annotated[
        SelectedStimulus,
        Field(
            description="Compiled native dependencies after date/time display projection."
        ),
    ]
    romanized_dependencies: Annotated[
        SelectedStimulus,
        Field(
            description="Compiled romanized dependencies after date/time display projection."
        ),
    ]
    source_guidance: Annotated[
        DifficultyGuidance, Field(description="Canonical safe shared guidance.")
    ]
    native_guidance: Annotated[
        DifficultyGuidance, Field(description="Native safe shared guidance.")
    ]
    romanized_guidance: Annotated[
        DifficultyGuidance, Field(description="Romanized safe shared guidance.")
    ]

    @model_validator(mode="after")
    def validate_dependencies(self):
        """Reject mismatched task, role, tier and answer evidence on deserialization."""
        for representation in ("source", "native", "romanized"):
            row = getattr(self, f"{representation}_dependencies")
            if (
                row.canonical_task_id,
                row.family,
                row.tier,
                row.field_name,
                row.written,
            ) != (
                self.canonical_task_id,
                self.family,
                self.tier,
                self.field_name,
                getattr(self, f"{representation}_value"),
            ):
                raise ValueError(
                    "Difficulty evidence differs from its exact selected slot"
                )
        if set(self.entry_sha256) != {
            self.measured_entry_id,
            *self.caller_entry_ids,
            *self.dependency_entry_ids,
        }:
            raise ValueError(
                "Difficulty evidence entry digests must cover exactly its entry IDs"
            )
        if len(set(self.dependency_entry_ids)) != len(self.dependency_entry_ids):
            raise ValueError("Duplicate difficulty dependency entry IDs")
        return self


class DifficultyCase(DifficultyComparator):
    """A Hard slot compared against every Easy slot of the same family."""

    tier: Annotated[
        Literal["hard"],
        Field(
            description="Frozen Hard assignment whose localized challenge must be audited."
        ),
    ]
    easy_comparators: Annotated[
        list[DifficultyComparator],
        Field(
            min_length=10,
            max_length=10,
            description="All ten Easy slots in the same source family, never a sampled comparison.",
        ),
    ]

    @model_validator(mode="after")
    def validate_comparators(self):
        """Require distinct same-family Easy controls for every Hard comparison."""
        if len({row.canonical_task_id for row in self.easy_comparators}) != 10 or any(
            row.family != self.family for row in self.easy_comparators
        ):
            raise ValueError(
                "Hard slot requires ten distinct same-family Easy comparators"
            )
        return self


class DifficultyInventoryPayload(LocalizationModel):
    """Complete source/map-bound evidence before adding its self-verifying digest."""

    version: Annotated[
        Literal["intake-difficulty-inventory-v1"],
        Field(
            description="Version of the exact selected-slot extraction and privacy contract."
        ),
    ] = INVENTORY_VERSION
    map_sha256: Digest
    source_sha256: Digest
    settings: Annotated[
        RegionalSettings,
        Field(description="Exact target locale settings used by compilation."),
    ]
    stimulus_version: Annotated[
        str, Field(description="Source selected-stimulus schema version.")
    ]
    date_format_version: Annotated[
        str, Field(description="Version of deterministic caller-facing date rendering.")
    ] = DATE_DISPLAY_VERSION
    time_format_version: Annotated[
        str, Field(description="Version of deterministic caller-facing time rendering.")
    ] = TIME_DISPLAY_VERSION
    cases: Annotated[
        list[DifficultyCase],
        Field(
            min_length=100,
            max_length=100,
            description="Every Hard slot: ten slots in each of ten families.",
        ),
    ]

    @model_validator(mode="after")
    def validate_frame(self):
        """Enforce full typed 100-Hard coverage and one consistent Easy frame."""
        if len({case.canonical_task_id for case in self.cases}) != 100 or Counter(
            case.family for case in self.cases
        ) != Counter({family: 10 for family in CANONICAL_BANKS}):
            raise ValueError("Difficulty inventory requires all 100 unique Hard slots")
        controls = {}
        for case in self.cases:
            values = {
                row.canonical_task_id: content_digest(row)
                for row in case.easy_comparators
            }
            if case.family in controls and controls[case.family] != values:
                raise ValueError(
                    "Same-family Hard slots must use identical complete Easy controls"
                )
            controls[case.family] = values
        return self


class DifficultyInventory(DifficultyInventoryPayload):
    """Immutable evidence manifest; no model or native review approval is implied."""

    payload_sha256: Digest

    @model_validator(mode="after")
    def validate_digest(self):
        """Reject stale or partially edited evidence artifacts."""
        payload = DifficultyInventoryPayload.model_validate(
            self.model_dump(exclude={"payload_sha256"})
        )
        if content_digest(payload) != self.payload_sha256:
            raise ValueError("Difficulty inventory payload digest differs")
        return self


class _CompiledView(LocalizationModel):
    """Local typed compile view, never serialized to an audit provider."""

    selected: Annotated[
        IntakeLocalizationData,
        Field(description="Validated selected-stimulus document."),
    ]
    tasks: Annotated[
        list[Task],
        Field(
            description="All tasks, retained only locally to extract the measured field."
        ),
    ]


def _view(
    documents: list[SourceDocument], language: str, representation: str
) -> _CompiledView:
    by_id = {document.id: document for document in documents}
    tasks = TypeAdapter(list[Task]).validate_python(by_id["tasks"].content)
    validate_intake_tasks(tasks, language, representation)
    selected = IntakeLocalizationData.model_validate(
        by_id["intake_localization"].content
    )
    if {row.canonical_task_id for row in selected.stimuli} != {
        task.id for task in tasks
    }:
        raise ValueError(
            "Selected stimuli do not exactly cover compiled canonical tasks"
        )
    if Counter((row.family, row.tier) for row in selected.stimuli) != Counter(
        {(family, tier): 10 for family in CANONICAL_BANKS for tier in ("easy", "hard")}
    ):
        raise ValueError(
            "Selected stimuli do not cover ten Easy and Hard slots per family"
        )
    return _CompiledView(selected=selected, tasks=tasks)


def _caller_payload(view: _CompiledView, row: SelectedStimulus) -> str:
    task = next(task for task in view.tasks if task.id == row.canonical_task_id)
    actions = [
        action
        for action in task.initial_state.initialization_actions
        if action.func_name == "set_entities"
    ]
    if len(actions) != 1:
        raise ValueError("Difficulty evidence requires exactly one caller entity setup")
    value = actions[0].arguments["entities"][row.field_name]
    if not isinstance(value, str):
        raise ValueError("Measured caller payload must be a string")
    return value


def _guidance(view: _CompiledView, family: Family) -> DifficultyGuidance:
    return DifficultyGuidance(
        pinned_clock=get_now().isoformat(),
        clarification_guidance=view.selected.clarification_guidance,
        time_guidance=TIME_CALLER_GUIDANCE if family == "times" else "",
    )


def build_difficulty_inventory(proposal: LocalizationMap) -> DifficultyInventory:
    """Compile the complete frozen frame, exporting only measured capture evidence.

    Owner names, organizations and scenario prose are deliberately not fields of
    the outgoing schema. Roles are resolved by exact source occurrences, never by
    mutable entry context text. Reused identities in measured slots fail closed.
    """
    if proposal.source.domain != "intake":
        raise ValueError("Difficulty inventory requires an intake localization map")
    proposal = validate_localization_source(proposal)
    source = _view(proposal.source.documents, "en", "native")
    native = _view(
        _compile_documents(proposal, "native"), proposal.settings.language, "native"
    )
    romanized = _view(
        _compile_documents(proposal, "romanized"),
        proposal.settings.language,
        "romanized",
    )
    owners = {}
    for entry in proposal.entries:
        for occurrence in entry.occurrences:
            key = (occurrence.document_id, *occurrence.path)
            if key in owners:
                raise ValueError(
                    "Duplicate exact occurrence ownership in difficulty inventory"
                )
            owners[key] = entry
    tasks = {task.id: index for index, task in enumerate(source.tasks)}
    raw_rows = []
    for index, row in enumerate(source.selected.stimuli):
        written_key = ("intake_localization", "stimuli", index, "written")
        measured = owners.get(written_key)
        if measured is None or measured.source != row.written:
            raise ValueError("Missing exact selected written-stimulus entry")
        task = source.tasks[tasks[row.canonical_task_id]]
        action_index = next(
            i
            for i, action in enumerate(task.initial_state.initialization_actions)
            if action.func_name == "set_entities"
        )
        caller_key = (
            "tasks",
            tasks[row.canonical_task_id],
            "initial_state",
            "initialization_actions",
            action_index,
            "arguments",
            "entities",
            row.field_name,
        )
        dependency_entries = {
            entry.id: entry
            for key, entry in owners.items()
            if (
                len(key) >= 4
                and key[:3] == ("intake_localization", "stimuli", index)
                and key[3]
                in {
                    "decoys",
                    "pronunciations",
                    "spelling_styles",
                    "omission_instruction",
                }
            )
        }
        if caller_key not in owners:
            raise ValueError("Missing measured caller occurrence owner")
        caller_entry = owners[caller_key]
        if (
            measured.reuse is not None
            or caller_entry.reuse is not None
            or any(entry.reuse is not None for entry in dependency_entries.values())
        ):
            raise ValueError(
                "Reused donor identities cannot enter measured difficulty evidence"
            )
        values = dict(
            canonical_task_id=row.canonical_task_id,
            family=row.family,
            tier=row.tier,
            field_name=row.field_name,
            measured_entry_id=measured.id,
            dependency_entry_ids=sorted(set(dependency_entries) - {measured.id}),
            caller_entry_ids=[caller_entry.id],
            entry_sha256={
                entry.id: content_digest(entry)
                for entry in [measured, caller_entry, *dependency_entries.values()]
            },
        )
        for name, view in (
            ("source", source),
            ("native", native),
            ("romanized", romanized),
        ):
            selected = view.selected.for_task(row.canonical_task_id)
            values[f"{name}_value"] = selected.written
            values[f"{name}_caller_payload"] = _caller_payload(view, selected)
            values[f"{name}_dependencies"] = selected
            values[f"{name}_guidance"] = _guidance(view, row.family)
        raw_rows.append(values)
    easy = [
        DifficultyComparator.model_validate(row)
        for row in raw_rows
        if row["tier"] == "easy"
    ]
    cases = [
        DifficultyCase.model_validate(
            {
                **row,
                "easy_comparators": [
                    control for control in easy if control.family == row["family"]
                ],
            }
        )
        for row in raw_rows
        if row["tier"] == "hard"
    ]
    payload = DifficultyInventoryPayload(
        map_sha256=content_digest(proposal),
        source_sha256=content_digest(proposal.source),
        settings=proposal.settings,
        stimulus_version=source.selected.version,
        cases=cases,
    )
    return DifficultyInventory(
        **payload.model_dump(), payload_sha256=content_digest(payload)
    )
