"""Apply audited entity and caller-surface repairs with independently bound stages."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from tau2.multilingual.factory.intake_difficulty_audit import (
    DifficultyAuditReport,
    DifficultyAuditRequest,
)
from tau2.multilingual.factory.localization_artifact import (
    read_localization_map,
    write_localization_map,
)
from tau2.multilingual.factory.localization_compile import validate_localization
from tau2.multilingual.localization import (
    GenerationProvenance,
    LocalizationMap,
    LocalizationModel,
)
from tau2.utils.artifacts import content_digest, write_artifact

REVISION_VERSION = "intake-difficulty-revision-v1"


class DifficultyRevisionRequest(DifficultyAuditRequest):
    """Explicit authoring settings and the exact audit whose findings are repaired."""

    audit_file: Annotated[
        Path, Field(description="Complete audit of the unchanged parent map.")
    ]


class DifficultyMergeRequest(DifficultyRevisionRequest):
    """Validated stage outputs, including explicitly repaired failed-stage descendants."""

    entity_map_file: Annotated[
        Path,
        Field(
            description="Validated entity-stage map or its scoped repair descendant."
        ),
    ]
    surface_map_file: Annotated[
        Path,
        Field(description="Validated time-surface stage against the original parent."),
    ]


class DifficultyRevisionMerge(LocalizationModel):
    """Reproducible lineage for joining disjoint, independently validated stages."""

    version: Annotated[
        Literal["intake-difficulty-revision-v1"],
        Field(description="Version of the disjoint-stage merge contract."),
    ] = REVISION_VERSION
    request: Annotated[
        DifficultyRevisionRequest, Field(description="Recorded authoring inputs.")
    ]
    parent_map_sha256: Annotated[
        str,
        Field(pattern=r"^[0-9a-f]{64}$", description="Unchanged parent map digest."),
    ]
    audit_sha256: Annotated[
        str,
        Field(pattern=r"^[0-9a-f]{64}$", description="Exact independent audit digest."),
    ]
    plan_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$", description="Exact targeted repair plan digest."
        ),
    ]
    entity_map_sha256: Annotated[
        str,
        Field(pattern=r"^[0-9a-f]{64}$", description="Validated entity stage digest."),
    ]
    surface_map_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Validated caller-surface stage digest.",
        ),
    ]
    selected_entry_ids: Annotated[
        list[str], Field(description="Exact map entries the entity stage may revise.")
    ]
    selected_surface_task_ids: Annotated[
        list[str],
        Field(description="Exact Hard time tasks the surface stage may revise."),
    ]


def merge_difficulty_revisions(
    parent: LocalizationMap,
    entities: LocalizationMap,
    surfaces: LocalizationMap,
    lineage: DifficultyRevisionMerge,
) -> LocalizationMap:
    """Join staged edits only when each branch changed exactly its permitted scope."""
    if (
        content_digest(parent) != lineage.parent_map_sha256
        or content_digest(entities) != lineage.entity_map_sha256
        or content_digest(surfaces) != lineage.surface_map_sha256
    ):
        raise ValueError("Revision merge input differs from recorded provenance")
    for candidate in (parent, entities, surfaces):
        validate_localization(candidate)
        if candidate.source != parent.source or candidate.settings != parent.settings:
            raise ValueError("Revision merge cannot change source or regional settings")
        unchanged_fields = set(LocalizationMap.model_fields) - {
            "revision",
            "provenance",
            "entries",
            "intake_caller_surfaces",
        }
        if any(
            getattr(candidate, field) != getattr(parent, field)
            for field in unchanged_fields
        ):
            raise ValueError(
                "Revision merge changed unselected map metadata or bindings"
            )
    if entities.intake_caller_surfaces != parent.intake_caller_surfaces:
        raise ValueError("Entity revision changed caller surfaces")
    if surfaces.entries != parent.entries:
        raise ValueError("Caller-surface revision changed entity entries")
    permitted = set(lineage.selected_entry_ids)
    original = {entry.id: entry for entry in parent.entries}
    if permitted - original.keys():
        raise ValueError("Revision merge contains unknown selected entries")
    for entry in entities.entries:
        if entry.id not in permitted and entry != original[entry.id]:
            raise ValueError("Entity revision changed an unselected entry")
        if original[entry.id].reuse is not None and entry != original[entry.id]:
            raise ValueError("Revision merge cannot change a reused identity")
    before = {item.canonical_task_id: item for item in parent.intake_caller_surfaces}
    after = {item.canonical_task_id: item for item in surfaces.intake_caller_surfaces}
    targets = set(lineage.selected_surface_task_ids)
    if not targets <= after.keys() or any(
        before.get(identifier) != after.get(identifier)
        for identifier in before.keys() | after.keys()
        if identifier not in targets
    ):
        raise ValueError("Caller-surface revision has missing or unselected changes")
    data = entities.model_dump()
    data.update(
        revision=lineage.request.revision,
        intake_caller_surfaces=surfaces.intake_caller_surfaces,
        provenance=GenerationProvenance(
            model=lineage.request.model,
            model_args=lineage.request.model_args,
            prompt_version=REVISION_VERSION,
            call_name="intake_difficulty_revision",
            input_sha256=content_digest(lineage),
            parent_map_sha256=content_digest(parent),
            email_rules_version=entities.provenance.email_rules_version,
            user_id_rules_version=entities.provenance.user_id_rules_version,
        ),
    )
    result = LocalizationMap.model_validate(data)
    validate_localization(result)
    return result


def revise_intake_difficulty(
    request: DifficultyRevisionRequest, output: Path, checkpoints: Path
) -> LocalizationMap:
    """Author auditable repairs, retaining source blockers and requiring a fresh audit."""
    from argparse import Namespace

    from tau2.multilingual.factory.intake_difficulty_revision import (
        build_difficulty_revision_plan,
        make_difficulty_revision_command,
        write_difficulty_revision_artifacts,
    )
    from tau2.multilingual.factory.intake_time_authoring import (
        IntakeTimeAuthoringRequest,
        revise_intake_time_surfaces,
    )
    from tau2.multilingual.factory.localization_cli import run_database_revise

    request = DifficultyRevisionRequest.model_validate(request.model_dump())
    parent = read_localization_map(request.map_file)
    audit = DifficultyAuditReport.model_validate_json(request.audit_file.read_text())
    plan = build_difficulty_revision_plan(parent, audit)
    write_artifact(request, checkpoints / "request.json")
    files = write_difficulty_revision_artifacts(plan, checkpoints / "plan")
    entity_request = make_difficulty_revision_command(
        plan, request, files.feedback_file
    )
    entities = parent
    if entity_request is not None:
        command_file = checkpoints / "entity-request.json"
        write_artifact(entity_request, command_file)
        entity_output = checkpoints / "entities.json"
        run_database_revise(
            Namespace(
                request=command_file,
                output=entity_output,
                checkpoints=checkpoints / "entity-checkpoints",
            )
        )
        entities = read_localization_map(entity_output)
    surfaces = parent
    if plan.caller_surface_task_ids:
        surface_request = IntakeTimeAuthoringRequest(
            **request.model_dump(),
            canonical_task_ids=plan.caller_surface_task_ids,
        )
        surfaces = revise_intake_time_surfaces(
            surface_request,
            checkpoints / "surfaces.json",
            checkpoints / "surface-checkpoints",
        )
    lineage = DifficultyRevisionMerge(
        request=request,
        parent_map_sha256=content_digest(parent),
        audit_sha256=content_digest(audit),
        plan_sha256=content_digest(plan),
        entity_map_sha256=content_digest(entities),
        surface_map_sha256=content_digest(surfaces),
        selected_entry_ids=plan.redraft_ids,
        selected_surface_task_ids=plan.caller_surface_task_ids,
    )
    result = merge_difficulty_revisions(parent, entities, surfaces, lineage)
    write_artifact(lineage, checkpoints / "merge.json")
    write_localization_map(result, output)
    return result


def merge_completed_difficulty_stages(
    request: DifficultyMergeRequest, output: Path, record: Path
) -> LocalizationMap:
    """Resume after a failed-stage repair without replacing prior evidence or checkpoints."""
    from tau2.multilingual.factory.intake_difficulty_revision import (
        build_difficulty_revision_plan,
    )

    parent = read_localization_map(request.map_file)
    audit = DifficultyAuditReport.model_validate_json(request.audit_file.read_text())
    plan = build_difficulty_revision_plan(parent, audit)
    entities = read_localization_map(request.entity_map_file)
    surfaces = read_localization_map(request.surface_map_file)
    original_request = DifficultyRevisionRequest.model_validate(
        request.model_dump(exclude={"entity_map_file", "surface_map_file"})
    )
    lineage = DifficultyRevisionMerge(
        request=original_request,
        parent_map_sha256=content_digest(parent),
        audit_sha256=content_digest(audit),
        plan_sha256=content_digest(plan),
        entity_map_sha256=content_digest(entities),
        surface_map_sha256=content_digest(surfaces),
        selected_entry_ids=plan.redraft_ids,
        selected_surface_task_ids=plan.caller_surface_task_ids,
    )
    result = merge_difficulty_revisions(parent, entities, surfaces, lineage)
    write_artifact(lineage, record)
    write_localization_map(result, output)
    return result
