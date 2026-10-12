"""Repairs cannot lose authored caller speech or overwrite unrelated identities."""

from pathlib import Path

import pytest

from tau2.multilingual.factory import database_localization as drafting
from tau2.multilingual.factory.generation_config import FactoryModelConfig
from tau2.multilingual.factory.intake_difficulty_apply import (
    DifficultyRevisionMerge,
    DifficultyRevisionRequest,
    merge_difficulty_revisions,
)
from tau2.multilingual.factory.localization_artifact import read_localization_map
from tau2.multilingual.localization import IntakeCallerSurface
from tau2.utils.artifacts import content_digest
from tau2.utils.utils import DATA_DIR


@pytest.fixture(scope="module")
def parent():
    return read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/ko/localization/intake.json"
    )


def surface_map(parent):
    changed = parent.model_copy(deep=True)
    changed.intake_caller_surfaces = [
        *[
            row
            for row in changed.intake_caller_surfaces
            if row.canonical_task_id != "intake_times_hard_09"
        ],
        IntakeCallerSurface(
            canonical_task_id="intake_times_hard_09",
            canonical_value="9:40 PM",
            native_spoken="열 시 이십 분 전",
            romanized_spoken="yeol si isip bun jeon",
            rationale="Minutes before next hour; caller knows the evening anchor.",
            provenance=parent.provenance,
        ),
    ]
    changed.intake_caller_surfaces.sort(key=lambda row: row.canonical_task_id)
    return changed


def lineage(parent, entities, surfaces, selected=None):
    return DifficultyRevisionMerge(
        request=DifficultyRevisionRequest(
            map_file=Path("parent.json"),
            audit_file=Path("audit.json"),
            revision="test-repair",
            model="fixture",
            model_args={"reasoning_effort": "low"},
        ),
        parent_map_sha256=content_digest(parent),
        entity_map_sha256=content_digest(entities),
        surface_map_sha256=content_digest(surfaces),
        audit_sha256="a" * 64,
        plan_sha256="b" * 64,
        selected_entry_ids=selected or [],
        selected_surface_task_ids=["intake_times_hard_09"],
    )


def test_merge_retains_exact_entities_source_and_new_time_knowledge(parent):
    surfaces = surface_map(parent)
    evidence = lineage(parent, parent, surfaces)
    merged = merge_difficulty_revisions(parent, parent, surfaces, evidence)
    assert merged.entries == parent.entries
    assert merged.source == parent.source
    assert merged.intake_caller_surfaces == surfaces.intake_caller_surfaces
    assert merged.provenance.input_sha256 == content_digest(evidence)


def test_merge_rejects_unselected_change_even_if_map_still_compiles(parent):
    entities = parent.model_copy(deep=True)
    entities.entries[0].notes += " Unrequested edit."
    surfaces = surface_map(parent)
    with pytest.raises(ValueError, match="unselected"):
        merge_difficulty_revisions(
            parent, entities, surfaces, lineage(parent, entities, surfaces)
        )


def test_merge_rejects_caller_stage_entity_drift(parent):
    surfaces = surface_map(parent)
    surfaces.entries[0].notes += " Undeclared edit."
    with pytest.raises(ValueError, match="entity entries"):
        merge_difficulty_revisions(
            parent, parent, surfaces, lineage(parent, parent, surfaces)
        )


def test_merge_rejects_unselected_prose_binding_that_would_add_csv_row(parent):
    from tau2.multilingual.localization import LocalizedForms, ProseEntityBinding

    entities = parent.model_copy(deep=True)
    entities.prose_bindings.append(
        ProseEntityBinding(
            entry_id=entities.entries[0].id,
            source="Unselected review identity",
            forms=LocalizedForms(native="가짜", romanized="gajja", english="Test only"),
            kind="generated",
            reason="Unrequested binding",
            provenance=parent.provenance,
        )
    )
    surfaces = surface_map(parent)
    with pytest.raises(ValueError, match="metadata or bindings"):
        merge_difficulty_revisions(
            parent, entities, surfaces, lineage(parent, entities, surfaces)
        )


def test_ordinary_database_revision_keeps_authored_time_surfaces(
    parent, monkeypatch, tmp_path
):
    previous = surface_map(parent)
    selected = next(
        entry
        for entry in previous.entries
        if entry.reuse is None
        and entry.context.startswith("intake field:second_guest_full_name")
    )

    def draft_entries(request, pending, completed, *args):
        assert [entry.id for entry in pending] == [selected.id]
        return [*completed, selected.model_copy(deep=True)]

    monkeypatch.setattr(drafting, "_draft_entries", draft_entries)
    revision = drafting.RegionalRevision(
        previous=previous,
        redraft_ids=[selected.id],
        request=drafting.RegionalDraftRequest(
            source=previous.source,
            settings=previous.settings,
            revision="unrelated-name-repair",
            generation=FactoryModelConfig(model="fixture", reasoning_effort="low"),
        ),
        feedback=[],
    )
    result = drafting.revise_database_localization(
        revision, tmp_path / "result.json", tmp_path / "checkpoints"
    )
    assert result.intake_caller_surfaces == previous.intake_caller_surfaces
    assert result.entries == previous.entries


def test_pipeline_joins_entity_and_time_stages_without_changing_easy_controls(
    parent, monkeypatch, tmp_path
):
    from tau2.multilingual.factory import intake_time_authoring, localization_cli
    from tau2.multilingual.factory.intake_difficulty_apply import (
        DifficultyMergeRequest,
        merge_completed_difficulty_stages,
        revise_intake_difficulty,
    )
    from tau2.multilingual.factory.intake_difficulty_inventory import (
        build_difficulty_inventory,
    )
    from tau2.multilingual.factory.localization_artifact import write_localization_map
    from tau2.multilingual.factory.localization_cli import DatabaseRevisionCommand
    from tau2.utils.artifacts import write_artifact
    from tests.test_multilingual.test_intake_difficulty_revision import report, revise

    inventory = build_difficulty_inventory(parent)
    first = inventory.cases[0]
    audit = report(
        inventory,
        {
            first.canonical_task_id: revise("omission_instruction"),
            "intake_times_hard_09": revise("caller_surface"),
        },
    )
    map_file = tmp_path / "parent.json"
    audit_file = tmp_path / "audit.json"
    write_localization_map(parent, map_file)
    write_artifact(audit, audit_file)
    touched = set()

    def revise_entities(args):
        command = DatabaseRevisionCommand.model_validate_json(args.request.read_text())
        assert command.previous_file == map_file
        assert first.measured_entry_id not in command.redraft_ids
        assert all(path.exists() for path in command.feedback_files)
        changed = parent.model_copy(deep=True)
        for entry in changed.entries:
            if entry.id in command.redraft_ids:
                entry.notes += " Mock targeted omission repair."
                touched.add(entry.id)
        write_localization_map(changed, args.output)

    def revise_times(request, output, checkpoints):
        assert request.map_file == map_file
        assert request.audit_file == audit_file
        assert request.canonical_task_ids == ["intake_times_hard_09"]
        changed = surface_map(parent)
        write_localization_map(changed, output)
        return changed

    monkeypatch.setattr(localization_cli, "run_database_revise", revise_entities)
    monkeypatch.setattr(
        intake_time_authoring, "revise_intake_time_surfaces", revise_times
    )
    request = DifficultyRevisionRequest(
        map_file=map_file,
        audit_file=audit_file,
        revision="combined-repair",
        model="fixture",
        model_args={"reasoning_effort": "low"},
    )
    output = tmp_path / "repaired.json"
    revised = revise_intake_difficulty(request, output, tmp_path / "checkpoints")
    assert touched
    assert revised.intake_caller_surfaces == surface_map(parent).intake_caller_surfaces
    assert all(
        before == after
        for before, after in zip(parent.entries, revised.entries, strict=True)
        if before.id not in touched
    )
    assert read_localization_map(output) == revised
    assert (tmp_path / "checkpoints/merge.json").exists()
    assert (tmp_path / "checkpoints/plan.ledger.csv").exists()
    resumed = merge_completed_difficulty_stages(
        DifficultyMergeRequest(
            **request.model_dump(),
            entity_map_file=tmp_path / "checkpoints/entities.json",
            surface_map_file=tmp_path / "checkpoints/surfaces.json",
        ),
        tmp_path / "resumed.json",
        tmp_path / "resumed-merge.json",
    )
    assert resumed == revised
