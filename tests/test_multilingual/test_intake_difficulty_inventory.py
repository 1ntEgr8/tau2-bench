"""Exhaustive difficulty evidence is task-bound and excludes callback identities."""

from collections import Counter

import pytest

from tau2.multilingual.factory.intake_difficulty_inventory import (
    DifficultyInventory,
    build_difficulty_inventory,
)
from tests.test_domains.test_intake.test_localization_intake import identity_map


@pytest.fixture(scope="module")
def proposal():
    return identity_map()


@pytest.fixture(scope="module")
def inventory(proposal):
    return build_difficulty_inventory(proposal)


def test_exhaustive_hard_slots_have_same_family_easy_comparators(inventory):
    assert len(inventory.cases) == 100
    assert set(Counter(case.family for case in inventory.cases).values()) == {10}
    assert len({case.canonical_task_id for case in inventory.cases}) == 100
    for case in inventory.cases:
        assert len(case.easy_comparators) == 10
        assert all(
            row.family == case.family and row.tier == "easy"
            for row in case.easy_comparators
        )
        assert case.source_dependencies.written == case.source_value
        assert case.native_dependencies.written == case.native_value
        assert case.romanized_dependencies.written == case.romanized_value
    assert (
        DifficultyInventory.model_validate_json(inventory.model_dump_json())
        == inventory
    )


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_family"])
def test_serialized_coverage_cannot_silently_drop_or_duplicate_slots(inventory, change):
    payload = inventory.model_dump()
    if change == "missing":
        payload["cases"].pop()
    elif change == "duplicate":
        payload["cases"][0] = payload["cases"][1]
    else:
        payload["cases"][0]["easy_comparators"][0] = payload["cases"][-1][
            "easy_comparators"
        ][0]
    with pytest.raises(ValueError):
        DifficultyInventory.model_validate(payload)


def test_source_changes_fail_before_evidence_is_exported(proposal):
    changed = proposal.model_copy(deep=True)
    document = next(
        d for d in changed.source.documents if d.id == "intake_localization"
    )
    document.content["stimuli"][0]["written"] = "SOURCE DRIFT"
    with pytest.raises(ValueError):
        build_difficulty_inventory(changed)


def test_every_measured_id_owns_exact_selected_written_occurrence(proposal, inventory):
    selected = next(
        d.content for d in proposal.source.documents if d.id == "intake_localization"
    )
    indices = {s["canonical_task_id"]: i for i, s in enumerate(selected["stimuli"])}
    entries = {e.id: e for e in proposal.entries}
    for case in inventory.cases:
        entry = entries[case.measured_entry_id]
        assert any(
            o.document_id == "intake_localization"
            and o.path == ["stimuli", indices[case.canonical_task_id], "written"]
            for o in entry.occurrences
        )
        assert entry.reuse is None


def test_callback_owner_values_are_not_exported(proposal):
    changed = proposal.model_copy(deep=True)
    owner_entries = [
        entry
        for entry in changed.entries
        if any(o.path and o.path[-1] == "callee_full_name" for o in entry.occurrences)
    ]
    assert owner_entries
    for index, entry in enumerate(owner_entries):
        entry.forms.native = entry.forms.romanized = f"PRIVATE_CALLBACK_OWNER_{index}"
        entry.operation = "adapt"
    exported = build_difficulty_inventory(changed).model_dump_json()
    assert "PRIVATE_CALLBACK_OWNER" not in exported


def test_payload_digest_changes_with_safe_evidence(inventory):
    changed = inventory.model_copy(deep=True)
    changed.cases[0].native_guidance.clarification_guidance += " Changed."
    with pytest.raises(ValueError, match="digest"):
        DifficultyInventory.model_validate(changed.model_dump())


def test_reused_measured_identity_is_refused_even_if_source_is_valid(proposal):
    from tau2.multilingual.localization import EntityReuseProvenance
    from tau2.utils.artifacts import ReviewApproval

    changed = proposal.model_copy(deep=True)
    measured = next(
        entry
        for entry in changed.entries
        if any(
            o.document_id == "intake_localization"
            and o.path == ["stimuli", 0, "written"]
            for o in entry.occurrences
        )
    )
    measured.reuse = EntityReuseProvenance(
        source_domain="retail",
        source_entry_ids=["private-source-id"],
        source_artifact_sha256="0" * 64,
        approval=ReviewApproval(content_sha256="0" * 64, reviewed_by="test"),
    )
    with pytest.raises(ValueError, match="donor"):
        build_difficulty_inventory(changed)


def test_caller_phrase_and_written_answer_have_separate_exact_entry_owners(inventory):
    dates = [case for case in inventory.cases if case.family == "dates"]
    assert any(case.measured_entry_id not in case.caller_entry_ids for case in dates)
    for case in dates:
        assert case.source_guidance.pinned_clock == "2025-06-12T10:30:00"
        assert set(case.entry_sha256) == {
            case.measured_entry_id,
            *case.caller_entry_ids,
            *case.dependency_entry_ids,
        }
