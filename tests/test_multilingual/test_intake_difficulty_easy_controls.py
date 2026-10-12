"""Ordinary repair authoring receives every Easy control without donor context."""

from pathlib import Path

import pytest

from tau2.multilingual.factory.intake_difficulty_inventory import (
    build_difficulty_inventory,
)
from tau2.multilingual.factory.intake_difficulty_revision import (
    DifficultyRevisionFeedback,
    DifficultyRevisionPlan,
    build_difficulty_revision_feedback,
    build_difficulty_revision_plan,
)
from tau2.multilingual.factory.localization_artifact import read_localization_map
from tau2.utils.artifacts import content_digest
from tau2.utils.utils import DATA_DIR
from tests.test_multilingual.test_intake_difficulty_revision import report, revise


@pytest.fixture(scope="module")
def evidence():
    mapping = read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/ko/localization/intake.json"
    )
    inventory = build_difficulty_inventory(mapping)
    cases = [c for c in inventory.cases if c.family == "person_names"]
    plan = build_difficulty_revision_plan(
        mapping,
        report(inventory, {c.canonical_task_id: revise("primary") for c in cases[:2]}),
        inventory,
    )
    return plan, cases


def test_primary_feedback_contains_all_ten_actual_easy_names_once(evidence):
    plan, cases = evidence
    feedback = build_difficulty_revision_feedback(plan)
    assert len(feedback.easy_controls) == 1
    group = feedback.easy_controls[0]
    assert group.family == "person_names"
    assert len(group.controls) == 10
    expected = {e.canonical_task_id: e for e in cases[0].easy_comparators}
    assert {c.canonical_task_id for c in group.controls} == set(expected)
    for control in group.controls:
        source = expected[control.canonical_task_id]
        assert control.native_value == source.native_value
        assert control.romanized_value == source.romanized_value
        assert control.native_caller_payload == source.native_caller_payload
        assert control.romanized_caller_payload == source.romanized_caller_payload
        assert control.native_decoys == source.native_dependencies.decoys
        assert control.source_evidence_sha256 == content_digest(source)
    assert feedback.plan_sha256 == content_digest(plan)
    assert all(f.family == "person_names" for f in feedback.findings)
    assert "callee_full_name" not in group.model_dump_json()
    assert "org_name" not in group.model_dump_json()
    assert len(plan.easy_controls) == 10


def test_controls_reject_stale_content_and_missing_feedback_family(evidence):
    plan, _ = evidence
    raw = plan.model_dump(mode="json")
    raw["easy_controls"][0]["controls"][0]["native_value"] += "stale"
    with pytest.raises(ValueError, match="digest"):
        DifficultyRevisionPlan.model_validate(raw)
    feedback = build_difficulty_revision_feedback(plan).model_dump(mode="json")
    feedback["easy_controls"] = []
    with pytest.raises(ValueError, match="control"):
        DifficultyRevisionFeedback.model_validate(feedback)


def test_feedback_rejects_duplicate_or_incorrect_family_controls(evidence):
    plan, _ = evidence
    feedback = build_difficulty_revision_feedback(plan).model_dump(mode="json")
    feedback["easy_controls"] *= 2
    with pytest.raises(ValueError, match="control"):
        DifficultyRevisionFeedback.model_validate(feedback)
    raw = plan.model_dump(mode="json")
    raw["easy_controls"].pop()
    with pytest.raises(ValueError, match="control"):
        DifficultyRevisionPlan.model_validate(raw)
