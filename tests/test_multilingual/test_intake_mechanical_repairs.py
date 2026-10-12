"""Clinical decoy invariants are repaired without rewriting model difficulty evidence."""

import re
from decimal import Decimal
from pathlib import Path

import pytest

from tau2.multilingual.factory.intake_difficulty_inventory import (
    build_difficulty_inventory,
)
from tau2.multilingual.factory.intake_difficulty_revision import (
    DifficultyMechanicalIssue,
    build_difficulty_revision_feedback,
    build_difficulty_revision_plan,
    medication_decoy_issues,
)
from tau2.multilingual.factory.localization_artifact import read_localization_map
from tau2.utils.utils import DATA_DIR
from tests.test_multilingual.test_intake_difficulty_revision import report


@pytest.fixture(scope="module")
def evidence():
    proposal = read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/hi/localization/intake.json"
    )
    return proposal, build_difficulty_inventory(proposal)


def bad_medication_decoy(proposal, tier="hard"):
    changed = proposal.model_copy(deep=True)
    data = next(
        d.content for d in changed.source.documents if d.id == "intake_localization"
    )
    index = next(
        i
        for i, row in enumerate(data["stimuli"])
        if row["family"] == "medications" and row["tier"] == tier
    )
    row = data["stimuli"][index]
    entry = next(
        e
        for e in changed.entries
        if any(
            o.document_id == "intake_localization"
            and o.path == ["stimuli", index, "decoys"]
            for o in e.occurrences
        )
    )
    for representation in ("native", "romanized"):
        values = getattr(entry.forms, representation)
        original = values[0]
        values[0] = re.sub(
            r"([0-9]+(?:[.,][0-9]+)?)(\s*(?:mg|mcg|g|mL|ml|%))",
            lambda match: str(Decimal(match[1].replace(",", ".")) + 1) + match[2],
            original,
            count=1,
        )
        assert values[0] != original
    return changed, row["canonical_task_id"], entry.id


@pytest.mark.parametrize(
    "disposition,expected_status", [("keep", "planned"), ("source_blocked", "partial")]
)
def test_mechanical_dose_error_overrides_keep_or_blocked_repair_scope_without_rewriting_audit(
    evidence, disposition, expected_status
):
    proposal, _ = evidence
    proposal, task_id, decoy_id = bad_medication_decoy(proposal)
    inventory = build_difficulty_inventory(proposal)
    changes = (
        {}
        if disposition == "keep"
        else {
            task_id: {
                "disposition": "source_blocked",
                "repair_targets": ["primary", "decoys", "pronunciations"],
                "source_blocked_reason": "Changing the primary drug or dose cannot establish Hard naturally.",
            }
        }
    )
    audit = report(inventory, changes)
    original = next(
        d for d in audit.decisions if d.canonical_task_id == task_id
    ).model_dump()
    plan = build_difficulty_revision_plan(proposal, audit, inventory)
    case = next(c for c in plan.cases if c.canonical_task_id == task_id)
    assert case.status == expected_status
    assert case.redraft_ids == [decoy_id]
    assert plan.redraft_ids == [decoy_id]
    assert case.decision.model_dump() == original
    assert len(case.mechanical_issues) == 2
    assert all(
        issue.entry_id == decoy_id and issue.source_strengths != issue.current_strengths
        for issue in case.mechanical_issues
    )
    if disposition == "source_blocked":
        assert original["source_blocked_reason"] in case.blockers
    feedback = build_difficulty_revision_feedback(plan)
    assert feedback.findings[0].decision.model_dump() == original
    assert len(feedback.findings[0].mechanical_issues) == 2
    assert len(feedback.easy_controls) == 1
    assert feedback.easy_controls[0].family == "medications"
    assert len(feedback.easy_controls[0].controls) == 10


def test_valid_source_blocked_medication_is_not_rewritten(evidence):
    proposal, inventory = evidence
    case = next(c for c in inventory.cases if c.family == "medications")
    audit = report(
        inventory,
        {
            case.canonical_task_id: {
                "disposition": "source_blocked",
                "source_blocked_reason": "Clinical identity is fixed.",
            }
        },
    )
    plan = build_difficulty_revision_plan(proposal, audit, inventory)
    case = next(
        c
        for c in plan.cases
        if c.canonical_task_id == case.canonical_task_id and c.family == "medications"
    )
    assert case.status == "source_blocked"
    assert not case.mechanical_issues
    assert not plan.redraft_ids


def test_easy_medication_dose_failure_is_reported_but_never_selected(evidence):
    proposal, _ = evidence
    proposal, task_id, decoy_id = bad_medication_decoy(proposal, "easy")
    inventory = build_difficulty_inventory(proposal)
    plan = build_difficulty_revision_plan(proposal, report(inventory), inventory)
    assert decoy_id not in plan.redraft_ids
    assert len(plan.protected_mechanical_issues) == 2
    assert all(
        issue.canonical_task_id == task_id
        and issue.entry_id == decoy_id
        and issue.tier == "easy"
        for issue in plan.protected_mechanical_issues
    )
    assert not build_difficulty_revision_feedback(plan).findings


def test_mechanical_evidence_rejects_invented_source_strength(evidence):
    proposal, _ = evidence
    proposal, _, _ = bad_medication_decoy(proposal)
    issue = medication_decoy_issues(proposal)[0].model_dump()
    issue["source_strengths"][0]["amount"] += 1
    with pytest.raises(ValueError, match="differs from its written value"):
        DifficultyMechanicalIssue.model_validate(issue)


def test_mechanical_repair_respects_donor_scope_guard(evidence):
    from tau2.multilingual.factory.intake_difficulty_inventory import (
        DifficultyInventory,
        DifficultyInventoryPayload,
    )
    from tau2.multilingual.localization import EntityReuseProvenance
    from tau2.utils.artifacts import ReviewApproval, content_digest

    proposal, _ = evidence
    proposal, task_id, decoy_id = bad_medication_decoy(proposal)
    inventory = build_difficulty_inventory(proposal)
    entry = next(entry for entry in proposal.entries if entry.id == decoy_id)
    entry.reuse = EntityReuseProvenance(
        source_domain="retail",
        source_entry_ids=["private-entry"],
        source_artifact_sha256="0" * 64,
        approval=ReviewApproval(content_sha256="0" * 64, reviewed_by="test"),
    )
    inventory.map_sha256 = content_digest(proposal)
    for case in inventory.cases:
        if decoy_id in case.entry_sha256:
            case.entry_sha256[decoy_id] = content_digest(entry)
    payload = DifficultyInventoryPayload.model_validate(
        inventory.model_dump(exclude={"payload_sha256"})
    )
    inventory = DifficultyInventory(
        **payload.model_dump(), payload_sha256=content_digest(payload)
    )
    plan = build_difficulty_revision_plan(proposal, report(inventory), inventory)
    case = next(case for case in plan.cases if case.canonical_task_id == task_id)
    assert case.status == "blocked"
    assert not plan.redraft_ids
    assert any("Donor-sourced" in blocker for blocker in case.blockers)
    assert not build_difficulty_revision_feedback(plan).findings
