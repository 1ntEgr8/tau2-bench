"""Easy repairs are exact-audit scoped and cannot mutate Hard dependencies."""

import re
from pathlib import Path

import pytest

from tau2.multilingual.factory import intake_easy_revision as bridge
from tau2.multilingual.factory.database_localization import _relevant_mappings, _stage
from tau2.multilingual.factory.intake_easy_audit import (
    EasyAuditReport,
    EasyAuditRequest,
    EasyDecision,
    extract_easy_cases,
)
from tau2.multilingual.factory.intake_easy_revision import (
    build_easy_revision_feedback,
    build_easy_revision_plan,
    validate_easy_revision,
)
from tau2.utils.artifacts import content_digest, write_artifact
from tests.test_multilingual import test_intake_difficulty_revision as fixtures

evidence = fixtures.evidence


def easy_report(inventory, changes=None):
    cases = extract_easy_cases(inventory)
    decisions = []
    for case in cases:
        data = dict(
            canonical_task_id=case.canonical_task_id,
            disposition="keep",
            source_difficulty="easy",
            localized_difficulty="easy",
            source_assessment="Synthetic source assessment.",
            ease_assessment="Synthetic local assessment.",
            evidence="Synthetic test evidence.",
            naturalness=True,
            known_answer_recoverable=True,
            source_semantics_preserved=True,
            dependency_issues=[],
            correction_instructions=[],
            repair_targets=[],
            source_blocked_reason=None,
            native_review_required=True,
        )
        data.update((changes or {}).get(case.canonical_task_id, {}))
        decisions.append(EasyDecision(**data))
    return EasyAuditReport(
        request=EasyAuditRequest(
            map_file=Path("map.json"),
            revision="audit-test",
            model="test",
            model_args={"reasoning_effort": "low"},
        ),
        prompt_version="test",
        map_sha256=inventory.map_sha256,
        source_sha256=inventory.source_sha256,
        inventory_sha256=content_digest(inventory),
        settings=inventory.settings,
        expected_task_ids=[case.canonical_task_id for case in cases],
        decisions=decisions,
        errors=[],
        keep_count=sum(d.disposition == "keep" for d in decisions),
        revise_count=sum(d.disposition == "revise" for d in decisions),
        source_blocked_count=sum(d.disposition == "source_blocked" for d in decisions),
        error_count=0,
        machine_gate_passed=all(d.disposition == "keep" for d in decisions),
    )


def repair(*targets):
    return dict(
        disposition="revise",
        repair_targets=list(targets),
        correction_instructions=[
            "Fix the exact local speech error; retain source semantics."
        ],
    )


def test_stale_easy_audit_rejected(evidence):
    proposal, inventory = evidence
    audit = easy_report(inventory)
    audit.map_sha256 = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        build_easy_revision_plan(proposal, audit, inventory)


def test_easy_primary_closes_dependencies_and_preserves_hard(evidence):
    proposal, inventory = evidence
    case = next(c for c in extract_easy_cases(inventory) if c.family == "person_names")
    plan = build_easy_revision_plan(
        proposal,
        easy_report(inventory, {case.canonical_task_id: repair("primary")}),
        inventory,
    )
    assert case.measured_entry_id in plan.redraft_ids
    assert set(case.caller_entry_ids) <= set(plan.redraft_ids)
    assert len(plan.redraft_ids) > 1
    hard_ids = {
        identifier for row in inventory.cases for identifier in row.entry_sha256
    }
    assert not hard_ids.intersection(plan.redraft_ids)
    assert len(plan.cases) == 100
    feedback = build_easy_revision_feedback(plan)
    assert feedback.plan_sha256 == content_digest(plan)
    assert feedback.redraft_ids == plan.redraft_ids
    assert {row.canonical_task_id for row in feedback.evidence} == set(
        plan.affected_easy_task_ids
    )
    assert "simpl" in feedback.instructions.lower()


def test_dependency_only_does_not_select_primary(evidence):
    proposal, inventory = evidence
    case = next(c for c in extract_easy_cases(inventory) if c.family == "person_names")
    plan = build_easy_revision_plan(
        proposal,
        easy_report(inventory, {case.canonical_task_id: repair("decoys")}),
        inventory,
    )
    assert plan.redraft_ids
    assert case.measured_entry_id not in plan.redraft_ids


@pytest.mark.parametrize("shared_tier", ["hard", "easy"])
def test_shared_ownership_blocks_whole_primary_group(
    evidence, monkeypatch, shared_tier
):
    proposal, inventory = evidence
    cases = extract_easy_cases(inventory)
    case = next(c for c in cases if c.family == "person_names")
    actual_owners = bridge._entry_owners(proposal, inventory)
    other = (inventory.cases if shared_tier == "hard" else cases)[-1]
    dependency = next(
        key
        for key in case.dependency_entry_ids
        if any(
            point.document_id == "intake_localization" and point.path[-1] == "decoys"
            for point in next(
                entry for entry in proposal.entries if entry.id == key
            ).occurrences
        )
    )
    actual_owners[dependency].add(other.canonical_task_id)
    monkeypatch.setattr(bridge, "_entry_owners", lambda *_: actual_owners)
    plan = build_easy_revision_plan(
        proposal,
        easy_report(inventory, {case.canonical_task_id: repair("primary")}),
        inventory,
    )
    row = next(
        row for row in plan.cases if row.canonical_task_id == case.canonical_task_id
    )
    assert row.status == "blocked" and not row.redraft_ids
    assert any(other.canonical_task_id in blocker for blocker in row.blockers)
    scope = next(scope for scope in plan.entry_scopes if scope.entry_id == dependency)
    assert other.canonical_task_id in getattr(scope, f"{shared_tier}_task_ids")


def test_all_easy_primaries_precede_dependencies_and_are_available_as_context(evidence):
    proposal, inventory = evidence
    entries = {entry.id: entry for entry in proposal.entries}
    for case in extract_easy_cases(inventory):
        primary = entries[case.measured_entry_id]
        dependencies = [
            entries[key]
            for key in case.dependency_entry_ids
            if key not in case.caller_entry_ids
        ]
        assert all(_stage(primary) < _stage(entry) for entry in dependencies)
        if primary.category not in {"protocol", "date_time", "money", "tool_text"}:
            for dependency in dependencies:
                if dependency.source not in ("", []):
                    context = _relevant_mappings(
                        [dependency], [primary], source=proposal.source
                    )
                    assert primary in context, case.canonical_task_id


def test_source_blocker_and_immutable_time_remain_explicit(evidence):
    proposal, inventory = evidence
    cases = extract_easy_cases(inventory)
    time = next(c for c in cases if c.family == "times")
    med = next(c for c in cases if c.family == "medications")
    plan = build_easy_revision_plan(
        proposal,
        easy_report(
            inventory,
            {
                time.canonical_task_id: repair("primary"),
                med.canonical_task_id: dict(
                    disposition="source_blocked",
                    source_blocked_reason="Frozen source medicine prevents the requested simplification.",
                ),
            },
        ),
        inventory,
    )
    rows = {row.canonical_task_id: row for row in plan.cases}
    assert rows[time.canonical_task_id].status == "blocked"
    assert rows[time.canonical_task_id].blockers
    assert rows[med.canonical_task_id].status == "source_blocked"
    assert not plan.redraft_ids


def test_source_blocker_retains_separate_dependency_repair(evidence):
    proposal, inventory = evidence
    case = next(c for c in extract_easy_cases(inventory) if c.family == "times")
    plan = build_easy_revision_plan(
        proposal,
        easy_report(
            inventory,
            {
                case.canonical_task_id: {
                    **repair("spelling_styles", "primary"),
                    "disposition": "source_blocked",
                    "source_blocked_reason": "Source time must stay fixed; a separate style conflicts with initial guidance.",
                }
            },
        ),
        inventory,
    )
    row = next(
        row for row in plan.cases if row.canonical_task_id == case.canonical_task_id
    )
    assert row.status == "partial" and row.redraft_ids
    assert case.measured_entry_id not in row.redraft_ids
    assert row.decision.source_blocked_reason in row.blockers
    assert row.decision.disposition == "source_blocked"


def test_factual_review_can_repair_keep_without_overwriting_audit(evidence):
    proposal, inventory = evidence
    case = next(c for c in extract_easy_cases(inventory) if c.family == "times")
    audit = easy_report(inventory)
    before = audit.model_dump()
    supplement = bridge.EasyReviewFeedback(
        canonical_task_id=case.canonical_task_id,
        map_sha256=content_digest(proposal),
        repair_targets=["spelling_styles"],
        concern="Synthetic exact style review: unconditional period speech conflicts with the active bare-initial guidance.",
    )
    plan = build_easy_revision_plan(
        proposal, audit, inventory, review_feedback=[supplement]
    )
    row = next(
        row for row in plan.cases if row.canonical_task_id == case.canonical_task_id
    )
    assert row.status == "planned" and row.redraft_ids
    assert row.decision.disposition == "keep" and row.review_feedback == supplement
    assert audit.model_dump() == before
    assert build_easy_revision_feedback(plan).findings[0].review_feedback == supplement
    assert case.measured_entry_id not in plan.redraft_ids


@pytest.mark.parametrize("defect", ["stale", "duplicate", "unknown"])
def test_review_feedback_requires_exact_map_and_task_scope(evidence, defect):
    proposal, inventory = evidence
    case = extract_easy_cases(inventory)[0]
    review = bridge.EasyReviewFeedback(
        canonical_task_id=case.canonical_task_id,
        map_sha256=content_digest(proposal),
        repair_targets=["decoys"],
        concern="Synthetic exact-source review.",
    )
    if defect == "stale":
        review.map_sha256 = "0" * 64
    elif defect == "unknown":
        review.canonical_task_id = "intake_person_names_easy_99"
    reviews = [review, review] if defect == "duplicate" else [review]
    with pytest.raises(ValueError, match="duplicate, stale or outside"):
        build_easy_revision_plan(
            proposal, easy_report(inventory), inventory, review_feedback=reviews
        )


@pytest.mark.parametrize("family", ["dates", "times"])
def test_compiler_derived_omission_repair_is_explicitly_blocked(evidence, family):
    proposal, inventory = evidence
    case = next(c for c in extract_easy_cases(inventory) if c.family == family)
    plan = build_easy_revision_plan(
        proposal,
        easy_report(
            inventory, {case.canonical_task_id: repair("omission_instruction")}
        ),
        inventory,
    )
    row = next(
        row for row in plan.cases if row.canonical_task_id == case.canonical_task_id
    )
    assert row.status == "blocked" and not plan.redraft_ids
    assert any("compiler-derived" in text for text in row.blockers)


@pytest.mark.parametrize("change", ["hard_entry", "caller_surface", "metadata"])
def test_candidate_rejects_changes_outside_easy_scope(evidence, change):
    proposal, inventory = evidence
    case = next(c for c in extract_easy_cases(inventory) if c.family == "person_names")
    plan = build_easy_revision_plan(
        proposal,
        easy_report(inventory, {case.canonical_task_id: repair("decoys")}),
        inventory,
    )
    candidate = proposal.model_copy(deep=True)
    if change == "hard_entry":
        target = inventory.cases[0].measured_entry_id
        next(
            entry for entry in candidate.entries if entry.id == target
        ).notes += " Unselected."
    elif change == "caller_surface":
        candidate.intake_caller_surfaces[0].rationale += " Unselected."
    else:
        candidate.status = "approved"
    with pytest.raises(ValueError, match="unselected|metadata|caller"):
        validate_easy_revision(proposal, candidate, plan)


@pytest.mark.parametrize("defect", ["dose", "phone_length", "drug_identity"])
def test_candidate_rejects_new_source_invariant_failures(evidence, defect):
    proposal, inventory = evidence
    family = "phones" if defect == "phone_length" else "medications"
    case = next(c for c in extract_easy_cases(inventory) if c.family == family)
    plan = build_easy_revision_plan(
        proposal,
        easy_report(inventory, {case.canonical_task_id: repair("decoys")}),
        inventory,
    )
    candidate = proposal.model_copy(deep=True)
    entry = next(entry for entry in candidate.entries if entry.id in plan.redraft_ids)
    if defect == "phone_length":
        entry.forms.native[0] += "9"
    elif defect == "dose":
        entry.forms.native[0] = re.sub(r"\d+", "999", entry.forms.native[0], count=1)
    else:
        report = bridge.medication_identity_report(proposal)
        decoy = next(
            row
            for row in report.observations
            if row.entry_id == entry.id and row.representation == "native"
        )
        other = next(
            row
            for row in report.observations
            if row.role == "primary"
            and row.representation == "native"
            and row.source_label != decoy.source_label
        )
        entry.forms.native[0] = re.sub(
            r"^[^0-9]+", other.current_label + " ", entry.forms.native[0]
        )
    with pytest.raises(ValueError, match="invariant|identity|strength|decoy"):
        validate_easy_revision(proposal, candidate, plan)


def test_bridge_generation_and_resume_are_bound_to_exact_safe_plan(
    evidence, monkeypatch, tmp_path
):
    proposal, inventory = evidence
    case = next(c for c in extract_easy_cases(inventory) if c.family == "person_names")
    audit = easy_report(inventory, {case.canonical_task_id: repair("decoys")})
    audit_file = tmp_path / "audit.json"
    write_artifact(audit, audit_file)
    request = bridge.EasyRevisionRequest(
        map_file=Path("input.json"),
        audit_file=audit_file,
        revision="easy-repair-test",
        model="test",
        model_args={"reasoning_effort": "low"},
        batch_size=1,
        max_concurrency=1,
    )
    original_reader = bridge.read_localization_map
    monkeypatch.setattr(
        bridge,
        "read_localization_map",
        lambda path: proposal if path == request.map_file else original_reader(path),
    )
    monkeypatch.setattr(bridge, "build_difficulty_inventory", lambda _: inventory)
    calls = []

    def generate(revision, output, checkpoints):
        calls.append(revision)
        feedback = bridge.EasyRevisionFeedback.model_validate_json(
            revision.feedback[0].content
        )
        assert feedback.redraft_ids == revision.redraft_ids
        assert feedback.map_sha256 == content_digest(proposal)
        assert all(row.tier == "easy" for row in feedback.evidence)
        candidate = proposal.model_copy(deep=True)
        candidate.revision = revision.request.revision
        candidate.provenance = candidate.provenance.model_copy(
            update={
                "input_sha256": content_digest(revision),
                "parent_map_sha256": content_digest(proposal),
            }
        )
        next(
            entry for entry in candidate.entries if entry.id in revision.redraft_ids
        ).notes += " Synthetic mocked repair review."
        return candidate

    monkeypatch.setattr(bridge, "revise_database_localization", generate)
    output = tmp_path / "repaired.json"
    checkpoints = tmp_path / "checkpoints"
    result = bridge.revise_intake_easy(request, output, checkpoints)
    assert (
        len(calls) == 1
        and result.intake_caller_surfaces == proposal.intake_caller_surfaces
    )
    assert bridge.revise_intake_easy(request, output, checkpoints) == result
    assert len(calls) == 1
    (checkpoints / "completed.json").unlink()
    with pytest.raises(ValueError, match="completed repair record"):
        bridge.revise_intake_easy(request, output, checkpoints)
    assert len(calls) == 1
