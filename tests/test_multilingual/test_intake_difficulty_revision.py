"""Difficulty repair scope preserves Easy controls and immutable capture gold."""

import csv
from pathlib import Path

import pytest

from tau2.multilingual.factory.intake_difficulty_audit import (
    DifficultyAuditReport,
    DifficultyAuditRequest,
)
from tau2.multilingual.factory.intake_difficulty_inventory import (
    build_difficulty_inventory,
)
from tau2.multilingual.factory.intake_difficulty_revision import (
    build_difficulty_revision_feedback,
    build_difficulty_revision_plan,
    make_difficulty_revision_command,
    write_difficulty_revision_artifacts,
)
from tau2.multilingual.factory.localization_artifact import read_localization_map
from tau2.utils.artifacts import content_digest
from tau2.utils.utils import DATA_DIR
from tests.test_multilingual.test_intake_difficulty_audit import decision


@pytest.fixture(scope="module")
def evidence():
    proposal = read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/hi/localization/intake.json"
    )
    inventory = build_difficulty_inventory(proposal)
    return proposal, inventory


def report(inventory, changes=None):
    decisions = [decision(case) for case in inventory.cases]
    for value in decisions:
        if changes and value.canonical_task_id in changes:
            for key, content in changes[value.canonical_task_id].items():
                setattr(value, key, content)
    return DifficultyAuditReport(
        request=DifficultyAuditRequest(
            map_file=Path("map.json"),
            revision="repair-v1",
            model="test",
            model_args={"reasoning_effort": "low"},
        ),
        prompt_version="test-v1",
        map_sha256=inventory.map_sha256,
        source_sha256=inventory.source_sha256,
        inventory_sha256=content_digest(inventory),
        settings=inventory.settings,
        expected_task_ids=[case.canonical_task_id for case in inventory.cases],
        decisions=decisions,
        errors=[],
        keep_count=sum(d.disposition == "keep" for d in decisions),
        revise_count=sum(d.disposition == "revise" for d in decisions),
        source_blocked_count=sum(d.disposition == "source_blocked" for d in decisions),
        error_count=0,
        machine_gate_passed=all(d.disposition == "keep" for d in decisions),
    )


def revise(*targets):
    return dict(
        disposition="revise",
        repair_targets=list(targets),
        correction_instructions=[
            "Restore the measured native capture challenge without changing source semantics."
        ],
    )


def test_exact_map_and_audit_binding_rejects_stale_input(evidence):
    proposal, inventory = evidence
    audit = report(inventory)
    audit.map_sha256 = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        build_difficulty_revision_plan(proposal, audit, inventory)


def test_primary_replacement_closes_only_mutable_used_dependencies(evidence):
    proposal, inventory = evidence
    case = next(c for c in inventory.cases if c.family == "person_names")
    plan = build_difficulty_revision_plan(
        proposal,
        report(inventory, {case.canonical_task_id: revise("primary")}),
        inventory,
    )
    assert case.measured_entry_id in plan.redraft_ids
    assert len(plan.redraft_ids) > 1
    easy_ids = {
        identifier
        for c in inventory.cases
        for e in c.easy_comparators
        for identifier in e.entry_sha256
    }
    assert not set(plan.redraft_ids) & easy_ids
    assert len(plan.cases) == 100
    assert (
        next(
            c for c in plan.cases if c.canonical_task_id == case.canonical_task_id
        ).status
        == "planned"
    )


def test_omission_only_repair_does_not_select_primary(evidence):
    proposal, inventory = evidence
    case = next(c for c in inventory.cases if c.family == "person_names")
    plan = build_difficulty_revision_plan(
        proposal,
        report(inventory, {case.canonical_task_id: revise("omission_instruction")}),
        inventory,
    )
    assert len(plan.redraft_ids) == 1
    assert case.measured_entry_id not in plan.redraft_ids


def test_dates_revise_relative_caller_phrase_without_canonical_gold(evidence):
    proposal, inventory = evidence
    case = next(c for c in inventory.cases if c.family == "dates")
    plan = build_difficulty_revision_plan(
        proposal,
        report(inventory, {case.canonical_task_id: revise("primary")}),
        inventory,
    )
    assert set(case.caller_entry_ids) <= set(plan.redraft_ids)
    assert case.measured_entry_id not in plan.redraft_ids


def test_time_primary_and_caller_repairs_route_to_task_surface_only(evidence):
    proposal, inventory = evidence
    case = next(c for c in inventory.cases if c.family == "times")
    plan = build_difficulty_revision_plan(
        proposal,
        report(
            inventory, {case.canonical_task_id: revise("primary", "caller_surface")}
        ),
        inventory,
    )
    assert plan.caller_surface_task_ids == [case.canonical_task_id]
    assert case.measured_entry_id not in plan.redraft_ids
    assert not plan.redraft_ids
    assert (
        make_difficulty_revision_command(
            plan, report(inventory).request, Path("feedback.json")
        )
        is None
    )


def test_absent_dependency_repair_is_explicitly_blocked(evidence):
    proposal, inventory = evidence
    case = next(c for c in inventory.cases if c.family == "dates")
    plan = build_difficulty_revision_plan(
        proposal,
        report(inventory, {case.canonical_task_id: revise("pronunciations")}),
        inventory,
    )
    result = next(
        c for c in plan.cases if c.canonical_task_id == case.canonical_task_id
    )
    assert result.status == "blocked" and result.blockers
    assert not plan.redraft_ids


def test_feedback_and_command_are_scoped_to_original_audit(evidence, tmp_path):
    proposal, inventory = evidence
    case = next(c for c in inventory.cases if c.family == "coined")
    audit = report(inventory, {case.canonical_task_id: revise("primary")})
    plan = build_difficulty_revision_plan(proposal, audit, inventory)
    feedback = build_difficulty_revision_feedback(plan)
    assert len(feedback.findings) == 1
    assert feedback.findings[0].canonical_task_id == case.canonical_task_id
    files = write_difficulty_revision_artifacts(plan, tmp_path / "plan.json")
    command = make_difficulty_revision_command(plan, audit.request, files.feedback_file)
    assert command.redraft_ids == plan.redraft_ids
    assert command.feedback_files == [files.feedback_file.resolve()]
    rows = list(csv.DictReader(files.ledger_csv.open()))
    assert len(rows) == 100
    assert {row["Family"] for row in rows} == {c.family for c in inventory.cases}
    assert all(row["Native review"] == "pending" for row in rows)
    before = files.ledger_csv.read_bytes()
    write_difficulty_revision_artifacts(plan, tmp_path / "plan.json")
    assert files.ledger_csv.read_bytes() == before


@pytest.mark.parametrize("family", ["dates", "times"])
def test_temporal_source_blocker_is_visible_during_conservative_repair(
    evidence, family
):
    proposal, inventory = evidence
    case = next(c for c in inventory.cases if c.family == family)
    audit = report(
        inventory,
        {
            case.canonical_task_id: {
                "disposition": "source_blocked",
                "source_blocked_reason": "Current source wording appears too simple.",
            }
        },
    )
    plan = build_difficulty_revision_plan(proposal, audit, inventory)
    result = next(
        c for c in plan.cases if c.canonical_task_id == case.canonical_task_id
    )
    assert result.status == "planned"
    assert result.decision.disposition == "source_blocked"
    assert (
        result.decision.source_blocked_reason
        == "Current source wording appears too simple."
    )
    assert case.measured_entry_id not in plan.redraft_ids
    if family == "dates":
        assert set(case.caller_entry_ids) <= set(plan.redraft_ids)
    else:
        assert plan.caller_surface_task_ids == [case.canonical_task_id]


def test_immutable_code_primary_is_blocked_without_disappearing(evidence):
    proposal, inventory = evidence
    case = next(c for c in inventory.cases if c.field_name in {"vin", "member_id"})
    plan = build_difficulty_revision_plan(
        proposal,
        report(inventory, {case.canonical_task_id: revise("primary")}),
        inventory,
    )
    result = next(
        c for c in plan.cases if c.canonical_task_id == case.canonical_task_id
    )
    assert result.status == "blocked"
    assert "immutable" in result.blockers[0]
    assert not plan.redraft_ids


def test_empty_style_noops_and_phone_protocols_are_explicit_feedback_constraints(
    evidence,
):
    proposal, inventory = evidence
    plan = build_difficulty_revision_plan(proposal, report(inventory), inventory)
    instructions = build_difficulty_revision_feedback(plan).instructions
    assert "no-op slots" in instructions
    assert "ext., x and +" in instructions
    assert "Do not force English oh, double or triple" in instructions


def test_absent_dependency_keeps_supported_primary_repair_visible_as_partial(evidence):
    proposal, inventory = evidence
    case = next(c for c in inventory.cases if c.family == "person_names")
    assert not case.source_dependencies.spelling_styles
    plan = build_difficulty_revision_plan(
        proposal,
        report(
            inventory, {case.canonical_task_id: revise("primary", "spelling_styles")}
        ),
        inventory,
    )
    result = next(
        c for c in plan.cases if c.canonical_task_id == case.canonical_task_id
    )
    assert result.status == "partial"
    assert result.blockers
    assert case.measured_entry_id in result.redraft_ids
    assert (
        build_difficulty_revision_feedback(plan).findings[0].redraft_ids
        == result.redraft_ids
    )


def test_unsupported_time_dependency_does_not_hide_supported_surface_repair(evidence):
    proposal, inventory = evidence
    case = next(c for c in inventory.cases if c.family == "times")
    plan = build_difficulty_revision_plan(
        proposal,
        report(
            inventory,
            {case.canonical_task_id: revise("caller_surface", "pronunciations")},
        ),
        inventory,
    )
    result = next(
        c for c in plan.cases if c.canonical_task_id == case.canonical_task_id
    )
    assert result.status == "partial" and result.blockers
    assert plan.caller_surface_task_ids == [case.canonical_task_id]
    assert not plan.redraft_ids


def test_unsafe_dependency_blocks_its_whole_primary_group_without_losing_other_cases(
    evidence,
):
    from tau2.multilingual.factory.intake_difficulty_inventory import (
        DifficultyInventory,
        DifficultyInventoryPayload,
    )
    from tau2.multilingual.localization import EntityReuseProvenance
    from tau2.utils.artifacts import ReviewApproval

    proposal, inventory = evidence
    proposal = proposal.model_copy(deep=True)
    inventory = inventory.model_copy(deep=True)
    case = next(c for c in inventory.cases if c.family == "person_names")
    second = next(c for c in inventory.cases if c.family == "coined")
    index = next(
        i
        for d in proposal.source.documents
        if d.id == "intake_localization"
        for i, row in enumerate(d.content["stimuli"])
        if row["canonical_task_id"] == case.canonical_task_id
    )
    dependency = next(
        entry
        for entry in proposal.entries
        if entry.id in case.dependency_entry_ids
        and any(
            o.document_id == "intake_localization"
            and o.path == ["stimuli", index, "pronunciations"]
            for o in entry.occurrences
        )
    )
    dependency.reuse = EntityReuseProvenance(
        source_domain="retail",
        source_entry_ids=["private-entry"],
        source_artifact_sha256="0" * 64,
        approval=ReviewApproval(content_sha256="0" * 64, reviewed_by="test"),
    )
    inventory.map_sha256 = content_digest(proposal)
    for value in inventory.cases:
        if dependency.id in value.entry_sha256:
            value.entry_sha256[dependency.id] = content_digest(dependency)
    payload = DifficultyInventoryPayload.model_validate(
        inventory.model_dump(exclude={"payload_sha256"})
    )
    inventory = DifficultyInventory(
        **payload.model_dump(), payload_sha256=content_digest(payload)
    )
    audit = report(
        inventory,
        {
            case.canonical_task_id: revise("pronunciations", "primary"),
            second.canonical_task_id: revise("primary"),
        },
    )
    plan = build_difficulty_revision_plan(proposal, audit, inventory)
    result = next(
        c for c in plan.cases if c.canonical_task_id == case.canonical_task_id
    )
    assert result.status == "blocked"
    assert case.measured_entry_id not in plan.redraft_ids
    assert dependency.id not in plan.redraft_ids
    assert second.measured_entry_id in plan.redraft_ids
    assert any("blocked together" in blocker for blocker in result.blockers)


@pytest.fixture
def five_plans(evidence):
    import hashlib

    proposal, inventory = evidence
    base = build_difficulty_revision_plan(proposal, report(inventory), inventory)
    plans = []
    for language in ("hi", "es", "pt", "zh", "ko"):
        plan = base.model_copy(deep=True)
        plan.settings.language = language
        plan.map_sha256 = hashlib.sha256(f"map-{language}".encode()).hexdigest()
        plan.audit_sha256 = hashlib.sha256(f"audit-{language}".encode()).hexdigest()
        plans.append(plan)
    return plans


def test_aggregate_covers_all_five_languages_and_500_cases(five_plans, tmp_path):
    from tau2.multilingual.factory.intake_difficulty_revision import (
        build_difficulty_aggregate,
        write_difficulty_aggregate,
    )

    summary = build_difficulty_aggregate(five_plans)
    assert len(summary.rows) == 500
    assert len(summary.counts) == 50
    assert all(cell.keep == 10 for cell in summary.counts)
    assert {item.language for item in summary.inputs} == {"hi", "es", "pt", "zh", "ko"}
    assert summary.native_review_status == "pending"
    assert all(row.native_review == "pending" for row in summary.rows)
    files = write_difficulty_aggregate(summary, tmp_path / "summary.json")
    rows = list(csv.DictReader(files.ledger_csv.open()))
    assert len(rows) == 500
    assert len({(row["Language"], row["Task"]) for row in rows}) == 500
    write_difficulty_aggregate(summary, tmp_path / "summary.json")


@pytest.mark.parametrize(
    "problem",
    ["missing", "duplicate", "unknown", "source_digest", "task_frame", "source_value"],
)
def test_aggregate_rejects_incomplete_languages_and_mixed_source_frames(
    five_plans, problem
):
    from tau2.multilingual.factory.intake_difficulty_revision import (
        build_difficulty_aggregate,
    )

    if problem == "missing":
        five_plans.pop()
    elif problem == "duplicate":
        five_plans[-1] = five_plans[0]
    elif problem == "unknown":
        five_plans[-1].settings.language = "en"
    elif problem == "source_digest":
        five_plans[-1].source_sha256 = "f" * 64
    elif problem == "task_frame":
        five_plans[-1].cases[0].canonical_task_id = "intake_person_names_hard_99"
        five_plans[-1].cases[
            0
        ].decision.canonical_task_id = "intake_person_names_hard_99"
    else:
        five_plans[-1].cases[0].source_value = "A different canonical source"
    with pytest.raises(ValueError):
        build_difficulty_aggregate(five_plans)


def test_aggregate_counters_and_csv_digest_cannot_drift(five_plans):
    from tau2.multilingual.factory.intake_difficulty_revision import (
        DifficultyAggregateSummary,
        build_difficulty_aggregate,
    )

    summary = build_difficulty_aggregate(five_plans)
    payload = summary.model_dump()
    payload["counts"][0]["keep"] = 0
    with pytest.raises(ValueError):
        DifficultyAggregateSummary.model_validate(payload)
    payload = summary.model_dump()
    payload["rows"][0]["native_value"] += " changed"
    with pytest.raises(ValueError, match="digest"):
        DifficultyAggregateSummary.model_validate(payload)


def test_aggregate_preserves_partial_and_source_blocked_rows(evidence, five_plans):
    from tau2.multilingual.factory.intake_difficulty_revision import (
        build_difficulty_aggregate,
    )

    proposal, inventory = evidence
    partial = next(c for c in inventory.cases if c.family == "person_names")
    blocked = next(c for c in inventory.cases if c.family == "medications")
    audit = report(
        inventory,
        {
            partial.canonical_task_id: revise("primary", "spelling_styles"),
            blocked.canonical_task_id: {
                "disposition": "source_blocked",
                "source_blocked_reason": "Changing the dose is forbidden.",
            },
        },
    )
    five_plans[0] = build_difficulty_revision_plan(proposal, audit, inventory)
    summary = build_difficulty_aggregate(five_plans)
    assert sum(cell.partial for cell in summary.counts) == 1
    assert sum(cell.source_blocked for cell in summary.counts) == 1
    rows = [row for row in summary.rows if row.language == "hi"]
    assert next(row for row in rows if row.task == partial.canonical_task_id).blockers
    assert (
        next(
            row for row in rows if row.task == blocked.canonical_task_id
        ).source_limitation
        == "Changing the dose is forbidden."
    )


def test_source_blocked_time_with_existing_authored_surface_is_not_retried(evidence):
    from tau2.domains.intake.time_display import display_time
    from tau2.multilingual.localization import IntakeCallerSurface

    proposal, original = evidence
    proposal = proposal.model_copy(deep=True)
    case = next(case for case in original.cases if case.family == "times")
    proposal.intake_caller_surfaces = [
        IntakeCallerSurface(
            canonical_task_id=case.canonical_task_id,
            canonical_value=case.source_value,
            native_spoken=display_time(case.source_value, "hi", include_period=False),
            romanized_spoken=display_time(
                case.source_value, "hi-latn", include_period=False
            ),
            rationale="Synthetic test surface already authored and independently reassessed.",
            provenance=proposal.provenance,
        )
    ]
    inventory = build_difficulty_inventory(proposal)
    audit = report(
        inventory,
        {
            case.canonical_task_id: {
                "disposition": "source_blocked",
                "source_blocked_reason": "The natural authored expression still does not distinguish Hard from Easy.",
            }
        },
    )
    plan = build_difficulty_revision_plan(proposal, audit, inventory)
    result = next(
        item for item in plan.cases if item.canonical_task_id == case.canonical_task_id
    )
    assert result.status == "source_blocked"
    assert result.decision.disposition == "source_blocked"
    assert result.blockers == [result.decision.source_blocked_reason]
    assert not result.caller_surface
    assert case.canonical_task_id not in plan.caller_surface_task_ids
    assert not result.redraft_ids
