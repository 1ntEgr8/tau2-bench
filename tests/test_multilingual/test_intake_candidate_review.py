"""Candidate audits retain valid entities and bind any repairs to source dependencies."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2.multilingual.factory import intake_candidate_review as review
from tau2.multilingual.factory.localization_artifact import read_localization_map
from tau2.utils.utils import DATA_DIR

REVIEW_FIXTURES = Path(__file__).parent / "fixtures/intake_review"


@pytest.fixture(scope="module")
def proposal():
    return read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/hi/localization/intake.json"
    )


def test_review_excludes_private_donors_and_selects_used_entities(proposal):
    rows = review.audit_rows(proposal)
    assert len(rows) == 193
    assert not (
        {row["id"] for row in rows} & {e.id for e in proposal.entries if e.reuse}
    )
    assert sum(row["role"] == "intake field:destination_property" for row in rows) == 20
    assert all("reuse" not in row for row in rows)


def test_revision_closes_dependencies_without_redrawing_retained_values(proposal):
    row = next(
        row
        for row in review.audit_rows(proposal)
        if row["role"] == "intake field:destination_property"
    )
    decision = review.AuditDecision(
        id=row["id"],
        decision="revise",
        revise_dependencies=True,
        rationale="Source-transplanted property name.",
    )
    selected = review.revision_ids(proposal, [decision])
    assert row["id"] in selected
    assert any(
        "dependency=decoys;" in e.context for e in proposal.entries if e.id in selected
    )
    assert all(e.reuse is None for e in proposal.entries if e.id in selected)
    assert not any(
        e.category == "identity_name" for e in proposal.entries if e.id in selected
    )
    kept = decision.model_copy(
        update={"decision": "keep", "revise_dependencies": False}
    )
    assert review.revision_ids(proposal, [kept]) == []


@pytest.mark.parametrize(
    "value",
    ["(011) 555-0107", "(080) 555-0195", "+1 (415) 555-0100", "9876543210 ext. 7"],
)
def test_indian_phone_check_rejects_transplanted_or_impossible_shapes(value):
    assert not review.is_indian_phone(value)


@pytest.mark.parametrize(
    "value",
    ["9876543210", "+91 98765 43210", "+91 11 2345 6789 ext. 7", "011 2345 6789"],
)
def test_indian_phone_check_preserves_plausible_local_formats(value):
    assert review.is_indian_phone(value)


def test_audit_uses_fixed_generate_contract_and_exact_cached_ids(
    monkeypatch, tmp_path, proposal
):
    rows = review.audit_rows(proposal)[:2]
    calls = []

    def fake_generate(**kwargs):
        payload = json.loads(kwargs["messages"][1].content)
        calls.append(payload)
        assert kwargs["call_name"] == "intake_candidate_locality_audit"
        return SimpleNamespace(
            content=json.dumps(
                {
                    "decisions": [
                        {
                            "id": row["id"],
                            "decision": "keep",
                            "revise_dependencies": False,
                            "rationale": "Plausible local entity; no source calque.",
                        }
                        for row in payload["entries"]
                    ]
                }
            )
        )

    monkeypatch.setattr(review, "generate", fake_generate)
    request = review.CandidateReviewRequest(
        map_file=Path("unused"),
        revision="test",
        model="test",
        model_args={"reasoning_effort": "medium"},
    )
    first = review.audit_batch(proposal, request, rows, tmp_path)
    assert review.audit_batch(proposal, request, rows, tmp_path) == first
    assert len(calls) == 1
    assert calls[0]["settings"]["country_code"] == "IN"
    assert len(review.local_reuse_roles(proposal)) == sum(
        entry.reuse is not None for entry in proposal.entries
    )


def test_source_review_overrides_are_explicit_without_rerunning_model(
    monkeypatch, tmp_path, proposal
):
    from tau2.utils.artifacts import content_digest, write_artifact

    rows = review.audit_rows(proposal)
    baseline = review.CandidateAudit(
        parent_sha256=content_digest(proposal),
        source_sha256=content_digest(proposal.source),
        settings=proposal.settings,
        model="test",
        model_args={"reasoning_effort": "medium"},
        decisions=[
            review.AuditDecision(
                id=row["id"],
                decision="keep",
                revise_dependencies=False,
                rationale="Prior model classification.",
            )
            for row in rows
        ],
        local_reuse_roles=[],
        redraft_ids=[],
    )
    prior_path = tmp_path / "prior.json"
    write_artifact(baseline, prior_path)

    def unexpected_call(**kwargs):
        raise AssertionError(
            "Source-review correction must not rerun or impersonate the model"
        )

    monkeypatch.setattr(review, "generate", unexpected_call)
    target = next(
        row["id"] for row in rows if row["role"] == "intake field:destination_property"
    )
    request = review.CandidateReviewRequest(
        map_file=Path(DATA_DIR) / "tau2/multilingual/hi/localization/intake.json",
        revision="source-review-test",
        model="test",
        model_args={"reasoning_effort": "medium"},
        prior_audit_file=prior_path,
        required_revision_ids=[target],
        required_revision_reason="Invented source identity remains transplanted; no local brand evidence.",
    )
    result = review.run_candidate_audit(
        request, tmp_path / "corrected.json", tmp_path / "checkpoints"
    )
    assert result.source_review_overrides[0].previous_decision == "keep"
    assert target in result.redraft_ids
    assert result.status == "pending_native_review"
    assert result.prior_audit_sha256 == content_digest(baseline)


def test_clean_audit_writes_evidence_without_invalid_empty_revision(
    monkeypatch, tmp_path, proposal
):
    monkeypatch.setattr(review, "is_indian_phone", lambda value: True)
    monkeypatch.setattr(
        review,
        "audit_batch",
        lambda proposal, request, rows, checkpoints: review.AuditBatch(
            decisions=[
                review.AuditDecision(
                    id=row["id"],
                    decision="keep",
                    revise_dependencies=False,
                    rationale="Local entity and dependencies remain appropriate.",
                )
                for row in rows
            ]
        ),
    )
    request = review.CandidateReviewRequest(
        map_file=Path(DATA_DIR) / "tau2/multilingual/hi/localization/intake.json",
        revision="clean-review",
        model="test",
        model_args={"reasoning_effort": "medium"},
    )
    output = tmp_path / "clean.json"
    output.with_suffix(".revision-request.json").write_text("stale revision")
    result = review.run_candidate_audit(request, output, tmp_path / "checks")
    assert result.redraft_ids == []
    assert output.exists()
    assert not output.with_suffix(".revision-request.json").exists()


def test_distinct_business_identities_cannot_collapse_across_audit_batches(proposal):
    changed = proposal.model_copy(deep=True)
    entities = [
        e
        for e in changed.entries
        if e.context.startswith("intake field:destination_property;")
    ]
    entities[1].forms = entities[0].forms.model_copy(deep=True)
    collisions = review.business_identity_collisions(changed)
    assert {entities[0].id, entities[1].id} <= set(collisions)
    assert "native" in collisions[entities[0].id]
    assert "romanized" in collisions[entities[0].id]


def test_hindi_candidates_have_local_phone_plate_shapes_and_distinct_businesses(
    proposal,
):
    import re

    phones = [
        e
        for e in proposal.entries
        if e.context.startswith("intake field:contact_phone;")
    ]
    plates = [
        e
        for e in proposal.entries
        if e.context.startswith("intake field:license_plate;")
    ]
    assert len(phones) == 50
    assert len(plates) == 5
    assert all(review.is_indian_phone(e.forms.native) for e in phones)
    for entry in plates:
        assert entry.forms.native == entry.forms.romanized
        compact = re.sub(r"[ -]", "", entry.forms.native)
        assert compact.isascii()
        # Delhi also uses DL1C-style prefixes (one district digit).
        # https://transport.delhi.gov.in/sites/default/files/Transport/generic_multiple_files/new_zonal_office_informations_0.pdf
        assert re.fullmatch(r"(?:DL\d{1,2}|[A-Z]{2}\d{2})[A-Z]{1,3}\d{4}", compact)
    assert review.business_identity_collisions(proposal) == {}


def test_targeted_followup_audit_records_exact_scope(monkeypatch, tmp_path, proposal):
    target = next(
        row["id"]
        for row in review.audit_rows(proposal)
        if row["role"] == "intake field:destination_property"
    )
    seen = []

    def fake_batch(proposal, request, rows, checkpoints):
        seen.extend(row["id"] for row in rows)
        return review.AuditBatch(
            decisions=[
                review.AuditDecision(
                    id=row["id"],
                    decision="keep",
                    revise_dependencies=False,
                    rationale="Localized identity and dependencies are plausible.",
                )
                for row in rows
            ]
        )

    monkeypatch.setattr(review, "audit_batch", fake_batch)
    request = review.CandidateReviewRequest(
        map_file=Path(DATA_DIR) / "tau2/multilingual/hi/localization/intake.json",
        revision="targeted",
        model="test",
        model_args={"reasoning_effort": "medium"},
        entry_ids=[target],
    )
    result = review.run_candidate_audit(
        request, tmp_path / "targeted.json", tmp_path / "checks"
    )
    assert seen == result.audited_ids == [target]
    assert len(result.decisions) == 1


def test_historical_hindi_followup_retains_repair_lineage_without_approval(proposal):
    from tau2.utils.artifacts import content_digest

    path = REVIEW_FIXTURES / "hi/intake.locality-followup-audit.json"
    audit = review.CandidateAudit.model_validate_json(path.read_text())
    from tau2.multilingual.factory.intake_pronunciation import ReferenceRepair

    repair = ReferenceRepair.model_validate_json(
        path.with_name("intake.native-reference-repair.json").read_text()
    )
    assert audit.parent_sha256 == repair.parent_map_sha256
    assert repair.status == "pending_native_review"
    # Gender metadata enriches the inventory; historical source-bound evidence
    # must remain visibly stale instead of being relabeled as a fresh audit.
    assert audit.source_sha256 != content_digest(proposal.source)
    assert audit.status == "pending_native_review"
    assert len(audit.audited_ids) == len(audit.decisions) == 10
    assert audit.redraft_ids == []
    assert all(
        d.decision == "keep" and not d.revise_dependencies for d in audit.decisions
    )


@pytest.mark.parametrize("language", ["hi", "es", "pt", "zh", "ko"])
def test_gender_repair_marks_prior_difficulty_review_stale_without_native_approval(
    language,
):
    from tau2.multilingual.factory.intake_difficulty_audit import DifficultyAuditReport
    from tau2.multilingual.factory.intake_difficulty_inventory import (
        build_difficulty_inventory,
    )
    from tau2.multilingual.factory.intake_easy_audit import (
        EasyAuditReport,
        extract_easy_cases,
    )
    from tau2.multilingual.factory.intake_gender import IntakeGenderReport
    from tau2.utils.artifacts import content_digest

    directory = Path(DATA_DIR) / f"tau2/multilingual/{language}/localization"
    evidence = REVIEW_FIXTURES / language
    candidate = read_localization_map(directory / "intake.json")
    repair = IntakeGenderReport.model_validate_json(
        (evidence / "intake.gender-preservation.json").read_text()
    )
    from tau2.multilingual.factory.intake_gender import validate_intake_gender_coverage

    if language in {"ko", "zh"}:
        from tau2.multilingual.factory.localization_corrections import (
            ReviewedCorrections,
            apply_corrections,
        )

        corrections = ReviewedCorrections.model_validate_json(
            (evidence / "approved-corrections.json").read_text()
        )
        assert repair.candidate_map_sha256 != content_digest(candidate)
        assert apply_corrections(candidate, corrections) == candidate
    elif candidate.address_revision is None:
        assert repair.candidate_map_sha256 == content_digest(candidate)
    else:
        assert (
            repair.candidate_map_sha256 == candidate.address_revision.parent_map_sha256
        )
    validate_intake_gender_coverage(candidate, Path(DATA_DIR))
    assert repair.prior_audits_stale
    assert repair.native_review_required
    assert not repair.review_exports_updated
    inventory = build_difficulty_inventory(candidate)
    for filename, model, cases in (
        ("intake.difficulty-audit.json", DifficultyAuditReport, inventory.cases),
        ("intake.easy-audit.json", EasyAuditReport, extract_easy_cases(inventory)),
    ):
        audit_path = evidence / filename
        audit = model.model_validate_json(audit_path.read_text())
        assert audit.map_sha256 != content_digest(candidate)
        assert audit.source_sha256 != content_digest(candidate.source)
        assert audit.inventory_sha256 != content_digest(inventory)
        assert {row.canonical_task_id for row in audit.decisions} == {
            row.canonical_task_id for row in cases
        }
        assert len(audit.decisions) == 100
        assert audit.error_count == 0
        assert audit.native_review_status == "pending"


@pytest.mark.parametrize(
    "language,country_prefix",
    [("hi", "91"), ("es", "34"), ("pt", "55"), ("zh", "86"), ("ko", "82")],
)
def test_localized_phone_difficulty_does_not_transplant_another_country(
    language, country_prefix
):
    import re

    candidate = read_localization_map(
        Path(DATA_DIR) / f"tau2/multilingual/{language}/localization/intake.json"
    )
    phones = [
        entry
        for entry in candidate.entries
        if entry.context.startswith("intake field:contact_phone;")
    ]
    assert len(phones) == 50
    for entry in phones:
        for representation in ("native", "romanized"):
            value = getattr(entry.forms, representation)
            if "+" in value:
                assert re.sub(r"\D", "", value).startswith(country_prefix), (
                    language,
                    entry.id,
                    value,
                )
