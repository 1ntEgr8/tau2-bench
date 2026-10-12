"""Source-role and dependency coverage for gender-preserving intake localization."""

from collections import Counter
from pathlib import Path

import pytest

from tau2.multilingual.factory.intake_gender import (
    IntakeGenderRequest,
    intake_gender_dependencies,
    intake_name_bindings,
    synchronize_intake_owner_emails,
    validate_intake_gender_coverage,
)
from tau2.multilingual.factory.localization_artifact import read_localization_map
from tau2.multilingual.factory.localization_corrections import (
    EntityCorrection,
    ReviewedCorrections,
    apply_corrections,
)
from tau2.multilingual.localization import LocalizedForms
from tau2.utils import DATA_DIR
from tau2.utils.artifacts import content_digest


@pytest.mark.parametrize("external", [False, True])
def test_report_request_has_portable_paths_without_mutating_execution(
    tmp_path, external
):
    data_root = tmp_path / "private-checkout/data"
    map_file = (
        tmp_path / "private-authoring/candidate.json"
        if external
        else data_root / "tau2/multilingual/es/localization/intake.json"
    )
    request = IntakeGenderRequest(
        map_file=map_file,
        data_root=data_root,
        revision="test",
        model="test",
        model_args={},
        max_concurrency=1,
        batch_size=5,
    )
    recorded = request.for_report()
    assert recorded.data_root == Path("data")
    assert recorded.map_file == (
        Path("inputs/candidate.json")
        if external
        else Path("data/tau2/multilingual/es/localization/intake.json")
    )
    assert str(tmp_path) not in recorded.model_dump_json()
    assert request.map_file == map_file
    assert request.data_root == data_root


@pytest.fixture(scope="module")
def proposal():
    return read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/es/localization/intake.json"
    )


def test_all_used_names_have_separate_source_roles(proposal):
    bindings = intake_name_bindings(proposal, Path(DATA_DIR))
    assert Counter(b.role for b in bindings) == {
        "callee": 39,
        "measured_name": 20,
        "name_decoy": 40,
    }
    assert len({b.id for b in bindings}) == 99
    entries = {e.id: e for e in proposal.entries}
    lisa = [b for b in bindings if entries[b.entry_id].source == "Lisa Sandoval"]
    assert {b.role for b in lisa} == {"callee", "measured_name"}
    assert {b.source_gender for b in lisa} == {"female"}
    # The frozen bank labels the primary, not the independent wrong-name decoy.
    gisela = next(
        b
        for b in bindings
        if b.index is not None
        and entries[b.entry_id].source[b.index] == "Gisela Lansbury-Cutugno"
    )
    assert gisela.source_gender == "unknown"
    assert "never copy the primary" in gisela.source_evidence


def test_reviewed_decoy_change_requires_matching_gender_evidence(proposal):
    entry = next(e for e in proposal.entries if len(e.name_gender_items) == 2)
    forms = entry.forms.model_copy(deep=True)
    forms.native[0] += "-Revisado"
    forms.romanized[0] += "-Revisado"
    review = ReviewedCorrections(
        domain="intake",
        language="es",
        revision="gender-decoy-review-test-v1",
        reviewed_by="test reviewer",
        evidence="Source-bound decoy review fixture.",
        corrections=[
            EntityCorrection(
                entry_id=entry.id,
                source=entry.source,
                before=entry.forms,
                after=forms,
                reason="Revise one decoy while retaining independent gender evidence.",
            )
        ],
    )
    with pytest.raises(ValueError, match="gender evidence no longer matches"):
        apply_corrections(proposal, review)

    evidence = []
    for item in entry.name_gender_items:
        item_forms = LocalizedForms(
            native=forms.native[item.index],
            romanized=forms.romanized[item.index],
            english=forms.english,
        )
        evidence.append(
            item.model_copy(
                update={
                    "evidence": item.evidence.model_copy(
                        update={"forms_sha256": content_digest(item_forms)}
                    )
                }
            )
        )
    review.corrections[0].name_gender_items = evidence
    revised = apply_corrections(proposal, review)
    assert apply_corrections(revised, review) == revised
    assert revised.entries != proposal.entries


def test_measured_name_repair_keeps_callee_and_email_separate(proposal):
    bindings = intake_name_bindings(proposal, Path(DATA_DIR))
    primary = next(
        b
        for b in bindings
        if b.role == "measured_name" and "intake_person_names_hard_01" in b.task_ids
    )
    selected = set(intake_gender_dependencies(proposal, {primary.entry_id}))
    entries = [e for e in proposal.entries if e.id in selected]
    assert entries
    assert all("intake_person_names_hard_01" in e.context for e in entries)
    assert not any("dependency=decoys" in e.context for e in entries)
    assert all(e.category != "identity_name" for e in entries)


def test_callee_repair_selects_only_owned_email_bundles(proposal):
    bindings = intake_name_bindings(proposal, Path(DATA_DIR))
    owner = next(
        b
        for b in bindings
        if b.role == "callee"
        and any(t.startswith("intake_emails_hard_") for t in b.task_ids)
    )
    selected = set(intake_gender_dependencies(proposal, {owner.entry_id}))
    entries = [e for e in proposal.entries if e.id in selected]
    assert entries
    assert all("intake_emails_" in e.context for e in entries)
    assert all("dependency=" in e.context for e in entries)
    assert not any("dependency=decoys" in e.context for e in entries)


def test_owned_email_uses_final_callee_and_keeps_frozen_pattern(proposal):
    candidate = proposal.model_copy(deep=True)
    stale = next(
        e for e in candidate.entries if e.source == "william.larson@quorvenhaven.net"
    )
    stale.forms.native = stale.forms.romanized = "stale.owner@naverra.es"
    changed = synchronize_intake_owner_emails(candidate, Path(DATA_DIR))
    original = next(
        e for e in proposal.entries if e.source == "william.larson@quorvenhaven.net"
    )
    updated = next(e for e in candidate.entries if e.id == original.id)
    assert updated.id in changed
    assert updated.forms.native == "javier.leon@naverra.es"
    assert updated.forms.romanized == updated.forms.native
    assert synchronize_intake_owner_emails(candidate, Path(DATA_DIR)) == set()
    # Email synchronization cannot touch the name assignment, address or frozen source.
    assert candidate.source == proposal.source
    for before, after in zip(proposal.entries, candidate.entries, strict=True):
        if after.id not in changed:
            assert before == after


def test_inventory_exposes_reviewed_gender_to_authoring(proposal):
    named = [e for e in proposal.source.entries if e.category == "identity_name"]
    assert named
    assert all("reviewed_source_gender=" in e.context for e in named)


def test_publication_refuses_missing_name_assessment(proposal):
    candidate = proposal.model_copy(deep=True)
    entry = next(e for e in candidate.entries if e.category == "identity_name")
    entry.name_gender = None
    with pytest.raises(ValueError, match="requires source-gender evidence"):
        validate_intake_gender_coverage(candidate, Path(DATA_DIR))


def test_publication_refuses_gender_evidence_that_changes_bank_label(proposal):
    from tau2.multilingual.localization import NameGenderEvidence
    from tau2.utils.artifacts import content_digest

    candidate = proposal.model_copy(deep=True)
    entry = next(e for e in candidate.entries if e.category == "identity_name")
    binding = next(
        b
        for b in intake_name_bindings(candidate, Path(DATA_DIR))
        if b.entry_id == entry.id
    )
    wrong = "male" if binding.source_gender == "female" else "female"
    entry.name_gender = NameGenderEvidence(
        source_gender=wrong,
        target_gender=wrong,
        source_basis="reviewed_metadata",
        source_evidence="Deliberately conflicting test evidence.",
        reason="Test fixture.",
        forms_sha256=content_digest(entry.forms),
        model="test",
        prompt_version="test-v1",
    )
    with pytest.raises(ValueError, match="differs from frozen bank"):
        validate_intake_gender_coverage(candidate, Path(DATA_DIR))


def test_email_owner_resolution_respects_explicit_family_first_order():
    candidate = read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/ko/localization/intake.json"
    )
    owner = next(
        e
        for e in candidate.entries
        if e.category == "identity_name" and e.source == "William Larson"
    )
    family, given = owner.forms.romanized.lower().split()
    email = next(
        e for e in candidate.entries if e.source == "william.larson@quorvenhaven.net"
    )
    domain = email.forms.native.split("@")[1]
    synchronize_intake_owner_emails(candidate, Path(DATA_DIR))
    assert email.forms.native == f"{given}.{family}@{domain}"


def test_replay_on_already_audited_map_replaces_indexed_evidence(
    monkeypatch, tmp_path, proposal
):
    from tau2.multilingual.factory import localization_gender as gender
    from tau2.multilingual.factory.intake_gender import (
        IntakeGenderReport,
        IntakeGenderRequest,
        revise_intake_gender,
    )

    def source_assessments(requests, *args, **kwargs):
        return [
            gender.SourceGenderResult(
                id=r.id,
                kind="person",
                gender=r.source_gender,
                basis=r.source_basis,
                reason=r.source_evidence,
            )
            for r in requests
        ]

    def retained_names(requests, *args, **kwargs):
        return [
            gender.GenderNameResult(
                id=r.id,
                source_gender=r.source_gender,
                target_gender=r.source_gender,
                decision="unresolved" if r.source_gender == "unknown" else "keep",
                forms=gender.GenderNameForms.model_validate(r.forms.model_dump()),
                reason="Offline unchanged-form assessment fixture.",
            )
            for r in requests
        ]

    monkeypatch.setattr(gender, "assess_source_names", source_assessments)
    monkeypatch.setattr(gender, "audit_gender_names", retained_names)
    assert sum(len(e.name_gender_items) for e in proposal.entries) == 40
    candidate = revise_intake_gender(
        IntakeGenderRequest(
            map_file=Path(DATA_DIR) / "tau2/multilingual/es/localization/intake.json",
            data_root=Path(DATA_DIR),
            revision="intake-es-offline-replay-test",
            model="test",
            model_args={},
            max_concurrency=1,
            batch_size=5,
        ),
        tmp_path / "replay.json",
        tmp_path / "checkpoints",
    )
    assert sum(len(e.name_gender_items) for e in candidate.entries) == 40
    assert all(
        len({item.index for item in e.name_gender_items}) == len(e.name_gender_items)
        for e in candidate.entries
    )
    assert [e.forms for e in candidate.entries] == [e.forms for e in proposal.entries]
    validate_intake_gender_coverage(candidate, Path(DATA_DIR))
    report = IntakeGenderReport.model_validate_json(
        (tmp_path / "replay.gender-report.json").read_text()
    )
    assert report.request.map_file == Path(
        "data/tau2/multilingual/es/localization/intake.json"
    )
    assert candidate.provenance.input_sha256 == content_digest(report.request)
