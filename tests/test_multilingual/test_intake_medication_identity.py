"""Medication label consistency catches drift without asserting medical synonyms."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from tau2.multilingual.factory.intake_medication_identity import (
    MedicationIdentityObservation,
    medication_identity_report,
)
from tau2.multilingual.factory.localization_artifact import read_localization_map
from tau2.utils.utils import DATA_DIR


@pytest.fixture(scope="module")
def proposal():
    return read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/es/localization/intake.json"
    )


def slot(proposal, task_id, field):
    source = next(
        d.content for d in proposal.source.documents if d.id == "intake_localization"
    )
    index = next(
        i
        for i, row in enumerate(source["stimuli"])
        if row["canonical_task_id"] == task_id
    )
    return next(
        e
        for e in proposal.entries
        if any(
            o.document_id == "intake_localization"
            and o.path == ["stimuli", index, field]
            for o in e.occurrences
        )
    )


def test_catalog_coverage_and_clinical_approval_never_implied(proposal):
    report = medication_identity_report(proposal)
    assert len(report.observations) == 80  # 20 primary + 20 decoys, two forms
    assert not report.unparsed
    assert report.clinical_identity_status == "unverified"
    assert all(o.source_label for o in report.observations)
    assert all(o.source_label == o.source_label.lower() for o in report.observations)


def test_same_target_different_generic_detected_despite_different_doses(proposal):
    changed = proposal.model_copy(deep=True)
    task = "intake_medications_hard_10"
    primary = slot(changed, task, "written")
    decoy = slot(changed, task, "decoys")
    decoy.forms.native = [primary.forms.native.replace("10 mg", "4 mg")]
    report = medication_identity_report(changed)
    issue = next(
        i
        for i in report.findings
        if i.kind == "distinct_source_label_collision"
        and any(o.entry_id == decoy.id for o in i.observations)
    )
    assert {o.role for o in issue.observations} == {"primary", "decoy"}
    assert {o.source_label for o in issue.observations} == {
        "cyclobenzaprine",
        "cyproheptadine",
    }
    assert len({o.current_label for o in issue.observations}) == 1
    assert issue.requires_review


def test_repeated_source_variants_reported_without_declaring_synonyms_invalid(proposal):
    changed = proposal.model_copy(deep=True)
    decoy = slot(changed, "intake_medications_hard_03", "decoys")
    decoy.forms.native = ["Nombre alternativo 100 mg comprimido"]
    report = medication_identity_report(changed)
    issue = next(
        i
        for i in report.findings
        if i.kind == "same_source_label_variant"
        and any(o.entry_id == decoy.id for o in i.observations)
    )
    assert {o.source_label for o in issue.observations} == {"lamotrigine"}
    assert {o.canonical_task_id for o in issue.observations} == {
        "intake_medications_hard_01",
        "intake_medications_hard_03",
        "intake_medications_hard_09",
    }
    assert issue.requires_review
    assert "synonym" in issue.explanation


def test_dose_form_case_and_unicode_width_are_not_identity_differences(proposal):
    changed = proposal.model_copy(deep=True)
    original = slot(changed, "intake_medications_hard_01", "written").forms.native
    prefix = original.split("100")[0].strip()
    decoy = slot(changed, "intake_medications_hard_03", "decoys")
    decoy.forms.native = [prefix.upper() + " １００ mg cápsula"]
    report = medication_identity_report(changed)
    assert not any(
        i.kind == "same_source_label_variant"
        and any(o.entry_id == decoy.id for o in i.observations)
        for i in report.findings
    )


def test_unknown_layout_explicitly_unparsed_not_silently_passed(proposal):
    changed = proposal.model_copy(deep=True)
    decoy = slot(changed, "intake_medications_hard_03", "decoys")
    decoy.forms.native = ["100 mg Lamotrigina"]
    report = medication_identity_report(changed)
    assert any(
        o.entry_id == decoy.id and o.current_label is None for o in report.unparsed
    )


def test_observation_rejects_fabricated_normalized_evidence(proposal):
    evidence = medication_identity_report(proposal).observations[0].model_dump()
    evidence["source_label"] = "different medicine"
    with pytest.raises(ValidationError, match="label evidence"):
        MedicationIdentityObservation.model_validate(evidence)


def test_scanner_rejects_stale_frozen_source(proposal):
    changed = proposal.model_copy(deep=True)
    source = next(d for d in changed.source.documents if d.id == "intake_localization")
    row = next(r for r in source.content["stimuli"] if r["family"] == "medications")
    row["written"] = "Different 4 mg tablet"
    with pytest.raises(ValueError):
        medication_identity_report(changed)


def test_package_cli_writes_typed_source_bound_report(proposal, tmp_path):
    import argparse

    from tau2.multilingual.factory.cli import add_factory_args
    from tau2.multilingual.factory.intake_medication_identity import (
        MedicationIdentityReport,
    )
    from tau2.utils.artifacts import content_digest

    parser = argparse.ArgumentParser()
    add_factory_args(parser)
    output = tmp_path / "identity.json"
    args = parser.parse_args(
        [
            "inspect-intake-medication-identities",
            "--map",
            str(Path(DATA_DIR) / "tau2/multilingual/es/localization/intake.json"),
            "--output",
            str(output),
        ]
    )
    args.func(args)
    result = MedicationIdentityReport.model_validate_json(output.read_text())
    assert result.map_sha256 == content_digest(proposal)
    assert result.source_sha256 == content_digest(proposal.source)
    assert result.clinical_identity_status == "unverified"
    before = output.read_bytes()
    args.func(args)
    assert output.read_bytes() == before


def test_finding_cannot_lose_implicated_slots(proposal):
    from tau2.multilingual.factory.intake_medication_identity import (
        MedicationIdentityReport,
    )

    changed = proposal.model_copy(deep=True)
    decoy = slot(changed, "intake_medications_hard_03", "decoys")
    decoy.forms.native = ["Nombre alternativo 100 mg comprimido"]
    report = medication_identity_report(changed).model_dump()
    issue = next(
        i
        for i in report["findings"]
        if i["kind"] == "same_source_label_variant"
        and any(o["entry_id"] == decoy.id for o in i["observations"])
    )
    issue["observations"] = [
        o
        for o in issue["observations"]
        if o["canonical_task_id"] != "intake_medications_hard_09"
    ]
    with pytest.raises(ValidationError, match="all observations"):
        MedicationIdentityReport.model_validate(report)


@pytest.mark.parametrize(
    "mutation", ["missing_slot", "wrong_tier", "unknown_task", "extra_decoy"]
)
def test_serialized_report_requires_exact_frozen_task_tier_and_slot_coverage(
    proposal, mutation
):
    from tau2.multilingual.factory.intake_medication_identity import (
        MedicationIdentityReport,
    )

    report = medication_identity_report(proposal).model_dump()
    if mutation == "missing_slot":
        report["observations"].pop()
    elif mutation == "wrong_tier":
        report["observations"][0]["tier"] = "hard"
    elif mutation == "unknown_task":
        report["observations"][0]["canonical_task_id"] = "intake_medications_easy_11"
    else:
        decoy = next(o for o in report["observations"] if o["role"] == "decoy")
        decoy["decoy_index"] = 1
    with pytest.raises(ValidationError, match="frozen tier|all 80 frozen"):
        MedicationIdentityReport.model_validate(report)
