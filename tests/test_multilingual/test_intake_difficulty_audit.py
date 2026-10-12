"""Exhaustive difficulty review rejects missing evidence and resumes exact inputs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tau2.multilingual.factory import intake_difficulty_audit as audit
from tau2.multilingual.factory.localization_artifact import read_localization_map
from tau2.utils.utils import DATA_DIR


@pytest.fixture(scope="module")
def inventory():
    proposal = read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/hi/localization/intake.json"
    )
    return audit.build_difficulty_inventory(proposal)


@pytest.fixture
def audit_request():
    return audit.DifficultyAuditRequest(
        map_file=Path(DATA_DIR) / "tau2/multilingual/hi/localization/intake.json",
        revision="test-review-v1",
        model="test",
        model_args={"temperature": 0},
        batch_size=10,
        max_concurrency=2,
    )


def decision(case):
    return audit.DifficultyDecision(
        canonical_task_id=case.canonical_task_id,
        disposition="keep",
        source_difficulty="hard",
        source_assessment="Source fixture has a capture contrast, assessed independently.",
        localized_difficulty="hard",
        naturalness=True,
        known_answer_recoverable=True,
        hazards=[
            audit.DifficultyHazard(
                tag="component_boundary",
                evidence=f"Fixture capture contrast in {case.native_value}.",
                resolving_clarification="Caller supplies each exact component on request.",
            )
        ],
        comparison_to_easy=f"Compared with {case.easy_comparators[0].canonical_task_id}, the fixture adds a specific component boundary.",
        dependency_issues=[],
        correction_instructions=[],
        repair_targets=[],
        source_blocked_reason=None,
        native_review_required=True,
    )


def install_mock(monkeypatch, transform=None):
    calls = []

    def generate(**kwargs):
        assert kwargs["call_name"] == "intake_difficulty_audit"
        assert kwargs["messages"][0].content == audit.AUDIT_PROMPT
        assert (
            kwargs["response_format"]["json_schema"]["schema"]
            == audit.DifficultyAuditBatch.model_json_schema()
        )
        payload = audit.DifficultyAuditPayload.model_validate_json(
            kwargs["messages"][1].content
        )
        assert len(payload.cases[0].easy_comparators) == 10
        calls.append(payload)
        result = audit.DifficultyAuditBatch(
            decisions=[decision(case) for case in payload.cases]
        ).model_dump(mode="json")
        if transform:
            transform(result)
        return SimpleNamespace(content=json.dumps(result))

    monkeypatch.setattr(audit, "generate", generate)
    return calls


@pytest.mark.parametrize(
    "patch",
    [
        {"localized_difficulty": "easy"},
        {"localized_difficulty": "uncertain"},
        {"naturalness": False},
        {"known_answer_recoverable": False},
        {"hazards": []},
        {"dependency_issues": ["Wrong pronunciation"]},
        {"correction_instructions": ["Repair pronunciation"]},
        {"native_review_required": False},
    ],
)
def test_keep_requires_supported_natural_hard_case(inventory, patch):
    raw = decision(inventory.cases[0]).model_dump(mode="json")
    raw.update(patch)
    with pytest.raises(ValidationError):
        audit.DifficultyDecision.model_validate(raw)


def test_revise_and_source_blocker_require_actionable_evidence(inventory):
    raw = decision(inventory.cases[0]).model_dump(mode="json")
    raw["disposition"] = "revise"
    with pytest.raises(ValidationError, match="correction instructions"):
        audit.DifficultyDecision.model_validate(raw)
    raw["correction_instructions"] = [
        "Restore the lost extension with coherent local grouping."
    ]
    raw["repair_targets"] = ["primary"]
    audit.DifficultyDecision.model_validate(raw)
    raw["disposition"] = "source_blocked"
    with pytest.raises(ValidationError, match="immutable-source reason"):
        audit.DifficultyDecision.model_validate(raw)
    raw["source_blocked_reason"] = (
        "Changing this fixed dose would change clinical identity."
    )
    audit.DifficultyDecision.model_validate(raw)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "extra", "uncited"])
def test_model_response_requires_exact_coverage_and_easy_comparison(
    inventory, audit_request, monkeypatch, tmp_path, mode
):
    def corrupt(raw):
        if mode == "missing":
            raw["decisions"].pop()
        elif mode == "duplicate":
            raw["decisions"][1] = raw["decisions"][0]
        elif mode == "extra":
            raw["decisions"].append(raw["decisions"][0])
        else:
            raw["decisions"][0]["comparison_to_easy"] = "Harder than Easy, trust me."

    install_mock(monkeypatch, corrupt)
    with pytest.raises(ValueError, match="exactly once|supplied same-family"):
        audit.audit_batch(audit_request, inventory, inventory.cases[:2], tmp_path)
    assert not list(tmp_path.glob("*.json"))


def test_checkpoints_bind_model_revision_payload_and_prompt(
    inventory, audit_request, monkeypatch, tmp_path
):
    calls = install_mock(monkeypatch)
    cases = inventory.cases[:2]
    first = audit.audit_batch(audit_request, inventory, cases, tmp_path)
    assert audit.audit_batch(audit_request, inventory, cases, tmp_path) == first
    assert len(calls) == 1
    changed = audit_request.model_copy(update={"revision": "test-review-v2"})
    audit.audit_batch(changed, inventory, cases, tmp_path)
    assert len(calls) == 2
    original = next(
        p
        for p in tmp_path.glob("*.json")
        if json.loads(p.read_text())["request"]["revision"] == audit_request.revision
    )
    saved = json.loads(original.read_text())
    saved["prompt_version"] = "stale-version"
    original.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="checkpoint differs"):
        audit.audit_batch(audit_request, inventory, cases, tmp_path)
    assert len(calls) == 2


def test_checkpoint_missing_case_rejected(
    inventory, audit_request, monkeypatch, tmp_path
):
    install_mock(monkeypatch)
    audit.audit_batch(audit_request, inventory, inventory.cases[:2], tmp_path)
    path = next(tmp_path.glob("*.json"))
    saved = json.loads(path.read_text())
    saved["response"]["decisions"].pop()
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="exactly once"):
        audit.audit_batch(audit_request, inventory, inventory.cases[:2], tmp_path)


def test_full_audit_resumes_and_never_native_approves(
    audit_request, monkeypatch, tmp_path
):
    calls = install_mock(monkeypatch)
    output, checkpoints = tmp_path / "audit.json", tmp_path / "checkpoints"
    report = audit.run_difficulty_audit(audit_request, output, checkpoints)
    assert len(calls) == 10
    assert report.keep_count == 100
    assert report.machine_gate_passed
    assert report.native_review_status == "pending"
    assert all(d.native_review_required for d in report.decisions)
    assert audit.run_difficulty_audit(audit_request, output, checkpoints) == report
    assert len(calls) == 10
    changed = audit_request.model_copy(update={"revision": "new-revision"})
    with pytest.raises(ValueError, match="stale"):
        audit.run_difficulty_audit(changed, output, checkpoints)


def test_one_bad_batch_accounted_for_and_good_batches_resumable(
    audit_request, monkeypatch, tmp_path
):
    calls = install_mock(monkeypatch)
    original = audit.generate
    first_id = (
        audit.build_difficulty_inventory(read_localization_map(audit_request.map_file))
        .cases[0]
        .canonical_task_id
    )

    def fail_one(**kwargs):
        payload = audit.DifficultyAuditPayload.model_validate_json(
            kwargs["messages"][1].content
        )
        if payload.cases[0].canonical_task_id == first_id:
            raise RuntimeError("injected provider failure")
        return original(**kwargs)

    monkeypatch.setattr(audit, "generate", fail_one)
    checkpoints = tmp_path / "checkpoints"
    report = audit.run_difficulty_audit(
        audit_request, tmp_path / "errors.json", checkpoints
    )
    assert report.keep_count == 90
    assert report.error_count == 10
    assert not report.machine_gate_passed
    assert len(calls) == 9
    monkeypatch.setattr(audit, "generate", original)
    resumed = audit.run_difficulty_audit(
        audit_request, tmp_path / "complete.json", checkpoints
    )
    assert resumed.keep_count == 100
    assert resumed.error_count == 0
    assert len(calls) == 10
    raw = resumed.model_dump(mode="json")
    raw["decisions"].pop()
    with pytest.raises(ValidationError, match="every task exactly once"):
        audit.DifficultyAuditReport.model_validate(raw)


def test_v2_prompt_preserves_representation_and_source_constraint_boundaries():
    assert audit.AUDIT_PROMPT_VERSION == "intake-difficulty-audit-v2"
    for requirement in (
        "intentional no-op placeholders",
        "Canonical ASCII phone/email/tool representations are not literal speech transcripts",
        "source relative phrase or relation, such as 'tomorrow', is NOT itself immutable",
        "English zero-as-oh or double/triple speech is NOT universally required",
        "coined family includes companies, employers and aftermarket brands",
        "formatter limits are not source blockers",
    ):
        assert requirement in audit.AUDIT_PROMPT


def test_inspection_preserves_only_valid_cached_cases_without_model_calls(
    inventory, audit_request, monkeypatch, tmp_path
):
    calls = install_mock(monkeypatch)
    checkpoints = tmp_path / "checkpoints"
    valid = audit.audit_batch(
        audit_request, inventory, inventory.cases[:10], checkpoints
    )
    audit.audit_batch(audit_request, inventory, inventory.cases[10:20], checkpoints)
    prepared = audit.prepare_difficulty_batch(
        audit_request, inventory, inventory.cases[10:20], checkpoints
    )
    saved = json.loads(prepared.path.read_text())
    saved["response"]["decisions"].pop()
    prepared.path.write_text(json.dumps(saved))

    def forbidden_generate(**kwargs):
        raise AssertionError("Checkpoint inspection must never call a model")

    monkeypatch.setattr(audit, "generate", forbidden_generate)
    output = tmp_path / "partial.json"
    report = audit.inspect_difficulty_audit(audit_request, output, checkpoints)
    assert len(calls) == 2
    assert report.decisions == valid.decisions
    assert report.keep_count == 10
    assert report.error_count == 90
    assert len(report.errors) == 90
    assert not report.machine_gate_passed
    assert report.native_review_status == "pending"
    assert report.map_sha256 == inventory.map_sha256
    assert report.source_sha256 == inventory.source_sha256
    assert all("exactly once" in e.error for e in report.errors[:10])
    assert all("incomplete" in e.error for e in report.errors[10:])
    assert audit.inspect_difficulty_audit(audit_request, output, checkpoints) == report


def test_checkpoint_inspection_cli_emits_all_missing_cases_as_errors(
    audit_request, monkeypatch, tmp_path
):
    import argparse

    from tau2.multilingual.factory.intake_difficulty_cli import (
        add_intake_difficulty_args,
    )

    def forbidden_generate(**kwargs):
        raise AssertionError("Read-only CLI must never call a model")

    monkeypatch.setattr(audit, "generate", forbidden_generate)
    request_file = tmp_path / "request.json"
    request_file.write_text(audit_request.model_dump_json())
    output = tmp_path / "missing.json"
    parser = argparse.ArgumentParser()
    add_intake_difficulty_args(parser.add_subparsers())
    args = parser.parse_args(
        [
            "inspect-intake-difficulty",
            "--request",
            str(request_file),
            "--output",
            str(output),
            "--checkpoints",
            str(tmp_path / "absent"),
        ]
    )
    args.func(args)
    report = audit.DifficultyAuditReport.model_validate_json(output.read_text())
    assert report.error_count == 100
    assert report.decisions == []
    assert not report.machine_gate_passed
