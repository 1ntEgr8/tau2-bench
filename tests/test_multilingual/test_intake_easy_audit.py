"""Independent Easy audits preserve exact coverage, privacy and review provenance."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tau2.multilingual.factory import intake_easy_audit as audit
from tau2.multilingual.factory.intake_difficulty_cli import add_intake_difficulty_args
from tau2.multilingual.factory.localization_artifact import read_localization_map
from tau2.utils.utils import DATA_DIR


@pytest.fixture(scope="module")
def proposal():
    return read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/hi/localization/intake.json"
    )


@pytest.fixture(scope="module")
def inventory(proposal):
    return audit.build_difficulty_inventory(proposal)


@pytest.fixture
def request_model():
    return audit.EasyAuditRequest(
        map_file=Path(DATA_DIR) / "tau2/multilingual/hi/localization/intake.json",
        revision="offline-easy-v1",
        model="test",
        model_args={"temperature": 0},
        batch_size=10,
        max_concurrency=2,
    )


def keep(case):
    return audit.EasyDecision(
        canonical_task_id=case.canonical_task_id,
        disposition="keep",
        source_difficulty="easy",
        source_assessment=f"The frozen source is straightforward: {case.source_value}.",
        localized_difficulty="easy",
        ease_assessment=f"Ordinary local capture of {case.native_value}.",
        evidence=f"Written {case.native_value} agrees with its exact caller lookup.",
        naturalness=True,
        known_answer_recoverable=True,
        source_semantics_preserved=True,
        dependency_issues=[],
        correction_instructions=[],
        repair_targets=[],
        source_blocked_reason=None,
        native_review_required=True,
    )


def mock_calls(monkeypatch, transform=None):
    calls = []

    def generate(**kwargs):
        assert kwargs["call_name"] == "intake_easy_audit"
        assert kwargs["messages"][0].content == audit.EASY_AUDIT_PROMPT
        assert kwargs["response_format"]["json_schema"]["schema"] == (
            audit.EasyAuditBatch.model_json_schema()
        )
        payload = audit.EasyAuditPayload.model_validate_json(
            kwargs["messages"][1].content
        )
        assert all(c.tier == "easy" for c in payload.cases)
        calls.append(payload)
        raw = audit.EasyAuditBatch(
            decisions=[keep(case) for case in payload.cases]
        ).model_dump(mode="json")
        if transform:
            transform(raw)
        return SimpleNamespace(content=json.dumps(raw))

    monkeypatch.setattr(audit, "generate", generate)
    return calls


def test_extracts_all_easy_once_and_rejects_inconsistent_repeated_controls(inventory):
    cases = audit.extract_easy_cases(inventory)
    assert len(cases) == len({c.canonical_task_id for c in cases}) == 100
    assert len(Counter(c.family for c in cases)) == 10
    assert set(Counter(c.family for c in cases).values()) == {10}
    assert all(c.tier == "easy" and c.entry_sha256 for c in cases)
    bad = inventory.model_copy(deep=True)
    bad.cases[1].easy_comparators[0].native_guidance.time_guidance += " altered"
    with pytest.raises(ValueError):
        audit.extract_easy_cases(bad)


def test_easy_payload_never_exports_callback_owners(proposal, request_model, tmp_path):
    changed = proposal.model_copy(deep=True)
    owners = [
        e
        for e in changed.entries
        if any(p.path and p.path[-1] == "callee_full_name" for p in e.occurrences)
    ]
    assert owners
    for n, entry in enumerate(owners):
        entry.name_gender = None
        entry.forms.native = entry.forms.romanized = f"PRIVATE_EASY_OWNER_{n}"
        entry.operation = "adapt"
    inventory = audit.build_difficulty_inventory(changed)
    prepared = audit.prepare_easy_batch(
        request_model, inventory, audit.extract_easy_cases(inventory)[:5], tmp_path
    )
    payload = prepared.payload.model_dump_json()
    assert "PRIVATE_EASY_OWNER" not in payload
    assert '"tier":"hard"' not in payload
    assert len(prepared.payload.family_controls) == 10


@pytest.mark.parametrize(
    "patch",
    [
        {"localized_difficulty": "too_hard"},
        {"localized_difficulty": "uncertain"},
        {"naturalness": False},
        {"known_answer_recoverable": False},
        {"source_semantics_preserved": False},
        {"evidence": " "},
        {"dependency_issues": ["Wrong speech"]},
        {"native_review_required": False},
    ],
)
def test_keep_cannot_mask_defects_or_uncertainty(inventory, patch):
    raw = keep(audit.extract_easy_cases(inventory)[0]).model_dump(mode="json")
    raw.update(patch)
    with pytest.raises(ValidationError):
        audit.EasyDecision.model_validate(raw)


def test_repair_and_source_blocker_need_precise_evidence(inventory):
    raw = keep(audit.extract_easy_cases(inventory)[0]).model_dump(mode="json")
    raw["disposition"] = "revise"
    with pytest.raises(ValidationError, match="repair targets"):
        audit.EasyDecision.model_validate(raw)
    raw.update(
        correction_instructions=["Fix only the mismatched pronunciation."],
        repair_targets=["pronunciations"],
    )
    audit.EasyDecision.model_validate(raw)
    raw["repair_targets"] *= 2
    with pytest.raises(ValidationError, match="distinct"):
        audit.EasyDecision.model_validate(raw)
    raw.update(
        disposition="source_blocked",
        repair_targets=[],
        correction_instructions=[],
        source_blocked_reason="Fixed drug identity is intrinsically complex.",
    )
    audit.EasyDecision.model_validate(raw)


@pytest.mark.parametrize("mode", ["missing", "duplicate", "extra", "hard_id"])
def test_every_response_covers_exact_easy_batch(
    inventory, request_model, monkeypatch, tmp_path, mode
):
    def corrupt(raw):
        if mode == "missing":
            raw["decisions"].pop()
        elif mode == "duplicate":
            raw["decisions"][1] = raw["decisions"][0]
        elif mode == "extra":
            raw["decisions"].append(raw["decisions"][0])
        else:
            raw["decisions"][0]["canonical_task_id"] = inventory.cases[
                0
            ].canonical_task_id

    mock_calls(monkeypatch, corrupt)
    with pytest.raises(ValueError, match="exactly once"):
        audit.audit_easy_batch(
            request_model, inventory, audit.extract_easy_cases(inventory)[:2], tmp_path
        )
    assert not list(tmp_path.glob("*.json"))


def test_checkpoints_bind_complete_request_map_and_prompt(
    inventory, request_model, monkeypatch, tmp_path
):
    calls = mock_calls(monkeypatch)
    cases = audit.extract_easy_cases(inventory)[:2]
    first = audit.audit_easy_batch(request_model, inventory, cases, tmp_path)
    assert audit.audit_easy_batch(request_model, inventory, cases, tmp_path) == first
    assert len(calls) == 1
    changed = request_model.model_copy(update={"model_args": {"temperature": 0.1}})
    audit.audit_easy_batch(changed, inventory, cases, tmp_path)
    assert len(calls) == 2
    prepared = audit.prepare_easy_batch(request_model, inventory, cases, tmp_path)
    raw = json.loads(prepared.path.read_text())
    raw["prompt_version"] = "stale"
    prepared.path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="checkpoint differs"):
        audit.audit_easy_batch(request_model, inventory, cases, tmp_path)
    assert len(calls) == 2


def test_all_cases_resume_and_ledger_is_model_backed(
    inventory, request_model, monkeypatch, tmp_path
):
    monkeypatch.setattr(audit, "build_difficulty_inventory", lambda _: inventory)
    calls = mock_calls(monkeypatch)
    output = tmp_path / "audit.json"
    checkpoints = tmp_path / "checkpoints"
    report = audit.run_easy_audit(request_model, output, checkpoints)
    assert report.keep_count == 100 and report.error_count == 0
    assert report.native_review_status == "pending" and report.machine_gate_passed
    assert len(calls) == 10
    assert audit.run_easy_audit(request_model, output, checkpoints) == report
    assert len(calls) == 10
    rows = list(csv.DictReader(output.with_suffix(".ledger.csv").open()))
    assert len(rows) == 100
    assert list(rows[0]) == list(audit.EasyLedgerRow.model_fields)
    assert len({r["canonical_task_id"] for r in rows}) == 100
    manifest = audit.EasyLedgerManifest.model_validate_json(
        output.with_suffix(".ledger.manifest.json").read_text()
    )
    assert manifest.map_sha256 == report.map_sha256
    assert manifest.audit_sha256 == audit.content_digest(report)
    assert manifest.row_count == 100
    raw = report.model_dump(mode="json")
    raw["decisions"].pop()
    with pytest.raises(ValidationError, match="exactly once"):
        audit.EasyAuditReport.model_validate(raw)
    raw = report.model_dump(mode="json")
    raw["keep_count"] -= 1
    with pytest.raises(ValidationError, match="counts"):
        audit.EasyAuditReport.model_validate(raw)
    changed = request_model.model_copy(update={"revision": "different"})
    with pytest.raises(ValueError, match="stale"):
        audit.run_easy_audit(changed, output, checkpoints)


def test_failures_are_explicit_and_only_missing_batch_retries(
    inventory, request_model, monkeypatch, tmp_path
):
    monkeypatch.setattr(audit, "build_difficulty_inventory", lambda _: inventory)
    calls = mock_calls(monkeypatch)
    success = audit.generate
    first = audit.extract_easy_cases(inventory)[0].canonical_task_id

    def fail(**kwargs):
        payload = audit.EasyAuditPayload.model_validate_json(
            kwargs["messages"][1].content
        )
        if payload.cases[0].canonical_task_id == first:
            raise RuntimeError("controlled failure")
        return success(**kwargs)

    monkeypatch.setattr(audit, "generate", fail)
    report = audit.run_easy_audit(
        request_model, tmp_path / "partial.json", tmp_path / "cp"
    )
    assert report.keep_count == 90 and report.error_count == 10
    assert not report.machine_gate_passed and len(calls) == 9
    monkeypatch.setattr(
        audit, "generate", lambda **_: pytest.fail("inspection called model")
    )
    inspected = audit.inspect_easy_audit(
        request_model, tmp_path / "inspect.json", tmp_path / "cp"
    )
    assert inspected.keep_count == 90 and inspected.error_count == 10
    monkeypatch.setattr(audit, "generate", success)
    final = audit.run_easy_audit(
        request_model, tmp_path / "complete.json", tmp_path / "cp"
    )
    assert final.keep_count == 100 and len(calls) == 10
    with pytest.raises(ValueError, match="new output"):
        audit.run_easy_audit(request_model, tmp_path / "partial.json", tmp_path / "cp")


def test_cli_owns_audit_and_no_call_inspection():
    parser = argparse.ArgumentParser()
    add_intake_difficulty_args(parser.add_subparsers())
    for verb in ("audit-intake-easy", "inspect-intake-easy"):
        args = parser.parse_args(
            [verb, "--request", "r.json", "--output", "o.json", "--checkpoints", "cp"]
        )
        assert callable(args.func)
    plan = parser.parse_args(
        [
            "plan-intake-easy",
            "--map",
            "map.json",
            "--audit",
            "audit.json",
            "--output",
            "plan.json",
            "--feedback",
            "feedback.json",
        ]
    )
    assert plan.feedback == Path("feedback.json") and callable(plan.func)
    revision = parser.parse_args(
        [
            "revise-intake-easy",
            "--request",
            "request.json",
            "--output",
            "map.json",
            "--checkpoints",
            "cp",
        ]
    )
    assert callable(revision.func)
    aggregate = parser.parse_args(
        [
            "report-intake-easy",
            "--report",
            "hi.json",
            "--report",
            "es.json",
            "--output",
            "all.json",
        ]
    )
    assert aggregate.report == [Path("hi.json"), Path("es.json")]


def test_prompt_covers_known_runtime_and_source_false_positives():
    for text in (
        "ONE complication",
        "alternatives",
        "private",
        "service agent",
        "English instructions",
        "ASCII",
        "empty",
        "clinical identity",
        "decoy",
        "Easy",
        "pronunciation",
    ):
        assert text in audit.EASY_AUDIT_PROMPT


def test_changed_map_digest_invalidates_checkpoint_identity(
    inventory, request_model, tmp_path
):
    from tau2.multilingual.factory.intake_difficulty_inventory import (
        DifficultyInventoryPayload,
    )

    original = audit.prepare_easy_batch(
        request_model, inventory, audit.extract_easy_cases(inventory)[:2], tmp_path
    )
    payload = DifficultyInventoryPayload.model_validate(
        inventory.model_dump(exclude={"payload_sha256"})
    )
    payload.map_sha256 = "0" * 64
    changed = audit.DifficultyInventory.model_validate(
        {
            **payload.model_dump(),
            "payload_sha256": audit.content_digest(payload),
        }
    )
    prepared = audit.prepare_easy_batch(
        request_model, changed, audit.extract_easy_cases(changed)[:2], tmp_path
    )
    assert original.input_sha256 != prepared.input_sha256
    assert original.payload.cases == prepared.payload.cases


def test_existing_report_requires_identical_checkpoint_evidence(
    inventory, request_model, monkeypatch, tmp_path
):
    monkeypatch.setattr(audit, "build_difficulty_inventory", lambda _: inventory)
    mock_calls(monkeypatch)
    output, checkpoints = tmp_path / "audit.json", tmp_path / "cp"
    report = audit.run_easy_audit(request_model, output, checkpoints)
    monkeypatch.setattr(
        audit, "generate", lambda **_: pytest.fail("resume called model")
    )
    raw = report.model_dump(mode="json")
    raw["decisions"][0]["evidence"] = "Tampered yet schema-valid review text."
    output.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="differs from its exact checkpoints"):
        audit.run_easy_audit(request_model, output, checkpoints)
    output.write_text(report.model_dump_json(indent=2) + "\n")
    next(checkpoints.glob("*.json")).unlink()
    with pytest.raises(FileNotFoundError, match="No completed Easy checkpoint"):
        audit.run_easy_audit(request_model, output, checkpoints)


@pytest.fixture
def aggregate_inputs(inventory, request_model):
    cases = audit.extract_easy_cases(inventory)
    reports, manifests = [], []
    base = audit.EasyAuditReport(
        request=request_model,
        prompt_version=audit.EASY_AUDIT_PROMPT_VERSION,
        map_sha256=inventory.map_sha256,
        source_sha256=inventory.source_sha256,
        inventory_sha256=audit.content_digest(inventory),
        settings=inventory.settings,
        expected_task_ids=[c.canonical_task_id for c in cases],
        decisions=[keep(c) for c in cases],
        errors=[],
        keep_count=100,
        revise_count=0,
        source_blocked_count=0,
        error_count=0,
        machine_gate_passed=True,
    )
    base_rows = audit.easy_ledger_rows(base, inventory)
    for n, language in enumerate(("hi", "es", "pt", "zh", "ko")):
        report = base.model_copy(
            update={
                "map_sha256": str(n) * 64,
                "settings": base.settings.model_copy(update={"language": language}),
            }
        )
        rows = [r.model_copy(update={"language": language}) for r in base_rows]
        reports.append(report)
        manifests.append(
            audit.EasyLedgerManifest(
                map_sha256=report.map_sha256,
                source_sha256=report.source_sha256,
                inventory_sha256=report.inventory_sha256,
                audit_sha256=audit.content_digest(report),
                csv_sha256=audit.hashlib.sha256(audit._render_ledger(rows)).hexdigest(),
                rows=rows,
            )
        )
    return reports, manifests


def test_five_language_aggregate_preserves_500_rows_and_input_provenance(
    aggregate_inputs, tmp_path
):
    reports, manifests = aggregate_inputs
    aggregate = audit.build_easy_aggregate(reports, manifests)
    assert len(aggregate.rows) == 500
    assert len(aggregate.family_counts) == 50
    assert all(c.keep == 10 for c in aggregate.family_counts)
    assert {i.audit_sha256 for i in aggregate.inputs} == {
        audit.content_digest(r) for r in reports
    }
    files = audit.write_easy_aggregate(aggregate, tmp_path / "all.json")
    assert len(list(csv.DictReader(files.ledger_csv.open()))) == 500
    assert audit.write_easy_aggregate(aggregate, tmp_path / "all.json") == files
    assert aggregate.native_review_status == "pending"


@pytest.mark.parametrize("mode", ["missing", "duplicate", "source", "audit", "row"])
def test_aggregate_rejects_missing_languages_or_unbound_rows(aggregate_inputs, mode):
    reports, manifests = aggregate_inputs
    if mode == "missing":
        reports.pop()
        manifests.pop()
    elif mode == "duplicate":
        reports[-1] = reports[0]
        manifests[-1] = manifests[0]
    elif mode == "source":
        reports[0] = reports[0].model_copy(update={"source_sha256": "f" * 64})
    elif mode == "audit":
        reports[0] = reports[0].model_copy(
            update={
                "request": reports[0].request.model_copy(update={"revision": "changed"})
            }
        )
    else:
        manifests[0] = manifests[0].model_copy(deep=True)
        manifests[0].rows[0].evidence = "Not the report decision."
        manifests[0].csv_sha256 = audit.hashlib.sha256(
            audit._render_ledger(manifests[0].rows)
        ).hexdigest()
    with pytest.raises(ValueError):
        audit.build_easy_aggregate(reports, manifests)
