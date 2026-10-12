"""Hard time repairs use exact audit evidence and immutable source truth."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2.multilingual.factory import intake_time_authoring as author
from tau2.multilingual.factory.intake_difficulty_audit import (
    AUDIT_PROMPT_VERSION,
    DifficultyAuditReport,
    DifficultyAuditRequest,
    build_difficulty_inventory,
)
from tau2.multilingual.factory.localization_artifact import (
    read_localization_map,
    write_localization_map,
)
from tau2.multilingual.localization import IntakeCallerSurface
from tau2.utils.artifacts import content_digest, write_artifact
from tau2.utils.utils import DATA_DIR
from tests.test_multilingual.test_intake_difficulty_audit import decision


@pytest.fixture(scope="module")
def proposal():
    return read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/ko/localization/intake.json"
    )


@pytest.fixture(scope="module")
def inventory(proposal):
    return build_difficulty_inventory(proposal)


@pytest.fixture
def author_request(tmp_path, inventory):
    map_file = Path(DATA_DIR) / "tau2/multilingual/ko/localization/intake.json"
    audit_file = tmp_path / "audit.json"
    decisions = [decision(case) for case in inventory.cases]
    for d in decisions:
        if d.canonical_task_id == "intake_times_hard_09":
            d.disposition = "revise"
            d.correction_instructions = [
                "Use a natural interpreted local time expression and omit its privately known period."
            ]
            d.repair_targets = ["caller_surface"]
    report = DifficultyAuditReport(
        request=DifficultyAuditRequest(
            map_file=map_file, revision="audit-v1", model="test", model_args={}
        ),
        prompt_version=AUDIT_PROMPT_VERSION,
        map_sha256=inventory.map_sha256,
        source_sha256=inventory.source_sha256,
        inventory_sha256=content_digest(inventory),
        settings=inventory.settings,
        expected_task_ids=[case.canonical_task_id for case in inventory.cases],
        decisions=decisions,
        errors=[],
        keep_count=99,
        revise_count=1,
        source_blocked_count=0,
        error_count=0,
        machine_gate_passed=False,
    )
    write_artifact(report, audit_file)
    return author.IntakeTimeAuthoringRequest(
        map_file=map_file,
        audit_file=audit_file,
        canonical_task_ids=["intake_times_hard_09"],
        revision="time-repair-v1",
        model="test",
        model_args={"temperature": 0},
        batch_size=1,
        max_concurrency=1,
    )


def install_mock(monkeypatch, transform=None):
    calls = []

    def generate(**kwargs):
        assert kwargs["call_name"] == "intake_hard_time_authoring"
        assert kwargs["messages"][0].content == author.TIME_AUTHORING_PROMPT
        assert (
            kwargs["response_format"]["json_schema"]["schema"]
            == author.TimeAuthoringBatch.model_json_schema()
        )
        payload = author.TimeAuthoringPayload.model_validate_json(
            kwargs["messages"][1].content
        )
        calls.append(payload)
        assert len(payload.cases[0].case.easy_comparators) == 10
        assert (
            payload.cases[0].review_feedback is not None
            or payload.cases[0].decision.correction_instructions
        )
        assert all(row.case.family == "times" for row in payload.cases)
        raw = {
            "surfaces": [
                {
                    "canonical_task_id": row.case.canonical_task_id,
                    "canonical_value": row.case.source_value,
                    "native_spoken": "열 시 이십 분 전",
                    "romanized_spoken": "yeol si isip bun jeon",
                    "rationale": "Minutes-before expression adds interpretation and explicit period clarification relative to simple Easy clocks.",
                    "feasibility": "feasible",
                    "limitation": "",
                }
                for row in payload.cases
            ]
        }
        if transform:
            transform(raw)
        return SimpleNamespace(content=json.dumps(raw))

    monkeypatch.setattr(author, "generate", generate)
    return calls


def test_fixed_authoring_preserves_source_entries_and_roundtrips(
    author_request, proposal, monkeypatch, tmp_path
):
    calls = install_mock(monkeypatch)
    output = tmp_path / "revised.json"
    revised = author.revise_intake_time_surfaces(
        author_request, output, tmp_path / "checkpoints"
    )
    assert len(calls) == 1
    assert revised.source == proposal.source
    assert revised.entries == proposal.entries
    assert revised.status == "draft"
    surfaces = {row.canonical_task_id: row for row in revised.intake_caller_surfaces}
    selected = surfaces["intake_times_hard_09"]
    assert selected.canonical_value == "9:40 PM"
    assert selected.provenance.parent_map_sha256 == content_digest(proposal)
    assert all(
        surfaces[row.canonical_task_id] == row
        for row in proposal.intake_caller_surfaces
        if row.canonical_task_id != selected.canonical_task_id
    )
    assert read_localization_map(output) == revised
    assert (
        author.revise_intake_time_surfaces(
            author_request, output, tmp_path / "checkpoints"
        )
        == revised
    )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "mode", ["wrong_time", "missing", "extra", "romanized_native", "period_leak"]
)
def test_invalid_generation_cannot_write_candidate(
    author_request, monkeypatch, tmp_path, mode
):
    def corrupt(raw):
        if mode == "wrong_time":
            raw["surfaces"][0]["canonical_value"] = "9:40 AM"
        elif mode == "missing":
            raw["surfaces"] = []
        elif mode == "extra":
            raw["surfaces"].append(raw["surfaces"][0])
        elif mode == "period_leak":
            raw["surfaces"][0]["native_spoken"] = "오후 열 시 이십 분 전"
        else:
            raw["surfaces"][0]["native_spoken"] = "yeol si isip bun jeon"

    install_mock(monkeypatch, corrupt)
    output = tmp_path / "bad.json"
    with pytest.raises(ValueError):
        author.revise_intake_time_surfaces(
            author_request, output, tmp_path / "checkpoints"
        )
    assert not output.exists()


def test_stale_audit_and_nonrepair_selection_fail_before_llm(
    author_request, monkeypatch, tmp_path
):
    calls = install_mock(monkeypatch)
    wrong = author_request.model_copy(
        update={"canonical_task_ids": ["intake_times_hard_08"]}
    )
    with pytest.raises(ValueError, match="caller_surface|revise"):
        author.revise_intake_time_surfaces(
            wrong, tmp_path / "wrong.json", tmp_path / "checkpoints"
        )
    raw = json.loads(author_request.audit_file.read_text())
    raw["map_sha256"] = "0" * 64
    author_request.audit_file.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="audit.*map|stale"):
        author.revise_intake_time_surfaces(
            author_request, tmp_path / "stale.json", tmp_path / "checkpoints"
        )
    assert calls == []


def test_checkpoint_reuse_and_corruption(author_request, monkeypatch, tmp_path):
    calls = install_mock(monkeypatch)
    first = author.revise_intake_time_surfaces(
        author_request, tmp_path / "one.json", tmp_path / "checkpoints"
    )
    assert (
        author.revise_intake_time_surfaces(
            author_request, tmp_path / "two.json", tmp_path / "checkpoints"
        )
        == first
    )
    assert len(calls) == 1

    path = next((tmp_path / "checkpoints").glob("*.json"))
    raw = json.loads(path.read_text())
    raw["prompt_version"] = "wrong"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="checkpoint"):
        author.revise_intake_time_surfaces(
            author_request, tmp_path / "three.json", tmp_path / "checkpoints"
        )
    assert len(calls) == 1


def test_untargeted_overrides_survive_a_targeted_repair(
    author_request, proposal, monkeypatch, tmp_path
):
    # Existing draft expressions are retained byte-for-byte, even when this
    # authoring stage adds a different task and updates the map's revision.
    selected = next(
        d for d in proposal.source.documents if d.id == "intake_localization"
    ).content["stimuli"]
    row = next(s for s in selected if s["canonical_task_id"] == "intake_times_hard_08")
    prior = IntakeCallerSurface(
        canonical_task_id=row["canonical_task_id"],
        canonical_value=row["written"],
        native_spoken="시험 표현",
        romanized_spoken="siheom pyohyeon",
        rationale="Synthetic existing test fixture, not production authoring.",
        provenance=proposal.provenance,
    )
    changed = proposal.model_copy(update={"intake_caller_surfaces": [prior]})
    map_file = tmp_path / "prior.json"
    write_localization_map(changed, map_file)
    inventory = build_difficulty_inventory(changed)
    report = DifficultyAuditReport.model_validate_json(
        author_request.audit_file.read_text()
    )
    report.map_sha256 = content_digest(changed)
    report.inventory_sha256 = content_digest(inventory)
    audit_file = tmp_path / "prior-audit.json"
    write_artifact(report, audit_file)
    revised_request = author_request.model_copy(
        update={"map_file": map_file, "audit_file": audit_file}
    )
    install_mock(monkeypatch)
    output = tmp_path / "with-retained.json"
    revised = author.revise_intake_time_surfaces(
        revised_request, output, tmp_path / "checkpoints"
    )
    assert (
        next(
            s
            for s in revised.intake_caller_surfaces
            if s.canonical_task_id == prior.canonical_task_id
        )
        == prior
    )
    assert len(revised.intake_caller_surfaces) == 2


@pytest.mark.parametrize(
    "disposition,targets",
    [
        ("revise", ["primary"]),
        ("revise", ["omission_instruction"]),
        ("source_blocked", []),
    ],
)
def test_primary_and_source_blocker_allow_surface_attempt_without_reclassification(
    inventory, disposition, targets
):
    case = next(
        c for c in inventory.cases if c.canonical_task_id == "intake_times_hard_09"
    )
    finding = decision(case).model_dump()
    finding.update(
        disposition=disposition,
        repair_targets=targets,
        correction_instructions=[
            "Try a natural expression without changing the fixed time."
        ]
        if targets
        else [],
        source_blocked_reason="The fixed source time may not support a locally Hard expression."
        if disposition == "source_blocked"
        else None,
    )
    typed = author.TimeAuthoringCase.model_validate({"case": case, "decision": finding})
    assert typed.decision.disposition == disposition


def test_source_limited_feedback_only_repair_retains_checkpoint_evidence(
    author_request, proposal, monkeypatch, tmp_path
):
    feedback = author.TimeReviewFeedback(
        canonical_task_id="intake_times_hard_09",
        map_sha256=content_digest(proposal),
        concern="The prior offset is contrived; prefer ordinary basic speech if Hard is infeasible.",
    )
    request = author_request.model_copy(
        update={"audit_file": None, "review_feedback": [feedback]}
    )

    def limited(raw):
        row = raw["surfaces"][0]
        row.update(
            native_spoken="아홉 시 사십 분",
            romanized_spoken="ahop si sasip bun",
            feasibility="source_limited",
            limitation="An ordinary expression for this fixed answer is not harder than the supplied Easy controls.",
        )

    calls = install_mock(monkeypatch, limited)
    revised = author.revise_intake_time_surfaces(
        request, tmp_path / "limited.json", tmp_path / "checkpoints"
    )
    assert calls[0].audit_sha256 is None
    assert calls[0].cases[0].decision is None
    assert calls[0].cases[0].review_feedback == feedback
    saved = author.TimeAuthoringCheckpoint.model_validate_json(
        next((tmp_path / "checkpoints").glob("*.json")).read_text()
    )
    assert saved.response.surfaces[0].feasibility == "source_limited"
    assert saved.response.surfaces[0].limitation
    assert revised.entries == proposal.entries
    assert revised.source == proposal.source
    selected = next(
        row
        for row in revised.intake_caller_surfaces
        if row.canonical_task_id == "intake_times_hard_09"
    )
    assert selected.canonical_value == "9:40 PM"
    # The new authoring metadata stays in the provenance-bearing checkpoint,
    # leaving the caller-surface/map serialization contract unchanged.
    assert "feasibility" not in selected.model_dump()


@pytest.mark.parametrize("mode", ["stale", "unselected", "duplicate", "missing"])
def test_review_feedback_must_bind_exact_selected_map_before_calls(
    author_request, proposal, monkeypatch, tmp_path, mode
):
    feedback = author.TimeReviewFeedback(
        canonical_task_id="intake_times_hard_09",
        map_sha256="0" * 64 if mode == "stale" else content_digest(proposal),
        concern="Observed arithmetic artificiality; retain natural basic speech if needed.",
    )
    if mode == "unselected":
        feedback.canonical_task_id = "intake_times_hard_08"
    rows = (
        []
        if mode == "missing"
        else [feedback, feedback]
        if mode == "duplicate"
        else [feedback]
    )
    request = author_request.model_copy(
        update={"audit_file": None, "review_feedback": rows}
    )
    calls = install_mock(monkeypatch)
    with pytest.raises(ValueError, match="feedback|selected"):
        author.revise_intake_time_surfaces(
            request, tmp_path / "invalid.json", tmp_path / "checkpoints"
        )
    assert not calls


@pytest.mark.parametrize(
    "feasibility,limitation",
    [("source_limited", ""), ("feasible", "Cannot be naturally Hard")],
)
def test_inconsistent_feasibility_is_rejected(
    author_request, monkeypatch, tmp_path, feasibility, limitation
):
    def inconsistent(raw):
        raw["surfaces"][0].update(feasibility=feasibility, limitation=limitation)

    install_mock(monkeypatch, inconsistent)
    with pytest.raises(ValueError, match="limitation"):
        author.revise_intake_time_surfaces(
            author_request, tmp_path / "invalid.json", tmp_path / "checkpoints"
        )
    assert not (tmp_path / "invalid.json").exists()
