"""Completed time authoring resumes require exact checkpoint and map evidence."""

import json

import pytest

from tau2.multilingual.factory import intake_time_authoring as author
from tau2.multilingual.factory.localization_artifact import write_localization_map
from tau2.multilingual.localization import LocalizedForms, ProseEntityBinding
from tests.test_multilingual import test_intake_time_authoring as fixtures

author_request = fixtures.author_request
inventory = fixtures.inventory
proposal = fixtures.proposal


@pytest.fixture
def completed(author_request, monkeypatch, tmp_path):
    calls = fixtures.install_mock(monkeypatch)
    output = tmp_path / "revised.json"
    checkpoints = tmp_path / "checkpoints"
    revised = author.revise_intake_time_surfaces(author_request, output, checkpoints)
    assert len(calls) == 1

    def forbidden_generate(**kwargs):
        pytest.fail("Existing-output resume must never call a provider")

    monkeypatch.setattr(author, "generate", forbidden_generate)
    return output, checkpoints, revised


def test_completed_resume_requires_and_reconstructs_evidence(author_request, completed):
    output, checkpoints, revised = completed
    before = output.read_bytes()
    assert (
        author.revise_intake_time_surfaces(author_request, output, checkpoints)
        == revised
    )
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "change", ["selected_surface", "untargeted_surface", "binding"]
)
def test_existing_output_rejects_tampering(author_request, completed, change):
    output, checkpoints, revised = completed
    altered = revised.model_copy(deep=True)
    if change == "binding":
        altered.prose_bindings.append(
            ProseEntityBinding(
                entry_id=altered.entries[0].id,
                source="Unselected review identity",
                forms=LocalizedForms(
                    native="가짜", romanized="gajja", english="Test only"
                ),
                kind="generated",
                reason="Unrequested test binding",
                provenance=altered.provenance,
            )
        )
    else:
        selected = change == "selected_surface"
        surface = next(
            row
            for row in altered.intake_caller_surfaces
            if (row.canonical_task_id in author_request.canonical_task_ids) == selected
        )
        surface.native_spoken = "시험 표현"
        surface.romanized_spoken = "siheom pyohyeon"
    output.unlink()
    write_localization_map(altered, output)
    before = output.read_bytes()
    with pytest.raises(ValueError, match="Existing time-authoring output differs"):
        author.revise_intake_time_surfaces(author_request, output, checkpoints)
    assert output.read_bytes() == before


@pytest.mark.parametrize(
    "change", ["missing", "request", "incomplete_feasibility", "invalid_surface"]
)
def test_existing_output_rejects_missing_or_invalid_checkpoints(
    author_request, completed, change
):
    output, checkpoints, _ = completed
    checkpoint = next(checkpoints.glob("*.json"))
    raw = json.loads(checkpoint.read_text())
    if change == "missing":
        checkpoint.unlink()
        error = "Missing completed.*checkpoint"
    else:
        if change == "request":
            raw["request"]["model"] = "different-model"
            error = "checkpoint differs"
        elif change == "incomplete_feasibility":
            raw["response"]["surfaces"][0]["feasibility"] = "source_limited"
            raw["response"]["surfaces"][0]["limitation"] = ""
            error = "limitation"
        else:
            raw["response"]["surfaces"][0]["native_spoken"] = "오후 열 시 이십 분 전"
            error = "omit its day period"
        checkpoint.write_text(json.dumps(raw))
    before = output.read_bytes()
    with pytest.raises(ValueError, match=error):
        author.revise_intake_time_surfaces(author_request, output, checkpoints)
    assert output.read_bytes() == before
