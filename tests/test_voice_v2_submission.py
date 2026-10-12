"""Offline checks for the separate Voice V2 publication contract."""

import csv
import json
import shutil
import socket
from pathlib import Path

import pytest
from pydantic import ValidationError

from tau2.data_model.simulation import RewardInfo, SimulationRun
from tau2.runner.presets import load_frame
from tau2.scripts.leaderboard.generate_schema import (
    generate_schema,
    generate_voice_v2_schema,
)
from tau2.scripts.leaderboard.voice_v2_publication import (
    _validate_cell_frame,
    _validate_simulation_files,
    build_voice_v2_submission,
    prepare_voice_v2_package,
    validate_voice_v2_package,
)
from tau2.scripts.leaderboard.voice_v2_submission import (
    CallOutcome,
    InteractionFacts,
    VoiceV2Cell,
    coverage,
    experience_outcome,
    interaction_outcome,
)


def _write_saved_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture
def forbid_paid_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Publication must use only saved offline inputs")

    monkeypatch.setattr("tau2.utils.llm_utils.generate", forbidden)
    monkeypatch.setattr("litellm.completion", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture(scope="module")
def saved_publication_inputs(tmp_path_factory):
    """Synthetic historical export: the real 18-cell frame, no provider calls.

    Judge records intentionally use the historical adapter's minimal wire format;
    this does not assert current catalog/identity validation for that adapter.
    """
    root = tmp_path_factory.mktemp("saved-voice-v2")
    source = root / "source"
    judges = root / "judges"
    metadata = {
        "model_name": "offline-fixture",
        "model_organization": "Test",
        "submitting_organization": "Test",
        "submission_date": "2026-10-02",
        "modality": "voice",
        "contact_info": {"email": "review@example.org"},
    }
    _write_saved_json(source / "submission.json", metadata)
    official = load_frame("voice", num_trials=1)
    facts = []
    for cell in official.cells:
        language, domain = cell.language, cell.domain
        directory = source / "trajectories" / language / domain
        index = []
        for task_id in cell.canonical_task_ids:
            simulation_id = f"{domain}-{task_id}"
            simulation = SimulationRun(
                id=simulation_id,
                task_id=task_id,
                trial=0,
                timestamp="2026-10-02T00:00:00",
                start_time="2026-10-02T00:00:00",
                end_time="2026-10-02T00:00:01",
                duration=1,
                termination_reason="agent_stop",
                reward_info=RewardInfo(reward=1.0),
            )
            _write_saved_json(
                directory / "simulations" / f"{simulation_id}.json",
                simulation.model_dump(mode="json"),
            )
            index.append(
                {
                    "id": simulation_id,
                    "task_id": task_id,
                    "trial": 0,
                    "reward": 1.0,
                    "termination_reason": "agent_stop",
                }
            )
            _write_saved_json(
                judges / "gemini" / language / f"{simulation_id}.json",
                [
                    {
                        "identity": {
                            "input": {"simulation_id": simulation_id},
                            "measure": measure,
                        },
                        "verdict": {
                            "units": [
                                {
                                    "unit_id": "live-segment:0",
                                    "opportunity": True,
                                    "violation": False,
                                }
                            ]
                        },
                    }
                    for measure in ("naturalness", "speech_fidelity")
                ],
            )
            facts.append(
                {
                    "provider": "gemini",
                    "language": language,
                    "domain": domain,
                    "simulation_id": simulation_id,
                    "had_unresponsive_period": False,
                    "nonresponse_failure": 0,
                    "interruption_count": 0,
                    "selectivity_error_count": 0,
                    "selectivity_failure": 0,
                    "monologue_failure": 0,
                    "tool_misuse_failure": 0,
                }
            )
        _write_saved_json(
            directory / "results.json",
            {
                "info": {
                    "benchmark": {
                        "benchmark_version": official.benchmark_version,
                        "language": language,
                        "domain": domain,
                        "selected_task_ids": cell.canonical_task_ids,
                        "canonical_task_ids": cell.canonical_task_ids,
                        "trials": [0],
                        "frame_revision": official.revision,
                        "binding_revision": cell.binding_revision,
                        "condition_fingerprint": "a" * 64,
                        "conditions": {
                            "audio_native_config": {
                                "provider": "gemini",
                                "model": "offline-fixture",
                            }
                        },
                    },
                    "language": language,
                    "environment_info": {"domain_name": domain},
                    "agent_info": {"llm": "offline-fixture"},
                    "user_info": {"llm": "offline-caller"},
                    "num_trials": 1,
                    "language_tasks": {
                        task_id: {"canonical_task_id": task_id}
                        for task_id in cell.canonical_task_ids
                    },
                },
                "tasks": [{"id": task_id} for task_id in cell.canonical_task_ids],
                "simulation_index": index,
            },
        )
    facts_path = root / "facts.csv"
    with facts_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(facts[0]))
        writer.writeheader()
        writer.writerows(facts)
    return source, facts_path, judges


@pytest.fixture
def prepared_publication(saved_publication_inputs, tmp_path, forbid_paid_calls):
    source, facts, judges = saved_publication_inputs
    destination = tmp_path / "package"
    submission = prepare_voice_v2_package(
        source, destination, "gemini", facts, judges, source / "submission.json"
    )
    return destination, submission


def test_saved_publication_round_trip_without_model_calls(prepared_publication):
    root, submission = prepared_publication
    assert validate_voice_v2_package(root) == submission
    rebuilt, provenance = build_voice_v2_submission(root, "gemini", submission)
    assert rebuilt == submission
    assert len(submission.benchmark_submission.trials) == 900
    assert len(submission.benchmark_submission.sources) == 18
    assert (
        len(provenance.files) == 1819
    )  # 900 simulations + 900 judges + 18 cells + CSV
    for domains in submission.results.values():
        for cell in domains.values():
            assert cell.selected == cell.scored == 50
            assert cell.pass_1 == 100
            assert cell.interaction.unscored == cell.experience.unscored == 0


@pytest.mark.parametrize("kind", ["simulation", "judge", "facts", "missing_judge"])
def test_saved_publication_rejects_changed_or_missing_sources(
    prepared_publication, kind
):
    root, _ = prepared_publication
    if kind == "simulation":
        path = next((root / "trajectories/en/airline/simulations").glob("*.json"))
    elif kind in {"judge", "missing_judge"}:
        path = next((root / "judges/gemini/en").glob("*.json"))
    else:
        path = root / "interaction_facts.csv"
    if kind == "missing_judge":
        path.unlink()
    else:
        path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="source SHA-256 mismatch"):
        validate_voice_v2_package(root)


def test_saved_publication_rejects_different_published_scores(prepared_publication):
    root, _ = prepared_publication
    path = root / "submission.json"
    data = json.loads(path.read_text())
    cell = data["results"]["en"]["airline"]
    cell.update({"pass": 49, "fail": 1, "pass_1": 98})
    _write_saved_json(path, data)
    with pytest.raises(ValueError):
        validate_voice_v2_package(root)


def test_prepare_rejects_missing_saved_facts_and_cleans_partial_output(
    saved_publication_inputs, tmp_path, forbid_paid_calls
):
    source, facts, judges = saved_publication_inputs
    incomplete = tmp_path / "facts.csv"
    rows = facts.read_text().splitlines()
    incomplete.write_text("\n".join(rows[:-1]) + "\n")
    destination = tmp_path / "package"
    with pytest.raises(ValueError, match="missing saved interaction facts"):
        prepare_voice_v2_package(
            source,
            destination,
            "gemini",
            incomplete,
            judges,
            source / "submission.json",
        )
    assert not destination.exists()
    assert facts.read_text().splitlines() == rows


def test_missing_measure_is_unscored_not_a_failed_call(
    saved_publication_inputs, tmp_path, forbid_paid_calls
):
    source, facts, judges = saved_publication_inputs
    partial_judges = tmp_path / "judges"
    shutil.copytree(judges, partial_judges)
    path = next((partial_judges / "gemini/en").glob("airline-*.json"))
    _write_saved_json(path, json.loads(path.read_text())[:1])
    submission = prepare_voice_v2_package(
        source,
        tmp_path / "package",
        "gemini",
        facts,
        partial_judges,
        source / "submission.json",
    )
    cell = submission.results["en"]["airline"]
    assert cell.pass_1 == 100
    assert cell.experience.unscored == 1
    assert cell.interaction.unscored == 0


def test_interaction_two_flags_and_missing_component() -> None:
    facts = InteractionFacts(
        nonresponse=False,
        interruption_count=4,
        selectivity_error_count=3,
        selectivity_available=True,
        monologue=True,
        tool_misuse=False,
    )
    assert interaction_outcome(facts) == CallOutcome.FAIL
    assert (
        interaction_outcome(facts.model_copy(update={"interruption_count": 3}))
        == CallOutcome.PASS
    )
    assert (
        interaction_outcome(facts.model_copy(update={"monologue": None}))
        == CallOutcome.UNSCORED
    )
    assert (
        interaction_outcome(
            facts.model_copy(update={"korean_unresponsive_period": True})
        )
        == CallOutcome.EXCLUDED
    )


def test_experience_strictly_more_than_quarter_and_complete() -> None:
    assert (
        experience_outcome(4, [True, False, False, False], [False] * 4)
        == CallOutcome.PASS
    )
    assert (
        experience_outcome(4, [True, True, False, False], [False] * 4)
        == CallOutcome.FAIL
    )
    assert (
        experience_outcome(4, [False] * 4, [False, None, False, False])
        == CallOutcome.UNSCORED
    )


def test_cell_counts_separate_infrastructure_and_missing_judges() -> None:
    base = {
        "selected": 50,
        "scored": 49,
        "pass": 30,
        "fail": 19,
        "infrastructure": 1,
        "pass_1": 100 * 30 / 49,
        "interaction": coverage(
            [CallOutcome.PASS] * 30 + [CallOutcome.FAIL] * 18 + [CallOutcome.UNSCORED],
            "voice-v2-interaction-call-v1",
        ).model_dump(),
        "experience": coverage(
            [CallOutcome.PASS] * 29
            + [CallOutcome.FAIL] * 17
            + [CallOutcome.UNSCORED] * 3,
            "voice-v2-experience-call-v1",
        ).model_dump(),
    }
    cell = VoiceV2Cell.model_validate(base)
    assert cell.pass_1 == pytest.approx(100 * 30 / 49)
    assert cell.experience.unscored == 3
    with pytest.raises(ValidationError, match="selected != scored"):
        VoiceV2Cell.model_validate({**base, "infrastructure": 0})


def test_frame_rejects_duplicate_task_trial_even_with_50_entries() -> None:
    official = load_frame("voice", num_trials=1)
    cell = next(
        item
        for item in official.cells
        if (item.language, item.domain) == ("en", "airline")
    )
    selected = cell.canonical_task_ids
    data = {
        "info": {
            "benchmark": {
                "benchmark_version": "voice-v2",
                "language": "en",
                "domain": "airline",
                "selected_task_ids": selected,
                "trials": [0],
                "frame_revision": official.revision,
                "binding_revision": cell.binding_revision,
                "canonical_task_ids": selected,
                "condition_fingerprint": "a" * 64,
            },
            "language": "en",
            "environment_info": {"domain_name": "airline"},
            "num_trials": 1,
            "language_tasks": {
                task_id: {"canonical_task_id": task_id} for task_id in selected
            },
        },
        "tasks": [{"id": task_id} for task_id in selected],
    }
    index = [
        {
            "id": f"simulation-{task_id}",
            "task_id": task_id,
            "trial": 0,
            "reward": 1.0,
            "termination_reason": "agent_stop",
        }
        for task_id in selected
    ]
    assert _validate_cell_frame(data, index, "en", "airline", official)["0"] == "0"
    index[-1] = {**index[-1], "task_id": "0"}
    with pytest.raises(ValueError, match="missing or duplicate task/trial"):
        _validate_cell_frame(data, index, "en", "airline", official)

    data["info"]["language_tasks"]["1"]["canonical_task_id"] = "0"
    with pytest.raises(ValueError, match="selected 50-task frame mismatch"):
        _validate_cell_frame(data, index, "en", "airline", official)


def test_voice_v2_schema_is_separate_from_legacy_schema() -> None:
    legacy = json.loads(generate_schema())
    voice_v2 = json.loads(generate_voice_v2_schema())
    assert "benchmark_submission" not in legacy["properties"]
    assert "benchmark_submission" in voice_v2["properties"]
    assert "provenance_sha256" in voice_v2["required"]


def test_index_must_agree_with_saved_simulation(tmp_path) -> None:
    simulation_dir = tmp_path / "trajectories/en/airline/simulations"
    simulation_dir.mkdir(parents=True)
    (simulation_dir / "sim-1.json").write_text(
        json.dumps(
            {
                "id": "sim-1",
                "task_id": "task-1",
                "trial": 0,
                "termination_reason": "agent_stop",
                "reward_info": {"reward": 1.0},
            }
        )
    )
    index = [
        {
            "id": "sim-1",
            "task_id": "task-1",
            "trial": 0,
            "termination_reason": "agent_stop",
            "reward": 1.0,
        }
    ]
    _validate_simulation_files(tmp_path, index, "en", "airline")
    with pytest.raises(ValueError, match="reward mismatch"):
        _validate_simulation_files(
            tmp_path, [{**index[0], "reward": 0.0}], "en", "airline"
        )
