"""Generic and crossed Pass^3 retain distinct, reproducible contracts."""

import argparse
from types import SimpleNamespace

import pytest

from tau2.cli import add_run_args
from tau2.data_model.persona import InterruptTendency, PersonaConfig
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import Results, SimulationIndexEntry, VoiceRunConfig
from tau2.data_model.tasks import Task
from tau2.data_model.voice import SynthesisConfig
from tau2.runner.cli import add_pass3_args
from tau2.runner.pass3 import CROSSED_CONDITIONS, summarize_crossed_pass3
from tau2.user_simulation_voice_presets import (
    VoiceSelection,
    get_or_load_task_voice_config,
    sample_voice_config,
)


def test_pass3_default_is_generic_and_crossed_accepts_other_voice_domains():
    parser = argparse.ArgumentParser()
    seen = []
    add_pass3_args(parser, add_run_args=add_run_args, run_command=seen.append)

    generic = parser.parse_args(["--domain", "airline", "--dry-run"])
    generic.func(generic)
    assert len(seen) == 1
    assert seen[0].num_trials == 3
    assert seen[0].domain == ["airline"]

    seen.clear()
    crossed = parser.parse_args(
        [
            "--pass3-mode",
            "crossed",
            "--audio-native",
            "--domain",
            "airline",
            "--save-to",
            "fixture",
            "--dry-run",
        ]
    )
    crossed.func(crossed)
    assert [call.pass3_condition for call in seen] == [
        condition.label for condition in CROSSED_CONDITIONS
    ]
    assert [call.seed for call in seen] == [crossed.seed] * 3
    assert [call.voice_sampling_seed for call in seen] == [
        condition.seed for condition in CROSSED_CONDITIONS
    ]
    assert all(call.domain == ["airline"] for call in seen)


def test_crossed_voice_modes_change_acoustics_and_selected_caller_behavior():
    source = SynthesisConfig()
    selection = VoiceSelection(
        persona_name="reviewed",
        persona_config=PersonaConfig(interrupt_tendency=InterruptTendency.WAITS),
        vocal_tics=[],
        non_directed_phrases=[],
        speech_insert_events_per_minute=0.7,
    )
    regular = sample_voice_config(9401, source, selection=selection)
    channel = sample_voice_config(
        9402,
        source,
        selection=selection,
        channel_effects_mode="heavy",
        speech_effects_mode="light",
    )
    speech = sample_voice_config(
        9403,
        source,
        selection=selection,
        channel_effects_mode="light",
        speech_effects_mode="heavy",
    )

    assert regular.speech_effects_config.speech_insert_events_per_minute == 0.7
    assert channel.channel_effects_config.frame_drop_rate == 0.03
    assert channel.source_effects_config.noise_snr_db == 10
    assert channel.speech_effects_config.speech_insert_events_per_minute == 0
    assert channel.backchannel_min_threshold is None
    assert not channel.enable_interruptions
    assert speech.channel_effects_config.frame_drop_rate == 0
    assert speech.speech_effects_config.speech_insert_events_per_minute == 1.5
    assert speech.persona_config.interrupt_tendency == InterruptTendency.INTERRUPTS
    assert speech.enable_interruptions


def test_crossed_voice_seeds_preserve_task_framing():
    from tau2.runner.batch import _load_run_tasks
    from tau2.runner.elicitation import sample_run_complication
    from tau2.runner.helpers import get_info

    def configs(domain, language):
        return [
            VoiceRunConfig(
                domain=domain,
                language=language,
                num_tasks=1,
                seed=300,
                voice_sampling_seed=condition.seed,
                audio_native_config=AudioNativeConfig(),
            )
            for condition in CROSSED_CONDITIONS
        ]

    intake = configs("intake", "en")
    task = _load_run_tasks(intake[0])[0]
    assert (
        len(
            {
                sample_run_complication(config, task).model_dump_json()
                for config in intake
            }
        )
        == 1
    )

    retail = configs("retail", "ko")
    task = _load_run_tasks(retail[0])[0]
    contexts = [
        get_info(config, tasks=[task]).language_tasks[task.id] for config in retail
    ]
    assert contexts[0] == contexts[1] == contexts[2]


def test_crossed_regular_resamples_instead_of_loading_a_presampled_file(
    monkeypatch, tmp_path
):
    from tau2 import user_simulation_voice_presets as presets

    saved = sample_voice_config(42, SynthesisConfig())
    file = tmp_path / "presampled.json"
    file.touch()
    monkeypatch.setattr(presets, "get_task_voice_configs_path", lambda _: file)
    monkeypatch.setattr(
        presets,
        "load_task_voice_configs",
        lambda _: SimpleNamespace(get_config=lambda *args: saved),
    )

    actual = get_or_load_task_voice_config(
        "airline", "task", 9401, "regular", SynthesisConfig(), force_sample=True
    )
    assert actual is not saved
    assert (
        actual.model_dump() == sample_voice_config(9401, SynthesisConfig()).model_dump()
    )


def _result(condition, rewards):
    info = SimpleNamespace(
        environment_info=SimpleNamespace(domain_name="airline"),
        language=None,
        git_commit="commit",
        localize_db=True,
        language_artifacts=None,
        language_tasks=None,
        semantic_communicate=None,
        elicitation=None,
        agent_info="agent",
        user_info="user",
        audio_native_config="audio",
        retrieval_config=None,
        retrieval_config_kwargs=None,
        seed=300,
        voice_sampling_seed=condition.seed,
        pass3_condition=condition.label,
        num_trials=1,
        channel_effects_mode=condition.channel_effects_mode,
        speech_effects_mode=condition.speech_effects_mode,
    )
    index = [
        SimulationIndexEntry(
            id=f"{condition.label}-{task}", task_id=task, trial=0, reward=reward
        )
        for task, reward in rewards.items()
    ]
    return Results.model_construct(
        info=info,
        tasks=[Task.model_construct(id=task) for task in rewards],
        simulation_index=index,
    )


def test_crossed_pass3_scores_only_aligned_complete_conditions():
    results = {
        condition.label: _result(
            condition,
            {"one": 1.0, "two": 0.0 if condition.label == "speech_heavy" else 1.0},
        )
        for condition in CROSSED_CONDITIONS
    }
    paths = {
        condition.label: f"/{condition.label}/results.json"
        for condition in CROSSED_CONDITIONS
    }
    report = summarize_crossed_pass3(results, paths)
    assert report.passed_all == 1
    assert report.total_tasks == 2
    assert report.pass3 == 0.5
    assert report.source_paths == paths

    results["speech_heavy"].info.seed = 301
    with pytest.raises(ValueError, match="run settings or task frames differ"):
        summarize_crossed_pass3(results, paths)
    results["speech_heavy"].info.seed = 300

    results["speech_heavy"].simulation_index = results["speech_heavy"].simulation_index[
        :1
    ]
    with pytest.raises(ValueError, match="missing or unexpected"):
        summarize_crossed_pass3(results, paths)

    results["speech_heavy"] = _result(CROSSED_CONDITIONS[2], {"one": 1.0, "two": 0.0})
    results["speech_heavy"].simulation_index[0].reward = None
    with pytest.raises(ValueError, match="unscored"):
        summarize_crossed_pass3(results, paths)
