"""Leaderboard task checks compare against the recorded language binding."""

from types import SimpleNamespace

import pytest

from tau2.data_model.elicitation import ElicitationConfig, ElicitationStrategy
from tau2.environment.environment import EnvironmentInfo
from tau2.multilingual.runtime import TaskSelection, load_language
from tau2.registry import registry
from tau2.scripts.leaderboard.verify_trajectories_public import check_tasks


def _results(domain, tasks, *, language, elicitation=None):
    info = SimpleNamespace(
        environment_info=EnvironmentInfo(domain_name=domain, policy=""),
        language=language,
        localize_db=None,
        pack_extension=None,
        benchmark=None,
        audio_native_config=None,
        elicitation=elicitation,
    )
    return SimpleNamespace(info=info, tasks=tasks)


@pytest.mark.parametrize("domain", ["retail", "telecom"])
def test_english_pack_replacements_pass(domain):
    pack_tasks = load_language(
        "en", domain, mode="text", selection=TaskSelection(task_split_name="base")
    ).tasks
    registry_tasks = registry.get_tasks_loader(domain)("base")
    assert [t.id for t in pack_tasks] == [t.id for t in registry_tasks]
    assert pack_tasks != registry_tasks

    assert check_tasks(_results(domain, pack_tasks, language="en")) == (True, "")
    ok, error = check_tasks(_results(domain, registry_tasks, language="en"))
    assert not ok and error.startswith("Task mismatches found in 50 tasks")


def test_results_without_language_use_registry_tasks():
    registry_tasks = registry.get_tasks_loader("telecom")("base")
    assert check_tasks(_results("telecom", registry_tasks, language=None)) == (
        True,
        "",
    )


@pytest.mark.parametrize("strategy", list(ElicitationStrategy))
def test_intake_results_use_recorded_strategy_variant(strategy):
    def tasks_for(chosen):
        return load_language(
            "en",
            "intake",
            mode="text",
            selection=TaskSelection(
                task_split_name="base", task_kwargs={"strategy": chosen.value}
            ),
        ).tasks

    config = ElicitationConfig(strategy=strategy)
    recorded = _results(
        "intake", tasks_for(strategy), language="en", elicitation=config
    )
    assert check_tasks(recorded) == (True, "")

    other = next(s for s in ElicitationStrategy if s != strategy)
    swapped = _results("intake", tasks_for(other), language="en", elicitation=config)
    assert not check_tasks(swapped)[0]
