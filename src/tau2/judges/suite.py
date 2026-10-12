"""Saved-run judging: load evidence, select implementations, plan and checkpoint."""

import json
import os
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from tau2.data_model.judges import (
    DiagnosticId,
    ImplementationFamily,
    JudgeCatalog,
    JudgeContract,
    JudgeInput,
    MeasureVerdict,
    PlanAction,
    PlannedVerdict,
    StoredVerdict,
    SuitePlan,
    SuiteResult,
    VerdictIdentity,
)
from tau2.judges.base import BaseJudge
from tau2.judges.run_evidence import SavedRun, progress_path


def _key(item: JudgeInput, measure: DiagnosticId) -> tuple[str, str, int, str]:
    return item.run_id, item.simulation_id, item.trial, measure


def plan_suite(
    inputs: list[JudgeInput],
    catalog: JudgeCatalog,
    stored: list[StoredVerdict],
    *,
    only: list[DiagnosticId] | None = None,
    trials: list[int] | None = None,
    force: bool = False,
) -> SuitePlan:
    """Plan missing/stale/forced work without accessing a model or mutating data."""
    measures = (
        only
        if only is not None
        else list(dict.fromkeys(pair.measure for pair in catalog.pairs))
    )
    if set(measures) - {pair.measure for pair in catalog.pairs}:
        raise ValueError("Selected diagnostic is absent from the judge registry")
    if len(set(measures)) != len(measures) or not measures:
        raise ValueError("Select distinct measures, at least one")
    if trials is not None and (
        not trials or min(trials) < 0 or len(set(trials)) != len(trials)
    ):
        raise ValueError("Select distinct nonnegative trials, at least one")
    existing = {_key(row.identity.input, row.identity.measure): row for row in stored}
    if len(existing) != len(stored):
        raise ValueError("Duplicate stored verdict identities")
    pairs = {(pair.language, pair.measure): pair for pair in catalog.pairs}
    contracts = {contract.contract_id: contract for contract in catalog.contracts}
    items = []
    seen = set()
    for source in inputs:
        if trials is not None and source.trial not in trials:
            continue
        for measure in measures:
            key = _key(source, measure)
            if key in seen:
                raise ValueError("Duplicate simulation/measure inputs")
            seen.add(key)
            pair = pairs.get((source.language, measure))
            identity = None
            if pair is None:
                action, reason = (
                    PlanAction.NOT_APPLICABLE,
                    "No selected language/measure contract",
                )
            elif pair.domains is not None and source.domain not in pair.domains:
                action, reason = (
                    PlanAction.NOT_APPLICABLE,
                    "Domain outside approved scope",
                )
            elif (
                pair.agent_gender is not None
                and pair.agent_gender != source.agent_gender
            ):
                action, reason = (
                    PlanAction.NOT_APPLICABLE,
                    f"Validation supports only the {pair.agent_gender}-agent frame",
                )
            else:
                contract = contracts[pair.contract_id]
                identity = VerdictIdentity(
                    input=(
                        source
                        if pair.needs_audio
                        else source.model_copy(
                            update={
                                "audio_fingerprint": None,
                                "audio_error": None,
                                "has_audio": False,
                            }
                        )
                    ),
                    measure=measure,
                    contract_id=contract.contract_id,
                    implementation=contract.implementation,
                    input_exclusions=contract.input_exclusions,
                    model=contract.model,
                    model_args=contract.model_args,
                    prompt_version=contract.prompt.prompt_version,
                    rubric_version=contract.prompt.rubric_version,
                    catalog_revision=catalog.revision,
                    transport_revision="multimodal-responses-v1",
                )
                previous = existing.get(key)
                if pair.needs_audio and not source.has_audio:
                    action, reason = (
                        PlanAction.MISSING_INPUT,
                        "Delivered audio/reference unavailable",
                    )
                elif force:
                    action, reason = (
                        PlanAction.FORCE,
                        "Explicit recomputation requested",
                    )
                elif previous is None:
                    action, reason = PlanAction.MISSING, "No saved verdict"
                elif previous.identity != identity:
                    action, reason = PlanAction.STALE, "Input or judge contract changed"
                elif previous.error is not None or (
                    previous.verdict is not None and previous.verdict.unit_errors
                ):
                    action, reason = (
                        PlanAction.RETRY_ERROR,
                        "Previous attempt recorded an error",
                    )
                else:
                    action, reason = PlanAction.REUSE, "Input and judge contract match"
            items.append(
                PlannedVerdict(
                    input=source,
                    measure=measure,
                    action=action,
                    identity=identity,
                    reason=reason,
                )
            )
    return SuitePlan(items=items)


def read_verdicts(path: Path) -> list[StoredVerdict]:
    """Read saved diagnostics; absent progress means no verdicts, not zero scores."""
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("Diagnostic progress must be a list of typed verdict records")
    try:
        return [StoredVerdict.model_validate(row) for row in payload]
    except ValidationError as error:
        raise ValueError(
            f"{path} was written by an incompatible judge-suite schema; move it "
            "aside and re-judge the run"
        ) from error


def _save_verdicts(path: Path, rows: list[StoredVerdict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "[\n" + ",\n".join(row.model_dump_json() for row in rows) + "\n]\n"
    )
    temporary.replace(path)


def execute_suite(
    inputs: list[JudgeInput],
    catalog: JudgeCatalog,
    path: Path,
    *,
    only: list[DiagnosticId] | None = None,
    trials: list[int] | None = None,
    force: bool = False,
    dry_run: bool = False,
    backend: Callable[[JudgeInput, JudgeContract, str], MeasureVerdict] | None = None,
    resume_backend: Callable[
        [JudgeInput, JudgeContract, str, MeasureVerdict], MeasureVerdict
    ]
    | None = None,
) -> SuitePlan:
    """Resume typed saved diagnostics, checkpointing each call/measure independently.

    The supplied judge returns a typed verdict without serialization. Failures are persisted
    and a later invocation retries them. Ordinary task rewards are never read or
    changed by this operation.
    """
    stored = read_verdicts(path)
    plan = plan_suite(inputs, catalog, stored, only=only, trials=trials, force=force)
    if dry_run:
        return plan
    work = [
        item
        for item in plan.items
        if item.action
        in {
            PlanAction.MISSING,
            PlanAction.STALE,
            PlanAction.FORCE,
            PlanAction.RETRY_ERROR,
        }
    ]
    if work and backend is None:
        raise ValueError(
            "A reviewed calibrated judge adapter is required for execution"
        )
    existing = {_key(row.identity.input, row.identity.measure): row for row in stored}
    contracts = {contract.contract_id: contract for contract in catalog.contracts}
    for item in work:
        assert item.identity is not None and backend is not None
        previous = existing.get(_key(item.input, item.measure))
        partial = (
            previous.verdict
            if (
                item.action == PlanAction.RETRY_ERROR
                and previous is not None
                and previous.identity == item.identity
                and previous.verdict is not None
            )
            else None
        )
        if partial is not None and resume_backend is None:
            raise ValueError(
                "Partial verdicts require a resume backend to preserve successful units"
            )
        try:
            if partial is not None:
                response = resume_backend(
                    item.input,
                    contracts[item.identity.contract_id],
                    item.measure,
                    partial,
                )
                if not isinstance(response, MeasureVerdict):
                    raise TypeError("Judge must return a typed MeasureVerdict")
                before_ids = {u.unit_id for u in [*partial.units, *partial.unit_errors]}
                after_ids = {
                    u.unit_id for u in [*response.units, *response.unit_errors]
                }
                if before_ids != after_ids:
                    raise ValueError("Partial retry changed diagnostic unit coverage")
                retained = {u.unit_id: u for u in response.units}
                if any(retained.get(u.unit_id) != u for u in partial.units):
                    raise ValueError(
                        "Partial retry changed or dropped a successful unit"
                    )
            else:
                response = backend(
                    item.input, contracts[item.identity.contract_id], item.measure
                )
            if not isinstance(response, MeasureVerdict):
                raise TypeError("Judge must return a typed MeasureVerdict")
            verdict = response
            saved = StoredVerdict(identity=item.identity, verdict=verdict)
        except Exception as exc:
            if partial is not None:
                saved = StoredVerdict(
                    identity=item.identity,
                    verdict=partial.model_copy(
                        update={
                            "unit_errors": [
                                error.model_copy(
                                    update={
                                        "error": (
                                            f"Retry failed: {type(exc).__name__}: {exc}"
                                        )
                                    }
                                )
                                for error in partial.unit_errors
                            ]
                        }
                    ),
                )
            else:
                saved = StoredVerdict(
                    identity=item.identity, error=f"{type(exc).__name__}: {exc}"
                )
        existing[_key(item.input, item.measure)] = saved
        _save_verdicts(path, list(existing.values()))
    return plan


def create_judge(contract: JudgeContract) -> BaseJudge:
    """Select the same calibrated implementation for production and candidates."""
    from tau2.judges.criteria import CriteriaJudge
    from tau2.judges.fidelity import FidelityJudge
    from tau2.judges.tool_use import ToolUseJudge

    if contract.implementation == ImplementationFamily.FIDELITY:
        return FidelityJudge(contract)
    if contract.implementation == ImplementationFamily.TOOL_USE:
        return ToolUseJudge(contract)
    if contract.implementation in {
        ImplementationFamily.NATIVENESS,
        ImplementationFamily.NATURALNESS,
        ImplementationFamily.HINDI_GENDER,
        ImplementationFamily.AGENT_TEXT,
    }:
        return CriteriaJudge(contract)
    raise NotImplementedError(f"No judge implementation for {contract.implementation}")


def judge_saved_run(
    results: Path,
    catalog: JudgeCatalog,
    *,
    only: list[DiagnosticId] | None = None,
    trials: list[int] | None = None,
    force: bool = False,
    execute: bool = False,
    output: Path | None = None,
) -> SuiteResult:
    """Plan or execute one saved run, with explicit separate candidate storage."""
    saved = SavedRun(results)
    path = output if output is not None else progress_path(saved.results_path)
    if execute:
        judges: dict[str, BaseJudge] = {}

        def evaluate(
            item: JudgeInput,
            contract: JudgeContract,
            measure: str,
            previous: MeasureVerdict | None = None,
        ) -> MeasureVerdict:
            judge = judges.get(contract.contract_id)
            if judge is None:
                judge = judges[contract.contract_id] = create_judge(contract)
            evidence = saved.load(item)
            if previous is not None:
                return judge.resume(evidence, measure, previous)
            return judge.evaluate(evidence, measure)

        execute_suite(
            saved.inputs,
            catalog,
            path,
            only=only,
            trials=trials,
            force=force,
            backend=evaluate,
            resume_backend=evaluate,
        )
    stored = read_verdicts(path)
    plan = plan_suite(
        saved.inputs,
        catalog,
        stored,
        only=only,
        trials=trials,
        force=force and not execute,
    )
    selected = {_key(item.input, item.measure) for item in plan.items}
    return SuiteResult(
        plan=plan,
        verdicts=[
            row
            for row in stored
            if _key(row.identity.input, row.identity.measure) in selected
        ],
    )
