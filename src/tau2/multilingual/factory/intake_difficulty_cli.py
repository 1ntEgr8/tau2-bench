"""Factory-owned commands for exhaustive intake difficulty coverage and review."""

import argparse
from pathlib import Path

from loguru import logger

from tau2.multilingual.factory.intake_difficulty_audit import (
    DifficultyAuditRequest,
    inspect_difficulty_audit,
    run_difficulty_audit,
)
from tau2.multilingual.factory.intake_difficulty_inventory import (
    build_difficulty_inventory,
)
from tau2.multilingual.factory.localization_artifact import read_localization_map
from tau2.utils.artifacts import write_artifact


def _inventory(args: argparse.Namespace) -> None:
    inventory = build_difficulty_inventory(read_localization_map(args.map))
    write_artifact(inventory, args.output)
    logger.info(
        "Inventoried {} Hard slots with complete Easy controls", len(inventory.cases)
    )


def _audit(args: argparse.Namespace) -> None:
    request = DifficultyAuditRequest.model_validate_json(args.request.read_text())
    if not request.map_file.is_absolute():
        request.map_file = (args.request.parent / request.map_file).resolve()
    report = run_difficulty_audit(request, args.output, args.checkpoints)
    logger.info(
        "Difficulty audit: {} keep, {} revise, {} source blockers, {} errors",
        report.keep_count,
        report.revise_count,
        report.source_blocked_count,
        report.error_count,
    )


def _inspect(args: argparse.Namespace) -> None:
    request = DifficultyAuditRequest.model_validate_json(args.request.read_text())
    if not request.map_file.is_absolute():
        request.map_file = (args.request.parent / request.map_file).resolve()
    report = inspect_difficulty_audit(request, args.output, args.checkpoints)
    logger.info(
        "Checkpoint inspection (no model calls): {} completed cases, {} incomplete/error cases; gate={}",
        len(report.decisions),
        report.error_count,
        report.machine_gate_passed,
    )


def _easy_audit(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.intake_easy_audit import (
        EasyAuditRequest,
        inspect_easy_audit,
        run_easy_audit,
    )

    request = EasyAuditRequest.model_validate_json(args.request.read_text())
    if not request.map_file.is_absolute():
        request.map_file = (args.request.parent / request.map_file).resolve()
    handler = inspect_easy_audit if args.inspect_only else run_easy_audit
    report = handler(request, args.output, args.checkpoints)
    logger.info(
        "Easy audit: {} keep, {} revise, {} source blockers, {} errors; native review pending",
        report.keep_count,
        report.revise_count,
        report.source_blocked_count,
        report.error_count,
    )


def _easy_plan(args: argparse.Namespace) -> None:
    from pydantic import TypeAdapter

    from tau2.multilingual.factory.intake_easy_audit import EasyAuditReport
    from tau2.multilingual.factory.intake_easy_revision import (
        EasyReviewFeedback,
        build_easy_revision_plan,
    )

    feedback = (
        TypeAdapter(list[EasyReviewFeedback]).validate_json(args.feedback.read_text())
        if args.feedback
        else None
    )
    plan = build_easy_revision_plan(
        read_localization_map(args.map),
        EasyAuditReport.model_validate_json(args.audit.read_text()),
        review_feedback=feedback,
    )
    write_artifact(plan, args.output)
    logger.info("Exact-source Easy repair plan: {}", args.output)


def _easy_revise(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.intake_easy_revision import (
        EasyRevisionRequest,
        revise_intake_easy,
    )

    request = EasyRevisionRequest.model_validate_json(args.request.read_text())
    for name in ("map_file", "audit_file"):
        path = getattr(request, name)
        if not path.is_absolute():
            setattr(request, name, (args.request.parent / path).resolve())
    revise_intake_easy(request, args.output.resolve(), args.checkpoints.resolve())
    logger.info("Revised Easy candidate requires a complete new audit: {}", args.output)


def _easy_report(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.intake_easy_audit import (
        EasyAuditReport,
        EasyLedgerManifest,
        build_easy_aggregate,
        write_easy_aggregate,
    )

    reports = [EasyAuditReport.model_validate_json(p.read_text()) for p in args.report]
    manifests = [
        EasyLedgerManifest.model_validate_json(
            p.with_suffix(".ledger.manifest.json").read_text()
        )
        for p in args.report
    ]
    files = write_easy_aggregate(build_easy_aggregate(reports, manifests), args.output)
    logger.info("Complete five-language Easy ledger: {}", files.ledger_csv)


def _plan(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.intake_difficulty_audit import DifficultyAuditReport
    from tau2.multilingual.factory.intake_difficulty_revision import (
        build_difficulty_revision_plan,
        write_difficulty_revision_artifacts,
    )

    plan = build_difficulty_revision_plan(
        read_localization_map(args.map),
        DifficultyAuditReport.model_validate_json(args.audit.read_text()),
    )
    files = write_difficulty_revision_artifacts(plan, args.output)
    logger.info("Difficulty plan and complete case ledger: {}", files.ledger_csv)


def _revise(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.intake_difficulty_apply import (
        DifficultyRevisionRequest,
        revise_intake_difficulty,
    )

    request = DifficultyRevisionRequest.model_validate_json(args.request.read_text())
    for name in ("map_file", "audit_file"):
        path = getattr(request, name)
        if not path.is_absolute():
            setattr(request, name, (args.request.parent / path).resolve())
    revise_intake_difficulty(request, args.output.resolve(), args.checkpoints.resolve())
    logger.info("Revised candidate requires a new complete audit: {}", args.output)


def _times(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.intake_time_authoring import (
        IntakeTimeAuthoringRequest,
        revise_intake_time_surfaces,
    )

    request = IntakeTimeAuthoringRequest.model_validate_json(args.request.read_text())
    for name in ("map_file", "audit_file"):
        path = getattr(request, name)
        if path is not None and not path.is_absolute():
            setattr(request, name, (args.request.parent / path).resolve())
    revise_intake_time_surfaces(request, args.output, args.checkpoints)
    logger.info("Authored time surfaces require a new complete audit: {}", args.output)


def _merge(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.intake_difficulty_apply import (
        DifficultyMergeRequest,
        merge_completed_difficulty_stages,
    )

    request = DifficultyMergeRequest.model_validate_json(args.request.read_text())
    for name in ("map_file", "audit_file", "entity_map_file", "surface_map_file"):
        path = getattr(request, name)
        if not path.is_absolute():
            setattr(request, name, (args.request.parent / path).resolve())
    merge_completed_difficulty_stages(request, args.output, args.record)
    logger.info(
        "Merged repaired stages; complete re-audit still required: {}", args.output
    )


def _report(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.intake_difficulty_revision import (
        DifficultyRevisionPlan,
        build_difficulty_aggregate,
        write_difficulty_aggregate,
    )

    plans = [
        DifficultyRevisionPlan.model_validate_json(path.read_text())
        for path in args.plan
    ]
    files = write_difficulty_aggregate(build_difficulty_aggregate(plans), args.output)
    logger.info("Complete five-language difficulty ledger: {}", files.ledger_csv)


def add_intake_difficulty_args(commands) -> None:
    """Register systematic difficulty stages on the existing factory command group."""
    inventory = commands.add_parser(
        "inventory-intake-difficulty",
        help="Inventory every Hard intake slot and all same-family Easy controls",
    )
    inventory.add_argument("--map", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    inventory.set_defaults(func=_inventory)
    audit = commands.add_parser(
        "audit-intake-difficulty",
        help="Audit all 100 Hard slots against their language's Easy controls",
    )
    audit.add_argument("--request", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--checkpoints", type=Path, required=True)
    audit.set_defaults(func=_audit)
    inspection = commands.add_parser(
        "inspect-intake-difficulty",
        help="Inspect exact audit checkpoints without model calls; expose missing/invalid cases",
    )
    for flag in ("request", "output", "checkpoints"):
        inspection.add_argument("--" + flag, type=Path, required=True)
    inspection.set_defaults(func=_inspect)
    for name, inspect_only in (
        ("audit-intake-easy", False),
        ("inspect-intake-easy", True),
    ):
        command = commands.add_parser(
            name,
            help=(
                "Inspect Easy audit checkpoints without model calls"
                if inspect_only
                else "Independently audit all 100 Easy cases and export a review ledger"
            ),
        )
        for flag in ("request", "output", "checkpoints"):
            command.add_argument("--" + flag, type=Path, required=True)
        command.set_defaults(func=_easy_audit, inspect_only=inspect_only)
    easy_plan = commands.add_parser(
        "plan-intake-easy", help="Plan exact audited Easy repairs"
    )
    for flag in ("map", "audit", "output"):
        easy_plan.add_argument("--" + flag, type=Path, required=True)
    easy_plan.add_argument(
        "--feedback",
        type=Path,
        help="Optional JSON array of exact-map factual EasyReviewFeedback records",
    )
    easy_plan.set_defaults(func=_easy_plan)
    easy_revise = commands.add_parser(
        "revise-intake-easy",
        help="Repair selected Easy entries while retaining Hard cases",
    )
    for flag in ("request", "output", "checkpoints"):
        easy_revise.add_argument("--" + flag, type=Path, required=True)
    easy_revise.set_defaults(func=_easy_revise)
    easy_report = commands.add_parser(
        "report-intake-easy",
        help="Combine five exact Easy audits into a 500-row review ledger",
    )
    easy_report.add_argument("--report", type=Path, action="append", required=True)
    easy_report.add_argument("--output", type=Path, required=True)
    easy_report.set_defaults(func=_easy_report)
    plan = commands.add_parser(
        "plan-intake-difficulty",
        help="Plan exact repairs and export every Hard case to a CSV ledger",
    )
    plan.add_argument("--map", type=Path, required=True)
    plan.add_argument("--audit", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.set_defaults(func=_plan)
    merge = commands.add_parser(
        "merge-intake-difficulty",
        help="Merge validated repair descendants without replacing failed-stage evidence",
    )
    for flag in ("request", "output", "record"):
        merge.add_argument("--" + flag, type=Path, required=True)
    merge.set_defaults(func=_merge)
    report = commands.add_parser(
        "report-intake-difficulty",
        help="Combine five language plans into an exhaustive 500-row CSV ledger",
    )
    report.add_argument("--plan", type=Path, action="append", required=True)
    report.add_argument("--output", type=Path, required=True)
    report.set_defaults(func=_report)
    for name, handler, help_text in (
        (
            "revise-intake-difficulty",
            _revise,
            "Apply audited entity and time-expression repairs with linked dependencies",
        ),
        (
            "revise-intake-times",
            _times,
            "Author selected time expressions while retaining exact canonical answers",
        ),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--request", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--checkpoints", type=Path, required=True)
        command.set_defaults(func=handler)
