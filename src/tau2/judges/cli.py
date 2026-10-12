"""Package-owned argument registration and dispatch for diagnostic workflows."""

import argparse
from pathlib import Path

from pydantic import TypeAdapter

from tau2.data_model.judges import DiagnosticId
from tau2.judges.summary import build_export


def _measures(value: str) -> list[DiagnosticId]:
    try:
        values = [
            TypeAdapter(DiagnosticId).validate_python(piece.strip())
            for piece in value.split(",")
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Use comma-separated selected measure names"
        ) from exc
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Measure names must be unique")
    return values


def add_judges_args(parser: argparse.ArgumentParser) -> None:
    """Register suite/summarize/export; no separate rejudge command."""
    commands = parser.add_subparsers(dest="judge_command", required=True)
    for name, help_text in (
        ("suite", "Plan or resume selected judges on a saved run"),
        ("summarize", "Read saved diagnostic scores and coverage without model calls"),
        ("export", "Export typed saved diagnostics and provenance without model calls"),
    ):
        child = commands.add_parser(name, help=help_text)
        child.add_argument(
            "results",
            type=Path,
            help="Ordinary Results JSON, run directory, or benchmark matrix root",
        )
        child.add_argument(
            "--only",
            type=_measures,
            default=None,
            help="Comma-separated measure names; default all selected measures",
        )
        child.add_argument(
            "--trials",
            type=int,
            nargs="+",
            default=None,
            help="Explicit stored trial indices; default all trials",
        )
        if name == "suite":
            child.add_argument(
                "--force", action="store_true", help="Recompute matching verdicts"
            )
            child.add_argument(
                "--dry-run",
                action="store_true",
                help="Plan without model calls or progress writes",
            )
        child.add_argument(
            "--registry",
            type=Path,
            action="append",
            help="Explicit verified promotion bundle; never an unvalidated candidate",
        )
        child.set_defaults(func=run_judges)
    _add_lifecycle_args(commands)


def run_judges(args: argparse.Namespace) -> None:
    """Emit diagnostics separately from ordinary task rewards."""
    from tau2.judges.validation import production_catalog

    output = build_export(
        args.results,
        production_catalog(args.registry or []),
        only=args.only,
        trials=args.trials,
        force=getattr(args, "force", False),
        execute=args.judge_command == "suite" and not args.dry_run,
    )
    print(output.model_dump_json(indent=2))


def _add_lifecycle_args(commands) -> None:
    """Register isolated candidate execution and human-reviewed publication."""
    candidate = commands.add_parser(
        "candidate", help="Run an unvalidated candidate in a separate output directory"
    )
    candidate.add_argument("results", type=Path)
    candidate.add_argument("--candidate", type=Path, required=True)
    candidate.add_argument("--output", type=Path, required=True)
    candidate.add_argument("--dry-run", action="store_true")
    candidate.set_defaults(func=_run_lifecycle)
    for name in ("validate", "promote"):
        child = commands.add_parser(
            name,
            help="Validate held-out evidence"
            if name == "validate"
            else "Export a reviewed validated registry bundle",
        )
        child.add_argument("--request", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.set_defaults(func=_run_lifecycle)


def _run_lifecycle(args: argparse.Namespace) -> None:
    from tau2.judges.authoring_models import CandidateJudge
    from tau2.judges.candidates import run_candidate
    from tau2.judges.validation import (
        PromotionRequest,
        ValidationRequest,
        promote_judge,
        validate_judge,
    )

    if args.judge_command == "candidate":
        result = run_candidate(
            args.results,
            CandidateJudge.model_validate_json(args.candidate.read_text()),
            args.output,
            dry_run=args.dry_run,
        )
    elif args.judge_command == "validate":
        result = validate_judge(
            ValidationRequest.model_validate_json(args.request.read_text()), args.output
        )
    else:
        result = promote_judge(
            PromotionRequest.model_validate_json(args.request.read_text()), args.output
        )
    print(result.model_dump_json(indent=2))
