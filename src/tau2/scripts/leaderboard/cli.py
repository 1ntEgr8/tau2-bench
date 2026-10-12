"""Package-owned leaderboard submission commands."""

import argparse
from pathlib import Path


def add_submit_args(parser: argparse.ArgumentParser) -> None:
    """Register submission preparation, independent validation, and trajectory tools."""
    submit_subparsers = parser.add_subparsers(
        dest="submit_command", help="Submit subcommands", required=True
    )

    # Prepare ordinary or frozen-matrix submissions.
    submit_prepare_parser = submit_subparsers.add_parser(
        "prepare", help="Prepare a submission for the leaderboard"
    )
    submit_prepare_parser.add_argument(
        "input_paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    submit_prepare_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory to save the submission and trajectories",
    )
    submit_prepare_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip trajectory verification step",
    )
    submit_prepare_parser.add_argument(
        "--voice",
        action="store_true",
        default=None,
        help="Force voice submission mode (auto-detected from input data if not specified)",
    )
    submit_prepare_parser.add_argument(
        "--benchmark",
        help="Verify this published preset; otherwise infer from matrix metadata",
    )
    submit_prepare_parser.add_argument(
        "--historical-voice-v2",
        action="store_true",
        help="Prepare the frozen 2026-10-02 Voice V2 release from saved exports",
    )
    submit_prepare_parser.add_argument(
        "--provider", choices=("gemini", "gpt_live", "grok")
    )
    submit_prepare_parser.add_argument("--interaction-facts", type=Path)
    submit_prepare_parser.add_argument("--judges-root", type=Path)
    submit_prepare_parser.add_argument("--metadata-file", type=Path)
    submit_prepare_parser.set_defaults(func=lambda args: run_prepare_submission(args))

    # Submit validate subcommand
    submit_validate_parser = submit_subparsers.add_parser(
        "validate", help="Validate an existing submission directory"
    )
    submit_validate_parser.add_argument(
        "submission_dir",
        help="Path to the submission directory to validate",
    )
    submit_validate_parser.add_argument(
        "--benchmark",
        help="Verify this published preset; otherwise infer from submission metadata",
    )
    submit_validate_parser.add_argument(
        "--results",
        type=Path,
        help="Original saved benchmark matrix for independent validation",
    )
    submit_validate_parser.set_defaults(func=lambda args: run_validate_submission(args))

    # Submit verify-trajs subcommand
    submit_verify_parser = submit_subparsers.add_parser(
        "verify-trajs", help="Verify trajectory files"
    )
    submit_verify_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    submit_verify_parser.set_defaults(func=lambda args: run_verify_trajectories(args))

    # Submit interaction-metrics subcommand
    submit_im_parser = submit_subparsers.add_parser(
        "interaction-metrics",
        help="Compute voice interaction metrics (latency, responsiveness, "
        "interrupts, selectivity) from full-duplex trajectories",
    )
    submit_im_parser.add_argument(
        "input_paths",
        nargs="+",
        help="Voice experiment directories (results.json + simulations/) or a "
        "parent directory such as a submission's trajectories/ dir",
    )
    submit_im_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Optional path to write the interaction_metrics JSON block",
    )
    submit_im_parser.set_defaults(func=lambda args: run_interaction_metrics(args))


def run_verify_trajectories(args):
    import sys

    from loguru import logger

    from tau2.scripts.leaderboard.verify_trajectories import (
        VerificationMode,
        verify_trajectories,
    )

    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])

    verify_trajectories(args.paths, VerificationMode.PUBLIC)


def run_prepare_submission(args):
    """Run the prepare submission command."""
    if args.historical_voice_v2:
        if args.benchmark or args.no_verify or len(args.input_paths) != 1:
            raise ValueError(
                "Historical Voice V2 requires one saved export and full verification"
            )
        required = ("provider", "interaction_facts", "judges_root", "metadata_file")
        missing = [
            name.replace("_", "-") for name in required if getattr(args, name) is None
        ]
        if missing:
            raise ValueError("Historical Voice V2 requires: " + ", ".join(missing))
        from tau2.scripts.leaderboard.voice_v2_publication import (
            prepare_voice_v2_package,
        )

        prepare_voice_v2_package(
            Path(args.input_paths[0]),
            Path(args.output),
            args.provider,
            args.interaction_facts,
            args.judges_root,
            args.metadata_file,
        )
        return
    if any(
        getattr(args, name) is not None
        for name in ("provider", "interaction_facts", "judges_root", "metadata_file")
    ):
        raise ValueError("Saved judge inputs require --historical-voice-v2")
    from tau2.scripts.leaderboard.prepare_submission import prepare_submission

    has_matrix = any(
        (Path(path) / "benchmark-frame.json").is_file() for path in args.input_paths
    )
    if args.benchmark or has_matrix:
        if len(args.input_paths) != 1 or args.no_verify:
            raise ValueError(
                "Benchmark submissions require one matrix root and full verification"
            )
        from tau2.scripts.leaderboard.prepare_submission import (
            prepare_benchmark_submission,
        )

        prepare_benchmark_submission(
            Path(args.input_paths[0]), Path(args.output), args.benchmark
        )
        return
    prepare_submission(
        input_paths=args.input_paths,
        output_dir=args.output,
        run_verification=not args.no_verify,
        voice=args.voice if args.voice else None,
    )


def run_validate_submission(args):
    """Run the validate submission command."""
    from tau2.scripts.leaderboard.prepare_submission import validate_submission

    directory = Path(args.submission_dir)
    submission_path = directory / "submission.json"
    if submission_path.is_file():
        import json

        if json.loads(submission_path.read_text()).get("format") == "voice-v2":
            from tau2.scripts.leaderboard.voice_v2_publication import (
                validate_voice_v2_package,
            )

            if args.benchmark or args.results:
                raise ValueError(
                    "Voice V2 package validation reads its saved source manifest"
                )
            submission = validate_voice_v2_package(directory)
            print(submission.benchmark_submission.summary.model_dump_json(indent=2))
            return
    if args.benchmark or (directory / "benchmark-submission.json").exists():
        if args.results is None:
            raise ValueError(
                "Benchmark validation requires --results with the original saved matrix"
            )
        from tau2.scripts.leaderboard.prepare_submission import (
            validate_benchmark_submission,
        )

        submission = validate_benchmark_submission(
            directory, args.results, args.benchmark
        )
        print(submission.summary.model_dump_json(indent=2))
        return
    if args.results is not None:
        raise ValueError("--results is only used for benchmark matrix submissions")
    validate_submission(submission_dir=args.submission_dir)


def run_interaction_metrics(args):
    """Run the interaction metrics computation command."""
    from tau2.scripts.leaderboard.compute_interaction_metrics import (
        compute_interaction_metrics,
    )

    compute_interaction_metrics(
        input_paths=args.input_paths,
        output_path=args.output,
    )
