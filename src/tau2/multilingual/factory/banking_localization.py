"""Bounded, resumable banking entity pilots before full locale authoring."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from tau2.multilingual.factory.database_localization import (
    DATABASE_LOCALIZATION_CALL_NAME,
    DATABASE_LOCALIZATION_PROMPT_VERSION,
    BatchInput,
    RegionalDraftRequest,
    _generate_batch,
)
from tau2.multilingual.factory.localization_artifact import (
    SourceManifest,
    _manifest,
    automatic_localization,
)
from tau2.multilingual.factory.localization_reuse import validate_reused_entries
from tau2.multilingual.localization import (
    GenerationProvenance,
    LocalizationEntry,
    LocalizationModel,
    RegionalSettings,
)
from tau2.utils.artifacts import content_digest, write_artifact


class BankingEntityPilot(LocalizationModel):
    """Incomplete candidate review artifact; never a publishable localization map."""

    schema_version: Literal["banking-entity-pilot-v1"] = "banking-entity-pilot-v1"
    revision: str = Field(min_length=1, description="Pilot artifact revision.")
    source: SourceManifest = Field(
        description="Complete pinned source manifest; only a subset is authored here."
    )
    settings: RegionalSettings = Field(
        description="Banking identity locale with USD scale 1."
    )
    entries: list[LocalizationEntry] = Field(
        description="Reused candidates and bounded newly generated entity candidates."
    )
    provenance: GenerationProvenance = Field(
        description="Actual fixed-prompt model call and inputs for the new candidates."
    )
    pending_entry_ids: list[str] = Field(
        description="Unwritten nonautomatic entries; this pilot is incomplete."
    )
    status: Literal["draft"] = "draft"


def draft_banking_entity_pilot(
    request: RegionalDraftRequest, output: Path, checkpoints: Path, *, per_category: int
) -> BankingEntityPilot:
    """Draft a bounded sample after reuse, preserving all source and call provenance."""
    if request.source.domain != "banking_knowledge" or per_category < 1:
        raise ValueError(
            "Banking pilot requires banking source and positive sample size"
        )
    if (
        request.settings.currency_code != "USD"
        or request.settings.currency_scale != "1"
    ):
        raise ValueError("Banking pilots preserve USD and scale 1")
    digest = content_digest(request)
    reused = validate_reused_entries(request.source, request.reused_entries)
    pending = [
        e
        for e in request.source.entries
        if e.id not in reused and automatic_localization(e, request.settings) is None
    ]
    selected = []
    for category in ("identity_name", "address", "contact", "other"):
        # IDs encode semantic source identity and remain stable across file ordering.
        selected.extend(
            sorted((e for e in pending if e.category == category), key=lambda e: e.id)[
                :per_category
            ]
        )
    batch = BatchInput(request_sha256=digest, entries=selected, completed=[])
    if output.exists():
        existing = BankingEntityPilot.model_validate_json(output.read_text())
        expected_ids = set(reused) | {e.id for e in selected}
        if (
            existing.provenance.input_sha256 != content_digest(batch)
            or {e.id for e in existing.entries} != expected_ids
        ):
            raise ValueError("Pilot output already exists for different inputs")
        return existing
    generated = _generate_batch(request, batch, checkpoints)
    entries = [*reused.values(), *generated]
    covered = {e.id for e in entries}
    pilot = BankingEntityPilot(
        revision=request.revision,
        settings=request.settings,
        source=_manifest(request.source),
        entries=entries,
        pending_entry_ids=[e.id for e in pending if e.id not in covered],
        provenance=GenerationProvenance(
            model=request.model,
            model_args=request.model_args,
            prompt_version=DATABASE_LOCALIZATION_PROMPT_VERSION,
            call_name=DATABASE_LOCALIZATION_CALL_NAME,
            input_sha256=content_digest(batch),
        ),
    )
    write_artifact(pilot, output)
    return pilot


def render_banking_pilots(pilots: list[BankingEntityPilot]) -> str:
    """Show every new and reused candidate without implying content approval."""
    import html
    import json

    def display(value):
        return html.escape(
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )

    sections = []
    for pilot in pilots:
        fresh = [e for e in pilot.entries if e.reuse is None]
        reused = [e for e in pilot.entries if e.reuse is not None]
        sections.append(
            f"<h2>{display(pilot.settings.language)}</h2><p>{len(fresh)} new candidates; {len(reused)} reused candidates; {len(pilot.pending_entry_ids)} authored entries still pending. Model: {display(pilot.provenance.model)}; prompt: {display(pilot.provenance.prompt_version)}.</p>"
        )
        for label, entries in (
            ("Newly generated candidates", fresh),
            ("Reused reviewed candidates — new banking context pending", reused),
        ):
            sections.append(
                f"<h3>{label}</h3><table><thead><tr><th>Original entity</th><th>Suggested localized entity</th><th>Approved / Disapproved</th><th>Proposed entity</th><th>Reason / context</th></tr></thead><tbody>"
            )
            for entry in entries:
                source_note = (
                    f"Prior map: {entry.reuse.source_domain}, {entry.reuse.source_artifact_sha256}"
                    if entry.reuse
                    else "New generation; no human approval."
                )
                sections.append(
                    f"<tr><td>{display(entry.source)}</td><td>{display(entry.forms.native)}<br><small>{display(entry.forms.romanized)}</small></td><td>Pending</td><td></td><td>{display(entry.notes)}<details><summary>Source and lineage</summary>{display(source_note)}<br>{display(entry.id)}<br>{display(entry.context)}</details></td></tr>"
                )
            sections.append("</tbody></table>")
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8"><title>Banking entity pilot review</title><style>body{font:16px/1.5 system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 24px;color:#17212b}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{border:1px solid #cbd5e1;padding:10px;text-align:left;vertical-align:top}th{background:#eaf0f6}small{color:#526476}details{font-size:12px;overflow-wrap:anywhere}</style><h1>Banking entity pilot: pending native review</h1><p>These are incomplete authoring samples, not published banking maps. The English knowledge base, USD amounts and rule-bearing values are unchanged. This preview records no approvals. Reused source approvals establish prior linguistic review only; the new banking usage still needs contextual review. Full map review uses the language-tab spreadsheet workflow.</p>"""
        + "".join(sections)
        + "</html>"
    )


def add_banking_pilot_review_args(commands) -> None:
    """Register the review renderer beside the existing regional factory stages."""

    def run(args):
        pilots = [
            BankingEntityPilot.model_validate_json(path.read_text())
            for path in args.pilot
        ]
        content = render_banking_pilots(pilots)
        if args.output.exists() and args.output.read_text() != content:
            raise FileExistsError("Pilot review already exists for different inputs")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)

    parser = commands.add_parser(
        "review-banking-localization-pilot",
        help="Render all new and reused candidates from incomplete banking entity pilots",
    )
    parser.add_argument("--pilot", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(func=run)
