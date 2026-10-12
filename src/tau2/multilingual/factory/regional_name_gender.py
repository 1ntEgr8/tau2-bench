"""Convention-level name audit for structured retail and airline identity maps.

This audits repeated given-name surfaces once without equating the people who
share them. It never assigns persona genders. Uncertain source conventions stay
unresolved; existing owner-specific evidence is retained and audited separately.
"""

import json
import re
from pathlib import Path

from loguru import logger
from pydantic import Field, JsonValue

from tau2.multilingual.factory.generation_config import regional_generation
from tau2.multilingual.factory.localization_gender import (
    GenderNameRequest,
    GenderNameResult,
    SourceGenderResult,
    assess_source_names,
    audit_gender_names,
    evidence_for,
)
from tau2.multilingual.localization import (
    LocalizationMap,
    LocalizationModel,
    LocalizedForms,
    NameGenderEvidence,
    json_content_digest,
)
from tau2.utils.artifacts import content_digest, write_artifact

AUDIT_VERSION = "regional-name-conventions-v1"


class NameSurfaceGroup(LocalizationModel):
    """Exact repeated given-name surface, with independent owner entry IDs."""

    request: GenderNameRequest
    entry_ids: list[str]
    source_id: str = Field(
        description="Target-independent key for the exact source evidence."
    )


class RegionalNameAudit(LocalizationModel):
    """Complete coverage and reproducible assessments, never native approval."""

    version: str = AUDIT_VERSION
    domain: str
    language: str
    input_map_sha256: str
    output_map_sha256: str
    model: str
    model_args: dict[str, JsonValue]
    sources: list[SourceGenderResult]
    requests: list[GenderNameRequest]
    results: list[GenderNameResult]
    covered_entry_ids: list[str]
    changed_entry_ids: list[str]
    unresolved_entry_ids: list[str]
    review_status: str = "pending_native_review"


def name_surface_groups(proposal: LocalizationMap) -> list[NameSurfaceGroup]:
    """Group only identical surfaces and evidence, never infer a surname's gender."""
    if proposal.source.domain not in {"retail", "airline"}:
        raise ValueError("Regional name audit requires retail or airline")
    groups = {}
    prose = [
        e.source
        for e in proposal.entries
        if e.category == "prose" and isinstance(e.source, str)
    ]
    name_owners: dict[str, set[str]] = {}
    for entry in proposal.entries:
        if entry.category != "identity_name" or "source_identity=" not in entry.context:
            continue
        identity, _ = json.JSONDecoder().raw_decode(
            entry.context.split("source_identity=", 1)[1]
        )
        full_name = " ".join(
            str(identity.get(part, "")) for part in ("first_name", "last_name")
        )
        owner = re.search(r"entity=([^;]+);", entry.context)
        if owner:
            name_owners.setdefault(full_name, set()).add(owner[1])
    for entry in proposal.entries:
        if entry.category != "identity_name":
            continue
        role = re.match(r"role=([^;]+);", entry.context)
        if role is None or role[1] not in {"first_name", "last_name"}:
            raise ValueError("Name audit requires structured given/family roles")
        if role[1] == "last_name":
            continue
        prior = entry.name_gender
        explicit = prior and prior.source_basis in {
            "reviewed_metadata",
            "explicit_context",
        }
        source_evidence = (
            prior.source_evidence
            if explicit
            else (
                "Assess only the given-name convention, not this person's verified gender. "
                "No reviewed owner gender is supplied; ambiguous conventions must remain unknown/unisex."
            )
        )
        context = [
            "Given-name component only; family-name components and persona genders are immutable."
        ]
        if explicit:
            context.append(entry.context)
        if "source_identity=" in entry.context:
            identity, _ = json.JSONDecoder().raw_decode(
                entry.context.split("source_identity=", 1)[1]
            )
            full_name = " ".join(
                str(identity.get(part, "")) for part in ("first_name", "last_name")
            )
            matching = [
                text
                for text in prose
                if full_name.strip()
                and len(name_owners.get(full_name, set())) == 1
                and re.search(r"(?<!\w)" + re.escape(full_name) + r"(?!\w)", text, re.I)
            ]
            if matching:
                context.extend(
                    ["Exact source identity: " + full_name, *sorted(set(matching))]
                )
        source_id = json_content_digest([entry.source, source_evidence, context])
        key = json_content_digest(
            [source_id, entry.forms.native, entry.forms.romanized]
        )
        if key not in groups:
            groups[key] = NameSurfaceGroup(
                request=GenderNameRequest(
                    id=key,
                    source=entry.source,
                    forms=LocalizedForms(
                        native=entry.forms.native,
                        romanized=entry.forms.romanized,
                        english="Given-name component of independent synthetic identities; family names are not supplied for revision.",
                    ),
                    context=context,
                    source_gender=prior.source_gender if explicit else "unknown",
                    source_basis=prior.source_basis if explicit else "unknown",
                    source_evidence=source_evidence,
                ),
                entry_ids=[],
                source_id=source_id,
            )
        groups[key].entry_ids.append(entry.id)
    return list(groups.values())


class NameAuditApplication(LocalizationModel):
    """Audited map plus unresolved given-name owners awaiting native review."""

    proposal: LocalizationMap
    unresolved_entry_ids: list[str]


def apply_name_assessments(
    proposal: LocalizationMap,
    groups: list[NameSurfaceGroup],
    results: list[GenderNameResult],
    model: str,
) -> NameAuditApplication:
    """Apply exact given-name results; bind evidence to each original owner record."""
    if {r.id for r in results} != {g.request.id for g in groups} or len(results) != len(
        groups
    ):
        raise ValueError("Name audit result coverage differs from requests")
    indexed = {r.id: r for r in results}
    edits = {}
    unresolved = []
    for group in groups:
        result = indexed[group.request.id]
        evidence = evidence_for(group.request, result, model)
        for identifier in group.entry_ids:
            original = next((e for e in proposal.entries if e.id == identifier), None)
            if original is None or (
                original.source,
                original.forms.native,
                original.forms.romanized,
            ) != (
                group.request.source,
                group.request.forms.native,
                group.request.forms.romanized,
            ):
                raise ValueError("Stale name audit owner or surface")
            if identifier in edits:
                raise ValueError("Duplicate name audit owner entry")
            edits[identifier] = (result, evidence)
            if result.decision == "unresolved" or group.request.source_gender in {
                "unknown",
                "unisex",
            }:
                unresolved.append(identifier)
    updated = proposal.model_copy(deep=True)
    for entry in updated.entries:
        if entry.category != "identity_name":
            continue
        if entry.id in edits:
            result, evidence = edits[entry.id]
            # Never replace an owner-specific explanation when the surface is kept.
            if result.decision == "revise":
                entry.forms = LocalizedForms.model_validate(result.forms.model_dump())
                entry.operation = "adapt"
                entry.reuse = None
                entry.notes = (
                    result.reason
                    + " Source convention assessment; native review pending."
                )
            entry.name_gender = evidence.model_copy(
                update={"forms_sha256": content_digest(entry.forms)}
            )
        elif entry.context.startswith("role=last_name;"):
            if entry.name_gender is None:
                entry.name_gender = NameGenderEvidence(
                    source_gender="unknown",
                    target_gender="unknown",
                    source_basis="unknown",
                    source_evidence="A standalone family-name component does not establish its owner's gender.",
                    reason="Family name unchanged; assess the separate given-name component and stronger owner evidence.",
                    forms_sha256=content_digest(entry.forms),
                    model="deterministic-role-check",
                    prompt_version=AUDIT_VERSION,
                )
        else:
            raise ValueError("Uncovered personal-name entry")
    return NameAuditApplication(proposal=updated, unresolved_entry_ids=unresolved)


def run_name_audit(args):
    """Run checkpointed shared gender prompts, then synchronize affected dependencies."""
    from tau2.multilingual.factory.database_localization import (
        LocalizationReference,
        RegionalDraftRequest,
        _draft_entries,
    )
    from tau2.multilingual.factory.localization_artifact import (
        read_localization_map,
        write_localization_map,
    )
    from tau2.multilingual.factory.localization_caller_names import (
        localized_caller_entries,
    )
    from tau2.multilingual.localization import CallerNameRevisionProvenance

    model_args = json.loads(args.model_args)
    for path in args.maps:
        proposal = read_localization_map(path)
        output = (
            args.output_dir
            / proposal.settings.language
            / f"{proposal.source.domain}.json"
        )
        if output.exists() and output.with_suffix(".gender.json").exists():
            report = RegionalNameAudit.model_validate_json(
                output.with_suffix(".gender.json").read_text()
            )
            if (
                report.input_map_sha256 != content_digest(proposal)
                or report.model != args.model
                or report.model_args != model_args
            ):
                raise ValueError("Existing audit output differs from requested inputs")
            if (
                content_digest(read_localization_map(output))
                != report.output_map_sha256
            ):
                raise ValueError("Completed name audit map differs from its report")
            logger.info("Retained completed name audit {}", output)
            continue
        groups = name_surface_groups(proposal)
        # Source-only cache keys omit target script; no target gender feeds source inference.
        source_requests = {
            g.source_id: g.request.model_copy(update={"id": g.source_id})
            for g in groups
        }
        sources = assess_source_names(
            list(source_requests.values()),
            args.model,
            model_args,
            args.checkpoints / "source",
            args.workers,
            offline=args.offline,
        )
        by_source = {s.id: s for s in sources}
        for group in groups:
            source = by_source[group.source_id]
            group.request.source_gender = source.gender
            group.request.source_basis = source.basis
            group.request.source_evidence = source.reason
        results = audit_gender_names(
            [g.request for g in groups],
            proposal.settings.language,
            proposal.settings.country_code,
            args.model,
            model_args,
            args.checkpoints / proposal.settings.language,
            args.workers,
            offline=args.offline,
        )
        applied = apply_name_assessments(proposal, groups, results, args.model)
        updated, unresolved = applied.proposal, applied.unresolved_entry_ids
        before = {e.id: e for e in proposal.entries}
        changed_names = [
            e
            for e in updated.entries
            if e.category == "identity_name" and e.forms != before[e.id].forms
        ]
        if changed_names:
            changed_sources = {str(e.source) for e in changed_names}
            selected = {
                e.id
                for e in updated.entries
                if e.category == "contact"
                or (
                    e.category == "prose"
                    and any(
                        re.search(
                            r"(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z])",
                            str(e.source),
                            re.I,
                        )
                        for name in changed_sources
                    )
                )
            }
            pending = [e for e in updated.source.entries if e.id in selected]
            retained = [e for e in updated.entries if e.id not in selected]
            if args.offline and any(e.category != "contact" for e in pending):
                raise ValueError(
                    "Offline name audit cannot author dependent prose; resume an already completed output or run online"
                )
            request = RegionalDraftRequest(
                source=proposal.source,
                settings=proposal.settings,
                revision=proposal.revision + "-gender-v1",
                generation=regional_generation(args.model, model_args),
                max_concurrency=args.workers,
                references=[
                    LocalizationReference(
                        name="Source-bound gender repair",
                        content="Only synchronize the supplied corrected given names and their dependent contacts/prose. Keep surrounding English and unrelated entity values unchanged.",
                    )
                ],
            )
            authored = _draft_entries(
                request,
                pending,
                retained,
                content_digest(updated),
                args.checkpoints / "dependencies",
            )
            by_id = {e.id: e for e in authored}
            updated.entries = [by_id[e.id] for e in proposal.entries]
            synchronized = localized_caller_entries(updated)
            if synchronized != updated.entries:
                updated.caller_name_revision = CallerNameRevisionProvenance(
                    parent_map_sha256=content_digest(updated),
                    changed_entry_count=sum(
                        a != b
                        for a, b in zip(updated.entries, synchronized, strict=True)
                    ),
                )
                updated.entries = synchronized
        from tau2.multilingual.factory.localization_user_ids import (
            synchronize_revised_user_ids,
        )

        updated = synchronize_revised_user_ids(updated)
        updated.revision = proposal.revision + "-gender-v1"
        write_localization_map(updated, output)
        report = RegionalNameAudit(
            domain=proposal.source.domain,
            language=proposal.settings.language,
            input_map_sha256=content_digest(proposal),
            output_map_sha256=content_digest(updated),
            model=args.model,
            model_args=model_args,
            sources=sources,
            requests=[g.request for g in groups],
            results=results,
            covered_entry_ids=[
                e.id for e in updated.entries if e.category == "identity_name"
            ],
            changed_entry_ids=[
                e.id for e in updated.entries if e.forms != before[e.id].forms
            ],
            unresolved_entry_ids=unresolved,
        )
        write_artifact(report, output.with_suffix(".gender.json"))
        logger.info(
            "Audited {}/{}: {} name groups; {} changed entries, {} unresolved names",
            proposal.settings.language,
            proposal.source.domain,
            len(groups),
            len(report.changed_entry_ids),
            len(unresolved),
        )


def add_regional_name_audit_args(commands):
    """Register the structured-domain adapter on the shared factory CLI."""
    parser = commands.add_parser("audit-regional-name-genders")
    parser.add_argument("--map", dest="maps", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-args", required=True)
    from tau2.config import DEFAULT_FACTORY_LOCALIZATION_CONCURRENCY

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_FACTORY_LOCALIZATION_CONCURRENCY,
        choices=range(1, 11),
    )
    parser.add_argument("--offline", action="store_true")
    parser.set_defaults(func=run_name_audit)
