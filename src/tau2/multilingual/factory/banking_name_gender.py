"""Audit and repair every banking person-name mapping without changing task genders."""

import re
from pathlib import Path

from pydantic import Field, JsonValue

from tau2.multilingual.factory.localization_gender import (
    NAME_GENDER_PROMPT_VERSION,
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
    json_content_digest,
)
from tau2.utils.artifacts import content_digest, write_artifact


class BankingNameGroup(LocalizationModel):
    """Exact same-owner aliases or one explicit source prose binding."""

    request: GenderNameRequest = Field(description="Current name and source evidence.")
    entry_ids: list[str] = Field(
        default_factory=list, description="Structured aliases of one source identity."
    )
    binding_indices: list[int] = Field(
        default_factory=list, description="Exact incidental binding locations."
    )


class BankingGenderRevision(LocalizationModel):
    """Complete audit coverage and source-bound repair provenance, pending native review."""

    version: str = NAME_GENDER_PROMPT_VERSION
    model: str = Field(description="Actual authoring model.")
    model_args: dict[str, JsonValue] = Field(description="Actual model settings.")
    input_map_sha256: str = Field(description="Exact original full map digest.")
    output_map_sha256: str = Field(description="Exact repaired full map digest.")
    source_assessments: list[SourceGenderResult] = Field(
        description="Every source classification, including non-person surfaces excluded from name repair."
    )
    requests: list[GenderNameRequest] = Field(
        description="All audited personal names and exact source evidence."
    )
    results: list[GenderNameResult] = Field(
        description="Every retained, repaired or unresolved person name."
    )
    changed_entry_ids: list[str] = Field(
        description="All changed map entries including dependent emails and prose."
    )
    changed_binding_count: int = Field(
        description="Number of exact prose bindings whose forms changed."
    )
    review_status: str = "pending_native_review"


def _snippets(value: str, source: str) -> list[str]:
    return [
        value[max(0, m.start() - 180) : m.end() + 240]
        for m in list(re.finditer(re.escape(source), value, re.I))[:4]
    ]


def banking_name_groups(proposal: LocalizationMap) -> list[BankingNameGroup]:
    """Inventory owner-specific structured aliases and every generated prose surface."""
    from tau2.domains.banking_knowledge.localization_prose import _tasks

    entries = {e.id: e for e in proposal.entries}
    grouped = {}
    for entry in proposal.entries:
        if entry.category != "identity_name":
            continue
        owner = re.search(r"entity=([^;]+)", entry.context)
        if not owner:
            raise ValueError("Banking personal name lacks source owner")
        key = json_content_digest(
            {"owner": owner[1], "source": entry.source.casefold()}
        )
        context = ["Structured personal-name role. Source owner: " + owner[1]]
        for prose in proposal.entries:
            if (
                prose.category == "prose"
                and isinstance(prose.source, str)
                and _tasks(entry) & _tasks(prose)
            ):
                context.extend(_snippets(prose.source, entry.source))
        context = sorted(set(context))
        if key not in grouped:
            grouped[key] = BankingNameGroup(
                request=GenderNameRequest(
                    id=key,
                    source=entry.source,
                    forms=entry.forms,
                    context=context,
                    source_gender="unknown",
                    source_basis="unknown",
                    source_evidence="No reviewed gender field in the banking source identity; assess explicit co-referential context or qualified source-name convention.",
                ),
                entry_ids=[],
            )
        group = grouped[key]
        if group.request.forms != entry.forms:
            # Same owner's spelling aliases must not conceal different local identities.
            if (group.request.forms.native, group.request.forms.romanized) != (
                entry.forms.native,
                entry.forms.romanized,
            ):
                raise ValueError(
                    "Same-owner source aliases have conflicting localized names"
                )
        group.entry_ids.append(entry.id)
        group.request.context = sorted(set(group.request.context + context))
    for index, binding in enumerate(proposal.prose_bindings):
        if binding.kind != "generated":
            continue
        key = json_content_digest(
            {"prose": binding.entry_id, "source": binding.source.casefold()}
        )
        context = _snippets(str(entries[binding.entry_id].source), binding.source)
        if key not in grouped:
            grouped[key] = BankingNameGroup(
                request=GenderNameRequest(
                    id=key,
                    source=binding.source,
                    forms=binding.forms,
                    context=context,
                    source_gender="unknown",
                    source_basis="unknown",
                    source_evidence="Incidental prose surface. Classify person versus non-person from the source context before any name repair.",
                )
            )
        group = grouped[key]
        if group.request.forms != binding.forms:
            raise ValueError("Same prose source binding has conflicting forms")
        group.binding_indices.append(index)
    return [grouped[key] for key in sorted(grouped)]


def repair_banking_name_genders(
    maps: list[Path],
    output_root: Path,
    checkpoints: Path,
    model: str,
    model_args: dict[str, JsonValue],
    workers: int,
    offline: bool = False,
) -> list[BankingGenderRevision]:
    """Audit complete source coverage, then synchronize names, emails and exact prose."""
    from tau2.multilingual.factory.banking_candidate_review import (
        BankingReviewRules,
        synchronize_banking_candidate,
    )
    from tau2.multilingual.factory.localization_artifact import (
        read_localization_map,
        write_localization_map,
    )
    from tau2.multilingual.factory.localization_compile import validate_localization

    proposals = [read_localization_map(p) for p in maps]
    if any(p.source.domain != "banking_knowledge" for p in proposals):
        raise ValueError("Banking name repair only accepts banking maps")
    groups = [banking_name_groups(p) for p in proposals]
    source_requests = {}
    for group_set in groups:
        for group in group_set:
            request = group.request
            if request.id in source_requests:
                old = source_requests[request.id]
                if (old.source.casefold(), old.context) != (
                    request.source.casefold(),
                    request.context,
                ):
                    raise ValueError("Cross-language source gender evidence differs")
            source_requests[request.id] = request
    sources = {
        r.id: r
        for r in assess_source_names(
            list(source_requests.values()),
            model,
            model_args,
            checkpoints / "source",
            workers,
            offline=offline,
        )
    }
    records = []
    for path, proposal, group_set in zip(maps, proposals, groups, strict=True):
        rules = BankingReviewRules.model_validate_json(
            path.with_name("banking_review_rules.json").read_text()
        )
        requests = []
        selected = []
        for group in group_set:
            source = sources[group.request.id]
            if source.kind == "non_person" and not group.entry_ids:
                continue
            request = group.request.model_copy(deep=True)
            request.source_gender = source.gender
            request.source_basis = source.basis
            request.source_evidence = source.reason
            request.context.append(
                f"Target name order: {rules.family_name_position} family name. Preserve the existing family token and same-owner identity. The source itself has no task/persona gender edits authorized."
            )
            requests.append(request)
            selected.append(group)
        results = audit_gender_names(
            requests,
            rules.language_name,
            rules.country_name,
            model,
            model_args,
            checkpoints / rules.language,
            workers,
            offline=offline,
        )
        updated = proposal.model_copy(deep=True)
        entries = {e.id: e for e in updated.entries}
        for group, request, result in zip(selected, requests, results, strict=True):
            for identifier in group.entry_ids:
                entry = entries[identifier]
                if result.decision == "revise":
                    entry.forms = LocalizedForms.model_validate(
                        result.forms.model_dump()
                    )
                    entry.reuse = None
                    entry.operation = "adapt"
                    entry.notes = (
                        "Gender-preserving name revision. "
                        + result.reason
                        + " Native review remains pending."
                    )
                # English rationale can differ across aliases with identical actual name.
                evidence = evidence_for(request, result, model)
                evidence.forms_sha256 = content_digest(entry.forms)
                entry.name_gender = evidence
            for index in group.binding_indices:
                binding = updated.prose_bindings[index]
                if result.decision == "revise":
                    binding.forms = LocalizedForms.model_validate(
                        result.forms.model_dump()
                    )
                    binding.reason = (
                        "Gender-preserving incidental name revision. " + result.reason
                    )
        # Refresh all identity-derived emails, including previously reused names, with
        # source punctuation and collision handling owned by the existing allocator.
        updated.revision = proposal.revision + "-gender-v1"
        from tau2.multilingual.factory.banking_candidate_review import (
            colliding_name_entries,
        )

        if colliding_name_entries(updated):
            raise ValueError("Gender repair introduced source-world name collisions")
        reconstruction, household = synchronize_banking_candidate(
            updated, updated, rules
        )
        updated = validate_localization(reconstruction.proposal)
        record = BankingGenderRevision(
            model=model,
            model_args=model_args,
            input_map_sha256=content_digest(proposal),
            output_map_sha256=content_digest(updated),
            source_assessments=[sources[g.request.id] for g in group_set],
            requests=requests,
            results=results,
            changed_entry_ids=[
                a.id
                for a, b in zip(updated.entries, proposal.entries, strict=True)
                if a.forms != b.forms
            ],
            changed_binding_count=sum(
                a.forms != b.forms
                for a, b in zip(
                    updated.prose_bindings, proposal.prose_bindings, strict=True
                )
            ),
        )
        output = output_root / rules.language / "banking_knowledge.json"
        write_localization_map(updated, output)
        write_artifact(record, output.with_suffix(".gender.json"))
        write_artifact(
            reconstruction.reconstruction, output.with_suffix(".reconstruction.json")
        )
        write_artifact(household.revision, output.with_suffix(".households.json"))
        records.append(record)
    return records


def _run(args):
    repair_banking_name_genders(
        args.map,
        args.output_root,
        args.checkpoints,
        args.model,
        {"temperature": 0, "reasoning_effort": args.reasoning_effort},
        args.workers,
        offline=args.offline,
    )


def add_banking_name_gender_args(commands):
    """Register reproducible full-name audit and dependency regeneration."""
    parser = commands.add_parser("repair-banking-name-genders")
    parser.add_argument("--map", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--reasoning-effort", required=True, choices=["low", "medium", "high"]
    )
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Replay exact completed checkpoints only; never call an external model",
    )
    parser.set_defaults(func=_run)
