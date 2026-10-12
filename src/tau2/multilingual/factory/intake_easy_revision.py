"""Exact-audit Easy repairs with explicit shared ownership and frozen Hard content."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from tau2.multilingual.factory.database_localization import (
    LocalizationReference,
    RegionalDraftRequest,
    RegionalRevision,
    _stage,
    revise_database_localization,
)
from tau2.multilingual.factory.generation_config import regional_generation
from tau2.multilingual.factory.intake_candidate_review import locale_rules
from tau2.multilingual.factory.intake_difficulty_inventory import (
    DifficultyComparator,
    DifficultyInventory,
    Digest,
    build_difficulty_inventory,
)
from tau2.multilingual.factory.intake_difficulty_revision import (
    medication_decoy_issues,
    phone_decoy_issues,
)
from tau2.multilingual.factory.intake_easy_audit import (
    EasyAuditReport,
    EasyAuditRequest,
    EasyDecision,
    EasyRepairTarget,
    extract_easy_cases,
)
from tau2.multilingual.factory.intake_medication_identity import (
    medication_identity_report,
)
from tau2.multilingual.factory.localization_artifact import (
    automatic_localization,
    read_localization_map,
    write_localization_map,
)
from tau2.multilingual.factory.localization_compile import validate_localization
from tau2.multilingual.localization import (
    LocalizationMap,
    LocalizationModel,
    RegionalSettings,
    SourceEntry,
)
from tau2.utils.artifacts import content_digest, write_artifact

EASY_REPAIR_VERSION = "intake-easy-revision-v1"
EASY_REPAIR_INSTRUCTIONS = """Repair ONLY the selected used Easy entries in this exact-map audit. Keep Easy entities naturally simple, familiar and easy to capture in the target locale; do not create artificial simplification, nonsensical names or unnatural speech to satisfy a difficulty label. Findings are model evidence, not native approval or replacement text. Separately attributed review_feedback contains exact-map factual corrections or runtime constraints and may identify a defect missed by an original keep verdict; retain the raw verdict and honor the scoped review facts without overriding source invariants. Preserve the full canonical task frame, protocol, known answer, source language semantics and all Hard content. A source_blocked decision is an unresolved source limitation, not permission to replace its identity or change difficulty; separately identified dependency errors may still be corrected while that blocker remains.
Primary replacements must keep every selected caller payload, decoy, pronunciation, spelling style and omission instruction coherent with the new identity. For dependency-only repairs retain the authoritative primary. Shared entry owners listed in the feedback all receive the same value; keep every affected Easy case coherent, and never mutate an unselected entry or donor identity. Dates and times retain their exact canonical truth and pinned clock. Do not swap medications: preserve the exact source drug identity, dose, unit, formulation and each decoy's source drug at its original array position. Preserve VIN/member checksums and protocol structure. For phone decoys preserve the source-relative primary/decoy main-number length difference and separate extension error shape, including genuine missing or extra digits. Use local telephone number conventions. Never manufacture a different clinical or protocol identity just to make a value easier.
This is an outbound intake call: the service agent is the caller and the user simulator is the callee/record owner. Runtime chooses ONE spelling style, and omission is applied only when that complication is selected. Always-active TIME_CALLER_GUIDANCE independently requires a bare initial time expression for annotated values while keeping the complete day period privately known. A style requiring the callee to say AM/PM whenever reading the time, including initially, is a real conflict and must be corrected. Private full time/date knowledge itself is correct and must remain complete. Empty spelling-style slots are deliberate no-ops. Written ext., x, + and email syntax are protocol, not English pronunciation instructions. Use natural local digit readings; never import English zero-as-oh, double or triple mechanically. Retain valid values when a finding rests only on mistaken runtime premises, recording the reason in ordinary generated notes. Return the existing ordinary revision schema; these remain draft candidates requiring fresh Easy and affected Hard audits plus native review."""


class EasyReviewFeedback(LocalizationModel):
    """Explicit exact-map factual review, retained separately from model verdicts."""

    canonical_task_id: Annotated[
        str,
        Field(
            pattern=r"^intake_[a-z_]+_easy_[0-9]{2}$",
            description="Exact used Easy task with a source-bound review concern.",
        ),
    ]
    map_sha256: Digest
    repair_targets: Annotated[
        list[EasyRepairTarget],
        Field(
            min_length=1,
            description="Only these primary or dependency roles require factual correction.",
        ),
    ]
    concern: Annotated[
        str,
        Field(
            min_length=1,
            description="Concrete current/source evidence and correction constraint, never authored replacement values.",
        ),
    ]

    @model_validator(mode="after")
    def validate_targets(self):
        if (
            len(set(self.repair_targets)) != len(self.repair_targets)
            or not self.concern.strip()
        ):
            raise ValueError(
                "Easy review requires distinct targets and a nonempty concern"
            )
        return self


class EasyRevisionRequest(EasyAuditRequest):
    """Explicit authoring settings and the complete exact-map Easy audit."""

    audit_file: Annotated[
        Path, Field(description="Completed Easy audit of the unchanged parent map.")
    ]
    review_feedback: Annotated[
        list[EasyReviewFeedback],
        Field(
            default_factory=list,
            description="Exact-map factual supplements; original audit verdicts remain unchanged.",
        ),
    ]


class EasyEntryScope(LocalizationModel):
    """All task owners affected by a candidate entry, including blocked owners."""

    entry_id: Annotated[
        str, Field(description="Exact source-owned proposed map entry.")
    ]
    easy_task_ids: Annotated[
        list[str],
        Field(
            description="Every Easy task using this entry, including incidental occurrences."
        ),
    ]
    hard_task_ids: Annotated[
        list[str],
        Field(
            description="Every protected Hard task using this entry; any owner blocks selection."
        ),
    ]


class EasyRepairCase(LocalizationModel):
    """One accounted-for Easy slot with original verdict and safe repair scope."""

    canonical_task_id: Annotated[str, Field(description="Frozen Easy task ID.")]
    decision: Annotated[
        EasyDecision | None,
        Field(description="Unmodified audit verdict; absent only for audit errors."),
    ]
    review_feedback: Annotated[
        EasyReviewFeedback | None,
        Field(
            default=None,
            description="Separately attributed exact-map factual review, never an overwritten audit decision.",
        ),
    ]
    status: Annotated[
        Literal[
            "keep", "planned", "partial", "blocked", "source_blocked", "audit_error"
        ],
        Field(description="Repair planning outcome, never approval."),
    ]
    redraft_ids: Annotated[
        list[str],
        Field(description="Exact mutable selected entries after ownership checks."),
    ]
    blockers: Annotated[
        list[str],
        Field(
            description="Unresolved source, immutable-entry or shared-scope reasons."
        ),
    ]


class EasyRevisionPlan(LocalizationModel):
    """Complete source-bound Easy accounting with protected Hard ownership."""

    version: Annotated[
        Literal["intake-easy-revision-v1"],
        Field(description="Fixed Easy repair contract version."),
    ] = EASY_REPAIR_VERSION
    map_sha256: Digest
    source_sha256: Digest
    audit_sha256: Digest
    inventory_sha256: Digest
    settings: Annotated[
        RegionalSettings, Field(description="Unchanged target locale settings.")
    ]
    cases: Annotated[
        list[EasyRepairCase],
        Field(
            min_length=100,
            max_length=100,
            description="All 100 used Easy slots, including errors and blockers.",
        ),
    ]
    evidence: Annotated[
        list[DifficultyComparator],
        Field(
            min_length=100,
            max_length=100,
            description="Safe exact measured Easy evidence without donor context.",
        ),
    ]
    entry_scopes: Annotated[
        list[EasyEntryScope],
        Field(
            description="Explicit ownership for all attempted entry selections, including blocked cross-tier dependencies."
        ),
    ]
    redraft_ids: Annotated[
        list[str], Field(description="Only these ordinary map entries may change.")
    ]
    affected_easy_task_ids: Annotated[
        list[str],
        Field(
            description="Union of Easy task owners for all selected entries, requiring re-audit."
        ),
    ]
    native_review_status: Annotated[
        Literal["pending"],
        Field(description="No machine repair grants native approval."),
    ] = "pending"

    @model_validator(mode="after")
    def validate_scope(self):
        """Keep serialized scope aligned with complete task evidence and ownership."""
        ids = [case.canonical_task_id for case in self.cases]
        if len(set(ids)) != 100 or set(ids) != {
            row.canonical_task_id for row in self.evidence
        }:
            raise ValueError("Easy repair plan requires complete unique task evidence")
        if set(self.redraft_ids) != {
            identifier for case in self.cases for identifier in case.redraft_ids
        } or len(set(self.redraft_ids)) != len(self.redraft_ids):
            raise ValueError("Easy plan selections differ from case scope")
        owners = {scope.entry_id: scope for scope in self.entry_scopes}
        if (
            len(owners) != len(self.entry_scopes)
            or not set(self.redraft_ids) <= owners.keys()
        ):
            raise ValueError("Easy plan requires unique entry ownership")
        if any(owners[key].hard_task_ids for key in self.redraft_ids):
            raise ValueError("Easy repair cannot select cross-tier shared entries")
        if set(self.affected_easy_task_ids) != {
            task for key in self.redraft_ids for task in owners[key].easy_task_ids
        }:
            raise ValueError("Easy plan affected tasks differ from selected ownership")
        return self


class EasyRevisionFeedback(LocalizationModel):
    """Safe measured evidence supplied to the ordinary fixed revision generator."""

    version: Annotated[
        Literal["intake-easy-revision-v1"],
        Field(description="Version of Easy repair instructions and scope."),
    ] = EASY_REPAIR_VERSION
    plan_sha256: Digest
    map_sha256: Digest
    audit_sha256: Digest
    instructions: Annotated[
        str,
        Field(
            description="Fixed versioned constraints, plus existing target locale rules."
        ),
    ]
    redraft_ids: Annotated[
        list[str], Field(description="Exact authorized ordinary revision targets.")
    ]
    findings: Annotated[
        list[EasyRepairCase],
        Field(
            description="Original decisions for the affected Easy slots, retaining source limitations."
        ),
    ]
    evidence: Annotated[
        list[DifficultyComparator],
        Field(
            description="Actual primary and dependency evidence for every affected Easy owner."
        ),
    ]
    entry_scopes: Annotated[
        list[EasyEntryScope],
        Field(description="Explicit shared Easy ownership for the selected entries."),
    ]


class EasyRevisionCompletion(LocalizationModel):
    """Exact validated result required when resuming a previously published draft."""

    version: Annotated[
        Literal["intake-easy-revision-v1"],
        Field(description="Easy repair completion contract version."),
    ] = EASY_REPAIR_VERSION
    request_sha256: Digest
    plan_sha256: Digest
    regional_revision_sha256: Digest
    candidate_sha256: Digest
    native_review_status: Annotated[
        Literal["pending"],
        Field(
            description="Successful generation and validation are not native approval."
        ),
    ] = "pending"


def _entry_owners(
    proposal: LocalizationMap, inventory: DifficultyInventory
) -> dict[str, set[str]]:
    """Include all task occurrences, not only measured comparator dependencies."""
    owners: dict[str, set[str]] = {}
    for row in [*inventory.cases, *extract_easy_cases(inventory)]:
        for identifier in row.entry_sha256:
            owners.setdefault(identifier, set()).add(row.canonical_task_id)
    documents = {
        document.id: document.content for document in proposal.source.documents
    }
    tasks = documents["tasks"]
    stimuli = documents["intake_localization"]["stimuli"]
    for entry in proposal.entries:
        for occurrence in entry.occurrences:
            if (
                occurrence.document_id == "tasks"
                and occurrence.path
                and isinstance(occurrence.path[0], int)
            ):
                owners.setdefault(entry.id, set()).add(tasks[occurrence.path[0]]["id"])
            elif (
                occurrence.document_id == "intake_localization"
                and len(occurrence.path) >= 2
                and occurrence.path[0] == "stimuli"
            ):
                owners.setdefault(entry.id, set()).add(
                    stimuli[occurrence.path[1]]["canonical_task_id"]
                )
    return owners


def build_easy_revision_plan(
    proposal: LocalizationMap,
    audit: EasyAuditReport,
    inventory: DifficultyInventory | None = None,
    review_feedback: list[EasyReviewFeedback] | None = None,
) -> EasyRevisionPlan:
    """Resolve exact Easy occurrences and block dependent groups with Hard owners."""
    audit = EasyAuditReport.model_validate(audit.model_dump())
    inventory = inventory or build_difficulty_inventory(proposal)
    if (
        content_digest(proposal),
        content_digest(proposal.source),
        content_digest(inventory),
        proposal.settings,
    ) != (
        audit.map_sha256,
        audit.source_sha256,
        audit.inventory_sha256,
        audit.settings,
    ) or inventory.map_sha256 != content_digest(proposal):
        raise ValueError("Easy repair input digest/settings differ from audited inputs")
    cases = extract_easy_cases(inventory)
    if set(audit.expected_task_ids) != {case.canonical_task_id for case in cases}:
        raise ValueError("Easy audit does not cover the exact inventory")
    feedback = [
        EasyReviewFeedback.model_validate(row.model_dump())
        for row in review_feedback or []
    ]
    supplements = {row.canonical_task_id: row for row in feedback}
    if (
        len(supplements) != len(feedback)
        or any(row.map_sha256 != content_digest(proposal) for row in feedback)
        or not supplements.keys() <= {case.canonical_task_id for case in cases}
    ):
        raise ValueError(
            "Easy review feedback is duplicate, stale or outside the exact Easy task scope"
        )
    decisions = {row.canonical_task_id: row for row in audit.decisions}
    errors = {row.canonical_task_id: row.error for row in audit.errors}
    entries = {entry.id: entry for entry in proposal.entries}
    owners = _entry_owners(proposal, inventory)
    stimuli = next(
        document.content
        for document in proposal.source.documents
        if document.id == "intake_localization"
    )["stimuli"]
    indices = {row["canonical_task_id"]: index for index, row in enumerate(stimuli)}
    immutable = {
        entry.id
        for entry in proposal.entries
        if automatic_localization(
            SourceEntry.model_validate(
                entry.model_dump(include=set(SourceEntry.model_fields))
            ),
            proposal.settings,
        )
        is not None
    }
    repairable = {
        task
        for task, verdict in decisions.items()
        if verdict.disposition == "revise"
        or (verdict.disposition == "source_blocked" and verdict.repair_targets)
    }
    repairable.update(supplements)
    planned = []
    attempted = set()
    for case in cases:
        decision = decisions.get(case.canonical_task_id)
        supplement = supplements.get(case.canonical_task_id)
        targets = list(
            dict.fromkeys(
                [
                    *(decision.repair_targets if decision else []),
                    *(supplement.repair_targets if supplement else []),
                ]
            )
        )
        selected: set[str] = set()
        primary_group: set[str] = set()
        blockers = []
        if decision is None:
            status = "audit_error"
            blockers.append(errors[case.canonical_task_id])
        elif decision.disposition == "source_blocked" and not targets:
            status = "source_blocked"
            blockers.append(decision.source_blocked_reason)
        elif decision.disposition == "keep" and supplement is None:
            status = "keep"
        else:
            status = "planned"
            if decision.disposition == "source_blocked":
                blockers.append(decision.source_blocked_reason)
            dependencies = {
                target: {
                    key
                    for key in case.dependency_entry_ids
                    if any(
                        point.document_id == "intake_localization"
                        and point.path[:3]
                        == ["stimuli", indices[case.canonical_task_id], target]
                        for point in entries[key].occurrences
                    )
                }
                for target in (
                    "decoys",
                    "pronunciations",
                    "spelling_styles",
                    "omission_instruction",
                )
            }
            for target in targets:
                if target in {"primary", "caller_surface"}:
                    if decision.disposition == "source_blocked":
                        blockers.append(
                            f"{target}: source blocker permits only separately repairable dependencies"
                        )
                        continue
                    if case.family == "times" or case.field_name in {
                        "vin",
                        "member_id",
                    }:
                        blockers.append(
                            f"{target}: immutable source time/protocol value; Easy caller-surface overrides are unsupported"
                        )
                        continue
                    if case.family == "dates":
                        group = set(case.caller_entry_ids) - {case.measured_entry_id}
                        if not group:
                            blockers.append(
                                "Date has no separately owned caller phrase; canonical date is immutable"
                            )
                            continue
                    else:
                        group = {case.measured_entry_id, *case.caller_entry_ids}
                    primary_ids = set(group)
                    group.update(
                        key
                        for dependency, ids in dependencies.items()
                        for key in ids
                        if entries[key].source not in ("", [])
                        and key not in immutable
                        and not (
                            case.family in {"dates", "times"}
                            and dependency == "omission_instruction"
                        )
                    )
                    if any(
                        _stage(entries[key])
                        <= max(_stage(entries[primary]) for primary in primary_ids)
                        for key in group - primary_ids
                    ):
                        blockers.append(
                            "Primary and dependencies lack safe ordinary generator stage ordering"
                        )
                        continue
                    primary_group.update(group)
                    selected.update(group)
                else:
                    if target == "omission_instruction" and case.family in {
                        "dates",
                        "times",
                    }:
                        blockers.append(
                            "Temporal omission instructions are compiler-derived; ordinary entry edits cannot repair runtime guidance"
                        )
                        continue
                    ids = dependencies[target]
                    if not ids:
                        blockers.append(
                            f"{target}: no exact source dependency entry exists"
                        )
                    for key in ids:
                        if key in immutable or entries[key].source in ("", []):
                            blockers.append(
                                f"{target}: fixed empty or automatic source dependency cannot be expanded"
                            )
                        else:
                            selected.add(key)
            attempted.update(selected)
            unsafe = set()
            for key in sorted(selected):
                hard = sorted(
                    task for task in owners.get(key, set()) if "_hard_" in task
                )
                outside = sorted(
                    task
                    for task in owners.get(key, set())
                    if task not in repairable and task not in hard
                )
                if hard:
                    blockers.append(
                        f"Cross-tier shared entry {key} affects protected Hard tasks: {', '.join(hard)}"
                    )
                if outside:
                    blockers.append(
                        f"Shared entry {key} affects Easy tasks outside repair scope: {', '.join(outside)}"
                    )
                if key in immutable or entries[key].reuse is not None:
                    blockers.append(f"Automatic or reused entry must stay fixed: {key}")
                if (
                    hard
                    or outside
                    or key in immutable
                    or entries[key].reuse is not None
                ):
                    unsafe.add(key)
            if unsafe & primary_group:
                selected.difference_update(primary_group)
                blockers.append(
                    "Primary and linked dependencies blocked together to preserve identity consistency"
                )
            selected.difference_update(unsafe)
            if not selected and not blockers:
                blockers.append("Audit requested no supported mutable repair")
            if blockers:
                status = "partial" if selected else "blocked"
        planned.append(
            EasyRepairCase(
                canonical_task_id=case.canonical_task_id,
                decision=decision,
                review_feedback=supplement,
                status=status,
                redraft_ids=sorted(selected),
                blockers=blockers,
            )
        )
    selected = {key for case in planned for key in case.redraft_ids}
    return EasyRevisionPlan(
        map_sha256=content_digest(proposal),
        source_sha256=content_digest(proposal.source),
        audit_sha256=content_digest(audit),
        inventory_sha256=content_digest(inventory),
        settings=proposal.settings,
        cases=planned,
        evidence=cases,
        entry_scopes=[
            EasyEntryScope(
                entry_id=key,
                easy_task_ids=sorted(task for task in owners[key] if "_easy_" in task),
                hard_task_ids=sorted(task for task in owners[key] if "_hard_" in task),
            )
            for key in sorted(attempted)
        ],
        redraft_ids=[entry.id for entry in proposal.entries if entry.id in selected],
        affected_easy_task_ids=sorted(
            {task for key in selected for task in owners[key]}
        ),
    )


def build_easy_revision_feedback(plan: EasyRevisionPlan) -> EasyRevisionFeedback:
    """Supply actual measured values for every affected owner without donor context."""
    affected = set(plan.affected_easy_task_ids)
    return EasyRevisionFeedback(
        plan_sha256=content_digest(plan),
        map_sha256=plan.map_sha256,
        audit_sha256=plan.audit_sha256,
        instructions=EASY_REPAIR_INSTRUCTIONS + "\n\n" + locale_rules(plan.settings),
        redraft_ids=plan.redraft_ids,
        findings=[row for row in plan.cases if row.canonical_task_id in affected],
        evidence=[row for row in plan.evidence if row.canonical_task_id in affected],
        entry_scopes=[
            row for row in plan.entry_scopes if row.entry_id in plan.redraft_ids
        ],
    )


def validate_easy_revision(
    parent: LocalizationMap, candidate: LocalizationMap, plan: EasyRevisionPlan
) -> LocalizationMap:
    """Reject unselected mutations and clinical/phone regressions before publication."""
    if content_digest(parent) != plan.map_sha256:
        raise ValueError("Easy revision parent differs from exact plan")
    for field in set(LocalizationMap.model_fields) - {
        "revision",
        "provenance",
        "entries",
    }:
        if getattr(candidate, field) != getattr(parent, field):
            raise ValueError(
                f"Easy revision changed unselected metadata or caller surfaces: {field}"
            )
    before = {entry.id: entry for entry in parent.entries}
    if [entry.id for entry in candidate.entries] != [
        entry.id for entry in parent.entries
    ]:
        raise ValueError("Easy revision changed source entry coverage or order")
    for entry in candidate.entries:
        if entry.id not in plan.redraft_ids and entry != before[entry.id]:
            raise ValueError(f"Easy revision changed an unselected entry: {entry.id}")
        if before[entry.id].reuse is not None and entry != before[entry.id]:
            raise ValueError("Easy revision changed a reused donor identity")
    candidate = validate_localization(candidate)
    for issue in [*medication_decoy_issues(candidate), *phone_decoy_issues(candidate)]:
        if issue.entry_id in plan.redraft_ids:
            raise ValueError(
                f"Easy revision violates source invariant {issue.invariant}: {issue.entry_id}"
            )
    previous_identity = medication_identity_report(parent)
    current_identity = medication_identity_report(candidate)

    def relation_keys(report):
        return {
            (
                finding.kind,
                tuple(
                    sorted(
                        (
                            row.canonical_task_id,
                            row.role,
                            row.decoy_index if row.decoy_index is not None else -1,
                            row.representation,
                            row.source_label,
                            row.current_label,
                        )
                        for row in finding.observations
                    )
                ),
            )
            for finding in report.findings
        }

    if relation_keys(current_identity) - relation_keys(previous_identity):
        raise ValueError(
            "Easy revision introduced a medication identity consistency finding; source-bound review required"
        )
    previous_unparsed = {
        (row.entry_id, row.decoy_index, row.representation)
        for row in previous_identity.unparsed
    }
    if any(
        (row.entry_id, row.decoy_index, row.representation) not in previous_unparsed
        for row in current_identity.unparsed
    ):
        raise ValueError(
            "Easy revision introduced an unparsed medication identity; review required"
        )
    return candidate


def revise_intake_easy(
    request: EasyRevisionRequest, output: Path, checkpoints: Path
) -> LocalizationMap:
    """Generate scoped Easy drafts through the ordinary fixed/versioned factory seam."""
    request = EasyRevisionRequest.model_validate(request.model_dump())
    parent = read_localization_map(request.map_file)
    audit = EasyAuditReport.model_validate_json(request.audit_file.read_text())
    plan = build_easy_revision_plan(
        parent, audit, review_feedback=request.review_feedback
    )
    feedback = build_easy_revision_feedback(plan)
    write_artifact(request, checkpoints / "request.json")
    write_artifact(plan, checkpoints / "plan.json")
    write_artifact(feedback, checkpoints / "feedback.json")
    if not plan.redraft_ids:
        raise ValueError(
            "Easy repair plan has no mutable targets; inspect its explicit keep/blocker accounting"
        )
    revision = RegionalRevision(
        previous=parent,
        request=RegionalDraftRequest(
            source=parent.source,
            settings=parent.settings,
            revision=request.revision,
            generation=regional_generation(request.model, request.model_args),
            batch_size=request.batch_size,
            max_concurrency=request.max_concurrency,
        ),
        redraft_ids=plan.redraft_ids,
        feedback=[
            LocalizationReference(
                name=EASY_REPAIR_VERSION, content=feedback.model_dump_json()
            )
        ],
    )
    if output.exists():
        saved = read_localization_map(output)
        record_file = checkpoints / "completed.json"
        if not record_file.is_file():
            raise ValueError(
                "Existing Easy output requires its exact completed repair record"
            )
        record = EasyRevisionCompletion.model_validate_json(record_file.read_text())
        if (
            saved.provenance.input_sha256 != content_digest(revision)
            or saved.provenance.parent_map_sha256 != content_digest(parent)
            or saved.revision != request.revision
            or record
            != EasyRevisionCompletion(
                request_sha256=content_digest(request),
                plan_sha256=content_digest(plan),
                regional_revision_sha256=content_digest(revision),
                candidate_sha256=content_digest(saved),
            )
        ):
            raise ValueError("Existing Easy repair output differs from exact request")
        return validate_easy_revision(parent, saved, plan)
    candidate = revise_database_localization(
        revision, checkpoints / "entities.json", checkpoints / "entities"
    )
    if (
        candidate.provenance.input_sha256 != content_digest(revision)
        or candidate.provenance.parent_map_sha256 != content_digest(parent)
        or candidate.revision != request.revision
    ):
        raise ValueError("Easy repair generation differs from exact regional revision")
    candidate = validate_easy_revision(parent, candidate, plan)
    write_artifact(
        EasyRevisionCompletion(
            request_sha256=content_digest(request),
            plan_sha256=content_digest(plan),
            regional_revision_sha256=content_digest(revision),
            candidate_sha256=content_digest(candidate),
        ),
        checkpoints / "completed.json",
    )
    write_localization_map(candidate, output)
    return candidate
