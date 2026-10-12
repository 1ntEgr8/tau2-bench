"""Source-pinned difficulty repair scope and exhaustive machine-review ledger."""

import csv
import hashlib
import io
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from tau2.multilingual.factory.generation_config import regional_generation
from tau2.multilingual.factory.intake_candidate_review import (
    REVISION_PROMPT,
    locale_rules,
)
from tau2.multilingual.factory.intake_difficulty_audit import (
    DifficultyAuditReport,
    DifficultyAuditRequest,
    DifficultyDecision,
)
from tau2.multilingual.factory.intake_difficulty_inventory import (
    DifficultyInventory,
    Family,
    build_difficulty_inventory,
)
from tau2.multilingual.factory.intake_phone_invariants import (
    PhoneDecoyLengthEvidence,
    phone_decoy_length_evidence,
)
from tau2.multilingual.factory.localization_artifact import automatic_localization
from tau2.multilingual.factory.localization_cli import DatabaseRevisionCommand
from tau2.multilingual.localization import (
    LocalizationMap,
    LocalizationModel,
    RegionalSettings,
    SourceEntry,
    json_content_digest,
)
from tau2.utils.artifacts import content_digest, write_artifact

REVISION_PLAN_VERSION = "intake-difficulty-revision-plan-v4"
FEEDBACK_VERSION = "intake-difficulty-revision-feedback-v4"
REPAIR_INSTRUCTIONS = """Repair only the exact selected entries for the listed frozen Hard task slots. The findings are source-pinned model audit evidence, not native approval and not authored replacement values. Keep every Easy comparator, reused donor identity, unselected field and canonical task fixed. Retain any primary entity when only its dependent omission, pronunciation, spelling or decoy is selected. For a replaced primary, make all selected dependencies coherent with that exact new identity. Realize a concrete naturally Hard local capture challenge relative to the supplied Easy controls, with correct caller knowledge and recoverable written answer. Do not introduce arbitrary length, implausible names or fabricated local spelling rules. Date repairs change only the relative spoken phrase while preserving its exact canonical-date annotation. Time caller surfaces are handled by a separate typed authoring stage: never rewrite canonical clock gold. Preserve VIN/member checksums and exact clinical identity, strength, unit and formulation. For a temporal source_blocked finding, the audit may have mistaken the original relative wording for an immutable source constraint. Attempt a new natural Hard spoken realization using the same canonical date/time and fixed source clock; keep the original source assessment visible and do not certify it as repaired until re-audited. Empty strings inside spelling_styles are deliberate no-op slots, not missing instructions; retain them when inapplicable locally. Written ASCII ext., x and + are protocol markers and do not force English pronunciation; preserve valid written phone syntax while using natural local spoken wording. Do not force English oh, double or triple into other languages. If a selected finding solely misreads a no-op or protocol marker, retain the valid value and document the rationale rather than fabricating complexity; genuine lost capture hazards still require repair. These are candidate revisions requiring complete re-audit and native review; never treat the model's old keep decisions as approval of changed content."""
RepairTarget = Literal[
    "primary",
    "caller_surface",
    "decoys",
    "pronunciations",
    "spelling_styles",
    "omission_instruction",
]
PlanStatus = Literal[
    "keep", "planned", "partial", "blocked", "source_blocked", "audit_error"
]


class MedicationStrengthEvidence(LocalizationModel):
    """One parsed numeric strength and its unchanged clinical unit."""

    amount: Annotated[
        Decimal,
        Field(
            ge=0, description="Exact decimal strength parsed from the written medicine."
        ),
    ]
    unit: Annotated[
        str,
        Field(
            description="Clinical unit recognized by the existing medication-strength contract."
        ),
    ]


class DifficultyMechanicalIssue(LocalizationModel):
    """A deterministic dose mismatch independent of the model difficulty judgment."""

    invariant: Annotated[
        Literal["medication_decoy_strength"],
        Field(
            description="Source invariant requiring repair without changing benchmark difficulty labels."
        ),
    ] = "medication_decoy_strength"
    canonical_task_id: Annotated[
        str,
        Field(
            description="Exact selected medication task owning the decoy occurrence."
        ),
    ]
    tier: Annotated[
        Literal["easy", "hard"],
        Field(
            description="Frozen assignment; Easy failures are reported but never selected."
        ),
    ]
    entry_id: Annotated[
        str, Field(description="Exact source entry that owns this decoy array.")
    ]
    decoy_index: Annotated[
        int,
        Field(ge=0, description="Position in the original fixed-length decoy array."),
    ]
    representation: Annotated[
        Literal["native", "romanized"],
        Field(description="Candidate representation with the dose mismatch."),
    ]
    source_value: Annotated[
        str, Field(description="Exact original wrong-answer medicine text.")
    ]
    current_value: Annotated[
        str, Field(description="Exact currently proposed wrong-answer medicine text.")
    ]
    source_strengths: Annotated[
        list[MedicationStrengthEvidence],
        Field(description="Per-position source numeric strengths and units."),
    ]
    current_strengths: Annotated[
        list[MedicationStrengthEvidence],
        Field(
            description="Current numeric strengths and units that differ from source."
        ),
    ]

    @model_validator(mode="after")
    def validate_strength_evidence(self):
        from tau2.domains.intake.localization import medication_strengths

        if not self.canonical_task_id.startswith(f"intake_medications_{self.tier}_"):
            raise ValueError(
                "Mechanical medication issue must name its exact family and tier"
            )
        for prefix in ("source", "current"):
            recorded = [
                (item.amount, item.unit)
                for item in getattr(self, f"{prefix}_strengths")
            ]
            if recorded != medication_strengths(getattr(self, f"{prefix}_value")):
                raise ValueError(
                    "Mechanical strength evidence differs from its written value"
                )
        if self.source_strengths == self.current_strengths:
            raise ValueError(
                "Mechanical medication issue requires a real strength or unit mismatch"
            )
        return self


class PhoneDecoyMechanicalIssue(PhoneDecoyLengthEvidence):
    """A source-relative phone length error introduced by localization."""

    invariant: Annotated[
        Literal["phone_decoy_length"],
        Field(
            description="Source-relative main-number or extension error shape mismatch."
        ),
    ] = "phone_decoy_length"
    canonical_task_id: Annotated[
        str, Field(description="Exact selected phone task owning the decoy occurrence.")
    ]
    tier: Annotated[
        Literal["easy", "hard"],
        Field(description="Frozen tier; Easy issues are report-only."),
    ]
    entry_id: Annotated[str, Field(description="Exact mutable decoy-array map entry.")]
    decoy_index: Annotated[
        int, Field(ge=0, description="Original position within the source decoy array.")
    ]
    representation: Annotated[
        Literal["native", "romanized"],
        Field(description="Representation with the introduced length error."),
    ]

    @model_validator(mode="after")
    def require_real_issue(self):
        if not self.canonical_task_id.startswith(f"intake_phones_{self.tier}_"):
            raise ValueError(
                "Mechanical phone issue must name its exact family and tier"
            )
        if self.preserves_source_error_shape:
            raise ValueError(
                "Mechanical phone issue requires a real source-relative length mismatch"
            )
        return self


MechanicalIssue = Annotated[
    DifficultyMechanicalIssue | PhoneDecoyMechanicalIssue,
    Field(discriminator="invariant"),
]


class DifficultyRepairCase(LocalizationModel):
    """One accounted-for Hard slot with an explicit safe scope or blocker."""

    canonical_task_id: Annotated[str, Field(description="Frozen Hard task identity.")]
    family: Annotated[
        Family, Field(description="Measured entity bank, including coined names.")
    ]
    source_value: Annotated[str, Field(description="Canonical written measurement.")]
    native_value: Annotated[
        str, Field(description="Current localized written measurement.")
    ]
    status: Annotated[
        PlanStatus, Field(description="Machine plan outcome; never human approval.")
    ]
    redraft_ids: Annotated[
        list[str], Field(description="Safe ordinary map entries selected by this slot.")
    ]
    caller_surface: Annotated[
        bool,
        Field(description="Whether this task needs separately authored time speech."),
    ]
    blockers: Annotated[
        list[str],
        Field(
            description="Explicit unresolved immutable, shared-scope or audit blockers."
        ),
    ]
    mechanical_issues: Annotated[
        list[MechanicalIssue],
        Field(
            default_factory=list,
            description="Exact deterministic source-invariant failures, retained independently of model disposition.",
        ),
    ]
    decision: Annotated[
        DifficultyDecision | None,
        Field(description="Exact model evidence, absent only for audit failures."),
    ]


class DifficultyEasyPronunciation(LocalizationModel):
    """Compact native speech contrast, without duplicated respelling metadata."""

    token: Annotated[str, Field(description="Exact native pronunciation token.")]
    distorted: Annotated[str, Field(description="Current native spoken error.")]


class DifficultyEasyControl(LocalizationModel):
    """Safe measured Easy evidence for ordinary entity authoring comparisons."""

    canonical_task_id: Annotated[str, Field(description="Exact frozen Easy task ID.")]
    native_value: Annotated[
        str, Field(description="Actual compiled native written answer.")
    ]
    romanized_value: Annotated[
        str, Field(description="Actual compiled romanized written answer.")
    ]
    native_caller_payload: Annotated[
        str,
        Field(
            description="Measured native lookup with private known-answer annotation if present."
        ),
    ]
    romanized_caller_payload: Annotated[
        str, Field(description="Corresponding romanized measured lookup.")
    ]
    native_decoys: Annotated[
        list[str],
        Field(
            description="Existing native wrong-answer neighbors for difficulty comparison."
        ),
    ]
    native_pronunciations: Annotated[
        list[DifficultyEasyPronunciation],
        Field(
            description="Compact native error contrasts already present in this Easy slot."
        ),
    ]
    source_evidence_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Digest of this complete safe DifficultyComparator in the bound inventory.",
        ),
    ]


class DifficultyEasyControls(LocalizationModel):
    """All ten measured Easy controls for one family, serialized once."""

    family: Annotated[
        Family, Field(description="Shared family of these Easy controls.")
    ]
    controls: Annotated[
        list[DifficultyEasyControl],
        Field(
            min_length=10,
            max_length=10,
            description="Exactly the ten same-family Easy slots; never selected examples only.",
        ),
    ]
    controls_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Digest of the exact compact comparison values.",
        ),
    ]

    @model_validator(mode="after")
    def validate_controls(self):
        """Reject partial, duplicate, mixed-family or stale comparison evidence."""
        ids = [control.canonical_task_id for control in self.controls]
        if len(set(ids)) != 10 or any(
            not identifier.startswith(f"intake_{self.family}_easy_")
            for identifier in ids
        ):
            raise ValueError(
                "Easy controls must cover ten distinct same-family Easy tasks"
            )
        if (
            json_content_digest(
                [control.model_dump(mode="json") for control in self.controls]
            )
            != self.controls_sha256
        ):
            raise ValueError(
                "Easy control content digest differs from recorded evidence"
            )
        return self


class DifficultyRevisionPlan(LocalizationModel):
    """Exhaustive repair accounting pinned to unchanged audit inputs."""

    version: Annotated[
        Literal["intake-difficulty-revision-plan-v4"],
        Field(description="Deterministic scope planner version."),
    ] = REVISION_PLAN_VERSION
    map_sha256: Annotated[str, Field(description="Exact full input map digest.")]
    source_sha256: Annotated[str, Field(description="Exact canonical source digest.")]
    audit_sha256: Annotated[str, Field(description="Exact typed audit report digest.")]
    inventory_sha256: Annotated[
        str, Field(description="Exact full measured inventory digest.")
    ]
    settings: Annotated[
        RegionalSettings, Field(description="Original target locale settings.")
    ]
    cases: Annotated[
        list[DifficultyRepairCase],
        Field(
            min_length=100,
            max_length=100,
            description="All 100 Hard slots; no silent omission of blocked or failed cases.",
        ),
    ]
    redraft_ids: Annotated[
        list[str],
        Field(description="Deterministically ordered union of safe ordinary changes."),
    ]
    caller_surface_task_ids: Annotated[
        list[str],
        Field(
            description="Distinct Hard time task IDs for separate typed caller-surface authoring."
        ),
    ]
    easy_controls: Annotated[
        list[DifficultyEasyControls],
        Field(
            min_length=10,
            max_length=10,
            description="One source-bound compact Easy comparison group per measured family.",
        ),
    ]
    protected_mechanical_issues: Annotated[
        list[MechanicalIssue],
        Field(
            default_factory=list,
            description="Easy-slot source-invariant failures recorded for review, never included in repair targets.",
        ),
    ]
    native_review_status: Annotated[
        Literal["pending"],
        Field(description="No native approval inferred from model evidence."),
    ] = "pending"

    @model_validator(mode="after")
    def validate_scope(self):
        families = [group.family for group in self.easy_controls]
        if len(set(families)) != 10 or set(families) != {
            case.family for case in self.cases
        }:
            raise ValueError(
                "Plan Easy controls must cover every measured family exactly once"
            )
        if (
            len({case.canonical_task_id for case in self.cases}) != 100
            or len(Counter(case.family for case in self.cases)) != 10
            or set(Counter(case.family for case in self.cases).values()) != {10}
        ):
            raise ValueError(
                "Repair plan must retain all 100 slots across all 10 families"
            )
        for case in self.cases:
            if not case.canonical_task_id.startswith(f"intake_{case.family}_hard_"):
                raise ValueError("Repair plan contains a non-Hard task")
            if case.status not in {"planned", "partial"} and (
                case.redraft_ids or case.caller_surface
            ):
                raise ValueError("Unplanned case cannot silently change entries")
            if case.status == "partial" and (
                not case.blockers or not (case.redraft_ids or case.caller_surface)
            ):
                raise ValueError(
                    "Partial repair must retain both a safe action and an explicit blocker"
                )
            if case.status == "planned" and (
                case.blockers or not (case.redraft_ids or case.caller_surface)
            ):
                raise ValueError(
                    "Planned repair requires an action and no unresolved scope blockers"
                )
            if case.caller_surface and case.family != "times":
                raise ValueError("Only time tasks use caller-surface overrides")
        for case in self.cases:
            if any(
                issue.tier != "hard"
                or issue.canonical_task_id != case.canonical_task_id
                for issue in case.mechanical_issues
            ):
                raise ValueError(
                    "Mechanical issues must belong to their exact Hard case"
                )
            if case.mechanical_issues and case.status == "keep":
                raise ValueError(
                    "A mechanical source-invariant failure cannot silently retain keep status"
                )
        if any(issue.tier != "easy" for issue in self.protected_mechanical_issues):
            raise ValueError(
                "Protected mechanical issues must remain outside Hard repair scope"
            )
        selected = {
            identifier for case in self.cases for identifier in case.redraft_ids
        }
        if selected & {issue.entry_id for issue in self.protected_mechanical_issues}:
            raise ValueError("Easy mechanical failures cannot be selected for repair")
        if sorted(selected) != sorted(self.redraft_ids) or len(
            set(self.redraft_ids)
        ) != len(self.redraft_ids):
            raise ValueError("Repair plan union differs from its per-case selection")
        surfaces = [
            case.canonical_task_id for case in self.cases if case.caller_surface
        ]
        if sorted(surfaces) != sorted(self.caller_surface_task_ids):
            raise ValueError("Caller-surface task union differs from the per-case plan")
        return self


class DifficultyRevisionFinding(LocalizationModel):
    """Model-produced constraints linked to the exact planned ordinary entry IDs."""

    canonical_task_id: Annotated[str, Field(description="Audited Hard slot.")]
    family: Annotated[
        Family,
        Field(description="Family used to select the complete Easy comparison group."),
    ]
    redraft_ids: Annotated[
        list[str],
        Field(description="Only these exact map entries may change for this finding."),
    ]
    decision: Annotated[
        DifficultyDecision | None,
        Field(
            description="Unmodified model assessment and repair constraints, absent only for failed audits."
        ),
    ]
    mechanical_issues: Annotated[
        list[MechanicalIssue],
        Field(
            default_factory=list,
            description="Factual source-invariant repairs for the exact selected mutable entries; never donor or Easy values.",
        ),
    ]


class DifficultyRevisionFeedback(LocalizationModel):
    """Ordinary revision feedback; candidate values remain fixed-prompt LLM output."""

    version: Annotated[
        Literal["intake-difficulty-revision-feedback-v4"],
        Field(description="Version of source-bound difficulty repair constraints."),
    ] = FEEDBACK_VERSION
    plan_sha256: Annotated[str, Field(description="Digest of the exact full plan.")]
    map_sha256: Annotated[str, Field(description="Exact parent map digest.")]
    audit_sha256: Annotated[str, Field(description="Exact model audit digest.")]
    instructions: Annotated[
        str,
        Field(
            description="Fixed versioned difficulty, locality and invariant constraints."
        ),
    ]
    findings: Annotated[
        list[DifficultyRevisionFinding],
        Field(
            description="Only ordinary map repair findings with exact selected entry IDs."
        ),
    ]
    easy_controls: Annotated[
        list[DifficultyEasyControls],
        Field(
            description="All ten Easy examples for each ordinary-repair family, once per family; no donor or callback context."
        ),
    ]

    @model_validator(mode="after")
    def require_comparison_controls(self):
        """A primary authoring request must actually supply its promised comparisons."""
        families = [group.family for group in self.easy_controls]
        if len(set(families)) != len(families) or set(families) != {
            finding.family for finding in self.findings
        }:
            raise ValueError(
                "Feedback Easy controls must cover exactly its repair families"
            )
        if any(
            not finding.canonical_task_id.startswith(f"intake_{finding.family}_hard_")
            for finding in self.findings
        ):
            raise ValueError(
                "Feedback task family differs from its comparison controls"
            )
        return self


class DifficultyLedgerRow(LocalizationModel):
    """Separate machine-review CSV contract; the entity approval CSV is unchanged."""

    model_config = ConfigDict(populate_by_name=True)

    language: Annotated[
        str, Field(alias="Language", description="Target locale language code.")
    ]
    task: Annotated[
        str, Field(alias="Task", description="Frozen canonical task identity.")
    ]
    family: Annotated[
        Family, Field(alias="Family", description="Measured bank family.")
    ]
    source_value: Annotated[
        str,
        Field(alias="Original entity", description="Canonical written measurement."),
    ]
    native_value: Annotated[
        str,
        Field(
            alias="Localized entity",
            description="Current written localized measurement.",
        ),
    ]
    source_difficulty: Annotated[
        str,
        Field(
            alias="Assessed source difficulty",
            description="Model assessment independent of source tier.",
        ),
    ]
    local_difficulty: Annotated[
        str,
        Field(
            alias="Assessed localized difficulty",
            description="Model assessment against all same-family Easy controls.",
        ),
    ]
    audit_disposition: Annotated[
        str,
        Field(
            alias="Audit disposition",
            description="Original model disposition, including temporal source blockers attempted conservatively.",
        ),
    ]
    source_limitation: Annotated[
        str,
        Field(
            alias="Source limitation",
            description="Unmodified source-blocker reason retained through repair planning.",
        ),
    ]
    status: Annotated[
        PlanStatus,
        Field(
            alias="Plan status",
            description="Keep, planned, blocked, source-blocked, or audit failure.",
        ),
    ]
    hazards: Annotated[
        str,
        Field(
            alias="Capture hazards",
            description="Specific model evidence and resolving clarification.",
        ),
    ]
    easy_comparison: Annotated[
        str,
        Field(
            alias="Easy comparison",
            description="Model comparison citing same-family Easy task IDs.",
        ),
    ]
    repairs: Annotated[
        str,
        Field(
            alias="Repair constraints",
            description="Model-produced constraints, never authored candidates.",
        ),
    ]
    blockers: Annotated[
        str,
        Field(
            alias="Blockers",
            description="Every unresolved source, audit or scope issue.",
        ),
    ]
    mechanical_issues: Annotated[
        str,
        Field(
            alias="Mechanical issues",
            description="Deterministic source/current dose evidence, independent of model disposition.",
        ),
    ]
    native_review: Annotated[
        Literal["pending"],
        Field(
            alias="Native review",
            description="Independent human review is still required.",
        ),
    ] = "pending"


class DifficultyLedgerManifest(LocalizationModel):
    """Digest-bearing accounting for every row in the separate ledger CSV."""

    version: Annotated[
        Literal["intake-difficulty-ledger-v2"],
        Field(description="Typed ledger format version."),
    ] = "intake-difficulty-ledger-v2"
    plan_sha256: Annotated[str, Field(description="Plan containing all row lineage.")]
    csv_sha256: Annotated[str, Field(description="Digest of exact UTF-8 CSV bytes.")]
    row_count: Annotated[
        Literal[100], Field(description="Exactly 100 frozen Hard slots.")
    ] = 100
    rows: Annotated[
        list[DifficultyLedgerRow],
        Field(
            min_length=100, max_length=100, description="Complete typed CSV contents."
        ),
    ]


class DifficultyRevisionFiles(LocalizationModel):
    """Paths returned by idempotent review artifact creation."""

    plan_file: Annotated[Path, Field(description="Typed repair plan.")]
    feedback_file: Annotated[
        Path,
        Field(description="Feedback compatible with ordinary DatabaseRevisionCommand."),
    ]
    ledger_csv: Annotated[
        Path, Field(description="Separate exhaustive difficulty machine ledger.")
    ]
    ledger_manifest_file: Annotated[
        Path, Field(description="Ledger provenance and exact content digest.")
    ]


def _easy_comparison_controls(
    inventory: DifficultyInventory,
) -> list[DifficultyEasyControls]:
    """Project safe measured comparisons once per family from the bound inventory."""
    groups = {}
    for case in inventory.cases:
        if case.family in groups:
            continue
        controls = [
            DifficultyEasyControl(
                canonical_task_id=control.canonical_task_id,
                native_value=control.native_value,
                romanized_value=control.romanized_value,
                native_caller_payload=control.native_caller_payload,
                romanized_caller_payload=control.romanized_caller_payload,
                native_decoys=control.native_dependencies.decoys,
                native_pronunciations=[
                    DifficultyEasyPronunciation(token=p.token, distorted=p.distorted)
                    for p in control.native_dependencies.pronunciations
                ],
                source_evidence_sha256=content_digest(control),
            )
            for control in case.easy_comparators
        ]
        groups[case.family] = DifficultyEasyControls(
            family=case.family,
            controls=controls,
            controls_sha256=json_content_digest(
                [control.model_dump(mode="json") for control in controls]
            ),
        )
    return [groups[family] for family in sorted(groups)]


def medication_decoy_issues(
    proposal: LocalizationMap,
) -> list[DifficultyMechanicalIssue]:
    """Check every selected medication decoy, including protected Easy controls."""
    from tau2.domains.intake.localization import medication_strengths
    from tau2.domains.intake.localized import IntakeLocalizationData
    from tau2.multilingual.factory.intake_authoring import _selected_decoy_family

    selected = IntakeLocalizationData.model_validate(
        next(
            document.content
            for document in proposal.source.documents
            if document.id == "intake_localization"
        )
    )
    issues = []
    for entry in proposal.entries:
        if _selected_decoy_family(entry) != "medications":
            continue
        rows = [
            selected.stimuli[point.path[1]]
            for point in entry.occurrences
            if point.document_id == "intake_localization"
            and len(point.path) == 3
            and point.path[0] == "stimuli"
            and point.path[2] == "decoys"
        ]
        for representation in ("native", "romanized"):
            values = getattr(entry.forms, representation)
            if (
                not isinstance(entry.source, list)
                or not isinstance(values, list)
                or len(entry.source) != len(values)
            ):
                raise ValueError(
                    "Mechanical medication inspection requires unchanged decoy array shape"
                )
            for index, (source, current) in enumerate(
                zip(entry.source, values, strict=True)
            ):
                if not isinstance(source, str) or not isinstance(current, str):
                    raise ValueError(
                        "Mechanical medication inspection requires string decoys"
                    )
                old, new = medication_strengths(source), medication_strengths(current)
                if old == new:
                    continue
                for row in rows:
                    issues.append(
                        DifficultyMechanicalIssue(
                            canonical_task_id=row.canonical_task_id,
                            tier=row.tier,
                            entry_id=entry.id,
                            decoy_index=index,
                            representation=representation,
                            source_value=source,
                            current_value=current,
                            source_strengths=[
                                MedicationStrengthEvidence(amount=amount, unit=unit)
                                for amount, unit in old
                            ],
                            current_strengths=[
                                MedicationStrengthEvidence(amount=amount, unit=unit)
                                for amount, unit in new
                            ],
                        )
                    )
    return issues


def phone_decoy_issues(proposal: LocalizationMap) -> list[PhoneDecoyMechanicalIssue]:
    """Compare every selected phone decoy with its exact source/current primary."""
    from tau2.domains.intake.localized import IntakeLocalizationData
    from tau2.multilingual.factory.intake_authoring import _selected_decoy_family

    selected = IntakeLocalizationData.model_validate(
        next(
            document.content
            for document in proposal.source.documents
            if document.id == "intake_localization"
        )
    )
    issues = []
    for entry in proposal.entries:
        if _selected_decoy_family(entry) != "phones":
            continue
        for point in entry.occurrences:
            if (
                point.document_id != "intake_localization"
                or len(point.path) != 3
                or point.path[0] != "stimuli"
                or point.path[2] != "decoys"
            ):
                continue
            row = selected.stimuli[point.path[1]]
            primary = next(
                item
                for item in proposal.entries
                if any(
                    p.document_id == "intake_localization"
                    and p.path == [*point.path[:-1], "written"]
                    for p in item.occurrences
                )
            )
            if primary.source != row.written or entry.source != row.decoys:
                raise ValueError(
                    "Phone mechanical inspection requires exact source occurrence values"
                )
            for representation in ("native", "romanized"):
                values = getattr(entry.forms, representation)
                if not isinstance(values, list) or len(values) != len(row.decoys):
                    raise ValueError(
                        "Phone mechanical inspection requires unchanged decoy array shape"
                    )
                for index, (source, current) in enumerate(
                    zip(row.decoys, values, strict=True)
                ):
                    evidence = phone_decoy_length_evidence(
                        row.written,
                        source,
                        getattr(primary.forms, representation),
                        current,
                    )
                    if not evidence.preserves_source_error_shape:
                        issues.append(
                            PhoneDecoyMechanicalIssue(
                                **evidence.model_dump(),
                                canonical_task_id=row.canonical_task_id,
                                tier=row.tier,
                                entry_id=entry.id,
                                decoy_index=index,
                                representation=representation,
                            )
                        )
    return issues


def build_difficulty_revision_plan(
    proposal: LocalizationMap,
    audit: DifficultyAuditReport,
    inventory: DifficultyInventory | None = None,
) -> DifficultyRevisionPlan:
    """Resolve only audited exact occurrences, closing changed-primary dependencies."""
    audit = DifficultyAuditReport.model_validate(audit.model_dump())
    inventory = inventory or build_difficulty_inventory(proposal)
    if (
        (
            content_digest(proposal),
            content_digest(proposal.source),
            content_digest(inventory),
            proposal.settings,
        )
        != (
            audit.map_sha256,
            audit.source_sha256,
            audit.inventory_sha256,
            audit.settings,
        )
        or inventory.map_sha256 != content_digest(proposal)
        or inventory.settings != proposal.settings
    ):
        raise ValueError(
            "Difficulty repair input digest/settings differ from audited inputs"
        )
    if set(audit.expected_task_ids) != {
        case.canonical_task_id for case in inventory.cases
    }:
        raise ValueError("Difficulty audit does not cover the exact inventory")
    entries = {entry.id: entry for entry in proposal.entries}
    decisions = {value.canonical_task_id: value for value in audit.decisions}
    errors = {value.canonical_task_id: value.error for value in audit.errors}
    easy_ids = {
        identifier
        for case in inventory.cases
        for easy in case.easy_comparators
        for identifier in easy.entry_sha256
    }
    source_tasks = next(
        document.content
        for document in proposal.source.documents
        if document.id == "tasks"
    )
    easy_task_indices = {
        index
        for index, task in enumerate(source_tasks)
        if task["id"].rsplit("_", 2)[1] == "easy"
    }
    easy_ids.update(
        entry.id
        for entry in proposal.entries
        if any(
            occurrence.document_id == "tasks"
            and occurrence.path
            and occurrence.path[0] in easy_task_indices
            for occurrence in entry.occurrences
        )
    )
    selected_source = next(
        document.content
        for document in proposal.source.documents
        if document.id == "intake_localization"
    )
    indices = {
        row["canonical_task_id"]: index
        for index, row in enumerate(selected_source["stimuli"])
    }
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
    mechanical = [*medication_decoy_issues(proposal), *phone_decoy_issues(proposal)]
    mechanical_by_task = {}
    for issue in mechanical:
        if issue.tier == "hard":
            mechanical_by_task.setdefault(issue.canonical_task_id, []).append(issue)
    task_owners = {}
    for case in inventory.cases:
        for identifier in case.entry_sha256:
            task_owners.setdefault(identifier, set()).add(case.canonical_task_id)
    authored_time_tasks = {
        surface.canonical_task_id for surface in proposal.intake_caller_surfaces
    }
    temporal_retry_tasks = {
        case.canonical_task_id
        for case in inventory.cases
        if case.family == "dates"
        or (
            case.family == "times" and case.canonical_task_id not in authored_time_tasks
        )
    }
    repairable_tasks = {
        case.canonical_task_id
        for case in inventory.cases
        if case.canonical_task_id in decisions
        and (
            decisions[case.canonical_task_id].disposition == "revise"
            or (
                decisions[case.canonical_task_id].disposition == "source_blocked"
                and case.canonical_task_id in temporal_retry_tasks
            )
        )
    }
    repairable_tasks.update(mechanical_by_task)
    planned = []
    for case in inventory.cases:
        assessment = decisions.get(case.canonical_task_id)
        selected = set()
        primary_group = set()
        blockers = []
        surface = False
        case_mechanical_issues = mechanical_by_task.get(case.canonical_task_id, [])
        if assessment is None:
            status = "audit_error"
            blockers = [errors[case.canonical_task_id]]
        elif (
            assessment.disposition == "source_blocked"
            and case.canonical_task_id not in temporal_retry_tasks
        ):
            status = "source_blocked"
            blockers = [assessment.source_blocked_reason]
        elif assessment.disposition == "keep":
            status = "keep"
        else:
            status = "planned"
            index = indices[case.canonical_task_id]
            dependencies = {
                target: {
                    identifier
                    for identifier in case.dependency_entry_ids
                    if any(
                        occurrence.document_id == "intake_localization"
                        and occurrence.path[:3] == ["stimuli", index, target]
                        for occurrence in entries[identifier].occurrences
                    )
                }
                for target in (
                    "decoys",
                    "pronunciations",
                    "spelling_styles",
                    "omission_instruction",
                )
            }
            targets = (
                (["caller_surface"] if case.family == "times" else ["primary"])
                if assessment.disposition == "source_blocked"
                else assessment.repair_targets
            )
            for target in targets:
                before_target = set(selected)
                if case.family == "times" and target in {
                    "primary",
                    "caller_surface",
                    "omission_instruction",
                }:
                    surface = True
                    continue
                if target in {"primary", "caller_surface"}:
                    if case.field_name in {"vin", "member_id"}:
                        blockers.append(
                            f"{target}: immutable {case.field_name} cannot be rewritten for difficulty; revise supported speech dependencies instead"
                        )
                        continue
                    if case.family == "dates":
                        if case.measured_entry_id in case.caller_entry_ids:
                            blockers.append(
                                "Date lacks a separately owned relative caller phrase; canonical date cannot change"
                            )
                            continue
                        else:
                            selected.update(case.caller_entry_ids)
                    else:
                        selected.update(
                            [case.measured_entry_id]
                            if target == "primary"
                            else case.caller_entry_ids
                        )
                    if (
                        target == "primary"
                        or case.measured_entry_id in case.caller_entry_ids
                    ):
                        selected.update(
                            identifier
                            for identifiers in dependencies.values()
                            for identifier in identifiers
                            if identifier not in immutable
                            and entries[identifier].source not in ("", [])
                        )
                        primary_group.update(selected - before_target)
                        primary_group.update(
                            identifier
                            for identifiers in dependencies.values()
                            for identifier in identifiers
                            if identifier in selected
                        )
                        primary_group.update(set(case.caller_entry_ids) & selected)
                        primary_group.update({case.measured_entry_id} & selected)
                else:
                    identifiers = dependencies[target]
                    if not identifiers:
                        blockers.append(
                            f"{target}: no exact source dependency entry exists"
                        )
                    for identifier in identifiers:
                        if identifier in immutable or entries[identifier].source in (
                            "",
                            [],
                        ):
                            blockers.append(
                                f"{target}: fixed empty or automatic source dependency cannot be expanded"
                            )
                        else:
                            selected.add(identifier)
        if case_mechanical_issues:
            selected.update(issue.entry_id for issue in case_mechanical_issues)
            status = "partial" if blockers else "planned"
        if status in {"planned", "partial"}:
            unsafe = set()
            for identifier in sorted(selected):
                if identifier in immutable:
                    unsafe.add(identifier)
                    blockers.append(
                        f"Automatic source entry is immutable: {identifier}"
                    )
                if entries[identifier].reuse is not None:
                    unsafe.add(identifier)
                    blockers.append(
                        f"Donor-sourced identity must remain unchanged: {identifier}"
                    )
                if identifier in easy_ids:
                    unsafe.add(identifier)
                    blockers.append(
                        f"Entry is shared with an Easy control: {identifier}"
                    )
                if any(
                    task_id not in repairable_tasks
                    for task_id in task_owners.get(identifier, set())
                ):
                    unsafe.add(identifier)
                    blockers.append(
                        f"Entry is shared with a Hard slot outside the repair scope: {identifier}"
                    )
            if unsafe & primary_group:
                selected.difference_update(primary_group)
                blockers.append(
                    "Primary replacement and its linked dependency group are blocked together to prevent incoherent identities"
                )
            selected.difference_update(unsafe)
            if not selected and not surface and not blockers:
                blockers.append("Audit requested no supported mutable repair")
            if blockers:
                status = "partial" if selected or surface else "blocked"
        planned.append(
            DifficultyRepairCase(
                canonical_task_id=case.canonical_task_id,
                family=case.family,
                source_value=case.source_value,
                native_value=case.native_value,
                status=status,
                redraft_ids=sorted(selected),
                caller_surface=surface,
                blockers=blockers,
                decision=assessment,
                mechanical_issues=case_mechanical_issues,
            )
        )
    identifiers = {identifier for case in planned for identifier in case.redraft_ids}
    return DifficultyRevisionPlan(
        map_sha256=content_digest(proposal),
        source_sha256=content_digest(proposal.source),
        audit_sha256=content_digest(audit),
        inventory_sha256=content_digest(inventory),
        settings=proposal.settings,
        cases=planned,
        redraft_ids=[entry.id for entry in proposal.entries if entry.id in identifiers],
        caller_surface_task_ids=[
            case.canonical_task_id for case in planned if case.caller_surface
        ],
        easy_controls=_easy_comparison_controls(inventory),
        protected_mechanical_issues=[
            issue for issue in mechanical if issue.tier == "easy"
        ],
    )


def build_difficulty_revision_feedback(
    plan: DifficultyRevisionPlan,
) -> DifficultyRevisionFeedback:
    """Attach original model constraints to only the exact ordinary repair scope."""
    repair_cases = [
        case
        for case in plan.cases
        if case.status in {"planned", "partial"} and case.redraft_ids
    ]
    families = {case.family for case in repair_cases}
    return DifficultyRevisionFeedback(
        plan_sha256=content_digest(plan),
        map_sha256=plan.map_sha256,
        audit_sha256=plan.audit_sha256,
        instructions=REPAIR_INSTRUCTIONS
        + "\n\nMechanical medication-decoy issues are factual source-invariant failures and take precedence over a model keep or source_blocked disposition for the selected decoy entry ONLY. Restore each exact source decoy clinical identity, strength, unit and formulation at its existing array position; decimal comma/point are equivalent. Retain the localized primary medicine and all unselected entries. A clinical difficulty source blocker remains unresolved after mechanical repair; never manufacture a new primary drug or dose to clear it. The original model decision is retained verbatim and does not authorize the observed dose drift. Audit suggestions NEVER authorize changing a source decoy clinical identity, formulation, dose, or array position. When source constraints make requested hardness infeasible, preserve clinical semantics and report the blocker; never invent a different drug."
        + "\n\nPhone mechanical issues compare each exact source decoy with its own source primary and the completed localized primary. Preserve the source-relative main-number digit-count difference and separate extension error shape at that array position; equal-length source pairs require equal-length localized pairs. Preserve genuine source missing/extra-digit errors. Do not blindly force all decoys to be valid or manufacture length errors because a model audit miscounted source digits. These factual constraints take precedence over contradictory audit suggestions; keep the authoritative primary unchanged for decoy-only repairs."
        + "\n\nFor each finding, compare against ALL TEN actual Easy controls in its same-family easy_controls group before choosing or retaining a Hard entity. These are read-only measured examples, not additional output targets. Compare written values, actual initial caller payloads, existing decoys and native pronunciation contrasts; the complete private answer is distinct from initial speech. A new native script or a hazard already equally present in Easy controls does not establish Hard difficulty."
        + "\n\n"
        + REVISION_PROMPT
        + "\n\n"
        + locale_rules(plan.settings),
        findings=[
            DifficultyRevisionFinding(
                canonical_task_id=case.canonical_task_id,
                family=case.family,
                redraft_ids=case.redraft_ids,
                decision=case.decision,
                mechanical_issues=[
                    issue
                    for issue in case.mechanical_issues
                    if issue.entry_id in case.redraft_ids
                ],
            )
            for case in repair_cases
        ],
        easy_controls=[
            group for group in plan.easy_controls if group.family in families
        ],
    )


def make_difficulty_revision_command(
    plan: DifficultyRevisionPlan, request: DifficultyAuditRequest, feedback_file: Path
) -> DatabaseRevisionCommand | None:
    """Create the existing database revision request, or none for surface-only plans."""
    if not plan.redraft_ids:
        return None
    return DatabaseRevisionCommand(
        previous_file=request.map_file.resolve(),
        revision=request.revision,
        generation=regional_generation(request.model, request.model_args),
        redraft_ids=plan.redraft_ids,
        feedback_files=[feedback_file.resolve()],
        batch_size=request.batch_size,
        max_concurrency=request.max_concurrency,
    )


def difficulty_ledger_rows(plan: DifficultyRevisionPlan) -> list[DifficultyLedgerRow]:
    """Render each source-pinned case with the shared per-language/aggregate schema."""
    rows = []
    for case in plan.cases:
        decision = case.decision
        rows.append(
            DifficultyLedgerRow.model_validate(
                {
                    "Language": plan.settings.language,
                    "Task": case.canonical_task_id,
                    "Family": case.family,
                    "Original entity": case.source_value,
                    "Localized entity": case.native_value,
                    "Assessed source difficulty": decision.source_difficulty
                    if decision
                    else "unassessed",
                    "Assessed localized difficulty": decision.localized_difficulty
                    if decision
                    else "unassessed",
                    "Audit disposition": decision.disposition
                    if decision
                    else "audit_error",
                    "Source limitation": decision.source_blocked_reason or ""
                    if decision
                    else "",
                    "Plan status": case.status,
                    "Capture hazards": "\n".join(
                        f"{hazard.tag}: {hazard.evidence} Clarification: {hazard.resolving_clarification}"
                        for hazard in decision.hazards
                    )
                    if decision
                    else "",
                    "Easy comparison": decision.comparison_to_easy if decision else "",
                    "Repair constraints": "\n".join(decision.correction_instructions)
                    if decision
                    else "",
                    "Blockers": "\n".join(case.blockers),
                    "Mechanical issues": "\n".join(
                        f"{issue.entry_id} [{issue.representation}] decoy {issue.decoy_index}: {issue.source_value!r} became {issue.current_value!r}; restore its source invariant ({issue.invariant})."
                        for issue in case.mechanical_issues
                    ),
                    "Native review": "pending",
                }
            )
        )
    return rows


def _render_ledger_csv(rows: list[DifficultyLedgerRow]) -> bytes:
    """Serialize the typed ledger contract identically for either report size."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            field.alias or name
            for name, field in DifficultyLedgerRow.model_fields.items()
        ],
    )
    writer.writeheader()
    writer.writerows(row.model_dump(by_alias=True) for row in rows)
    return buffer.getvalue().encode("utf-8")


def write_difficulty_revision_artifacts(
    plan: DifficultyRevisionPlan, output: Path
) -> DifficultyRevisionFiles:
    """Write immutable plan/feedback and a separate CSV with all 100 case outcomes."""
    files = DifficultyRevisionFiles(
        plan_file=output,
        feedback_file=output.with_suffix(".feedback.json"),
        ledger_csv=output.with_suffix(".ledger.csv"),
        ledger_manifest_file=output.with_suffix(".ledger.manifest.json"),
    )
    plan = DifficultyRevisionPlan.model_validate(plan.model_dump())
    rows = difficulty_ledger_rows(plan)
    data = _render_ledger_csv(rows)
    if files.ledger_csv.exists() and files.ledger_csv.read_bytes() != data:
        raise ValueError("Refusing to overwrite a different difficulty ledger")
    write_artifact(plan, files.plan_file)
    write_artifact(build_difficulty_revision_feedback(plan), files.feedback_file)
    write_artifact(
        DifficultyLedgerManifest(
            plan_sha256=content_digest(plan),
            csv_sha256=hashlib.sha256(data).hexdigest(),
            rows=rows,
        ),
        files.ledger_manifest_file,
    )
    if not files.ledger_csv.exists():
        files.ledger_csv.write_bytes(data)
    return files


AggregateLanguage = Literal["hi", "es", "pt", "zh", "ko"]
AGGREGATE_LANGUAGES: tuple[AggregateLanguage, ...] = ("hi", "es", "pt", "zh", "ko")


class DifficultyAggregateInput(LocalizationModel):
    """Exact per-language provenance for a row set in the combined checklist."""

    language: Annotated[
        AggregateLanguage,
        Field(description="One of the five required localization languages."),
    ]
    settings: Annotated[
        RegionalSettings,
        Field(description="Locale settings recorded by this exact plan."),
    ]
    source_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Shared canonical source frame digest.",
        ),
    ]
    map_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Exact full candidate map evaluated by the audit.",
        ),
    ]
    audit_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Exact complete audit report used for this plan.",
        ),
    ]
    inventory_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$", description="Full selected-slot evidence digest."
        ),
    ]
    plan_sha256: Annotated[
        str,
        Field(pattern=r"^[0-9a-f]{64}$", description="Exact input repair-plan digest."),
    ]

    protected_mechanical_issues: Annotated[
        list[MechanicalIssue],
        Field(
            description="Concrete protected Easy findings retained for review outside the 500-row Hard ledger."
        ),
    ]
    protected_mechanical_issue_count: Annotated[
        int,
        Field(
            ge=0,
            description="Number of per-representation Easy invariant findings; never repair or approval counts.",
        ),
    ]

    @model_validator(mode="after")
    def verify_protected_issues(self):
        if self.protected_mechanical_issue_count != len(
            self.protected_mechanical_issues
        ):
            raise ValueError(
                "Protected mechanical issue count differs from its evidence"
            )
        if any(issue.tier != "easy" for issue in self.protected_mechanical_issues):
            raise ValueError("Aggregate protected findings must belong to Easy slots")
        return self


class DifficultyFamilyCounts(LocalizationModel):
    """Complete machine-plan dispositions for one language and measured family."""

    language: Annotated[
        AggregateLanguage, Field(description="Target language of this ten-case cell.")
    ]
    family: Annotated[
        Family, Field(description="Measured family, including coined names.")
    ]
    keep: Annotated[
        int,
        Field(ge=0, le=10, description="Cases currently retained by the model audit."),
    ] = 0
    planned: Annotated[
        int,
        Field(
            ge=0,
            le=10,
            description="Cases with a supported repair and no unresolved scope blocker.",
        ),
    ] = 0
    partial: Annotated[
        int,
        Field(
            ge=0,
            le=10,
            description="Cases with both supported work and explicit unresolved targets.",
        ),
    ] = 0
    blocked: Annotated[
        int,
        Field(
            ge=0,
            le=10,
            description="Cases with no safe supported repair under current scope.",
        ),
    ] = 0
    source_blocked: Annotated[
        int,
        Field(ge=0, le=10, description="Cases requiring a source benchmark decision."),
    ] = 0
    audit_error: Annotated[
        int,
        Field(
            ge=0, le=10, description="Cases still lacking a valid model audit decision."
        ),
    ] = 0

    @model_validator(mode="after")
    def require_ten_cases(self):
        if (
            sum(
                (
                    self.keep,
                    self.planned,
                    self.partial,
                    self.blocked,
                    self.source_blocked,
                    self.audit_error,
                )
            )
            != 10
        ):
            raise ValueError(
                "Each language and family must account for exactly ten Hard cases"
            )
        return self


class DifficultyAggregateSummary(LocalizationModel):
    """One exhaustive source-bound checklist; model evidence never implies approval."""

    version: Annotated[
        Literal["intake-difficulty-aggregate-v3"],
        Field(description="Combined summary and ledger contract version."),
    ] = "intake-difficulty-aggregate-v3"
    source_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Single shared source frame across all five languages.",
        ),
    ]
    inputs: Annotated[
        list[DifficultyAggregateInput],
        Field(
            min_length=5,
            max_length=5,
            description="One exact provenance record per required language.",
        ),
    ]
    counts: Annotated[
        list[DifficultyFamilyCounts],
        Field(
            min_length=50,
            max_length=50,
            description="Every language by family disposition count, including unresolved cases.",
        ),
    ]
    rows: Annotated[
        list[DifficultyLedgerRow],
        Field(
            min_length=500,
            max_length=500,
            description="All 100 Hard cases in each language, using the existing ledger schema.",
        ),
    ]
    csv_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Digest of the exact combined CSV bytes.",
        ),
    ]
    native_review_status: Annotated[
        Literal["pending"],
        Field(description="Independent native and speech review remains pending."),
    ] = "pending"

    @model_validator(mode="after")
    def validate_complete_summary(self):
        languages = [item.language for item in self.inputs]
        if len(set(languages)) != 5 or set(languages) != set(AGGREGATE_LANGUAGES):
            raise ValueError(
                "Aggregate requires exactly hi, es, pt, zh and ko, without duplicates"
            )
        if any(
            item.settings.language != item.language
            or item.source_sha256 != self.source_sha256
            for item in self.inputs
        ):
            raise ValueError("Aggregate input locale or source frame differs")
        if len({(row.language, row.task) for row in self.rows}) != 500:
            raise ValueError(
                "Aggregate ledger must contain 500 unique language/task pairs"
            )
        frames = []
        for language in AGGREGATE_LANGUAGES:
            frame = {
                row.task: (row.family, row.source_value)
                for row in self.rows
                if row.language == language
            }
            if len(frame) != 100 or any(
                not task.startswith(f"intake_{family}_hard_")
                for task, (family, _) in frame.items()
            ):
                raise ValueError(
                    "Aggregate language is missing a canonical Hard task frame"
                )
            frames.append(frame)
        if any(frame != frames[0] for frame in frames[1:]):
            raise ValueError(
                "Aggregate cannot mix canonical task or source-value frames"
            )
        expected_cells = {
            (language, family)
            for language in AGGREGATE_LANGUAGES
            for family, _ in frames[0].values()
        }
        cells = {(cell.language, cell.family): cell for cell in self.counts}
        if len(cells) != 50 or set(cells) != expected_cells:
            raise ValueError(
                "Aggregate must contain every language by family count exactly once"
            )
        for (language, family), cell in cells.items():
            counts = Counter(
                row.status
                for row in self.rows
                if row.language == language and row.family == family
            )
            expected = DifficultyFamilyCounts(
                language=language, family=family, **counts
            )
            if cell != expected:
                raise ValueError(
                    "Aggregate counts disagree with the complete ledger rows"
                )
        if hashlib.sha256(_render_ledger_csv(self.rows)).hexdigest() != self.csv_sha256:
            raise ValueError("Aggregate CSV digest differs from its typed rows")
        return self


class DifficultyAggregateFiles(LocalizationModel):
    """The complete combined checklist and its typed, digest-bearing summary."""

    summary_file: Annotated[
        Path,
        Field(
            description="Typed summary JSON including counts, provenance and all rows."
        ),
    ]
    ledger_csv: Annotated[
        Path,
        Field(
            description="Combined 500-row ledger with the unchanged per-case CSV schema."
        ),
    ]


def build_difficulty_aggregate(
    plans: list[DifficultyRevisionPlan],
) -> DifficultyAggregateSummary:
    """Combine exactly the five current plans without mixing frozen source frames."""
    plans = [DifficultyRevisionPlan.model_validate(plan.model_dump()) for plan in plans]
    by_language = {plan.settings.language: plan for plan in plans}
    if (
        len(plans) != 5
        or len(by_language) != 5
        or set(by_language) != set(AGGREGATE_LANGUAGES)
    ):
        raise ValueError(
            "Aggregate requires exactly one plan for each of hi, es, pt, zh and ko"
        )
    if len({plan.source_sha256 for plan in plans}) != 1:
        raise ValueError("Aggregate cannot mix source frame digests")
    ordered = [by_language[language] for language in AGGREGATE_LANGUAGES]
    rows = [row for plan in ordered for row in difficulty_ledger_rows(plan)]
    counts = []
    for plan in ordered:
        for family in sorted({case.family for case in plan.cases}):
            counts.append(
                DifficultyFamilyCounts(
                    language=plan.settings.language,
                    family=family,
                    **Counter(
                        case.status for case in plan.cases if case.family == family
                    ),
                )
            )
    return DifficultyAggregateSummary(
        source_sha256=ordered[0].source_sha256,
        inputs=[
            DifficultyAggregateInput(
                language=plan.settings.language,
                settings=plan.settings,
                source_sha256=plan.source_sha256,
                map_sha256=plan.map_sha256,
                audit_sha256=plan.audit_sha256,
                inventory_sha256=plan.inventory_sha256,
                plan_sha256=content_digest(plan),
                protected_mechanical_issues=plan.protected_mechanical_issues,
                protected_mechanical_issue_count=len(plan.protected_mechanical_issues),
            )
            for plan in ordered
        ],
        counts=counts,
        rows=rows,
        csv_sha256=hashlib.sha256(_render_ledger_csv(rows)).hexdigest(),
    )


def write_difficulty_aggregate(
    summary: DifficultyAggregateSummary, output: Path
) -> DifficultyAggregateFiles:
    """Write exactly one combined CSV and its complete typed summary, idempotently."""
    summary = DifficultyAggregateSummary.model_validate(summary.model_dump())
    files = DifficultyAggregateFiles(
        summary_file=output, ledger_csv=output.with_suffix(".ledger.csv")
    )
    data = _render_ledger_csv(summary.rows)
    if files.ledger_csv.exists() and files.ledger_csv.read_bytes() != data:
        raise ValueError(
            "Refusing to overwrite a different aggregate difficulty ledger"
        )
    write_artifact(summary, files.summary_file)
    if not files.ledger_csv.exists():
        files.ledger_csv.write_bytes(data)
    return files
