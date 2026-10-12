"""Apply exact, approved entity corrections through the existing regional compiler."""

from pathlib import Path

from loguru import logger
from pydantic import Field, JsonValue, model_validator

from tau2.multilingual.localization import (
    Domain,
    IntakeCallerSurface,
    LocalizationMap,
    LocalizationModel,
    LocalizedForms,
    NameGenderEvidence,
    NameGenderItemEvidence,
    ProseEntityBinding,
)
from tau2.utils.artifacts import ReviewApproval, content_digest, write_artifact


class EntityCorrection(LocalizationModel):
    """One guarded entry or exact prose span, never a global text replacement."""

    entry_id: str
    source: JsonValue
    binding: bool = False
    before: LocalizedForms | None
    after: LocalizedForms
    reason: str = Field(min_length=1)
    name_gender: NameGenderEvidence | None = None
    name_gender_items: list[NameGenderItemEvidence] | None = None


class CallerSurfaceCorrection(LocalizationModel):
    """Reviewed speech change with an immutable task, answer and generation record."""

    before: IntakeCallerSurface = Field(description="Exact surface before review.")
    after: IntakeCallerSurface = Field(
        description="Corrected wording with original anchors and lineage."
    )
    reason: str = Field(
        min_length=1, description="Reason for this approved correction."
    )

    @model_validator(mode="after")
    def preserve_anchor(self):
        """Only wording and rationale may change; never canonical truth or lineage."""
        for field in ("canonical_task_id", "canonical_value", "provenance"):
            if getattr(self.before, field) != getattr(self.after, field):
                raise ValueError(f"Caller correction cannot change {field}")
        if self.after.review is not None:
            raise ValueError("Caller correction after value must omit prior review")
        return self


class ReviewedCorrections(LocalizationModel):
    """Approved source-bound edits; they do not approve the complete language pack."""

    version: str = "reviewed-localization-corrections-v1"
    domain: Domain
    language: str
    revision: str
    reviewed_by: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    corrections: list[EntityCorrection] = Field(default_factory=list)
    caller_surfaces: list[CallerSurfaceCorrection] = Field(
        default_factory=list,
        description="Exact reviewed intake caller-wording corrections.",
    )

    @model_validator(mode="after")
    def validate_edits(self):
        """Require a nonempty review and keep caller surfaces in their owning domain."""
        if not self.corrections and not self.caller_surfaces:
            raise ValueError("Review requires at least one correction")
        if self.caller_surfaces and self.domain != "intake":
            raise ValueError("Caller surface corrections require intake")
        return self


def apply_corrections(
    proposal: LocalizationMap, review: ReviewedCorrections
) -> LocalizationMap:
    """Reject stale/duplicate edits and apply exact reviewed values to a copy."""
    if (proposal.source.domain, proposal.settings.language) != (
        review.domain,
        review.language,
    ):
        raise ValueError("Review domain/language does not match the map")
    result = proposal.model_copy(deep=True)
    seen = set()
    for edit in review.corrections:
        key = (edit.entry_id, str(edit.source), edit.binding)
        if key in seen:
            raise ValueError("Duplicate reviewed correction")
        seen.add(key)
        matches = [
            entry
            for entry in (result.prose_bindings if edit.binding else result.entries)
            if getattr(entry, "entry_id" if edit.binding else "id") == edit.entry_id
            and entry.source == edit.source
        ]
        if not matches and edit.binding and edit.before is None:
            parent = next((e for e in result.entries if e.id == edit.entry_id), None)
            if (
                parent is None
                or not isinstance(parent.source, str)
                or not isinstance(edit.source, str)
                or edit.source not in parent.source
            ):
                raise ValueError("New reviewed binding is not an exact source span")
            created = ProseEntityBinding(
                entry_id=edit.entry_id,
                source=edit.source,
                forms=edit.after,
                kind="reviewed",
                reason=edit.reason,
                review=ReviewApproval(
                    content_sha256=content_digest(edit.after),
                    reviewed_by=review.reviewed_by,
                ),
            )
            result.prose_bindings.append(created)
            matches = [created]
        if len(matches) != 1:
            raise ValueError(f"Review source no longer resolves: {edit.entry_id}")
        entry = matches[0]
        if entry.forms not in (edit.before, edit.after):
            raise ValueError(f"Stale reviewed value: {edit.entry_id}")
        entry.forms = edit.after.model_copy(deep=True)
        if edit.binding:
            if edit.name_gender is not None:
                raise ValueError("Binding name edits require structured owner evidence")
            if edit.name_gender_items is not None:
                raise ValueError("Binding name edits require structured owner evidence")
            entry.kind = "reviewed"
            entry.source_entry_id = None
            entry.source_part = None
            entry.provenance = None
            entry.review = ReviewApproval(
                content_sha256=content_digest(entry.forms),
                reviewed_by=review.reviewed_by,
            )
            entry.reason = edit.reason
        else:
            entry.reuse = None
            if (
                entry.forms.native != entry.source
                or entry.forms.romanized != entry.source
            ):
                entry.operation = "adapt"
            entry.notes = edit.reason
            if edit.name_gender is not None:
                edit.name_gender.require(entry.forms)
                entry.name_gender = edit.name_gender
            elif entry.name_gender is not None:
                entry.name_gender.require(entry.forms)
            if edit.name_gender_items is not None:
                for item in edit.name_gender_items:
                    item.require(entry.forms)
                entry.name_gender_items = edit.name_gender_items
            else:
                for item in entry.name_gender_items:
                    item.require(entry.forms)
    seen_surfaces = set()
    for edit in review.caller_surfaces:
        task_id = edit.before.canonical_task_id
        if task_id in seen_surfaces:
            raise ValueError("Duplicate reviewed caller surface")
        seen_surfaces.add(task_id)
        matches = [
            i
            for i, surface in enumerate(result.intake_caller_surfaces)
            if surface.canonical_task_id == task_id
        ]
        if len(matches) != 1:
            raise ValueError(f"Review caller task no longer resolves: {task_id}")
        corrected = edit.after.model_copy(deep=True)
        corrected.review = ReviewApproval(
            content_sha256=content_digest(edit.after), reviewed_by=review.reviewed_by
        )
        if result.intake_caller_surfaces[matches[0]] not in (edit.before, corrected):
            raise ValueError(f"Stale reviewed caller surface: {task_id}")
        result.intake_caller_surfaces[matches[0]] = corrected
    result.revision = review.revision
    return type(result).model_validate(result.model_dump())


def _run(args) -> None:
    from tau2.multilingual.factory.localization_artifact import (
        RegionalMapArtifact,
        restore_localization_map,
        write_localization_map,
    )
    from tau2.multilingual.factory.localization_inventory import inventory_localization
    from tau2.utils import DATA_DIR

    review = ReviewedCorrections.model_validate_json(args.review.read_text())
    artifact = RegionalMapArtifact.model_validate_json(args.mapping.read_text())
    source = inventory_localization(artifact.source.domain, DATA_DIR)
    proposal = restore_localization_map(artifact, source, validate_candidate=False)
    result = apply_corrections(proposal, review)
    if review.domain == "banking_knowledge":
        from tau2.multilingual.factory.banking_candidate_review import (
            BankingReviewRules,
            synchronize_banking_candidate,
        )

        if args.rules is None:
            raise ValueError("Banking corrections require the existing locale rules")
        rules = BankingReviewRules.model_validate_json(args.rules.read_text())
        reconstruction, _ = synchronize_banking_candidate(result, result, rules)
        result = reconstruction.proposal
        if reconstruction.reconstruction.unresolved:
            raise ValueError("Reviewed corrections leave unresolved banking prose")
    write_localization_map(result, args.output)
    if review.domain == "banking_knowledge":
        write_artifact(
            reconstruction.reconstruction,
            args.output.with_suffix(".reconstruction.json"),
        )
    write_artifact(review, args.output.with_suffix(".review.json"))
    logger.info("Applied reviewed corrections to {}", args.output)


def add_correction_args(commands) -> None:
    """Register deterministic, reviewed source-bound corrections on tau2 factory."""
    parser = commands.add_parser("apply-localization-corrections")
    parser.add_argument("--map", dest="mapping", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(func=_run)
