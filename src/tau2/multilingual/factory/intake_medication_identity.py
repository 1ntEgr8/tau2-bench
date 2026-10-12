"""Conservative, source-bound medication label consistency, not medical translation QA."""

import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from tau2.domains.intake.localized import IntakeLocalizationData
from tau2.multilingual.factory.localization_compile import validate_localization_source
from tau2.multilingual.localization import LocalizationMap, LocalizationModel
from tau2.utils.artifacts import content_digest, write_artifact

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Nonempty = Annotated[str, Field(min_length=1)]
Kind = Literal["distinct_source_label_collision", "same_source_label_variant"]


def _label(value: str) -> str | None:
    """Read the name before one explicit strength; leave unfamiliar layouts unparsed.

    Every selected frozen source drug uses this layout. No medical synonym,
    transliteration, ingredient, or local dosage-form dictionary is inferred.
    """
    value = unicodedata.normalize("NFKC", value)
    strengths = list(
        re.finditer(r"[0-9]+(?:[.,][0-9]+)?\s*(?:mcg|mg|mL|ml|g|%)(?!\w)", value)
    )
    if len(strengths) != 1:
        return None
    prefix = " ".join(value[: strengths[0].start()].casefold().split())
    if not prefix or any(c.isdigit() for c in prefix):
        return None
    return prefix


class MedicationIdentityObservation(LocalizationModel):
    """One exact selected primary or indexed decoy, with reproducible label parsing."""

    canonical_task_id: Nonempty = Field(description="Frozen selected medication task.")
    tier: Literal["easy", "hard"] = Field(description="Original difficulty assignment.")
    role: Literal["primary", "decoy"] = Field(
        description="Selected value's source role."
    )
    entry_id: Nonempty = Field(description="Exact map entry owning this value.")
    entry_sha256: Digest = Field(
        description="Digest of the complete source-bound entry."
    )
    decoy_index: Annotated[int, Field(ge=0)] | None = Field(
        description="Decoy position, or null for primary."
    )
    representation: Literal["native", "romanized"] = Field(
        description="Candidate script representation inspected independently."
    )
    source_value: Nonempty = Field(description="Exact frozen written source medicine.")
    current_value: Nonempty = Field(description="Exact candidate written medicine.")
    source_label: str | None = Field(
        description="Normalized prefix before strength, or unparsed."
    )
    current_label: str | None = Field(
        description="Normalized candidate prefix before strength, or unparsed."
    )

    @model_validator(mode="after")
    def verify_labels(self):
        if self.canonical_task_id not in {
            f"intake_medications_{self.tier}_{index:02d}" for index in range(1, 11)
        }:
            raise ValueError("Medication task identity and frozen tier disagree")
        if (self.role == "primary") != (self.decoy_index is None):
            raise ValueError("Medication identity role and decoy index disagree")
        if self.source_label != _label(
            self.source_value
        ) or self.current_label != _label(self.current_value):
            raise ValueError("Medication label evidence differs from the exact values")
        return self


class MedicationIdentityFinding(LocalizationModel):
    """A consistency observation requiring review; no suggested replacement or culprit."""

    kind: Kind = Field(description="Reproducible label relation requiring review.")
    observations: Annotated[
        list[MedicationIdentityObservation],
        Field(
            min_length=2,
            description="All implicated selected slots in one representation.",
        ),
    ]
    requires_review: Literal[True] = Field(
        default=True,
        description="A consistency finding never authorizes an automatic repair.",
    )
    explanation: Nonempty = Field(
        description="Meaning and limits of the observed relation, without a proposed replacement."
    )

    @model_validator(mode="after")
    def verify_relation(self):
        if len({o.representation for o in self.observations}) != 1:
            raise ValueError(
                "Medication identity comparisons cannot mix representations"
            )
        source = {o.source_label for o in self.observations}
        current = {o.current_label for o in self.observations}
        if None in source or None in current:
            raise ValueError("Unparsed medication labels cannot support a finding")
        valid = (
            len(source) > 1 and len(current) == 1
            if self.kind == "distinct_source_label_collision"
            else len(source) == 1 and len(current) > 1
        )
        if not valid:
            raise ValueError(
                "Medication identity finding is unsupported by its label evidence"
            )
        return self


def _findings(
    observations: list[MedicationIdentityObservation],
) -> list[MedicationIdentityFinding]:
    findings = []
    for kind, key in (
        ("distinct_source_label_collision", "current_label"),
        ("same_source_label_variant", "source_label"),
    ):
        grouped = defaultdict(list)
        for observation in observations:
            if (
                observation.source_label is not None
                and observation.current_label is not None
            ):
                grouped[(observation.representation, getattr(observation, key))].append(
                    observation
                )
        for _, group in sorted(grouped.items()):
            other = "source_label" if key == "current_label" else "current_label"
            if len({getattr(o, other) for o in group}) < 2:
                continue
            findings.append(
                MedicationIdentityFinding(
                    kind=kind,
                    observations=group,
                    explanation=(
                        "Different frozen source drug labels collapse to the same local label. Review all implicated slots; this does not identify the incorrect translation or rule out source synonyms."
                        if kind == "distinct_source_label_collision"
                        else "The same frozen source drug label has multiple local labels. Review consistency; legitimate synonym or orthographic variants may explain this, so no automatic redraft is authorized."
                    ),
                )
            )
    return findings


class MedicationIdentityReport(LocalizationModel):
    """Deterministic evidence with complete accounting, never clinical approval."""

    version: Literal["intake-medication-identity-v1"] = Field(
        default="intake-medication-identity-v1",
        description="Versioned contract for the frozen twenty-task medication frame.",
    )
    map_sha256: Digest = Field(description="Exact candidate map digest.")
    source_sha256: Digest = Field(description="Exact frozen source digest.")
    language: Nonempty = Field(description="Candidate map's target language.")
    observations: list[MedicationIdentityObservation] = Field(
        description="Complete eighty-slot frame: twenty tasks, primary and one decoy, both representations."
    )
    findings: list[MedicationIdentityFinding] = Field(
        description="Every supported collision or variant derived from the full observations."
    )
    unparsed: list[MedicationIdentityObservation] = Field(
        description="Every observation with an unrecognized source or candidate label layout."
    )
    clinical_identity_status: Literal["unverified"] = Field(
        default="unverified",
        description="Consistency alone can never verify clinical translation.",
    )
    limitations: Literal[
        "Label consistency does not verify medical translation: unique or consistently wrong mappings can pass; synonyms and unrecognized name-before-strength layouts require review. Dose and formulation validity are separate checks."
    ] = Field(
        default="Label consistency does not verify medical translation: unique or consistently wrong mappings can pass; synonyms and unrecognized name-before-strength layouts require review. Dose and formulation validity are separate checks.",
        description="Fixed limits that remain visible even when no consistency finding exists.",
    )

    @model_validator(mode="after")
    def verify_accounting(self):
        keys = [
            (o.canonical_task_id, o.role, o.decoy_index, o.representation)
            for o in self.observations
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate medication identity observations")
        expected = {
            (
                f"intake_medications_{tier}_{index:02d}",
                role,
                decoy_index,
                representation,
            )
            for tier in ("easy", "hard")
            for index in range(1, 11)
            for role, decoy_index in (("primary", None), ("decoy", 0))
            for representation in ("native", "romanized")
        }
        if set(keys) != expected:
            raise ValueError(
                "Medication identity report requires all 80 frozen medication slots"
            )
        if self.findings != _findings(self.observations):
            raise ValueError(
                "Medication identity findings do not match all observations"
            )
        if self.unparsed != [
            o
            for o in self.observations
            if o.source_label is None or o.current_label is None
        ]:
            raise ValueError("Medication identity unparsed accounting differs")
        return self


def medication_identity_report(proposal: LocalizationMap) -> MedicationIdentityReport:
    """Check all selected medicines without external calls or choosing repair targets."""
    if proposal.source.domain != "intake":
        raise ValueError("Medication identity inspection requires an intake map")
    proposal = validate_localization_source(proposal)
    selected = IntakeLocalizationData.model_validate(
        next(
            d.content
            for d in proposal.source.documents
            if d.id == "intake_localization"
        )
    )
    owners = {}
    for entry in proposal.entries:
        for occurrence in entry.occurrences:
            if occurrence.document_id != "intake_localization":
                continue
            key = tuple(occurrence.path)
            if key in owners:
                raise ValueError("Duplicate medication identity occurrence owner")
            owners[key] = entry
    observations = []
    for index, row in enumerate(selected.stimuli):
        if row.family != "medications":
            continue
        for field, sources in (("written", [row.written]), ("decoys", row.decoys)):
            entry = owners.get(("stimuli", index, field))
            if entry is None or entry.source != (
                row.written if field == "written" else row.decoys
            ):
                raise ValueError(
                    "Missing or mismatched source medication identity entry"
                )
            if entry.reuse is not None:
                raise ValueError(
                    "Medication identity inspection excludes donor identities"
                )
            for representation in ("native", "romanized"):
                forms = getattr(entry.forms, representation)
                values = [forms] if field == "written" else forms
                if (
                    not isinstance(values, list)
                    or len(values) != len(sources)
                    or any(not isinstance(v, str) for v in values)
                ):
                    raise ValueError(
                        "Medication identity inspection requires unchanged string slot shape"
                    )
                for position, (source, current) in enumerate(
                    zip(sources, values, strict=True)
                ):
                    observations.append(
                        MedicationIdentityObservation(
                            canonical_task_id=row.canonical_task_id,
                            tier=row.tier,
                            role="primary" if field == "written" else "decoy",
                            entry_id=entry.id,
                            entry_sha256=content_digest(entry),
                            decoy_index=None if field == "written" else position,
                            representation=representation,
                            source_value=source,
                            current_value=current,
                            source_label=_label(source),
                            current_label=_label(current),
                        )
                    )
    return MedicationIdentityReport(
        map_sha256=content_digest(proposal),
        source_sha256=content_digest(proposal.source),
        language=proposal.settings.language,
        observations=observations,
        findings=_findings(observations),
        unparsed=[
            o for o in observations if o.source_label is None or o.current_label is None
        ],
    )


def _run_inspection(args) -> None:
    from tau2.multilingual.factory.localization_artifact import read_localization_map

    write_artifact(
        medication_identity_report(read_localization_map(args.map)), args.output
    )


def add_intake_medication_identity_args(commands) -> None:
    """Register the deterministic, local-only medication identity inspection verb."""
    parser = commands.add_parser(
        "inspect-intake-medication-identities",
        help="Report source-bound medicine label consistency; no model calls or approval",
    )
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(func=_run_inspection)
