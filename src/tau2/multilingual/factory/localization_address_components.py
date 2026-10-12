"""Source-bound repair of malformed road components in shared regional maps."""

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import Field, TypeAdapter

from tau2.multilingual.address_format import AddressRevisionProvenance
from tau2.multilingual.localization import (
    LocalizationEntry,
    LocalizationMap,
    LocalizationModel,
    LocalizedForms,
    SourceRefreshProvenance,
)
from tau2.utils.artifacts import content_digest, write_artifact

ADDRESS_COMPONENT_REPAIR_VERSION = "regional-address-components-v1"


class AddressComponentChange(LocalizationModel):
    """Exact source record and before/after forms for a deterministic repair."""

    entry_id: str = Field(description="Stable source-bound localization entry ID.")
    before: LocalizedForms = Field(
        description="Exact prior forms, including explanation."
    )
    after: LocalizedForms = Field(
        description="Repaired forms with conserved identifiers."
    )
    reason: str = Field(description="Rule applied to this source role.")


class ToolEvidenceTransition(LocalizationModel):
    """Exact reviewed code-evidence transition; never a blanket source refresh."""

    domain: Literal["retail", "airline", "telecom"] = Field(
        description="Domain whose existing runtime wiring changed the evidence."
    )
    document_id: str = Field(
        pattern=r"^tool_text:", description="Automatic source-code evidence document."
    )
    before_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Exact evidence digest stored in the old map.",
    )
    after_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Exact inspected current evidence digest.",
    )
    reason: str = Field(
        min_length=1,
        description="Explanation of the reviewed code change; no native-content approval.",
    )


class AddressComponentReport(LocalizationModel):
    """Repair evidence; never confers native or postal approval."""

    version: Literal["regional-address-components-v1"] = (
        ADDRESS_COMPONENT_REPAIR_VERSION
    )
    parent_map_sha256: str = Field(description="Digest of expanded input map.")
    output_map_sha256: str = Field(description="Digest of expanded output map.")
    tool_evidence_transitions: list[ToolEvidenceTransition] = Field(
        default_factory=list,
        description="Only the exact inspected evidence changes accepted for this map.",
    )
    prior_source_refresh: SourceRefreshProvenance | None = Field(
        default=None,
        description="Preserved historical line-refresh provenance superseded by explicit current code evidence.",
    )
    input_artifact_sha256: str | None = Field(
        default=None,
        description="Exact compact input artifact, including its prior source manifest.",
    )
    refreshed_tool_document_ids: list[str] = Field(
        default_factory=list,
        description="Explicitly refreshed automatic code-evidence documents; never task/DB data.",
    )
    changes: list[AddressComponentChange] = Field(description="Exact changed records.")
    review_findings: list[str] = Field(
        description="Ambiguous occurrences left untouched."
    )


def repair_street_line(value: str, language: str, representation: str) -> str:
    """Reorder an unambiguous number-before-road defect, preserving all digits.

    This operates only on known street/address1 roles. It is not an address parser:
    units, postal codes, P.O. boxes and unsupported road shapes remain untouched.
    Latin forms retain native reference ordering, not international mailing order.
    """
    if language not in {"es", "pt", "zh", "ko"}:
        return value
    match = re.fullmatch(r"(\d+)\s*(?:号\s*|\s+)(.+)", value)
    if match is None:
        return value
    number, road = match.groups()
    if language == "es":
        if not re.match(
            r"(?:Calle|Carrer|Carretera|Avenida|Avinguda|Gran V[ií]a|Rúa|Paseo|Passeig|Camino|Ronda|Rambla|Plaza|Plaça)\b",
            road,
        ):
            return value
        result = f"{road} {number}"
    elif language == "pt":
        if not re.match(
            r"(?:Rua|Avenida|Travessa|Alameda|Estrada|Viela|Passagem|Praça|Rodovia|Ladeira|Servidão)\b",
            road,
        ):
            return value
        result = f"{road}, {number}"
    elif language == "zh":
        if representation == "native":
            if not re.fullmatch(r"[^\d,，]+[路街巷道]", road):
                return value
            result = f"{road}{number}号"
        else:
            road = re.sub(r"^Hao\s+", "", road, flags=re.IGNORECASE)
            if not re.search(r"(?:lu|jie|xiang|dao)$", road, re.IGNORECASE):
                return value
            result = f"{road} {number} Hao"
    else:
        suffix = r"(?:로|길)$" if representation == "native" else r"(?:ro|gil)$"
        if not re.search(suffix, road, re.IGNORECASE):
            return value
        result = f"{road} {number}"
    if Counter(re.findall(r"\d+", value)) != Counter(re.findall(r"\d+", result)):
        raise ValueError("Address repair changed numeric identifiers")
    return result


def _street(entry: LocalizationEntry) -> bool:
    return entry.category == "address" and bool(
        re.match(r"role=(?:address1|street);", entry.context)
    )


def _span(value: str):
    return re.compile(r"(?<!\w)" + re.escape(value) + r"(?!\w)")


def _owner(entry: LocalizationEntry) -> str | None:
    match = re.search(
        r"(?:entity=|Scenario/evaluation prose for )([^;]+);", entry.context
    )
    return match[1] if match else None


def repair_address_components(
    proposal: LocalizationMap,
) -> tuple[LocalizationMap, AddressComponentReport]:
    """Repair structured street entries and only source-bound prose references."""
    if proposal.source.domain not in {"retail", "airline", "telecom"}:
        raise ValueError(
            "Structured address repair supports retail, airline and telecom"
        )
    result = proposal.model_copy(deep=True)
    changes = []
    findings = []
    streets = []
    for old, new in zip(proposal.entries, result.entries, strict=True):
        if not _street(old):
            continue
        for representation in ("native", "romanized"):
            value = getattr(old.forms, representation)
            if not isinstance(value, str):
                raise ValueError(f"Street component must be a string: {old.id}")
            setattr(
                new.forms,
                representation,
                repair_street_line(value, proposal.settings.language, representation),
            )
        if old.forms != new.forms:
            new.reuse = None
            new.notes = f"{old.notes}\n{ADDRESS_COMPONENT_REPAIR_VERSION}: reordered road/premise components without changing identifiers; native review remains pending."
            changes.append(
                AddressComponentChange(
                    entry_id=old.id,
                    before=old.forms,
                    after=new.forms,
                    reason="Locale road-before-premise ordering on an explicit street role.",
                )
            )
            streets.append((old, new))
    # A street reference in prose must be grounded in its canonical source string.
    # Never globally replace an equal-looking localized substring across owners.
    for old, new in zip(proposal.entries, result.entries, strict=True):
        if old.category != "prose" or not isinstance(old.source, str):
            continue
        for representation in ("native", "romanized"):
            value = getattr(old.forms, representation)
            if not isinstance(value, str):
                continue
            substitutions = {}
            for before, after in streets:
                original = getattr(before.forms, representation)
                replacement = getattr(after.forms, representation)
                if original == replacement or not _span(original).search(value):
                    continue
                if (
                    not isinstance(before.source, str)
                    or not _span(before.source).search(old.source)
                    or _owner(before) is None
                    or _owner(before) != _owner(old)
                ):
                    findings.append(
                        f"{old.id}:{representation}: unbound address reference to {before.id}; review required"
                    )
                    continue
                if original in substitutions and substitutions[original] != replacement:
                    raise ValueError(
                        f"Conflicting source-bound address references in {old.id}"
                    )
                substitutions[original] = replacement
            if substitutions:
                pattern = re.compile(
                    "|".join(
                        r"(?<!\w)" + re.escape(s) + r"(?!\w)"
                        for s in sorted(substitutions, key=len, reverse=True)
                    )
                )
                setattr(
                    new.forms,
                    representation,
                    pattern.sub(lambda m: substitutions[m[0]], value),
                )
        if old.forms != new.forms:
            new.notes = f"{old.notes}\n{ADDRESS_COMPONENT_REPAIR_VERSION}: synchronized source-bound street references."
            changes.append(
                AddressComponentChange(
                    entry_id=old.id,
                    before=old.forms,
                    after=new.forms,
                    reason="Canonical source and exact old regional street both bind the changed prose span.",
                )
            )
    if changes:
        # The report binds original generation provenance to this deterministic edit.
        # Changing forms also changes the map digest, invalidating old approvals.
        result.revision = f"{proposal.revision}-{ADDRESS_COMPONENT_REPAIR_VERSION}"
        result.address_revision = AddressRevisionProvenance(
            parent_map_sha256=content_digest(proposal),
            changed_entry_ids=[change.entry_id for change in changes],
        )
    report = AddressComponentReport(
        parent_map_sha256=content_digest(proposal),
        output_map_sha256=content_digest(result),
        changes=changes,
        review_findings=sorted(set(findings)),
    )
    return result, report


def read_address_repair_input(
    path: Path,
    refresh_tool_evidence: bool = False,
    transitions: list[ToolEvidenceTransition] | None = None,
) -> tuple[LocalizationMap, list[str]]:
    """Optionally repin automatic tool evidence while refusing any data/source drift."""
    from tau2.multilingual.factory.localization_artifact import (
        RegionalMapArtifact,
        SourceDescriptor,
        SourceManifest,
        read_localization_map,
        restore_localization_map,
    )
    from tau2.multilingual.factory.localization_inventory import inventory_localization
    from tau2.utils import DATA_DIR

    if not refresh_tool_evidence:
        return read_localization_map(path), []
    artifact = RegionalMapArtifact.model_validate_json(path.read_text())
    source = inventory_localization(artifact.source.domain, DATA_DIR)
    old = {doc.id: doc for doc in artifact.source.documents}
    current = {
        doc.id: SourceDescriptor.model_validate(doc.model_dump(exclude={"content"}))
        for doc in source.documents
    }
    if old.keys() != current.keys():
        raise ValueError("Tool-evidence refresh changed source document membership")
    changed = []
    for key, before in old.items():
        after = current[key]
        if before == after:
            continue
        if (
            before.kind != "tool_text"
            or after.kind != "tool_text"
            or before.source_path != after.source_path
        ):
            raise ValueError(
                "Tool-evidence refresh cannot change database, tasks, policy or document identity"
            )
        if not any(
            t.domain == source.domain
            and t.document_id == key
            and t.before_sha256 == before.sha256
            and t.after_sha256 == after.sha256
            for t in transitions or []
        ):
            raise ValueError(
                "Changed tool evidence requires an exact inspected transition; no automatic semantic refresh"
            )
        changed.append(key)
    rebound = artifact.model_copy(
        update={
            "source_refresh": None,
            "source": SourceManifest(
                domain=source.domain,
                documents=list(current.values()),
            ),
        }
    )
    # Exact authored category/source/ID coverage is still checked here; only
    # automatic tool evidence is rebuilt. This is an explicit CLI action, never
    # a fallback for a failed source digest check.
    return restore_localization_map(rebound, source), changed


def _run(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.localization_artifact import (
        RegionalMapArtifact,
        write_localization_map,
    )

    original = RegionalMapArtifact.model_validate_json(args.map.read_text())
    transitions = (
        TypeAdapter(list[ToolEvidenceTransition]).validate_json(
            args.tool_evidence_transition.read_text()
        )
        if args.tool_evidence_transition
        else []
    )
    loaded, refreshed = read_address_repair_input(
        args.map, args.refresh_tool_evidence, transitions
    )
    proposal, report = repair_address_components(loaded)
    report.input_artifact_sha256 = content_digest(original)
    report.prior_source_refresh = original.source_refresh
    report.tool_evidence_transitions = [
        t
        for t in transitions
        if t.domain == original.source.domain and t.document_id in refreshed
    ]
    report.refreshed_tool_document_ids = refreshed
    if report.review_findings:
        write_artifact(report, args.report)
        raise ValueError(
            "Unbound address references need review; report written, map not emitted"
        )
    write_localization_map(proposal, args.output)
    write_artifact(report, args.report)
    logger.info(
        "Address component repair: {} changed entries; CSVs untouched",
        len(report.changes),
    )


def add_address_component_repair_args(commands) -> None:
    """Register the deterministic, source-bound structured address repair stage."""
    command = commands.add_parser(
        "repair-address-components",
        help="Repair known locale street-order defects without changing address identities",
    )
    command.add_argument("--map", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--report", type=Path, required=True)
    command.add_argument(
        "--refresh-tool-evidence",
        action="store_true",
        help="Explicitly repin only automatic code evidence to this checkout; refuse changed task/DB/policy sources",
    )
    command.add_argument(
        "--tool-evidence-transition",
        type=Path,
        help="Typed exact before/after evidence digests for an inspected existing code transition",
    )
    command.set_defaults(func=_run)
