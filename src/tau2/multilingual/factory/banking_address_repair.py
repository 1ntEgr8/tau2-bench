"""Exact-component migration of banking candidates into locale-aware addresses."""

import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field

from tau2.domains.banking_knowledge.localization_prose import (
    BankingProseRepairProvenance,
)
from tau2.multilingual.address_format import (
    AddressForms,
    AddressParts,
    AddressRevisionProvenance,
    format_address,
)
from tau2.multilingual.localization import (
    LocalizationMap,
    LocalizationModel,
    LocalizedForms,
)
from tau2.utils.artifacts import content_digest, write_artifact


class AddressRepairFinding(LocalizationModel):
    """Unresolved source evidence, never an invented administrative component."""

    entry_id: str = Field(description="Exact source entry or binding owner.")
    reason: str = Field(description="Remaining native-review question.")


class BankingAddressRepairReport(LocalizationModel):
    """Exact input/output evidence for a deterministic repair, not human approval."""

    version: Literal["banking-address-components-v1"] = "banking-address-components-v1"
    parent_map_sha256: str = Field(description="Exact expanded input map digest.")
    output_map_sha256: str = Field(description="Exact expanded output map digest.")
    changed_entry_ids: list[str] = Field(
        description="Changed entries, including dependent prose."
    )
    changed_binding_count: int = Field(
        description="Changed source-bound prose binding occurrences."
    )
    findings: list[AddressRepairFinding] = Field(
        description="Explicit unresolved geography/role evidence."
    )
    reconstruction: BankingProseRepairProvenance = Field(
        description="Original and repaired dependent prose, retaining source reconstruction lineage."
    )
    status: Literal["pending_native_review"] = "pending_native_review"


def decompose_address(value: str, language: str, representation: str) -> AddressParts:
    """Migrate the prior authoring contract once, never parse display text at runtime.

    The old prompts explicitly stored the final city/region/postcode/country
    components. Recognize only that recorded format; unknown layouts fail closed.
    No new geographic names, street labels, premise numbers or postal codes enter.
    """
    pieces = [piece.strip() for piece in value.split(",")]
    if len(pieces) >= 5 and re.fullmatch(r"[0-9]{5,6}|[0-9]{5}-[0-9]{3}", pieces[-2]):
        streets, city, region, postal, country = pieces[:-4], *pieces[-4:]
    elif (
        language == "hi"
        and len(pieces) >= 4
        and (match := re.fullmatch(r"(.+?)\s+([0-9]{6})", pieces[-2]))
    ):
        streets, city, region, postal, country = (
            pieces[:-3],
            pieces[-3],
            match[1],
            match[2],
            pieces[-1],
        )
    else:
        raise ValueError(
            "Address does not match the source-bound prior component contract"
        )
    if not all([streets, city, region, country]):
        raise ValueError("Missing source address component")
    district = ""
    if language == "ko":
        is_district = (
            city.endswith("구")
            if representation == "native"
            else city.casefold().endswith("-gu")
        )
        if is_district:
            # The suffix supplies explicit district evidence. The next component
            # is the supplied municipality; a missing province remains missing.
            district, city, region = city, region, ""
            metropolitan = (
                city.endswith(("특별시", "광역시", "특별자치시"))
                if representation == "native"
                else city.casefold().endswith(
                    ("-teukbyeolsi", "-gwangyeoksi", "-teukbyeoljachisi")
                )
            )
            if metropolitan:
                region = city
    if language in {"zh", "ko"}:
        # Remove only exact redundant administrative prefixes, never fuzzy words
        # or the same characters occurring inside a road/property proper name.
        for prefix in dict.fromkeys([region, city, district]):
            if prefix and streets[0].startswith(prefix):
                remaining = streets[0][len(prefix) :]
                if (
                    language == "zh"
                    and representation == "native"
                    or remaining.startswith(" ")
                ):
                    streets[0] = remaining.strip()
    return AddressParts(
        street_lines=streets,
        district=district,
        city=city,
        region=region,
        postal_code=postal,
        country=country,
    )


def repair_address_forms(
    forms: LocalizedForms, language: str
) -> tuple[LocalizedForms, list[str]]:
    """Reorder exact components and record unresolved roles without inventing data."""
    if forms.address is not None:
        if forms.address.language != language:
            raise ValueError("Address metadata belongs to another language")
        for representation in ("native", "romanized"):
            if language in {"zh", "ko", "es"} and getattr(
                forms, representation
            ) != format_address(
                getattr(forms.address, representation), language, representation
            ):
                raise ValueError(
                    "Rendered address no longer matches explicit component evidence"
                )
        findings = []
        if not forms.address.native.region or not forms.address.romanized.region:
            findings.append(
                "The existing candidate supplies a district and municipality but no province; province remains absent rather than inferred. Native geographic review required."
            )
        return forms.model_copy(deep=True), findings
    parts = AddressForms(
        language=language,
        native=decompose_address(forms.native, language, "native"),
        romanized=decompose_address(forms.romanized, language, "romanized"),
    )
    result = forms.model_copy(deep=True)
    result.address = parts
    for representation in ("native", "romanized"):
        previous = getattr(forms, representation)
        if language in {"zh", "ko", "es"}:
            updated = format_address(
                getattr(parts, representation), language, representation
            )
            if Counter(re.findall(r"[0-9]+", previous)) != Counter(
                re.findall(r"[0-9]+", updated)
            ):
                raise ValueError("Address rendering changed numeric components")
            setattr(result, representation, updated)
    findings = []
    if not parts.native.region or not parts.romanized.region:
        findings.append(
            "The existing candidate supplies a district and municipality but no province; province remains absent rather than inferred. Native geographic review required."
        )
    return result, findings


def repair_banking_addresses(
    proposal: LocalizationMap,
) -> tuple[LocalizationMap, BankingAddressRepairReport]:
    """Repair complete addresses and every source-bound dependent prose occurrence."""
    from tau2.domains.banking_knowledge.localization_bindings import address_component
    from tau2.domains.banking_knowledge.localization_prose import (
        reconstruct_banking_prose,
    )
    from tau2.multilingual.factory.localization_compile import validate_localization

    if proposal.source.domain != "banking_knowledge":
        raise ValueError("Banking address repair requires the banking domain")
    result = proposal.model_copy(deep=True)
    findings = []
    language = proposal.settings.language
    for entry in result.entries:
        if entry.category != "address":
            continue
        # Invalid verification challenges are deliberate non-address values.
        if (
            entry.operation == "preserve"
            and entry.forms.native == entry.source == entry.forms.romanized
        ):
            entry.notes = "Intentional invalid address-verification value preserved exactly; no residential address was authored."
            continue
        previous_forms = entry.forms
        entry.forms, unresolved = repair_address_forms(entry.forms, language)
        if entry.forms != previous_forms:
            entry.reuse = None
        findings.extend(
            AddressRepairFinding(entry_id=entry.id, reason=reason)
            for reason in unresolved
        )
    indexed = {entry.id: entry for entry in result.entries}
    for binding in result.prose_bindings:
        if binding.kind == "reuse":
            reference = indexed[binding.source_entry_id]
            binding.forms = (
                address_component(reference, binding.source_part)
                if binding.source_part
                else reference.forms
            )
        elif binding.kind == "generated" and re.match(r"^\d+\s+.+,", binding.source):
            binding.forms, unresolved = repair_address_forms(binding.forms, language)
            findings.extend(
                AddressRepairFinding(entry_id=binding.entry_id, reason=reason)
                for reason in unresolved
            )
    reconstruction = reconstruct_banking_prose(result)
    result = reconstruction.proposal
    before = {entry.id: entry for entry in proposal.entries}
    changed = [entry.id for entry in result.entries if entry != before[entry.id]]
    binding_changes = sum(
        left != right
        for left, right in zip(
            proposal.prose_bindings, result.prose_bindings, strict=True
        )
    )
    if changed or binding_changes:
        result.address_revision = AddressRevisionProvenance(
            parent_map_sha256=content_digest(proposal), changed_entry_ids=changed
        )
    validate_localization(result)
    return result, BankingAddressRepairReport(
        parent_map_sha256=content_digest(proposal),
        output_map_sha256=content_digest(result),
        changed_entry_ids=changed,
        changed_binding_count=binding_changes,
        findings=findings,
        reconstruction=reconstruction.reconstruction,
    )


def _run_repair(args) -> None:
    from tau2.multilingual.factory.localization_artifact import (
        RegionalMapArtifact,
        restore_localization_map,
        write_localization_map,
    )
    from tau2.multilingual.factory.localization_inventory import inventory_localization
    from tau2.utils import DATA_DIR

    artifact = RegionalMapArtifact.model_validate_json(args.map.read_text())
    source = inventory_localization(artifact.source.domain, DATA_DIR)
    # Exact source coverage is still checked. Only candidate validation is
    # deferred until the obsolete address layout and its bindings are repaired.
    proposal = restore_localization_map(artifact, source, validate_candidate=False)
    result, report = repair_banking_addresses(proposal)
    write_localization_map(result, args.output)
    write_artifact(report, args.report)


def add_banking_address_repair_args(commands) -> None:
    """Register the source-bound address migration in the existing factory."""
    parser = commands.add_parser(
        "repair-banking-addresses",
        help="Repair exact address components and banking prose without generating entities",
    )
    for name in ("map", "output", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.set_defaults(func=_run_repair)
