"""Reorder selected intake address components and their used dependencies."""

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from tau2.multilingual.localization import LocalizationMap, LocalizedForms, SourceEntry
from tau2.utils.artifacts import content_digest, write_artifact

if TYPE_CHECKING:
    from tau2.multilingual.factory.localization_address_components import (
        AddressComponentReport,
    )

INTAKE_ADDRESS_REPAIR_VERSION = "intake-address-order-v1"


def reorder_intake_address(value: str, language: str, task_id: str) -> str:
    """Apply proven component permutations, never infer an address from punctuation.

    The selected task ID bounds the reviewed repair. Unknown layouts stay unchanged;
    no province, postcode, unit or premise identifier is invented or discarded.
    """
    parts = value.split(", ")
    prefix = []
    if language == "pt" and task_id in {
        f"intake_addresses_hard_{i:02d}" for i in (2, 3, 4, 5, 6, 10)
    }:
        if (
            len(parts) >= 5
            and re.fullmatch(r"nº \d+", parts[1])
            and re.match(r"(?:Rua|Viela|Travessa|Passagem)\b", parts[2])
            and re.match(r"(?:Lote|Casa|Unidade|Galpão|Apto)\b", parts[0])
        ):
            prefix = [2, 1, 0]
    elif language == "es":
        rules = {
            "intake_addresses_hard_01": ([1, 2, 0], 5),
            "intake_addresses_hard_03": ([3, 4, 0, 1, 2], 8),
            "intake_addresses_hard_04": ([2, 3, 0, 1], 6),
            "intake_addresses_hard_05": ([2, 3, 0, 1], 5),
            "intake_addresses_hard_10": ([2, 3, 0, 1], 5),
        }
        rule = rules.get(task_id)
        if rule is not None:
            order, count = rule
            if (
                len(parts) == count
                and re.match(r"(?:Calle|Calleja|Travesía)\b", parts[order[0]])
                and re.fullmatch(r"\d+", parts[order[1]])
            ):
                prefix = order
        if prefix and task_id == "intake_addresses_hard_03":
            # Exact audited locality pair: official Asturias/Correos directories
            # identify 33940 with El Entrego. Only move these existing tokens.
            if parts[5:] != [
                "El Entrego",
                "San Martín del Rey Aurelio",
                "33940 Asturias",
            ]:
                raise ValueError(
                    "Spanish Hard 03 locality changed; review the bound repair"
                )
            parts[5], parts[7] = "33940 El Entrego", "Asturias"
    if not prefix:
        return value
    ordered = [parts[i] for i in prefix] + parts[len(prefix) :]
    result = ", ".join(ordered)
    if Counter(re.findall(r"\w+", value)) != Counter(re.findall(r"\w+", result)):
        raise ValueError("Intake address repair changed component tokens")
    return result


def validate_intake_address_forms(
    entry: SourceEntry, forms: LocalizedForms, language: str | None
) -> None:
    """Reject recurrence of the exact selected address-order defects at authoring."""
    if language not in {"es", "pt"}:
        return
    if not (
        entry.context.startswith("intake field:home_address;")
        or "dependency=decoys;" in entry.context
    ):
        return
    tasks = set(re.findall(r"intake_addresses_(?:easy|hard)_[0-9]{2}", entry.context))
    if not tasks:
        return
    for representation in ("native", "romanized"):
        value = getattr(forms, representation)
        values = value if isinstance(value, list) else [value]
        for text in values:
            if not isinstance(text, str):
                continue
            if any(
                reorder_intake_address(text, language, task_id) != text
                for task_id in tasks
            ):
                raise ValueError(
                    "Intake address uses source-style unit/street or postcode ordering; use native component order while preserving every identifier"
                )


def _address_tasks(proposal: LocalizationMap, entry: SourceEntry) -> set[str]:
    source = next(
        doc for doc in proposal.source.documents if doc.id == "intake_localization"
    )
    return {
        source.content["stimuli"][occ.path[1]]["canonical_task_id"]
        for occ in entry.occurrences
        if occ.document_id == "intake_localization"
        and len(occ.path) == 3
        and occ.path[0] == "stimuli"
        and source.content["stimuli"][occ.path[1]]["family"] == "addresses"
    }


def repair_intake_addresses(
    proposal: LocalizationMap,
) -> tuple[LocalizationMap, "AddressComponentReport"]:
    """Recompile source-bound address values and retain every intentional decoy error."""
    from tau2.multilingual.address_format import AddressRevisionProvenance
    from tau2.multilingual.factory.localization_address_components import (
        AddressComponentChange,
        AddressComponentReport,
    )

    if proposal.source.domain != "intake":
        raise ValueError("Intake address repair requires the intake domain")
    result = proposal.model_copy(deep=True)
    replacements = {"native": {}, "romanized": {}}
    changes = []
    for old, new in zip(proposal.entries, result.entries, strict=True):
        if "intake field:home_address;" not in old.context:
            continue
        tasks = _address_tasks(proposal, old)
        for representation in ("native", "romanized"):
            before = getattr(old.forms, representation)
            after = before
            for task_id in sorted(tasks):
                after = reorder_intake_address(
                    after, proposal.settings.language, task_id
                )
            if after != before:
                for task_id in tasks:
                    scoped = replacements[representation].setdefault(task_id, {})
                    if scoped.get(before, after) != after:
                        raise ValueError("Ambiguous shared intake address value")
                    scoped[before] = after
                setattr(new.forms, representation, after)
        if old.forms != new.forms:
            new.reuse = None
            new.notes += f"\n{INTAKE_ADDRESS_REPAIR_VERSION}: reordered existing components; numeric identifiers, roles and difficulty retained."
            changes.append(
                AddressComponentChange(
                    entry_id=old.id,
                    before=old.forms,
                    after=new.forms,
                    reason="Exact selected address component permutation.",
                )
            )
    for old, new in zip(proposal.entries, result.entries, strict=True):
        if "dependency=" not in old.context or not (
            tasks := _address_tasks(proposal, old)
        ):
            continue
        for representation in ("native", "romanized"):
            scoped_replacements = {}
            for task_id in tasks:
                for before, after in (
                    replacements[representation].get(task_id, {}).items()
                ):
                    if (
                        before in scoped_replacements
                        and scoped_replacements[before] != after
                    ):
                        raise ValueError("Conflicting address dependency ownership")
                    scoped_replacements[before] = after

            def revise(value):
                if isinstance(value, str):
                    if value in scoped_replacements:
                        return scoped_replacements[value]
                    # Near-miss decoys use the same scoped permutation while
                    # retaining their changed road/unit tokens.
                    if "dependency=decoys;" in old.context:
                        for task_id in sorted(tasks):
                            value = reorder_intake_address(
                                value, proposal.settings.language, task_id
                            )
                    for before, after in sorted(
                        scoped_replacements.items(),
                        key=lambda pair: len(pair[0]),
                        reverse=True,
                    ):
                        value = re.sub(
                            r"(?<!\w)" + re.escape(before) + r"(?!\w)",
                            lambda _: after,
                            value,
                        )
                    return value
                if isinstance(value, list):
                    return [revise(item) for item in value]
                if isinstance(value, dict):
                    return {key: revise(item) for key, item in value.items()}
                return value

            setattr(
                new.forms, representation, revise(getattr(old.forms, representation))
            )
        if old.forms != new.forms:
            new.reuse = None
            new.notes += f"\n{INTAKE_ADDRESS_REPAIR_VERSION}: synchronized used address dependencies; decoy wrongness retained."
            changes.append(
                AddressComponentChange(
                    entry_id=old.id,
                    before=old.forms,
                    after=new.forms,
                    reason="Used source-bound address decoy/guidance synchronization.",
                )
            )
    if changes:
        result.revision += f"-{INTAKE_ADDRESS_REPAIR_VERSION}"
        result.address_revision = AddressRevisionProvenance(
            parent_map_sha256=content_digest(proposal),
            changed_entry_ids=[change.entry_id for change in changes],
        )
    return result, AddressComponentReport(
        parent_map_sha256=content_digest(proposal),
        output_map_sha256=content_digest(result),
        changes=changes,
        review_findings=[],
    )


def _run(args: argparse.Namespace) -> None:
    from tau2.multilingual.factory.localization_artifact import (
        RegionalMapArtifact,
        restore_localization_map,
        write_localization_map,
    )
    from tau2.multilingual.factory.localization_inventory import inventory_localization
    from tau2.utils import DATA_DIR

    artifact = RegionalMapArtifact.model_validate_json(args.map.read_text())
    source = inventory_localization("intake", DATA_DIR)
    proposal = restore_localization_map(artifact, source, validate_candidate=False)
    repaired, report = repair_intake_addresses(proposal)
    write_localization_map(repaired, args.output)
    write_artifact(report, args.report)
    logger.info(
        "Intake address repair: {} changed entries; review CSVs unchanged",
        len(report.changes),
    )


def add_intake_address_repair_args(commands) -> None:
    """Register selected-bank repair through the existing language factory."""
    command = commands.add_parser(
        "repair-intake-addresses",
        help="Reorder proven address components and used decoys without changing difficulty",
    )
    command.add_argument("--map", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--report", type=Path, required=True)
    command.set_defaults(func=_run)
