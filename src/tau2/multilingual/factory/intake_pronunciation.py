"""Source-bound deterministic repair of redundant localized pronunciation guides."""

import argparse
from pathlib import Path
from typing import Literal

from pydantic import Field

from tau2.multilingual.factory.localization_artifact import (
    RegionalMapArtifact,
    restore_localization_map,
)
from tau2.multilingual.factory.localization_inventory import inventory_localization
from tau2.multilingual.localization import LocalizationModel
from tau2.utils.artifacts import content_digest, write_artifact
from tau2.utils.utils import DATA_DIR

REFERENCE_REPAIR_VERSION = "intake-plain-pronunciation-references-v1"


class ReferenceChange(LocalizationModel):
    """One exact reference replacement; tokens and distortions are immutable."""

    entry_id: str
    representation: Literal["native", "romanized"]
    index: int = Field(ge=0)
    previous: str
    reference: str


class ReferenceRepair(LocalizationModel):
    """Deterministic map lineage, separate from prior model audits and approval."""

    version: Literal["intake-plain-pronunciation-references-v1"] = (
        REFERENCE_REPAIR_VERSION
    )
    parent_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    changes: list[ReferenceChange]
    status: Literal["pending_native_review"] = "pending_native_review"


def repair_native_references(
    artifact: RegionalMapArtifact, revision: str
) -> tuple[RegionalMapArtifact, ReferenceRepair]:
    """Reuse each existing written token as its guide without authoring phonetics."""
    if artifact.source.domain != "intake":
        raise ValueError("This repair is scoped to the intake candidate map")
    source = inventory_localization("intake", DATA_DIR)
    selected = {
        e.id for e in source.entries if "dependency=pronunciations;" in e.context
    }
    result = artifact.model_copy(deep=True)
    changes = []
    for entry in result.entries:
        if entry.id not in selected:
            continue
        for representation in ("native", "romanized"):
            for index, pronunciation in enumerate(getattr(entry.forms, representation)):
                if pronunciation["reference"] == pronunciation["token"]:
                    continue
                changes.append(
                    ReferenceChange(
                        entry_id=entry.id,
                        representation=representation,
                        index=index,
                        previous=pronunciation["reference"],
                        reference=pronunciation["token"],
                    )
                )
                pronunciation["reference"] = pronunciation["token"]
    if changes:
        if revision == artifact.revision:
            raise ValueError("Changed pronunciation guides require a new map revision")
        result.revision = revision
    # Restore verifies exact canonical documents, source entries, shapes and both-form
    # full-task compilation. The old artifact may fail the newly tightened guide rule.
    proposal = restore_localization_map(result, source)
    previous = proposal.model_copy(deep=True)
    previous.revision = artifact.revision
    original = {e.id: e for e in artifact.entries}
    for entry in previous.entries:
        if entry.id in original:
            entry.forms = original[entry.id].forms.model_copy(deep=True)
    record = ReferenceRepair(
        parent_artifact_sha256=content_digest(artifact),
        artifact_sha256=content_digest(result),
        parent_map_sha256=content_digest(previous),
        map_sha256=content_digest(proposal),
        source_sha256=content_digest(source),
        changes=changes,
    )
    return result, record


def _run(args: argparse.Namespace) -> None:
    artifact = RegionalMapArtifact.model_validate_json(args.map.read_text())
    result, record = repair_native_references(artifact, args.revision)
    write_artifact(result, args.output)
    write_artifact(record, args.record)


def add_intake_pronunciation_args(commands: argparse._SubParsersAction) -> None:
    """Register the source-checked deterministic repair with the factory package."""
    parser = commands.add_parser(
        "repair-intake-pronunciation-references",
        help="Replace localized pronunciation guides with their existing exact written tokens.",
    )
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.set_defaults(func=_run)
