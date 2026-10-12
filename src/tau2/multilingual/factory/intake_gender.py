"""Preserve frozen source name genders and propagate intake identity repairs."""

import argparse
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, RootModel

from tau2.domains.intake.localized import IntakeLocalizationData
from tau2.domains.intake.tasks.banks import load_banks
from tau2.domains.intake.tasks.generator import FreezeManifest
from tau2.multilingual.factory.generation_config import regional_generation
from tau2.multilingual.factory.localization_gender import (
    GenderNameRequest,
    GenderNameResult,
)
from tau2.multilingual.localization import LocalizationMap, LocalizationModel

INTAKE_GENDER_VERSION = "intake-gender-preservation-v1"
INTAKE_GENDER_DEPENDENCY_RULES = """Propagate the supplied gender-preserving intake name repairs.
Completed name mappings are authoritative. Preserve every frozen task, family and
Easy/Hard tier. Keep native and romanized forms the same identity. For changed
measured names, refresh pronunciation tokens/distortions and applicable spelling or
omission guidance against that exact new answer. Decoy names have independently
assessed source gender: retain each supplied decoy and its gender, even where it
differs from the primary person. Do not change authoritative primary or decoy names.
For emails whose callee changed, preserve the existing localized domain, punctuation,
digit runs, separators and letter/number capture hazards, updating only owner-linked
name components to match the supplied callee. Refresh each email's dependent decoys,
pronunciation and clarification instructions. Keep ASCII protocol values. A name
appearing as the measured second guest is not the speaking callee. Never change
task persona/gender metadata, unrelated addresses or other fields. Existing source
shape, native pronunciation script and no-stress-capital rules remain binding.
These are model-authored draft repairs, never native-speaker approval.
"""
INTAKE_GENDER_DECOY_RULES = """Intake name-decoy distinctness repair v1. Two independently
gender-preserved wrong-name candidates collapsed onto one another. Keep the first
array element exactly unchanged, and change only later colliding decoy given-name
tokens to plausible distinct capture errors of the authoritative completed primary.
Preserve EACH decoy's supplied source gender separately, the native script,
faithful romanization, family name, array length and Easy/Hard difficulty. All
decoys must differ from one another and the primary after actual name normalization
in both native and romanized forms. Never change a correct primary or unrelated
entries. Do not invent native approval. Return the ordinary requested map entries.
"""


class IntakeNameBinding(LocalizationModel):
    """Exact name slot with reviewed source evidence and separate semantic role."""

    id: str = Field(description="Unique audit slot ID; decoys include array position.")
    entry_id: str = Field(description="Owning localization map entry.")
    index: int | None = Field(default=None, description="Name decoy array position.")
    role: Literal["callee", "measured_name", "name_decoy"] = Field(
        description="Identity role; measured names never inherit callee gender."
    )
    source_gender: Literal["male", "female", "unknown"] = Field(
        description="Reviewed bank gender, or unknown for independently assessed decoys."
    )
    source_evidence: str = Field(
        description="Exact frozen-bank evidence or its absence."
    )
    task_ids: list[str] = Field(description="Canonical tasks using this exact role.")


class IntakeGenderRequest(LocalizationModel):
    """Explicit resumable model and artifact inputs for an intake gender repair."""

    map_file: Path = Field(description="Parent candidate map, never a review CSV.")
    data_root: Path = Field(description="Canonical data root containing frozen banks.")
    revision: str = Field(min_length=1, description="New immutable map revision.")
    model: str = Field(min_length=1, description="Explicit authoring model.")
    model_args: dict[str, JsonValue] = Field(description="Recorded model settings.")
    max_concurrency: int = Field(ge=1, le=10, description="Concurrent model calls.")
    batch_size: int = Field(
        ge=1, le=100, description="Dependency authoring batch size."
    )

    def for_report(self) -> "IntakeGenderRequest":
        """Record portable input labels; content digests identify the exact maps.

        Maps outside the data tree use an ``inputs/`` label. These labels are
        descriptive, not a promise that the original execution paths exist.
        Resumable checkpoints retain the original execution request locally.
        """
        map_path = self.map_file.resolve()
        data_path = self.data_root.resolve()
        if map_path.is_relative_to(data_path):
            map_label = Path("data") / map_path.relative_to(data_path)
        else:
            map_label = Path("inputs") / map_path.name
        return self.model_copy(
            update={"map_file": map_label, "data_root": Path("data")}
        )


class IntakeGenderReport(LocalizationModel):
    """Exact changed-name and dependency accounting; never native review approval."""

    version: Literal["intake-gender-preservation-v1"] = INTAKE_GENDER_VERSION
    request: IntakeGenderRequest = Field(
        description="Recorded model and input settings."
    )
    parent_map_sha256: str = Field(description="Exact parent map content digest.")
    candidate_map_sha256: str = Field(description="Exact resulting map content digest.")
    bindings: list[IntakeNameBinding] = Field(
        description="Every used independent name slot."
    )
    names: list[GenderNameRequest] = Field(
        description="Exact source assessments and candidate forms."
    )
    results: list[GenderNameResult] = Field(
        description="Typed per-name results, including unresolved cases."
    )
    changed_entry_ids: list[str] = Field(
        description="Entries with changed localized forms."
    )
    dependency_entry_ids: list[str] = Field(
        description="Dependent entries selected for reauthoring."
    )
    native_review_required: Literal[True] = True
    review_exports_updated: Literal[False] = False
    prior_audits_stale: Literal[True] = True


def intake_name_bindings(
    proposal: LocalizationMap, data_root: Path
) -> list[IntakeNameBinding]:
    """Join used identities to pinned bank labels without guessing gender from names."""
    if proposal.source.domain != "intake":
        raise ValueError("Intake gender preservation requires the intake domain")
    directory = data_root / "tau2/domains/intake"
    banks = load_banks(directory / "banks")
    manifest = FreezeManifest.model_validate_json(
        (directory / "tasks.manifest.json").read_text()
    )
    if banks.shas["person_names"] != manifest.bank_shas["person_names"]:
        raise ValueError("Frozen person-name bank changed")
    names = {}
    for name in banks.person_names:
        for value in [name.value, name.alt_name]:
            if value is not None:
                if value in names and names[value].gender != name.gender:
                    raise ValueError("Conflicting reviewed source name gender")
                names[value] = name
    tasks = next(d.content for d in proposal.source.documents if d.id == "tasks")
    selected = IntakeLocalizationData.model_validate(
        next(
            d.content
            for d in proposal.source.documents
            if d.id == "intake_localization"
        )
    )
    result = []
    for entry in proposal.entries:
        rows = {
            point.path[1]
            for point in entry.occurrences
            if point.document_id == "intake_localization"
            and len(point.path) == 3
            and point.path[0] == "stimuli"
            and selected.stimuli[point.path[1]].family == "person_names"
            and point.path[2] in {"written", "decoys"}
        }
        if entry.category == "identity_name":
            role = "callee"
            task_ids = sorted(
                {
                    tasks[p.path[0]]["id"]
                    for p in entry.occurrences
                    if p.document_id == "tasks"
                }
            )
        elif rows:
            role = "name_decoy" if isinstance(entry.source, list) else "measured_name"
            task_ids = sorted(
                selected.stimuli[index].canonical_task_id for index in rows
            )
        else:
            continue
        values = (
            enumerate(entry.source) if role == "name_decoy" else [(None, entry.source)]
        )
        for index, value in values:
            bank = names.get(value)
            if role != "name_decoy" and bank is None:
                raise ValueError(f"Used identity has no reviewed bank gender: {value}")
            evidence = (
                f"Frozen person_names bank: {bank.value!r}, gender={bank.gender.value}; sha256={banks.shas['person_names']}."
                if bank is not None
                else "This selected source decoy has no explicit reviewed gender label. Assess its own naming convention independently; never copy the primary's gender. Mark ambiguous or unisex names explicitly."
            )
            result.append(
                IntakeNameBinding(
                    id=entry.id if index is None else f"{entry.id}:decoy:{index}",
                    entry_id=entry.id,
                    index=index,
                    role=role,
                    source_gender=bank.gender.value if bank is not None else "unknown",
                    source_evidence=evidence,
                    task_ids=task_ids,
                )
            )
    return result


def intake_gender_dependencies(
    proposal: LocalizationMap, changed_ids: set[str]
) -> list[str]:
    """Select only changed measured-name dependencies and owner-linked email bundles."""
    tasks = next(d.content for d in proposal.source.documents if d.id == "tasks")
    selected = IntakeLocalizationData.model_validate(
        next(
            d.content
            for d in proposal.source.documents
            if d.id == "intake_localization"
        )
    )
    name_rows = set()
    email_task_ids = set()
    for entry in proposal.entries:
        if entry.id not in changed_ids:
            continue
        for point in entry.occurrences:
            if (
                point.document_id == "intake_localization"
                and len(point.path) == 3
                and point.path[0] == "stimuli"
                and point.path[2] == "written"
            ):
                if selected.stimuli[point.path[1]].family == "person_names":
                    name_rows.add(point.path[1])
                elif selected.stimuli[point.path[1]].family == "emails":
                    email_task_ids.add(
                        selected.stimuli[point.path[1]].canonical_task_id
                    )
            if entry.category == "identity_name" and point.document_id == "tasks":
                task_id = tasks[point.path[0]]["id"]
                if task_id.startswith("intake_emails_"):
                    email_task_ids.add(task_id)
    result = []
    for entry in proposal.entries:
        if entry.category == "protocol":
            continue
        for point in entry.occurrences:
            if (
                point.document_id != "intake_localization"
                or len(point.path) != 3
                or point.path[0] != "stimuli"
            ):
                continue
            row, field = point.path[1:]
            if field in {
                "pronunciations",
                "spelling_styles",
                "omission_instruction",
            } and (
                row in name_rows
                or selected.stimuli[row].canonical_task_id in email_task_ids
            ):
                result.append(entry.id)
                break
    return sorted(set(result) - changed_ids)


def synchronize_intake_owner_emails(
    proposal: LocalizationMap, data_root: Path
) -> set[str]:
    """Instantiate frozen owner patterns from finalized names, retaining local domains."""
    from tau2.data_model.language import NameOrder
    from tau2.domains.intake.tasks.banks import instantiate_email
    from tau2.multilingual.localization import LocalizedForms

    directory = data_root / "tau2/domains/intake"
    manifest = FreezeManifest.model_validate_json(
        (directory / "tasks.manifest.json").read_text()
    )
    banks = load_banks(directory / "banks")
    if banks.shas["emails"] != manifest.bank_shas["emails"]:
        raise ValueError("Frozen email bank changed")
    draws = {draw.task_id: draw for draw in manifest.tasks if draw.bank == "emails"}
    tasks = next(d.content for d in proposal.source.documents if d.id == "tasks")
    selected = IntakeLocalizationData.model_validate(
        next(
            d.content
            for d in proposal.source.documents
            if d.id == "intake_localization"
        )
    )
    owners = {}
    slots = {}
    for entry in proposal.entries:
        for point in entry.occurrences:
            if entry.category == "identity_name" and point.document_id == "tasks":
                owners[tasks[point.path[0]]["id"]] = entry
            if (
                point.document_id == "intake_localization"
                and len(point.path) == 3
                and point.path[0] == "stimuli"
            ):
                slots[
                    (selected.stimuli[point.path[1]].canonical_task_id, point.path[2])
                ] = entry
    changed = set()
    for task_id, draw in draws.items():
        owner = owners[task_id]
        parts = owner.forms.romanized.split()
        # The reviewed callee donor contract is exactly one given/family token.
        # Refuse ambiguous new structures rather than silently guessing boundaries.
        if len(parts) != 2:
            raise ValueError(
                f"Owner email requires reviewed name components: {task_id}"
            )
        if proposal.settings.name_order == NameOrder.FAMILY_FIRST:
            parts.reverse()
        written = slots[(task_id, "written")]
        decoys = slots[(task_id, "decoys")]
        if len(decoys.forms.native) != 1 or len(decoys.forms.romanized) != 1:
            raise ValueError("Frozen owner-linked email requires one decoy")
        bank = next(
            e for e in banks.emails if f"{e.pattern}@{e.domain}" == draw.email_pattern
        )
        localized = bank.model_copy(
            update={
                "domain": written.forms.native.rsplit("@", 1)[1],
                "decoy_domain": decoys.forms.native[0].rsplit("@", 1)[1],
            }
        )
        resolved = instantiate_email(localized, " ".join(parts))
        for entry, value in ((written, resolved.value), (decoys, resolved.decoys)):
            if entry.forms.native == value and entry.forms.romanized == value:
                continue
            entry.forms = LocalizedForms(
                native=value,
                romanized=value,
                english=f"Owner-linked email {'decoy' if isinstance(value, list) else 'answer'} instantiated from frozen source pattern and finalized callee {owner.forms.romanized}; existing localized domain retained.",
            )
            entry.notes += " Owner name synchronized using the frozen email pattern; punctuation, digits, tier and localized domain retained."
            changed.add(entry.id)
    return changed


def _colliding_name_decoys(proposal: LocalizationMap) -> list[str]:
    from tau2.domains.intake.folds import FoldKind, fold_value

    result = []
    for entry in proposal.entries:
        if (
            "intake stimulus:intake_person_names_" not in entry.context
            or "dependency=decoys;" not in entry.context
        ):
            continue
        for representation in ("native", "romanized"):
            values = [
                fold_value(FoldKind.NATIVE_NAME, value)
                for value in getattr(entry.forms, representation)
            ]
            if len(values) != len(set(values)):
                result.append(entry.id)
                break
    return result


def _repair_name_decoy_collisions(
    candidate: LocalizationMap,
    request: IntakeGenderRequest,
    checkpoints: Path,
    names: list[GenderNameRequest],
    results: list[GenderNameResult],
    changed_ids: set[str],
) -> LocalizationMap:
    from tau2.multilingual.factory.database_localization import (
        LocalizationReference,
        RegionalDraftRequest,
        RegionalRevision,
        revise_database_localization,
    )
    from tau2.multilingual.factory.localization_gender import audit_gender_names
    from tau2.multilingual.localization import LocalizedForms

    collisions = _colliding_name_decoys(candidate)
    if not collisions:
        return candidate
    changed_measured = {
        entry.id
        for entry in candidate.entries
        if entry.id in changed_ids
        and entry.context.startswith("intake field:second_guest_full_name;")
    }
    # The ordinary compiler validates every task after each revision. Repair
    # changed-primary speech fields in this same stage, before that validation.
    dependencies = intake_gender_dependencies(candidate, changed_measured)
    selected = [name for name in names if name.id.rsplit(":decoy:", 1)[0] in collisions]
    current = {entry.id: entry for entry in candidate.entries}
    selected = [
        name.model_copy(
            update={
                "forms": LocalizedForms(
                    native=current[name.id.rsplit(":decoy:", 1)[0]].forms.native[
                        int(name.id.rsplit(":decoy:", 1)[1])
                    ],
                    romanized=current[name.id.rsplit(":decoy:", 1)[0]].forms.romanized[
                        int(name.id.rsplit(":decoy:", 1)[1])
                    ],
                    english="Selected wrong-name decoy; source-bound independent identity.",
                )
            }
        )
        for name in selected
    ]
    candidate = revise_database_localization(
        RegionalRevision(
            previous=candidate,
            request=RegionalDraftRequest(
                source=candidate.source,
                settings=candidate.settings,
                revision=request.revision + "-distinct-decoys",
                generation=regional_generation(request.model, request.model_args),
                batch_size=request.batch_size,
                max_concurrency=request.max_concurrency,
            ),
            redraft_ids=sorted(set(collisions + dependencies)),
            feedback=[
                LocalizationReference(
                    name="intake-gender-decoy-distinctness-v1",
                    content=INTAKE_GENDER_DECOY_RULES
                    + "\n"
                    + "\n".join(item.model_dump_json() for item in selected),
                )
            ],
        ),
        checkpoints / "distinct-decoys-complete.json",
        checkpoints / "distinct-decoys-complete",
    )
    indexed = {entry.id: entry for entry in candidate.entries}
    refreshed = []
    for original in selected:
        entry_id, index = original.id.rsplit(":decoy:", 1)
        entry = indexed[entry_id]
        refreshed.append(
            original.model_copy(
                update={
                    "forms": LocalizedForms(
                        native=entry.forms.native[int(index)],
                        romanized=entry.forms.romanized[int(index)],
                        english="Selected wrong-name decoy; source-bound independent identity.",
                    ),
                    "context": original.context
                    + [
                        "Other finalized sibling decoys that this independent slot MUST differ from in both native and romanized forms: "
                        + str(
                            [
                                (native, romanized)
                                for sibling, (native, romanized) in enumerate(
                                    zip(
                                        entry.forms.native,
                                        entry.forms.romanized,
                                        strict=True,
                                    )
                                )
                                if sibling != int(index)
                            ]
                        )
                        + ". If a gender repair is needed, choose a compatible distinct name, never reuse a listed sibling."
                    ],
                }
            )
        )
    audited = audit_gender_names(
        refreshed,
        candidate.settings.language,
        candidate.settings.country_code,
        request.model,
        request.model_args,
        checkpoints / "distinct-decoy-genders",
        workers=request.max_concurrency,
    )
    positions = {name.id: index for index, name in enumerate(names)}
    for name, result in zip(refreshed, audited, strict=True):
        entry_id, slot = name.id.rsplit(":decoy:", 1)
        indexed[entry_id].forms.native[int(slot)] = result.forms.native
        indexed[entry_id].forms.romanized[int(slot)] = result.forms.romanized
        names[positions[name.id]] = name
        results[positions[name.id]] = result
    if _colliding_name_decoys(candidate):
        raise ValueError(
            "Gender-preserving decoy repair still collapses sibling identities"
        )
    return candidate


def _name_requests(
    proposal: LocalizationMap, bindings: list[IntakeNameBinding]
) -> list[GenderNameRequest]:
    from tau2.multilingual.localization import LocalizedForms

    indexed = {entry.id: entry for entry in proposal.entries}
    requests = []
    for binding in bindings:
        entry = indexed[binding.entry_id]
        forms = (
            entry.forms
            if binding.index is None
            else LocalizedForms(
                native=entry.forms.native[binding.index],
                romanized=entry.forms.romanized[binding.index],
                english="Selected wrong-name decoy; source-bound independent identity.",
            )
        )
        requests.append(
            GenderNameRequest(
                id=binding.id,
                source=entry.source
                if binding.index is None
                else entry.source[binding.index],
                forms=forms,
                context=[
                    entry.context,
                    f"Role={binding.role}; tasks={binding.task_ids}.",
                ],
                source_gender=binding.source_gender,
                source_basis="reviewed_metadata"
                if binding.source_gender != "unknown"
                else "unknown",
                source_evidence=binding.source_evidence,
            )
        )
    return requests


def revise_intake_gender(
    request: IntakeGenderRequest, output: Path, checkpoints: Path
) -> LocalizationMap:
    """Repair all used names, propagate dependencies and validate both task views."""
    from tau2.multilingual.factory.database_localization import (
        LocalizationReference,
        RegionalDraftRequest,
        RegionalRevision,
        revise_database_localization,
    )
    from tau2.multilingual.factory.localization_artifact import (
        read_localization_map,
        write_localization_map,
    )
    from tau2.multilingual.factory.localization_compile import validate_localization
    from tau2.multilingual.factory.localization_gender import (
        assess_source_names,
        audit_gender_names,
        evidence_for,
    )
    from tau2.multilingual.localization import GenerationProvenance, LocalizedForms
    from tau2.utils.artifacts import content_digest, write_artifact

    parent = read_localization_map(request.map_file, request.data_root)
    candidate = parent.model_copy(deep=True)
    bindings = intake_name_bindings(parent, request.data_root)
    decoy_ids = {binding.entry_id for binding in bindings if binding.index is not None}
    for entry in candidate.entries:
        if entry.id in decoy_ids:
            # Every selected slot is assessed again below. Invalidate previous
            # hashes before a changed decoy reaches an intermediate compiler.
            entry.name_gender_items = []
    source_requests = _name_requests(parent, bindings)
    source_results = assess_source_names(
        source_requests,
        request.model,
        request.model_args,
        checkpoints / "source",
        workers=request.max_concurrency,
    )
    source_by_id = {item.id: item for item in source_results}
    write_artifact(request, checkpoints / "request.json")
    write_artifact(
        RootModel[list[IntakeNameBinding]](bindings), checkpoints / "bindings.json"
    )
    changed = set()
    all_results = []
    all_requests = []
    for decoys in (False, True):
        active = [b for b in bindings if (b.index is not None) == decoys]
        pending = _name_requests(candidate, active)
        for item, binding in zip(pending, active, strict=True):
            source = source_by_id[item.id]
            item.source_gender = source.gender
            item.source_basis = source.basis
            item.source_evidence = source.reason
            if decoys:
                # The source assessment above never sees localized target evidence.
                primary = next(
                    e
                    for e in candidate.entries
                    if any(
                        p.document_id == "intake_localization"
                        and len(p.path) == 3
                        and p.path[2] == "written"
                        and p.path[:-1]
                        in [
                            q.path[:-1]
                            for q in next(
                                e for e in candidate.entries if e.id == binding.entry_id
                            ).occurrences
                        ]
                        for p in e.occurrences
                    )
                )
                item.context.append(
                    "Authoritative localized primary: "
                    + primary.forms.model_dump_json()
                    + f"; primary_changed={primary.id in changed}. The decoy must be a plausible wrong capture of this exact primary while preserving its own source gender, even if that gender differs from the primary. Rebuild an obsolete decoy when primary_changed=true. Both native and romanized normalized forms must differ from the primary."
                )
        results = audit_gender_names(
            pending,
            parent.settings.language,
            parent.settings.country_code,
            request.model,
            request.model_args,
            checkpoints / ("decoys" if decoys else "names"),
            workers=request.max_concurrency,
        )
        indexed = {e.id: e for e in candidate.entries}
        for binding, item, result in zip(active, pending, results, strict=True):
            entry = indexed[binding.entry_id]
            if binding.index is None:
                if entry.forms.model_dump() != result.forms.model_dump():
                    changed.add(entry.id)
                    entry.reuse = None
                entry.forms = LocalizedForms.model_validate(result.forms.model_dump())
                entry.name_gender = evidence_for(item, result, request.model)
            elif result.forms.model_dump() != item.forms.model_dump():
                entry.forms.native[binding.index] = result.forms.native
                entry.forms.romanized[binding.index] = result.forms.romanized
                changed.add(entry.id)
            if result.decision == "revise":
                entry.notes += " Gender-preserving repair: " + result.reason
        all_requests.extend(pending)
        all_results.extend(results)
    candidate = _repair_name_decoy_collisions(
        candidate, request, checkpoints, all_requests, all_results, changed
    )
    from tau2.multilingual.localization import NameGenderItemEvidence

    entry_index = {entry.id: entry for entry in candidate.entries}
    for name, result in zip(all_requests, all_results, strict=True):
        if ":decoy:" not in name.id:
            continue
        entry_id, slot = name.id.rsplit(":decoy:", 1)
        entry = entry_index[entry_id]
        evidence = evidence_for(name, result, request.model)
        evidence.forms_sha256 = content_digest(
            LocalizedForms(
                native=entry.forms.native[int(slot)],
                romanized=entry.forms.romanized[int(slot)],
                english=entry.forms.english,
            )
        )
        entry.name_gender_items.append(
            NameGenderItemEvidence(index=int(slot), evidence=evidence)
        )
    changed.update(synchronize_intake_owner_emails(candidate, request.data_root))
    dependencies = intake_gender_dependencies(candidate, changed)
    write_artifact(
        RootModel[list[GenderNameRequest]](all_requests),
        checkpoints / "name-requests.json",
    )
    write_artifact(
        RootModel[list[GenderNameResult]](all_results),
        checkpoints / "name-results.json",
    )
    if dependencies:
        owners = [
            entry for entry in candidate.entries if entry.category == "identity_name"
        ]
        feedback = LocalizationReference(
            name=INTAKE_GENDER_VERSION,
            content=INTAKE_GENDER_DEPENDENCY_RULES
            + "\nAuthoritative callee identities (emails already synchronized by the frozen owner patterns):\n"
            + "\n".join(e.model_dump_json(include={"source", "forms"}) for e in owners),
        )
        candidate = revise_database_localization(
            RegionalRevision(
                previous=candidate,
                request=RegionalDraftRequest(
                    source=candidate.source,
                    settings=candidate.settings,
                    revision=request.revision,
                    generation=regional_generation(request.model, request.model_args),
                    batch_size=request.batch_size,
                    max_concurrency=request.max_concurrency,
                ),
                redraft_ids=dependencies,
                feedback=[feedback],
            ),
            checkpoints / "dependencies.json",
            checkpoints / "dependencies",
        )
    else:
        candidate.revision = request.revision
    candidate.provenance = GenerationProvenance(
        model=request.model,
        model_args=request.model_args,
        prompt_version=INTAKE_GENDER_VERSION,
        call_name="factory_localized_name_gender",
        input_sha256=content_digest(request.for_report()),
        parent_map_sha256=content_digest(parent),
    )
    validate_localization(candidate)
    validate_intake_gender_coverage(candidate, request.data_root)
    write_localization_map(candidate, output)
    write_artifact(
        IntakeGenderReport(
            request=request.for_report(),
            parent_map_sha256=content_digest(parent),
            candidate_map_sha256=content_digest(candidate),
            bindings=bindings,
            names=all_requests,
            results=all_results,
            changed_entry_ids=[
                after.id
                for before, after in zip(parent.entries, candidate.entries, strict=True)
                if before.forms != after.forms
            ],
            dependency_entry_ids=dependencies,
        ),
        output.with_suffix(".gender-report.json"),
    )
    return candidate


def validate_intake_gender_coverage(proposal: LocalizationMap, data_root: Path) -> None:
    """Refuse publication with missing, stale or conflicting source-name evidence."""
    indexed = {entry.id: entry for entry in proposal.entries}
    for binding in intake_name_bindings(proposal, data_root):
        entry = indexed[binding.entry_id]
        if (
            proposal.settings.language == "en"
            and entry.forms.native == entry.source
            and entry.forms.romanized == entry.source
        ):
            # Exact source controls preserve identity without a model assessment.
            continue
        if binding.index is None:
            evidence = entry.name_gender
            if evidence is not None:
                evidence.require(entry.forms)
        else:
            item = next(
                (
                    item
                    for item in entry.name_gender_items
                    if item.index == binding.index
                ),
                None,
            )
            evidence = item.evidence if item is not None else None
            if item is not None:
                item.require(entry.forms)
        if evidence is None:
            raise ValueError(
                f"Intake name requires source-gender evidence before publication: {binding.id}"
            )
        if binding.source_gender != "unknown" and (
            evidence.source_gender != binding.source_gender
            or evidence.source_basis != "reviewed_metadata"
        ):
            raise ValueError(
                f"Intake source gender differs from frozen bank: {binding.id}"
            )
    if _colliding_name_decoys(proposal):
        raise ValueError(
            "Intake name decoys collapse independent wrong-name identities"
        )


def _run(args):
    request = IntakeGenderRequest.model_validate_json(args.request.read_text())
    revise_intake_gender(request, args.output, args.checkpoints)


def add_intake_gender_args(commands: argparse._SubParsersAction) -> None:
    """Register the complete resumable gender-preserving intake repair stage."""
    parser = commands.add_parser(
        "revise-intake-gender",
        help="Preserve source name genders and dependent identity fields without changing review CSVs.",
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.set_defaults(func=_run)
