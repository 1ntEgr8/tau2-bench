"""Source-bound locality audit and dependency-aware revision requests for intake."""

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import Field, JsonValue

from tau2.config import (
    DEFAULT_FACTORY_LOCALIZATION_BATCH_SIZE,
    DEFAULT_FACTORY_LOCALIZATION_CONCURRENCY,
)
from tau2.data_model.message import SystemMessage, UserMessage
from tau2.multilingual.factory.generation_config import regional_generation
from tau2.multilingual.factory.localization_artifact import read_localization_map
from tau2.multilingual.factory.localization_cli import DatabaseRevisionCommand
from tau2.multilingual.localization import (
    LocalizationMap,
    LocalizationModel,
    RegionalSettings,
    json_content_digest,
)
from tau2.utils.artifacts import content_digest, write_artifact
from tau2.utils.llm_utils import extract_json_from_llm_response, generate

AUDIT_VERSION = "intake-local-entities-audit-v4"
REVISION_VERSION = "intake-local-entities-revision-v5"
LOCALE_RULES_VERSION = "intake-locality-constraints-v3"
AUDIT_PROMPT = """Audit existing synthetic intake benchmark candidates for actual local plausibility in the supplied language/country. This is a model audit, never native-speaker approval. Return one decision for every requested ID, exactly once. Keep good existing local names and entities. Revise transplanted foreign proper names, phonetic source-name copies, literal business/property-name translations and foreign address/phone conventions. Localizing an invented source hotel or business means choosing a plausible local proper name with the same business role, not translating the source's words. Familiar generic technical borrowings can remain when normal locally; explain why. Review linked decoys/pronunciation examples for the same defects and for consistency with the written entity; set revise_dependencies even when the primary can stay. Preserve source task roles and fixed easy/hard measurement assignments. Hard entities must remain naturally difficult in the target language through plausible spelling/phonetic/address complexity, not nonsense, unnatural source calques or arbitrary foreign syllables. Medicines retain ingredient, dosage, strength, units and release form; legitimate standard pharmaceutical borrowings remain, with explicit rationale. Protocol dates/times, VINs and member-ID checksums are not an excuse to change clinical/scenario semantics. Invented source hotel/company/brand proper-name identities must be replaced, even if their English words could plausibly appear locally. Mere English plausibility or a new script is not localization. A retained international brand requires evidence of established local presence; these source fixture brands are invented. Generic industry borrowings can remain, but their proper-name identity must be genuinely localized. Synthetic foreign email provider identities and foreign country-code domains also require local replacement, while already plausible local-part names can remain. License plates are localizable capture entities, not immutable checksummed VIN/member IDs. Output only the requested JSON decisions. Do not generate replacement suggestions in this audit."""
REVISION_PROMPT = """Revise the specifically audited intake entities into plausible local entities. This is candidate authoring, not approval. Reject transliterated foreign proper names and literal hotel/clinic/company name calques. Create a plausible local entity of the same role rather than translating each source word. Preserve exact canonical tasks, all easy/hard assignments and source capture difficulty, but realize that difficulty naturally in the target language. For hard names/properties/brands use credible compound names, spelling/phonetic confusability, abbreviations or script-specific conjuncts; do not mechanically transplant foreign syllables or add unnatural hyphens. Use coherent local city/neighborhood/street/postcode addresses, with hard addresses retaining meaningful multi-component and script-specific capture challenges. Generic local technical borrowings are legitimate when conventional. Medicines retain identical active ingredient, strength, unit, release form and clinical identity; never replace a drug for local flavor. ASCII VIN/member identifiers and required ISO dates/AM-PM times remain exact. Local plates follow plausible country-specific formats including Hangul for Korean and Hanzi for Chinese native forms, with corresponding romanized identity. Phone values remain ASCII and must follow actual country number shape; do not merely swap a prefix onto a US seven-digit subscriber. Preserve legitimate extension/country-code complexity where the measured source tier requires it. Email values remain ASCII and use plausible local aliases and domains, preserving easy/hard punctuation/digit/spelling hazards. All dependent decoys, pronunciation tokens, distortions, spelling styles and omission instructions must be coherent with the exact completed written mapping; discard foreign proper-name remnants. Every decoy is wrong after normalization in BOTH native and romanized forms. Pronunciation tokens occur exactly in the written answer and distortions are audible non-noops in both representations. Internal simulator instructions remain English, with localized spoken examples. Keep all source list lengths, object keys and explicit protocol syntax. Unselected valid entries remain unchanged. Invented source proper-name identities must be replaced, even if their English wording might be plausible locally. English plausibility alone is NOT evidence for retaining a source hotel/company/brand; preserving an international brand requires evidence of established local presence. These fixture names are invented, not established international brands. Retain natural industry terms (hotel, residency, clinic, robotics, finance), while replacing the proper-name identity with a plausible local one. Do not create artificial dictionary translations instead of natural local market names. Email revisions must retain already plausible local-part names and punctuation hazards, but replace invented foreign provider identities/country domains with plausible local synthetic domains. All output requires native and speech review."""
# Written capture neighbors and deliberately mispronounced speech have different contracts.
_DECOY_SPEECH_DISTINCTION = " Written decoys may be homophones or spelling-confusable variants: keep them when their actual written capture normalization differs from the correct answer in BOTH native and romanized forms. Homophony alone is NOT a decoy defect. Only intentional pronunciation token/distorted pairs require an audible sound change. Do not erase useful spelling ambiguity from the hard benchmark."
AUDIT_PROMPT += _DECOY_SPEECH_DISTINCTION
REVISION_PROMPT += _DECOY_SPEECH_DISTINCTION
INDIA_RULES = """For India/Hindi: use plausible Indian personal names, local business/property names and Indian neighborhood/address conventions, written naturally in Devanagari with Latin romanization of the same identity. Do not retain Elmsworth, foreign Vennel/Wynd/Snicket place morphology, transplanted Scandinavian/Celtic/French surnames or literal wolf/moon/tree hotel calques. Standard words such as hotel, clinic, auto service, capsule and technical brand suffixes may remain when natural; localize the actual proper-name component. Invented local brand names must sound plausible in India, not like the source rewritten in Devanagari. Hard measurement needs real Hindi difficulties such as consonant conjuncts, retroflex/dental or vowel distinctions and credible longer multi-part names/addresses; easy names should remain common and straightforward. Use common Indian mobile presentation (10 ASCII digits starting 6–9, optionally +91), or coherent Indian landline STD plus 8-digit subscriber for major cities. A city code followed by a US 555-xxxx seven-digit subscriber is invalid. Extensions belong on landlines, not bare mobile numbers. Keep source number/extension complexity where measured; do not preserve source US fictional-number patterns at the expense of Indian structure. Keep all medicine identities and strengths; legitimate pharmaceutical transliterations are expected, unlike local hotel/business names. Synthetic values are fixtures, not a claim that a particular number or establishment exists."""
_ENTITY_FIELDS = {
    "second_guest_full_name",
    "contact_phone",
    "home_address",
    "destination_property",
    "contact_email",
    "current_medication",
    "company_name",
    "employer_name",
    "aftermarket_brand",
    "license_plate",
}


class CandidateReviewRequest(LocalizationModel):
    """Explicit map, revision and model settings for a resumable locale audit."""

    map_file: Path
    revision: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_args: dict[str, JsonValue]
    prior_audit_file: Path | None = None
    entry_ids: list[str] = Field(default_factory=list)
    required_revision_ids: list[str] = Field(default_factory=list)
    required_revision_reason: str | None = None
    batch_size: int = Field(
        default=DEFAULT_FACTORY_LOCALIZATION_BATCH_SIZE, ge=1, le=100
    )
    max_concurrency: int = Field(
        default=DEFAULT_FACTORY_LOCALIZATION_CONCURRENCY, ge=1, le=10
    )


class AuditDecision(LocalizationModel):
    """A proposal to retain or regenerate a source-bound entity and its dependencies."""

    id: str
    decision: Literal["keep", "revise"]
    revise_dependencies: bool
    rationale: str = Field(min_length=1)


class SourceReviewOverride(LocalizationModel):
    """Explicit source-review correction to a model audit, never native approval."""

    entry_id: str
    previous_decision: Literal["keep", "revise"]
    reason: str = Field(min_length=1)


class AuditBatch(LocalizationModel):
    """Strict audit reply; coverage is checked against the exact requested IDs."""

    decisions: list[AuditDecision]


class AuditCheckpoint(LocalizationModel):
    """Immutable model audit result tied to the exact outgoing source payload."""

    input_sha256: str
    model: str
    model_args: dict[str, JsonValue]
    prompt_version: str = AUDIT_VERSION
    response: AuditBatch


class CandidateRevisionFeedback(LocalizationModel):
    """Versioned source-bound constraints supplied to the ordinary revision stage."""

    version: str = REVISION_VERSION
    locale_rules_version: str = LOCALE_RULES_VERSION
    instructions: str
    findings: list[AuditDecision]


class LocalReuseRole(LocalizationModel):
    """Local-only task-role evidence for an identity whose donor forms stay private."""

    entry_id: str
    tasks: list[str]
    verticals: list[str]
    role: str = "Record owner receiving an outbound callback; preserve the reviewed source-bank gender independently from any measured second guest. No source-imposed kinship or profession. Prior identity review is not approval of this intake use."


class CandidateAudit(LocalizationModel):
    """Model findings and local source evidence, explicitly not human approval."""

    parent_sha256: str
    source_sha256: str
    settings: RegionalSettings
    model: str
    model_args: dict[str, JsonValue]
    prompt_version: str = AUDIT_VERSION
    locale_rules_version: str = LOCALE_RULES_VERSION
    decisions: list[AuditDecision]
    audited_ids: list[str] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )
    local_reuse_roles: list[LocalReuseRole]
    redraft_ids: list[str]
    prior_audit_sha256: str | None = None
    source_review_overrides: list[SourceReviewOverride] = Field(default_factory=list)
    revision_rules_version: str = REVISION_VERSION
    status: Literal["pending_native_review"] = "pending_native_review"


COUNTRY_RULES = {
    "IN": INDIA_RULES,
    "ES": "Spain/Spanish: use natural Spanish names and accents, locally plausible property/business identities, streets and residential floor/door conventions, coherent municipality/province/five-digit postal code. Do not calque invented English hotels/brands or transplant foreign surname clusters solely for difficulty. Hard capture can use credible multi-part surnames, accent and consonant distinctions, similar local proper names and multi-component addresses. Phones have nine national ASCII digits; preserve source international/extension complexity only where measured, using +34 and a plausible fixed line for extensions. Never US555 subscriber patterns. Native and romanized forms use ordinary Spanish orthography, including accents, not stress or syllable guides. Intentional mispronunciations must be AUDIBLE in Spain: b/v, silent h, g/j before e/i, ll/y under yeismo and i/y glide-only spelling changes are not audible errors. Keep written decoys and intentional speech errors distinct in purpose.",
    "BR": "Brazil/Portuguese: use natural Brazilian names, including plausible multicultural surnames, and locally plausible property/business identities, not translations or phonetic copies of invented foreign brands. Coherent street/number/unit, municipality/UF and eight-digit CEP (12345-678). Phones use two-digit DDD plus nine-digit mobile starting9 or eight-digit landline starting2-5; optional +55 for measured complexity, landlines for extensions. Hard capture can use natural compound names, nasal vowels, lh/nh, accented spelling and multi-component addresses. Preserve native accents and plain written reference tokens; no stress guides. Intentional mispronunciations must be AUDIBLE in Brazilian Portuguese: silent h, g/j before e/i, intervocalic s/z spellings and unstressed final o/u or e/i spellings often pronounce identically and must not be used as the sole distortion. Evaluate the sound, not merely different letters.",
    "KR": "Korea/Korean: use natural Hangul names and local business/property identities, not Hangul copies of invented Western names or literal business calques. Family name first. Coherent road-name/building/unit, municipality/province and five-digit postcode. Phones use domestic010 plus eight digits for mobiles, or credible city area code/subscriber landlines; extension belongs to landline. Optional +82 must drop domestic trunk0. Hard capture uses credible Korean names, batchim, consonant/vowel distinctions and multi-component addresses; no contrived foreign syllables. All native pronunciation tokens/references/distortions use Hangul, never Latin romanization. Romanized metadata must refer to same identity and preserve audible error distinction.",
    "CN": "Mainland China/Mandarin: use Simplified Chinese personal names, local properties/businesses and coherent province/city/district/street/building/unit and six-digit postcode. Family name first; no phonetic Hanzi copies of arbitrary foreign invented names or literal property calques. Mobiles have11 digits starting1; use credible landline city area code/subscriber for extensions; optional +86 where source complexity requires. Hard capture uses natural multi-character names, confusable initials/finals/tones and meaningful address complexity, not foreign syllables. Native pronunciation tokens/references/distortions use Hanzi, not pinyin; romanized metadata identifies the same entity. Wrong decoys and distortions must remain distinct under BOTH native and romanized normalization; do not use a homophone that collapses in pinyin. All reference fields equal written token without syllable guides.",
}


def locale_rules(settings: RegionalSettings) -> str:
    """Return versioned country constraints from actual recorded map settings."""
    return COUNTRY_RULES.get(
        settings.country_code,
        "Apply the supplied target country's ordinary naming, address, phone and entity conventions; preserve task protocols and measured difficulty.",
    )


def _indices(entry) -> set[int]:
    return {
        o.path[1]
        for o in entry.occurrences
        if o.document_id == "intake_localization"
        and len(o.path) == 3
        and o.path[0] == "stimuli"
        and o.path[-1] == "written"
    }


def audit_rows(proposal: LocalizationMap) -> list[dict]:
    """Select all used non-protocol entities, with measured context and dependent entities."""
    if proposal.source.domain != "intake":
        raise ValueError("Candidate review requires an intake map")
    rows = []
    for entry in proposal.entries:
        role = entry.context.split(";", 1)[0]
        if entry.reuse is not None or not (
            role == "intake organization"
            or role.removeprefix("intake field:") in _ENTITY_FIELDS
        ):
            continue
        indices = _indices(entry)
        dependencies = [
            {"id": e.id, "source": e.source, "forms": e.forms.model_dump(mode="json")}
            for e in proposal.entries
            if any(
                o.document_id == "intake_localization"
                and len(o.path) == 3
                and o.path[1] in indices
                and o.path[-1] in {"decoys", "pronunciations"}
                for o in e.occurrences
            )
            and e.source
        ]
        rows.append(
            {
                "id": entry.id,
                "role": role,
                "source": entry.source,
                "forms": entry.forms.model_dump(mode="json"),
                "measurement": entry.context.split("; measured_selection=", 1)[-1]
                if "; measured_selection=" in entry.context
                else "Incidental context, not an independently measured slot",
                "dependencies": dependencies,
            }
        )
    return rows


def business_identity_collisions(proposal: LocalizationMap) -> dict[str, list[str]]:
    """Find distinct source businesses collapsed to one full localized identity."""
    roles = {
        "intake organization",
        "intake field:destination_property",
        "intake field:company_name",
        "intake field:employer_name",
        "intake field:aftermarket_brand",
    }
    entries = [e for e in proposal.entries if e.context.split(";", 1)[0] in roles]
    collisions: dict[str, list[str]] = defaultdict(list)
    for form in ("native", "romanized"):
        groups = defaultdict(list)
        for entry in entries:
            value = getattr(entry.forms, form)
            if not isinstance(value, str):
                raise ValueError("Business identity must be a string")
            key = " ".join(unicodedata.normalize("NFC", value).casefold().split())
            groups[key].append(entry)
        for group in groups.values():
            if len({e.source for e in group}) > 1:
                for entry in group:
                    collisions[entry.id].append(form)
    return dict(collisions)


def local_reuse_roles(proposal: LocalizationMap) -> list[LocalReuseRole]:
    """Check source roles locally without sending reviewed donor names to a model."""
    tasks = next(d.content for d in proposal.source.documents if d.id == "tasks")
    result = []
    for entry in proposal.entries:
        if entry.reuse is None:
            continue
        indices = sorted(
            {
                o.path[0]
                for o in entry.occurrences
                if o.document_id == "tasks" and o.path and isinstance(o.path[0], int)
            }
        )
        records = [
            next(
                a["arguments"]["record"]
                for a in tasks[i]["initial_state"]["initialization_actions"]
                if a["func_name"] == "seed_record"
            )
            for i in indices
        ]
        if entry.context.split(";", 1)[0] != "intake callee" or any(
            r["callee_full_name"] != entry.source for r in records
        ):
            raise ValueError("Reused intake identity has an unverified source role")
        result.append(
            LocalReuseRole(
                entry_id=entry.id,
                tasks=[tasks[i]["id"] for i in indices],
                verticals=sorted({r["vertical"] for r in records}),
            )
        )
    return result


def is_indian_phone(value: str) -> bool:
    """Check the candidate's number shape; this does not assert subscriber existence."""
    if not value.isascii():
        return False
    parts = re.split(r"\s+ext\.?\s*", value, flags=re.I)
    if len(parts) > 2 or (len(parts) == 2 and not parts[1].isdigit()):
        return False
    digits = re.sub(r"\D", "", parts[0])
    international = parts[0].lstrip().startswith(("+", "(+"))
    if international:
        if not digits.startswith("91"):
            return False
        digits = digits[2:]
    elif digits.startswith("0"):
        digits = digits[1:]
    mobile = bool(re.fullmatch(r"[6-9]\d{9}", digits))
    landline = bool(re.fullmatch(r"(?:11|22|33|44|20|40|80)\d{8}", digits))
    return landline if len(parts) == 2 else mobile or landline


def revision_ids(
    proposal: LocalizationMap, decisions: list[AuditDecision]
) -> list[str]:
    """Close only changed measurement dependencies while retaining valid primary values."""
    by_id = {e.id: e for e in proposal.entries}
    selected = {d.id for d in decisions if d.decision == "revise"}
    indices = set().union(
        *[
            _indices(by_id[d.id])
            for d in decisions
            if d.decision == "revise" or d.revise_dependencies
        ]
    )
    selected.update(
        e.id
        for e in proposal.entries
        if e.category != "protocol"
        and e.source not in ([], "")
        and any(
            o.document_id == "intake_localization"
            and len(o.path) == 3
            and o.path[1] in indices
            and o.path[-1] != "written"
            for o in e.occurrences
        )
    )
    return [e.id for e in proposal.source.entries if e.id in selected]


def audit_batch(proposal, request, rows, checkpoints: Path) -> AuditBatch:
    """Call the fixed audit prompt once per source-pinned batch and resume exactly."""
    payload = {
        "settings": proposal.settings.model_dump(mode="json"),
        "locale_rules_version": LOCALE_RULES_VERSION,
        "locale_rules": locale_rules(proposal.settings),
        "entries": rows,
    }
    digest = json_content_digest(
        {
            "payload": payload,
            "model": request.model,
            "model_args": request.model_args,
            "prompt_version": AUDIT_VERSION,
        }
    )
    path = checkpoints / f"{digest}.json"
    expected = [row["id"] for row in rows]
    if path.exists():
        artifact = AuditCheckpoint.model_validate_json(path.read_text())
        if artifact.input_sha256 != digest:
            raise ValueError("Candidate audit checkpoint differs from source input")
        result = artifact.response
    else:
        messages = [
            SystemMessage(role="system", content=AUDIT_PROMPT),
            UserMessage(role="user", content=json.dumps(payload, ensure_ascii=False)),
        ]
        for attempt in range(3):
            message = generate(
                model=request.model,
                messages=messages,
                call_name="intake_candidate_locality_audit",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "intake_locality_audit",
                        "strict": True,
                        "schema": AuditBatch.model_json_schema(),
                    },
                },
                **request.model_args,
            )
            try:
                result = AuditBatch.model_validate_json(
                    extract_json_from_llm_response(message.content or "")
                )
                if sorted(d.id for d in result.decisions) != sorted(expected):
                    raise ValueError("Audit must return each requested ID exactly once")
                break
            except ValueError as error:
                if attempt == 2:
                    raise
                messages.append(
                    UserMessage(
                        role="user",
                        content=f"Repair the audit JSON against the exact requested IDs and schema: {error}",
                    )
                )
        write_artifact(
            AuditCheckpoint(
                input_sha256=digest,
                model=request.model,
                model_args=request.model_args,
                response=result,
            ),
            path,
        )
    if sorted(d.id for d in result.decisions) != sorted(expected):
        raise ValueError("Cached audit does not cover requested IDs")
    return result


def run_candidate_audit(
    request: CandidateReviewRequest, output: Path, checkpoints: Path
) -> CandidateAudit:
    """Write audit evidence and an ordinary source-aware revision request, never approve."""
    proposal = read_localization_map(request.map_file)
    rows = audit_rows(proposal)
    if request.entry_ids:
        if len(set(request.entry_ids)) != len(request.entry_ids) or set(
            request.entry_ids
        ) - {row["id"] for row in rows}:
            raise ValueError("Targeted audit requires distinct selected entity IDs")
        rows = [row for row in rows if row["id"] in request.entry_ids]
    batches = [
        rows[i : i + request.batch_size]
        for i in range(0, len(rows), request.batch_size)
    ]
    logger.info(
        "Auditing {} selected entities and their linked decoys/pronunciations in {} batches",
        len(rows),
        len(batches),
    )
    previous = None
    if request.prior_audit_file is not None:
        previous = CandidateAudit.model_validate_json(
            request.prior_audit_file.read_text()
        )
        if previous.parent_sha256 != content_digest(
            proposal
        ) or previous.source_sha256 != content_digest(proposal.source):
            raise ValueError("Prior audit does not match the exact current map/source")
        if previous.model != request.model or previous.model_args != request.model_args:
            raise ValueError(
                "Reusing audit evidence requires unchanged recorded model settings"
            )
        decisions = [d.model_copy(deep=True) for d in previous.decisions]
    else:
        with ThreadPoolExecutor(max_workers=request.max_concurrency) as pool:
            decisions = [
                decision
                for result in pool.map(
                    lambda batch: audit_batch(proposal, request, batch, checkpoints),
                    batches,
                )
                for decision in result.decisions
            ]
    indexed = {d.id: d for d in decisions}
    if set(indexed) != {row["id"] for row in rows}:
        raise ValueError("Audit must cover the current selected entity inventory")
    if set(request.required_revision_ids) - set(indexed):
        raise ValueError("Source review refers to unknown audited entity IDs")
    overrides = []
    for identifier in request.required_revision_ids:
        if not request.required_revision_reason:
            raise ValueError(
                "Required source revisions need an explicit source-review reason"
            )
        decision = indexed[identifier]
        overrides.append(
            SourceReviewOverride(
                entry_id=identifier,
                previous_decision=decision.decision,
                reason=request.required_revision_reason,
            )
        )
        decision.decision = "revise"
        decision.revise_dependencies = True
        decision.rationale += " Source review: " + request.required_revision_reason
    entries = {e.id: e for e in proposal.entries}
    if proposal.settings.country_code == "IN":
        for decision in decisions:
            entry = entries[decision.id]
            if entry.context.startswith(
                "intake field:contact_phone;"
            ) and not is_indian_phone(entry.forms.native):
                decision.decision = "revise"
                decision.revise_dependencies = True
                decision.rationale += (
                    " Deterministic country check: incompatible Indian phone shape."
                )
    for identifier, forms in business_identity_collisions(proposal).items():
        if identifier not in indexed:
            continue
        decision = indexed[identifier]
        decision.decision = "revise"
        decision.revise_dependencies = True
        decision.rationale += (
            " Deterministic identity check: distinct source businesses collapse to the same full localized identity in "
            + ", ".join(forms)
            + "; retain distinct proper-name identities."
        )
    result = CandidateAudit(
        parent_sha256=content_digest(proposal),
        source_sha256=content_digest(proposal.source),
        settings=proposal.settings,
        model=request.model,
        model_args=request.model_args,
        decisions=decisions,
        audited_ids=[row["id"] for row in rows],
        prompt_version=previous.prompt_version if previous else AUDIT_VERSION,
        locale_rules_version=previous.locale_rules_version
        if previous
        else LOCALE_RULES_VERSION,
        prior_audit_sha256=content_digest(previous) if previous else None,
        source_review_overrides=overrides,
        local_reuse_roles=local_reuse_roles(proposal),
        redraft_ids=revision_ids(proposal, decisions),
    )
    write_artifact(result, output)
    feedback = output.with_suffix(".feedback.json")
    write_artifact(
        CandidateRevisionFeedback(
            instructions=REVISION_PROMPT + "\n\n" + locale_rules(proposal.settings),
            findings=[
                d for d in decisions if d.decision == "revise" or d.revise_dependencies
            ],
        ),
        feedback,
    )
    if result.redraft_ids:
        command = DatabaseRevisionCommand(
            previous_file=request.map_file.resolve(),
            revision=request.revision,
            generation=regional_generation(request.model, request.model_args),
            redraft_ids=result.redraft_ids,
            feedback_files=[feedback.resolve()],
            batch_size=request.batch_size,
            max_concurrency=request.max_concurrency,
        )
        write_artifact(command, output.with_suffix(".revision-request.json"))
    else:
        output.with_suffix(".revision-request.json").unlink(missing_ok=True)
    logger.info(
        "Selected {} source entries for revision; {} primary entities retained",
        len(result.redraft_ids),
        sum(d.decision == "keep" for d in decisions),
    )
    return result


def _run(args):
    request = CandidateReviewRequest.model_validate_json(args.request.read_text())
    if not request.map_file.is_absolute():
        request.map_file = (args.request.parent / request.map_file).resolve()
    if (
        request.prior_audit_file is not None
        and not request.prior_audit_file.is_absolute()
    ):
        request.prior_audit_file = (
            args.request.parent / request.prior_audit_file
        ).resolve()
    run_candidate_audit(request, args.output, args.checkpoints)


def add_intake_candidate_review_args(commands: argparse._SubParsersAction) -> None:
    """Register source-aware local-entity audits under the existing factory CLI."""
    from tau2.multilingual.factory.intake_pronunciation import (
        add_intake_pronunciation_args,
    )

    add_intake_pronunciation_args(commands)
    from tau2.multilingual.factory.intake_gender import add_intake_gender_args

    add_intake_gender_args(commands)
    parser = commands.add_parser(
        "audit-intake-candidates",
        help="Audit used intake entities against local conventions and prepare a dependency-aware revision.",
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.set_defaults(func=_run)
