"""Source-bound banking candidate review using explicit versioned locale rules."""

import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import regex
from pydantic import Field, JsonValue, create_model

from tau2.data_model.message import SystemMessage, UserMessage
from tau2.domains.banking_knowledge.localization_prose import _tasks
from tau2.multilingual.localization import (
    LocalizationMap,
    LocalizationModel,
    json_content_digest,
)
from tau2.utils.artifacts import write_artifact


class EntitySpan(LocalizationModel):
    """One exact source surface, classified without authoring target-language text."""

    source: str = Field(min_length=1, description="Verbatim source entity substring.")
    role: str = Field(
        min_length=1, description="Source-supported entity role or industry."
    )
    disposition: Literal[
        "generate", "preserve", "address_city", "address_city_region"
    ] = Field(description="Source-grounded treatment, not an approval.")
    reference_id: str | None = Field(
        description="Exact structured address entry for a derived residence; otherwise null."
    )
    reason: str = Field(
        min_length=1,
        description="English source evidence and relationship constraints.",
    )


class BankingReviewRules(LocalizationModel):
    """Recorded locale policy; the execution engine is shared across banking locales."""

    schema_version: Literal["banking-candidate-review-rules-v1"] = Field(
        description="Versioned locale review policy contract."
    )
    language: str = Field(description="Target pack language.")
    country_code: str = Field(description="Target country code.")
    country_name: str = Field(description="Target country in prompt metadata.")
    language_name: str = Field(description="Target language in prompt metadata.")
    discovery_version: str = Field(
        description="Version of the checked-in source discovery prompt."
    )
    entity_version: str = Field(
        description="Version of the checked-in entity revision prompt."
    )
    discovery_call_name: str = Field(
        description="Stable observable discovery call name."
    )
    entity_call_name: str = Field(
        description="Stable observable entity authoring call name."
    )
    family_name_position: Literal["first", "last"] = Field(
        default="last",
        description="Position of the family token in the reviewed donor names; verify the actual locale corpus.",
    )
    compact_family_name_length: int | None = Field(
        default=None,
        ge=1,
        le=4,
        description="Verified native surname character count for the selected source identities, only when family-first names have no spaces. Never inferred from general nationality.",
    )
    phone_pattern: str = Field(
        description="Validated national contact shape for the source role."
    )
    phone_role: str = Field(
        description="Source-evidenced phone role; do not silently convert explicit business landlines to mobile."
    )
    imported_street_pattern: str = Field(
        description="Explicit prior-address audit selection pattern."
    )
    name_role_evidence: dict[str, str] = Field(
        description="Source-supported role corrections; never infer role from names alone."
    )
    family_sources: dict[str, str] = Field(
        description="Explicit source-relative whose existing family form is required."
    )

    span_overrides: dict[str, list[EntitySpan]] = Field(
        default_factory=dict,
        description="Reviewed exact source-scope classification corrections, not localized values.",
    )
    foreign_atm_country_codes: list[str] = Field(
        default_factory=list,
        description="Explicit country codes distinguishing protected foreign ATM locations from domestic US state codes.",
    )
    source_aliases: dict[str, str] = Field(
        default_factory=dict,
        description="Reviewed exact source spelling/descriptor aliases; no localized content.",
    )
    protected_sources: dict[str, str] = Field(
        default_factory=dict,
        description="KB/protocol source identities with explicit preservation reasons.",
    )
    organization_role_evidence: dict[str, str] = Field(
        default_factory=dict,
        description="Source-keyed semantic role and collision findings from review; instructions only, never hand-authored replacement entities.",
    )
    organization_aliases: list[list[str]] = Field(
        default_factory=list,
        description="Explicit source aliases permitted to share a localized identity.",
    )
    organization_review_sources: list[str] = Field(
        default_factory=list,
        description="Source merchants selected for evidence/rationale correction.",
    )

    def family_token(self, value: str, representation: str) -> str:
        """Extract only the configured, source-reviewed family component."""
        words = value.split()
        if self.family_name_position == "last":
            return words[-1]
        if len(words) > 1 or representation == "romanized":
            return words[0]
        if (
            self.compact_family_name_length is None
            or len(value) <= self.compact_family_name_length
        ):
            raise ValueError(
                "Compact family-first name requires a source-verified surname length"
            )
        return value[: self.compact_family_name_length]

    @property
    def discovery_prompt(self) -> str:
        """Resolve only a checked-in versioned discovery prompt."""
        from tau2.multilingual.factory.banking_candidate_prompts import (
            BANKING_REVIEW_PROMPTS,
        )

        return BANKING_REVIEW_PROMPTS[self.discovery_version]

    @property
    def entity_prompt(self) -> str:
        """Resolve only a checked-in versioned entity prompt."""
        from tau2.multilingual.factory.banking_candidate_prompts import (
            BANKING_REVIEW_PROMPTS,
        )

        return BANKING_REVIEW_PROMPTS[self.entity_version]

    def require(self, proposal: LocalizationMap) -> None:
        """Prevent a locale policy from touching a different language or domain."""
        if (
            proposal.source.domain != "banking_knowledge"
            or proposal.settings.language != self.language
            or proposal.settings.country_code != self.country_code
        ):
            raise ValueError("Banking candidate rules differ from the requested locale")


class DiscoveryCheckpoint(LocalizationModel):
    """Immutable actual discovery call inputs and typed reply."""

    input_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Actual source request digest."
    )
    model: str = Field(min_length=1, description="Actual source discovery model.")
    model_args: dict[str, JsonValue] = Field(
        description="Actual source discovery model settings."
    )
    prompt_version: str = Field(description="Actual discovery prompt version.")
    entries: dict[str, list[EntitySpan]] = Field(
        description="Exact source entity spans by assigned prose ID."
    )
    child_inputs: list[str] = Field(
        default_factory=list,
        description="Strictly validated child requests after bounded subdivision.",
    )


def normalized_phone(value: str) -> str:
    """Compare actual digits across written scripts and presentation separators."""
    return "".join(str(unicodedata.digit(c)) for c in value if c.isdecimal())


def validate_discovered_spans(
    proposal: LocalizationMap, rows: dict[str, list[EntitySpan]]
) -> None:
    """Require exact source grounding and valid address links before authoring."""
    entries = {entry.id: entry for entry in proposal.entries}
    for identifier, spans in rows.items():
        entry = entries.get(identifier)
        if entry is None or entry.category != "prose":
            raise ValueError("Unknown prose source entry")
        seen = set()
        for span in spans:
            if span.source not in entry.source:
                raise ValueError(
                    f"Invented source surface: {identifier}: {span.source}"
                )
            if span.source.casefold() in seen:
                raise ValueError(
                    f"Duplicate source surface: {identifier}: {span.source}"
                )
            seen.add(span.source.casefold())
            if (
                re.search(
                    r"Rho[- ]Bank|Rewards Card|(?:Green|Blue|Red|Gold|Silver|Bronze|Platinum|Navy|Evergreen|Turquoise|Orange|Purple|Yellow|Light Green) Account",
                    span.source,
                    re.I,
                )
                and span.disposition != "preserve"
            ):
                raise ValueError(f"Cannot replace protected KB identity: {span.source}")
            if span.disposition.startswith("address_"):
                ref = entries.get(span.reference_id)
                if (
                    ref is None
                    or ref.category != "address"
                    or not (
                        _tasks(entry) & _tasks(ref)
                        or any(o.document_id == "database" for o in ref.occurrences)
                    )
                ):
                    raise ValueError("Residence has no scoped source address reference")
            elif span.reference_id is not None:
                raise ValueError("Unexpected source address reference")


def _discover_batch(proposal, entries, checkpoints, model, model_args, rules):
    from tau2.utils.llm_utils import generate

    entities = [
        e
        for e in proposal.entries
        if e.category in {"identity_name", "address", "contact", "other"}
    ]
    payload = {
        "country": rules.country_name,
        "language": rules.language_name,
        "entries": [],
    }
    for entry in entries:
        refs = [
            {
                "id": e.id,
                "source": e.source,
                "category": e.category,
                "source_identity": e.context.split("; role=")[0],
            }
            for e in entities
            if isinstance(e.source, str)
            and (
                e.source.casefold() in entry.source.casefold()
                or (e.category == "address" and _tasks(e) & _tasks(entry))
            )
        ]
        payload["entries"].append(
            {"id": entry.id, "text": entry.source, "structured_source_references": refs}
        )
    digest = json_content_digest(
        {
            "payload": payload,
            "model": model,
            "model_args": model_args,
            "version": rules.discovery_version,
        }
    )
    path = checkpoints / f"{digest}.json"
    if path.exists():
        cached = DiscoveryCheckpoint.model_validate_json(path.read_text())
        if cached.input_sha256 != digest:
            raise ValueError("Discovery checkpoint mismatch")
        validate_discovered_spans(proposal, cached.entries)
        return cached
    rows_model = create_model(
        "BankingSourceSpanRows",
        __base__=LocalizationModel,
        **{e.id: (list[EntitySpan], Field(...)) for e in entries},
    )
    response_model = create_model(
        "BankingSourceSpans",
        __base__=LocalizationModel,
        entries=(rows_model, Field(...)),
    )
    schema = response_model.model_json_schema()
    feedback = ""
    for attempt in range(3):
        response = generate(
            model=model,
            messages=[
                SystemMessage(role="system", content=rules.discovery_prompt),
                UserMessage(
                    role="user",
                    content=json.dumps(payload, ensure_ascii=False) + feedback,
                ),
            ],
            call_name=rules.discovery_call_name,
            **model_args,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "banking_source_spans",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        try:
            rows = response_model.model_validate_json(response.content).model_dump()[
                "entries"
            ]
            typed = {
                key: [EntitySpan.model_validate(span) for span in spans]
                for key, spans in rows.items()
            }
            validate_discovered_spans(proposal, typed)
            result = DiscoveryCheckpoint(
                input_sha256=digest,
                model=model,
                model_args=model_args,
                prompt_version=rules.discovery_version,
                entries=typed,
            )
            write_artifact(result, path)
            return result
        except ValueError as error:
            if attempt == 2:
                if len(entries) == 1:
                    raise
                middle = len(entries) // 2
                children = [
                    _discover_batch(
                        proposal, part, checkpoints, model, model_args, rules
                    )
                    for part in (entries[:middle], entries[middle:])
                ]
                result = DiscoveryCheckpoint(
                    input_sha256=digest,
                    model=model,
                    model_args=model_args,
                    prompt_version=rules.discovery_version,
                    entries={
                        key: value
                        for child in children
                        for key, value in child.entries.items()
                    },
                    child_inputs=[child.input_sha256 for child in children],
                )
                write_artifact(result, path)
                return result
            feedback = (
                "\nPrevious attempt failed validation. Correct this exact issue: "
                + str(error)
            )


def discover_entities(
    proposal, checkpoints: Path, model: str, model_args: dict, rules: BankingReviewRules
) -> dict[str, list[EntitySpan]]:
    """Audit all English prose except typed tool-argument JSON with resumable calls."""
    rules.require(proposal)
    pending = [
        e
        for e in proposal.entries
        if e.category == "prose" and "role=arguments;" not in e.context
    ]
    batches = []
    batch = []
    length = 0
    for entry in pending:
        if batch and (len(batch) >= 16 or length + len(entry.source) > 16000):
            batches.append(batch)
            batch = []
            length = 0
        batch.append(entry)
        length += len(entry.source)
    if batch:
        batches.append(batch)
    checkpoints.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda items: _discover_batch(
                    proposal, items, checkpoints, model, model_args, rules
                ),
                batches,
            )
        )
    return {
        identifier: spans
        for result in results
        for identifier, spans in result.entries.items()
    }


def _run_discovery(args):
    from tau2.multilingual.factory.localization_artifact import read_localization_map

    proposal = read_localization_map(args.map)
    rules = BankingReviewRules.model_validate_json(args.rules.read_text())
    rows = discover_entities(
        proposal,
        args.checkpoints,
        args.model,
        {"temperature": 0, "reasoning_effort": "medium"},
        rules,
    )
    write_artifact(
        DiscoveryCheckpoint(
            input_sha256=json_content_digest(
                [
                    content.model_dump(mode="json")
                    for content in proposal.source.documents
                ]
            ),
            model=args.model,
            model_args={"temperature": 0, "reasoning_effort": "medium"},
            prompt_version=rules.discovery_version,
            entries=rows,
        ),
        args.output,
    )


def add_banking_candidate_review_args(commands):
    """Register source-only discovery and locale-configured candidate revision."""
    add_banking_structured_review_args(commands)
    add_banking_prose_review_args(commands)
    add_banking_name_collision_args(commands)
    add_banking_merchant_review_args(commands)
    add_banking_finalize_review_args(commands)
    add_banking_inline_address_args(commands)
    parser = commands.add_parser("discover-banking-entities")
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.set_defaults(func=_run_discovery)


class EntityCheckpoint(LocalizationModel):
    """One actual provider revision response, bound to masked request inputs."""

    input_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Masked request and configuration digest.",
    )
    model: str = Field(description="Actual authoring model.")
    model_args: dict[str, JsonValue] = Field(description="Actual provider settings.")
    prompt_version: str = Field(description="Actual entity prompt version.")
    entries: dict[str, JsonValue] = Field(
        description="Validated generated string-form entries, locally restored."
    )


def _source_at(value, path):
    for part in path:
        value = value[part]
    return value


def _entity_requests(proposal, rules):
    """Select all businesses/phones and source-evidenced identity/address defects."""
    docs = {d.id: d.content for d in proposal.source.documents}
    requests = []
    families = []
    members = {}
    for entry in proposal.entries:
        kind = None
        context = ""
        if entry.category == "other":
            kind = "organization"
            categories = set()
            for occurrence in entry.occurrences:
                try:
                    row = _source_at(docs[occurrence.document_id], occurrence.path[:-1])
                    if isinstance(row, dict) and row.get("category"):
                        categories.add(str(row["category"]))
                except (KeyError, IndexError, TypeError):
                    pass
            context = "Source spending categories: " + ", ".join(sorted(categories))
            if entry.source.casefold() in rules.organization_role_evidence:
                context += (
                    " Source review: "
                    + rules.organization_role_evidence[entry.source.casefold()]
                )
        elif entry.category == "contact" and re.fullmatch(
            r"[+\d ()-]+", str(entry.source)
        ):
            kind = "phone"
        elif entry.category == "address" and re.search(
            rules.imported_street_pattern,
            entry.forms.romanized,
            re.I,
        ):
            kind = "address"
            context = "Replace imported street labels with plausible target-country locality, city/region/postcode. Prior donor forms intentionally omitted."
        elif entry.category == "identity_name":
            evidence = rules.name_role_evidence
            if entry.source.casefold() in evidence:
                kind = "name"
                context = evidence[entry.source.casefold()]
        if kind is None:
            continue
        key = (
            json_content_digest({"kind": kind, "source": entry.source.casefold()})
            if kind == "name"
            else entry.id
        )
        members.setdefault(key, []).append(entry.id)
        if any(r["id"] == key for r in requests):
            continue
        item = {"id": key, "source": entry.source, "kind": kind, "context": context}
        if kind == "name":
            family_entry = entry
            if entry.source.casefold() in rules.family_sources:
                family_entry = next(
                    e
                    for e in proposal.entries
                    if e.category == "identity_name"
                    and e.source == rules.family_sources[entry.source.casefold()]
                )
            forms = {
                rep: rules.family_token(getattr(family_entry.forms, rep), rep)
                for rep in ("native", "romanized")
            }
            forms["english"] = "Existing family name retained locally."
            family_id = "family:" + key
            families.append(
                {
                    "id": family_id,
                    "source": "Required existing localized family name for "
                    + entry.source,
                    "category": "other",
                    "forms": forms,
                }
            )
            item["required_family_reference"] = family_id
        requests.append(item)
    return requests, families, members


def _revise_batch(requests, completed, checkpoints, model, model_args, rules):
    from tau2.domains.banking_knowledge.localization_tokens import (
        BankingReferenceTokens,
        banking_response_model,
        banking_response_schema,
    )
    from tau2.utils.llm_utils import generate

    payload = {
        "country": rules.country_name,
        "language": rules.language_name,
        "country_code": rules.country_code,
        "phone_pattern": rules.phone_pattern,
        "family_name_position": rules.family_name_position,
        "entries": requests,
        "completed_mappings": json.loads(json.dumps(completed, ensure_ascii=False)),
    }
    tokens = BankingReferenceTokens.protect(payload)
    digest = json_content_digest(
        {
            "payload": payload,
            "version": rules.entity_version,
            "model": model,
            "model_args": model_args,
        }
    )
    path = checkpoints / f"{digest}.json"
    if path.exists():
        cached = EntityCheckpoint.model_validate_json(path.read_text())
        if (
            cached.input_sha256 != digest
            or cached.model != model
            or cached.model_args != model_args
            or cached.prompt_version != rules.entity_version
            or set(cached.entries) != {r["id"] for r in requests}
        ):
            raise ValueError(
                "Entity checkpoint differs from exact requested source/configuration"
            )
        for request in requests:
            for component, reference_id in request.get(
                "required_address_references", {}
            ).items():
                reference = next(
                    c["forms"] for c in completed if c["id"] == reference_id
                )
                address = cached.entries[request["id"]]["forms"].get("address")
                if address is None or any(
                    address[rep][component] != reference[rep]
                    for rep in ("native", "romanized")
                ):
                    raise ValueError(
                        "Cached address components differ from exact local references"
                    )
            for key in (
                "required_family_reference",
                "required_city_reference",
                "required_street_reference",
                "required_postal_reference",
            ):
                if key in request:
                    reference = next(
                        c["forms"] for c in completed if c["id"] == request[key]
                    )
                    if any(
                        reference[rep]
                        not in cached.entries[request["id"]]["forms"][rep]
                        for rep in ("native", "romanized")
                    ):
                        raise ValueError(
                            "Cached local references changed; use a fresh checkpoint directory"
                        )
        return cached
    response_model = banking_response_model([r["id"] for r in requests])
    feedback = ""
    for attempt in range(3):
        message = generate(
            model=model,
            messages=[
                SystemMessage(role="system", content=rules.entity_prompt),
                UserMessage(
                    role="user",
                    content=json.dumps(payload, ensure_ascii=False) + feedback,
                ),
            ],
            call_name=rules.entity_call_name,
            **model_args,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "banking_hindi_entities",
                    "strict": True,
                    "schema": banking_response_schema(response_model),
                },
            },
        )
        try:
            parsed = response_model.model_validate_json(message.content).model_dump()[
                "entries"
            ]
            restored = tokens.restore(
                {"entries": [{"id": key, **value} for key, value in parsed.items()]}
            )["entries"]
            result = {row.pop("id"): row for row in restored}
            phones = []
            businesses = []
            for request in requests:
                forms = result[request["id"]]["forms"]
                if request["kind"] == "address":
                    from tau2.multilingual.address_format import (
                        AddressForms,
                        format_address,
                    )

                    components = AddressForms.model_validate(forms.get("address"))
                    if components.language != rules.language:
                        raise ValueError(
                            "Generated address components have the wrong language"
                        )
                    for representation in ("native", "romanized"):
                        forms[representation] = format_address(
                            getattr(components, representation),
                            rules.language,
                            representation,
                        )
                    for component, reference_id in request.get(
                        "required_address_references", {}
                    ).items():
                        required = next(
                            c["forms"] for c in completed if c["id"] == reference_id
                        )
                        for representation in ("native", "romanized"):
                            if (
                                getattr(getattr(components, representation), component)
                                != required[representation]
                            ):
                                raise ValueError(
                                    "Inline address changed a required semantic component"
                                )
                if any(
                    unicodedata.category(c).startswith("L")
                    and not regex.fullmatch(r"\p{Script=Latin}", c)
                    for c in forms["romanized"]
                ):
                    raise ValueError(
                        "Romanized values must contain Latin-script letters only"
                    )
                if request.get("distinct_business"):
                    names = {forms[rep].casefold() for rep in ("native", "romanized")}
                    reserved = {
                        x.casefold()
                        for x in request["reserved_generated_business_names"]
                    }
                    if names.intersection(reserved) or names.intersection(businesses):
                        raise ValueError("Distinct incidental businesses collided")
                    businesses.extend(names)
                if request.get("required_new_identity") and forms[
                    "romanized"
                ].casefold() in {
                    request["source"].casefold(),
                    request["previous_generated_forms"]["romanized"].casefold(),
                }:
                    raise ValueError(
                        "Flagged merchant requires an actual new target-country identity"
                    )
                for reference_key in (
                    "required_street_reference",
                    "required_postal_reference",
                ):
                    if reference_key in request:
                        required = next(
                            c["forms"]
                            for c in completed
                            if c["id"] == request[reference_key]
                        )
                        if any(
                            required[rep] not in forms[rep]
                            for rep in ("native", "romanized")
                        ):
                            raise ValueError(
                                "Address must preserve exact required local street/postal tokens"
                            )
                if "required_city_reference" in request:
                    required = next(
                        c["forms"]
                        for c in completed
                        if c["id"] == request["required_city_reference"]
                    )
                    if any(
                        required[rep] not in forms[rep]
                        for rep in ("native", "romanized")
                    ):
                        raise ValueError(
                            "Inline address must retain exact required locality token"
                        )
                if request["kind"] == "phone":
                    if (
                        not re.fullmatch(rules.phone_pattern, forms["native"])
                        or forms["native"] != forms["romanized"]
                    ):
                        raise ValueError(
                            "Phone must have identical ASCII national digits matching the locale profile"
                        )
                    phones.append(forms["native"])
            if len(phones) != len(set(phones)):
                raise ValueError("Distinct source phones collided")
            checkpoint = EntityCheckpoint(
                input_sha256=digest,
                model=model,
                model_args=model_args,
                prompt_version=rules.entity_version,
                entries=result,
            )
            write_artifact(checkpoint, path)
            return checkpoint
        except ValueError as error:
            if attempt == 2:
                raise
            feedback = (
                "\nStrict validation failed: "
                + str(error)
                + ". Regenerate the exact required IDs and retain all other requirements."
            )


def revise_structured_entities(
    proposal, checkpoints, model, model_args, rules, selected_ids=None
):
    """Revise eligible source entities through typed checkpointed generation."""
    from tau2.multilingual.localization import GenerationProvenance, LocalizedForms
    from tau2.utils.artifacts import content_digest

    requests, families, members = _entity_requests(proposal, rules)
    if selected_ids is not None:
        requests = [r for r in requests if set(members[r["id"]]) & selected_ids]
        if {
            identifier for r in requests for identifier in members[r["id"]]
        } != selected_ids:
            raise ValueError(
                "Selected source entries are not exactly eligible for revision"
            )
    result = proposal.model_copy(deep=True)
    entries = {e.id: e for e in result.entries}
    checkpoints.mkdir(parents=True, exist_ok=True)
    # Phone uniqueness is checked jointly; business references remain coherent across batches.
    groups = [[r for r in requests if r["kind"] == "phone"]]
    other = sorted(
        (r for r in requests if r["kind"] != "phone"),
        key=lambda r: (r["kind"], r["source"].casefold()),
    )
    groups.extend(other[index : index + 40] for index in range(0, len(other), 40))
    completed = list(families)
    calls = []
    for batch in groups:
        if not batch:
            continue
        reply = _revise_batch(batch, completed, checkpoints, model, model_args, rules)
        calls.append(reply)
        for request in batch:
            value = reply.entries[request["id"]]
            for identifier in members[request["id"]]:
                entry = entries[identifier]
                entry.forms = LocalizedForms.model_validate(value["forms"])
                entry.notes = value["notes"]
                entry.operation = "adapt"
                entry.reuse = None
            completed.append(
                {
                    "id": request["id"],
                    "source": request["source"],
                    "category": request["kind"],
                    "forms": value["forms"],
                }
            )
    # Deliberately wrong protocol tokens remain exact even in a name-shaped field.
    for entry in result.entries:
        if entry.source == "9K2X7M4P1N8Q3R5T6A" and entry.category != "protocol":
            entry.forms = LocalizedForms(
                native=entry.source,
                romanized=entry.source,
                english="Intentional invalid verification token preserved exactly.",
            )
            entry.operation = "preserve"
            entry.reuse = None
    from tau2.multilingual.factory.localization_emails import (
        _is_email,
        localize_email_entries,
    )

    changed_names = {
        identifier
        for request in requests
        if request["kind"] == "name"
        for identifier in members[request["id"]]
    }
    owners = {
        re.search(r"entity=([^;]+);", entries[key].context)[1] for key in changed_names
    }
    email_sources = [
        e
        for e in result.source.entries
        if _is_email(e) and any(f"entity={owner};" in e.context for owner in owners)
    ]
    email_ids = {e.id for e in email_sources}
    for entry in localize_email_entries(
        email_sources, [e for e in result.entries if e.id not in email_ids]
    ):
        entries[entry.id] = entry
    result.entries = [entries[e.id] for e in result.entries]
    result.revision = proposal.revision + "-" + rules.entity_version
    result.provenance = GenerationProvenance(
        model=model,
        model_args=model_args,
        prompt_version=rules.entity_version,
        call_name=rules.entity_call_name,
        input_sha256=json_content_digest([c.input_sha256 for c in calls]),
        parent_map_sha256=content_digest(proposal),
    )
    return result, calls


def _run_structured(args):
    from tau2.multilingual.factory.localization_artifact import read_localization_map
    from tau2.multilingual.factory.localization_compile import (
        validate_localization_source,
    )

    raw = json.loads(args.map.read_text())
    proposal = (
        validate_localization_source(LocalizationMap.model_validate(raw))
        if raw.get("schema_version") == "regional-localization-v1"
        else read_localization_map(args.map)
    )
    rules = BankingReviewRules.model_validate_json(args.rules.read_text())
    rules.require(proposal)
    revised, calls = revise_structured_entities(
        proposal,
        args.checkpoints,
        args.model,
        {"temperature": 0, "reasoning_effort": "medium"},
        rules,
        set(args.entry_id) if args.entry_id else None,
    )
    write_artifact(revised, args.output)


def add_banking_structured_review_args(commands):
    """Register bounded structured-entity authoring before prose synchronization."""
    parser = commands.add_parser("revise-banking-entities")
    parser.add_argument(
        "--entry-id",
        action="append",
        help="Revise only these exact eligible source entries.",
    )
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.set_defaults(func=_run_structured)


def _surface_key(value):
    return "".join(c for c in value.casefold() if c.isalnum())


def _matching_entity(proposal, prose, surface):
    """Resolve spelling/descriptor variants only against scoped source identities."""
    from tau2.domains.banking_knowledge.localization_prose import _owner

    candidates = []
    needle = _surface_key(surface)
    for entry in proposal.entries:
        if entry.category not in {
            "identity_name",
            "address",
            "contact",
            "other",
        } or not isinstance(entry.source, str):
            continue
        exact = entry.source.casefold() == surface.casefold()
        folded = _surface_key(entry.source) == needle
        suffix_alias = (
            entry.category == "other"
            and (
                needle.startswith(_surface_key(entry.source))
                or _surface_key(entry.source).startswith(needle)
            )
            and min(len(needle), len(_surface_key(entry.source))) >= 7
        )
        if not (exact or folded or suffix_alias):
            continue
        same_task = bool(_tasks(prose) & _tasks(entry))
        in_db = any(o.document_id == "database" for o in entry.occurrences)
        if entry.category == "identity_name" and not (same_task or in_db):
            continue
        rank = (
            3 if exact else 2 if folded else 1,
            2 if same_task else 1 if in_db else 0,
            1 if _owner(entry) == _owner(prose) else 0,
        )
        candidates.append((rank, entry))
    if not candidates:
        return None
    best = max(rank for rank, _ in candidates)
    matches = [entry for rank, entry in candidates if rank == best]
    if len({(e.forms.native, e.forms.romanized) for e in matches}) != 1:
        return None
    return min(matches, key=lambda e: e.id)


def source_person_owners(proposal, prose):
    """Resolve source people from explicit persona names or transaction/account joins."""
    from tau2.domains.banking_knowledge.localization_prose import _owner

    header = prose.source.split("\n\n", 1)[0]
    owners = {
        _owner(e)
        for e in proposal.entries
        if e.category == "identity_name"
        and e.source in header
        and (
            _tasks(e) & _tasks(prose)
            or any(o.document_id == "database" for o in e.occurrences)
        )
    }
    docs = {d.id: d.content for d in proposal.source.documents}
    database = docs["database"]
    accounts = dict(database["accounts"]["data"])
    for occurrence in prose.occurrences:
        if occurrence.document_id == "tasks":
            task = docs["tasks"][occurrence.path[0]]
            extra = (
                (task.get("initial_state") or {}).get("initialization_data") or {}
            ).get("agent_data") or {}
            accounts.update(extra.get("accounts", {}).get("data", {}))
        try:
            row = _source_at(docs[occurrence.document_id], occurrence.path[:-1])
        except (TypeError, KeyError, IndexError):
            continue
        if isinstance(row, dict):
            user = row.get("user_id")
            account = accounts.get(row.get("account_id"), {})
            user = user or account.get("user_id")
            if user:
                owners.add(str(user))
    return owners


def _residence_binding(proposal, prose, span):
    """Prefer source-proven home locality over independent generated city names."""
    from tau2.domains.banking_knowledge.localization_bindings import address_component
    from tau2.domains.banking_knowledge.localization_prose import _owner
    from tau2.multilingual.localization import ProseEntityBinding

    if span.disposition == "generate" and (
        re.search(r"\d", span.source)
        or re.search(r"friend|parents|colleague|partner", span.role, re.I)
    ):
        return None
    entries = {e.id: e for e in proposal.entries}
    reference = entries.get(span.reference_id) if span.reference_id else None
    owners = source_person_owners(proposal, prose)
    if reference is not None and _owner(reference) not in owners:
        equivalent = [
            e
            for e in proposal.entries
            if e.category == "address"
            and _owner(e) in owners
            and e.source == reference.source
            and _tasks(e) & _tasks(prose)
        ]
        if not equivalent:
            raise ValueError(
                "Address binding lacks explicit source person/account evidence: "
                + prose.id
                + " "
                + span.source
            )
        reference = min(equivalent, key=lambda e: e.id)
    if reference is None and re.search(r"location|city|residen|home", span.role, re.I):
        candidates = [
            e
            for e in proposal.entries
            if e.category == "address"
            and _owner(e) in owners
            and _surface_key(e.source.split(",")[-2].strip())
            in _surface_key(span.source)
        ]
        candidates.sort(key=lambda e: bool(_tasks(e) & _tasks(prose)), reverse=True)
        if candidates:
            reference = candidates[0]
    if reference is None and re.search(
        r"residen|home|customer.*city|character.*city", span.role, re.I
    ):
        header = prose.source.split("\n\n", 1)[0]
        people = [
            e
            for e in proposal.entries
            if e.category == "identity_name"
            and e.source in header
            and (
                _tasks(e) & _tasks(prose)
                or any(o.document_id == "database" for o in e.occurrences)
            )
        ]
        if people:
            person = min(people, key=lambda e: header.index(e.source))
            addresses = [
                e
                for e in proposal.entries
                if e.category == "address" and _owner(e) == _owner(person)
            ]
            addresses.sort(
                key=lambda e: (
                    e.source in prose.source,
                    bool(_tasks(e) & _tasks(prose)),
                ),
                reverse=True,
            )
            if addresses:
                city = addresses[0].source.split(",")[-2].strip()
                if city.casefold() in span.source.casefold():
                    reference = addresses[0]
    if reference is None:
        return None
    part = (
        span.disposition
        if span.disposition.startswith("address_")
        else "address_city_region"
        if "," in span.source
        else "address_city"
    )
    return ProseEntityBinding(
        entry_id=prose.id,
        source=span.source,
        forms=address_component(reference, part),
        kind="reuse",
        source_entry_id=reference.id,
        source_part=part,
        reason="Source-proven residence/locality derived from the same localized address. "
        + span.reason,
    )


def _binding_plan(proposal, discovery, rules):
    """Turn discovered source spans into exact reuse or independently authored gaps."""
    from tau2.multilingual.localization import LocalizedForms, ProseEntityBinding

    entries = {e.id: e for e in proposal.entries}
    rows = {}
    for binding in proposal.prose_bindings:
        if binding.kind == "generated":
            rows[(binding.entry_id, binding.source.casefold())] = EntitySpan(
                source=binding.source,
                role="incidental entity",
                disposition="generate",
                reference_id=None,
                reason=binding.reason,
            )
        elif binding.kind == "protected" and binding.source == "9K2X7M4P1N8Q3R5T6A":
            rows[(binding.entry_id, binding.source.casefold())] = EntitySpan(
                source=binding.source,
                role="protocol token",
                disposition="preserve",
                reference_id=None,
                reason=binding.reason,
            )
    for entry_id, spans in discovery.entries.items():
        for span in spans:
            rows[(entry_id, span.source.casefold())] = span
    for entry_id, spans in rules.span_overrides.items():
        for span in spans:
            rows[(entry_id, span.source.casefold())] = span
    ready = []
    requests = {}
    members = {}
    previous = {
        (b.entry_id, b.source.casefold()): b
        for b in proposal.prose_bindings
        if b.kind == "generated"
    }
    source_aliases = {k.casefold(): v for k, v in rules.source_aliases.items()}
    protected_sources = {k.casefold(): v for k, v in rules.protected_sources.items()}
    for (entry_id, _), span in rows.items():
        prose = entries[entry_id]
        # Terminal/store identifiers remain outside the replaceable named surface.
        surface = re.sub(r"\s+#\d+$", "", span.source)
        span = span.model_copy(update={"source": surface})
        canonical = source_aliases.get(surface.casefold(), surface)
        foreign_atm = (
            bool(rules.foreign_atm_country_codes)
            and "ATM" in prose.source.upper()
            and bool(
                re.search(
                    r"\b(?:"
                    + "|".join(map(re.escape, rules.foreign_atm_country_codes))
                    + r")\b",
                    prose.source,
                )
            )
        )
        if foreign_atm and re.search(
            r"bank|ATM|location|city|country|operator|merchant", span.role, re.I
        ):
            span = span.model_copy(
                update={
                    "disposition": "preserve",
                    "reference_id": None,
                    "reason": "Source foreign ATM geography and operator identity preserve international fee semantics and avoid fictitious domestic branches.",
                }
            )
        if surface.casefold() in protected_sources:
            span = span.model_copy(
                update={
                    "disposition": "preserve",
                    "reference_id": None,
                    "reason": protected_sources[surface.casefold()],
                }
            )

        if span.disposition != "preserve" and (
            span.disposition.startswith("address_")
            or re.search(r"residen|home|customer.*city|location", span.role, re.I)
        ):
            binding = _residence_binding(proposal, prose, span)
            if binding is not None:
                ready.append(binding)
                continue
        ref = (
            None
            if span.disposition == "preserve"
            else _matching_entity(proposal, prose, canonical)
        )
        if ref is None and re.search(
            r"daughter\s+" + re.escape(span.source) + r"\b", prose.source, re.I
        ):
            matches = [
                e
                for e in proposal.entries
                if e.category == "identity_name"
                and e.source.casefold().startswith(span.source.casefold() + " ")
                and _tasks(e) & _tasks(prose)
            ]
            if len(matches) == 1:
                ref = matches[0]
        if ref is not None:
            ready.append(
                ProseEntityBinding(
                    entry_id=entry_id,
                    source=span.source,
                    forms=ref.forms,
                    kind="reuse",
                    source_entry_id=ref.id,
                    reason="Explicit scoped source identity or merchant spelling/descriptor alias. "
                    + span.reason,
                )
            )
            continue
        if span.disposition == "preserve" and not re.search(
            r"person|given name|family name", span.role, re.I
        ):
            ready.append(
                ProseEntityBinding(
                    entry_id=entry_id,
                    source=span.source,
                    forms=LocalizedForms(
                        native=span.source, romanized=span.source, english=span.reason
                    ),
                    kind="protected",
                    reason=span.reason,
                )
            )
            continue
        key = json_content_digest({"source": canonical.casefold()})
        request = {
            "id": key,
            "source": canonical,
            "kind": "incidental",
            "context": span.role + ". " + span.reason,
        }
        old = previous.get((entry_id, span.source.casefold()))
        evidence = rules.organization_role_evidence.get(
            canonical.casefold()
        ) or rules.organization_role_evidence.get(surface.casefold())
        if evidence:
            request["context"] += " Source review: " + evidence
        if old is not None:
            request["previous_generated_forms"] = old.forms.model_dump(mode="json")
            request["context"] += (
                " Retain only if already a plausible target-country entity; otherwise replace the prior transliteration/calque."
            )
        if key not in requests:
            requests[key] = request
        elif request["context"] not in requests[key]["context"]:
            requests[key]["context"] += " Additional source role: " + request["context"]
        members.setdefault(key, []).append((entry_id, span))
    for key, request in requests.items():
        if not (
            re.search(r"\baddress\b", request["context"], re.I)
            and re.search(r"\d", request["source"])
        ):
            continue
        entry_ids = {identifier for identifier, _ in members[key]}
        references = {
            b.source_entry_id
            for b in ready
            if b.entry_id in entry_ids
            and b.source_part in {"address_city", "address_city_region"}
        }
        # The source explicitly describes work or an across-town residence, establishing same-city scope.
        if len(references) == 1 and re.search(
            r"work|shipping|across town|roommate", request["context"], re.I
        ):
            request["required_city_reference"] = "locality:" + next(iter(references))
            request["context"] += (
                " Use exactly the required localized city/region reference for this distinct work/friend premise. The source establishes the same city; keep a different street/premise and a locally plausible postal code."
            )
    for key, request in requests.items():
        if "taxi route destination" in request["context"]:
            entry_ids = {identifier for identifier, _ in members[key]}
            refs = {
                b.source_entry_id
                for b in ready
                if b.entry_id in entry_ids and b.source_part == "address_city"
            }
            if len(refs) == 1:
                request["required_city_reference"] = "locality:" + next(iter(refs))
                request["context"] += (
                    " This is a short local taxi journey. The destination must be a "
                    "distinct plausible neighborhood or local destination within the "
                    "required origin city/region. Include the exact required city/region "
                    "reference token in the destination, preceded by the distinct local "
                    "place name. Do not choose a distant city or use only the origin itself."
                )
    return ready, list(requests.values()), members


def reserved_prose_business_names(proposal, bindings, source, rules) -> list[str]:
    """Keep unrelated completed businesses distinct during bounded prose repairs."""
    aliases = {k.casefold(): v.casefold() for k, v in rules.source_aliases.items()}

    def canonical(value):
        return aliases.get(value.casefold(), value.casefold())

    target = canonical(source)
    forms = [
        e.forms
        for e in proposal.entries
        if e.category == "other" and e.reuse is None and canonical(e.source) != target
    ] + [
        b.forms
        for b in bindings
        if b.kind == "generated" and canonical(b.source) != target
    ]
    return sorted(
        {getattr(form, rep) for form in forms for rep in ("native", "romanized")}
    )


def revise_prose_entities(
    proposal, discovery, checkpoints, model, model_args, rules, selected_sources=None
):
    """Author discovered gaps and reconstruct English with exact entity substitutions."""
    from tau2.domains.banking_knowledge.localization_bindings import (
        bind_explicit_person_aliases,
    )
    from tau2.domains.banking_knowledge.localization_households import (
        harmonize_banking_households,
    )
    from tau2.domains.banking_knowledge.localization_prose import (
        reconstruct_banking_prose,
    )
    from tau2.multilingual.localization import (
        GenerationProvenance,
        LocalizedForms,
        ProseEntityBinding,
    )

    rules.require(proposal)
    validate_discovered_spans(proposal, discovery.entries)
    household = harmonize_banking_households(proposal)
    proposal = household.proposal
    ready, requests, members = _binding_plan(proposal, discovery, rules)
    if selected_sources is not None:
        selected_sources = {value.casefold() for value in selected_sources}
        existing = {
            (b.entry_id, b.source.casefold()): b for b in proposal.prose_bindings
        }
        keep_requests = []
        for request in requests:
            assigned = members[request["id"]]
            if request["source"].casefold() in selected_sources or any(
                span.source.casefold() in selected_sources for _, span in assigned
            ):
                keep_requests.append(request)
            else:
                for entry_id, span in assigned:
                    old = existing.get((entry_id, span.source.casefold()))
                    if old is None:
                        raise ValueError(
                            "Bounded prose repair cannot omit an unauthored source surface"
                        )
                    ready.append(old.model_copy(deep=True))
        requests = keep_requests

    completed = [
        {
            "id": e.id,
            "source": e.source,
            "category": e.category,
            "forms": e.forms.model_dump(mode="json"),
        }
        for e in proposal.entries
        if e.category == "other"
    ]
    from tau2.domains.banking_knowledge.localization_bindings import address_component

    required_ids = {
        r[key].removeprefix("locality:")
        for r in requests
        for key in ("required_city_reference", "nearby_origin_reference")
        if key in r
    }
    completed.extend(
        {
            "id": "locality:" + e.id,
            "source": "Required exact city and region from linked source address "
            + e.source,
            "category": "other",
            "forms": address_component(e, "address_city_region").model_dump(
                mode="json"
            ),
        }
        for e in proposal.entries
        if e.id in required_ids
    )
    checkpoints.mkdir(parents=True, exist_ok=True)
    calls = []
    for start in range(0, len(requests), 32):
        batch = requests[start : start + 32]
        for request in batch:
            if re.search(
                r"bank|merchant|company|employer|business|institution|retail|restaurant|service",
                request["context"],
                re.I,
            ) and not re.search(
                r"\b(?:city|location|address|person|payee alias)\b",
                request["context"],
                re.I,
            ):
                request["distinct_business"] = True
                request["reserved_generated_business_names"] = (
                    reserved_prose_business_names(
                        proposal, ready, request["source"], rules
                    )
                )
                request["context"] += (
                    " Choose a distinct full business name, different from reserved generated business names and other unrelated requested businesses; explicitly fictional target-country businesses are acceptable. Source aliases have already been grouped."
                )
        reply = _revise_batch(batch, completed, checkpoints, model, model_args, rules)
        calls.append(reply)
        provenance = GenerationProvenance(
            model=model,
            model_args=model_args,
            prompt_version=rules.entity_version,
            call_name=rules.entity_call_name,
            input_sha256=reply.input_sha256,
        )
        for request in batch:
            value = reply.entries[request["id"]]
            forms = LocalizedForms.model_validate(value["forms"])
            for entry_id, span in members[request["id"]]:
                ready.append(
                    ProseEntityBinding(
                        entry_id=entry_id,
                        source=span.source,
                        forms=forms,
                        kind="generated",
                        reason=value["notes"] + " Source role: " + span.role,
                        provenance=provenance,
                    )
                )
            completed.append(
                {
                    "id": request["id"],
                    "source": request["source"],
                    "category": "other",
                    "forms": value["forms"],
                }
            )
    # Exact inline shipping addresses must reach corresponding encoded tool arguments.
    for e in proposal.entries:
        if e.category != "prose" or "role=arguments;" not in e.context:
            continue
        for binding in list(ready):
            if binding.source in e.source and _tasks(
                entries_by_id(proposal)[binding.entry_id]
            ) & _tasks(e):
                if not any(
                    b.entry_id == e.id
                    and b.source.casefold() == binding.source.casefold()
                    for b in ready
                ):
                    ready.append(binding.model_copy(update={"entry_id": e.id}))
    ready = _unique_bindings(ready)
    revised = proposal.model_copy(update={"prose_bindings": ready})
    revised = bind_explicit_person_aliases(revised)
    reconstruction = reconstruct_banking_prose(revised)
    return reconstruction, household, calls


def _run_prose(args):
    from tau2.multilingual.factory.database_localization import _finish_draft
    from tau2.multilingual.factory.localization_compile import (
        validate_localization_source,
    )

    proposal = validate_localization_source(
        LocalizationMap.model_validate_json(args.map.read_text())
    )
    discovery = DiscoveryCheckpoint.model_validate_json(args.discovery.read_text())
    rules = BankingReviewRules.model_validate_json(args.rules.read_text())
    reconstruction, household, calls = revise_prose_entities(
        proposal,
        discovery,
        args.checkpoints,
        args.model,
        {"temperature": 0, "reasoning_effort": "medium"},
        rules,
        set(args.source) if args.source else None,
    )
    write_artifact(household.revision, args.output.with_suffix(".households.json"))
    write_artifact(
        reconstruction.reconstruction, args.output.with_suffix(".reconstruction.json")
    )
    write_artifact(reconstruction.proposal, args.output.with_suffix(".snapshot.json"))
    _finish_draft(reconstruction.proposal, args.output)


def add_banking_prose_review_args(commands):
    """Register source-aware incidental review and deterministic reconstruction."""
    parser = commands.add_parser("revise-banking-prose-entities")
    parser.add_argument(
        "--source",
        action="append",
        help="Regenerate only these existing source entity surfaces; keep other generated bindings.",
    )
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.set_defaults(func=_run_prose)


def colliding_name_entries(proposal):
    """Find distinct source people merged in any source task world or base DB."""
    names = [e for e in proposal.entries if e.category == "identity_name"]
    collisions = set()
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if left.source.casefold() == right.source.casefold():
                continue
            together = bool(_tasks(left) & _tasks(right)) or any(
                o.document_id == "database"
                for o in left.occurrences + right.occurrences
            )
            if together and any(
                getattr(left.forms, rep).casefold()
                == getattr(right.forms, rep).casefold()
                for rep in ("native", "romanized")
            ):
                collisions.update((left.id, right.id))
    return collisions


def _run_name_collision(args):
    from tau2.multilingual.factory.localization_compile import (
        validate_localization_source,
    )
    from tau2.multilingual.localization import GenerationProvenance, LocalizedForms
    from tau2.utils.artifacts import content_digest

    proposal = validate_localization_source(
        LocalizationMap.model_validate_json(args.map.read_text())
    )
    rules = BankingReviewRules.model_validate_json(args.rules.read_text())
    rules.require(proposal)
    requests, families, members = _entity_requests(proposal, rules)
    collisions = colliding_name_entries(proposal)
    result = proposal.model_copy(deep=True)
    indexed = {e.id: e for e in result.entries}
    inputs = []
    for request in requests:
        if request["kind"] != "name" or not (collisions & set(members[request["id"]])):
            continue
        alternatives = [
            {
                **request,
                "id": request["id"] + ":alternative:" + str(i),
                "context": request["context"]
                + " Produce one distinct alternative target-country given name. Each requested alternative must differ from the other alternatives; existing family token remains exact.",
            }
            for i in range(8)
        ]
        args.checkpoints.mkdir(parents=True, exist_ok=True)
        reply = _revise_batch(
            alternatives,
            families,
            args.checkpoints,
            args.model,
            {"temperature": 0, "reasoning_effort": "medium"},
            rules,
        )
        inputs.append(reply.input_sha256)
        forbidden = {
            getattr(e.forms, rep).casefold()
            for e in result.entries
            if e.category == "identity_name" and e.id not in members[request["id"]]
            for rep in ("native", "romanized")
        }
        candidate = next(
            (
                reply.entries[item["id"]]
                for item in alternatives
                if all(
                    reply.entries[item["id"]]["forms"][rep].casefold() not in forbidden
                    for rep in ("native", "romanized")
                )
            ),
            None,
        )
        if candidate is None:
            raise ValueError(
                "All generated name alternatives collide; no candidate applied"
            )
        for identifier in members[request["id"]]:
            indexed[identifier].forms = LocalizedForms.model_validate(
                candidate["forms"]
            )
            indexed[identifier].notes = candidate["notes"]
            indexed[identifier].reuse = None
    if colliding_name_entries(result):
        raise ValueError("Source-world name collisions remain")
    result.provenance = GenerationProvenance(
        model=args.model,
        model_args={"temperature": 0, "reasoning_effort": "medium"},
        prompt_version=rules.entity_version,
        call_name=rules.entity_call_name,
        input_sha256=json_content_digest(inputs),
        parent_map_sha256=content_digest(proposal),
    )
    write_artifact(result, args.output)


def add_banking_name_collision_args(commands):
    """Register bounded source-aware name collision correction."""
    parser = commands.add_parser("resolve-banking-name-collisions")
    for argument in ("map", "output", "checkpoints", "rules"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.set_defaults(func=_run_name_collision)


def merchant_collisions(proposal, aliases):
    """Reject collapsed distinct source businesses while honoring explicit aliases."""
    canonical = {
        name.casefold(): group[0].casefold() for group in aliases for name in group
    }
    collisions = set()
    for representation in ("native", "romanized"):
        grouped = {}
        for entry in proposal.entries:
            if entry.category == "other":
                key = _surface_key(
                    unicodedata.normalize("NFC", getattr(entry.forms, representation))
                )
                grouped.setdefault(key, []).append(entry)
        collisions.update(
            e.id
            for group in grouped.values()
            if len(
                {canonical.get(e.source.casefold(), e.source.casefold()) for e in group}
            )
            > 1
            for e in group
        )
    return collisions


def _run_merchant_review(args):
    from tau2.multilingual.factory.localization_compile import (
        validate_localization_source,
    )
    from tau2.multilingual.localization import GenerationProvenance, LocalizedForms
    from tau2.utils.artifacts import content_digest

    proposal = validate_localization_source(
        LocalizationMap.model_validate_json(args.map.read_text())
    )
    rules = BankingReviewRules.model_validate_json(args.rules.read_text())
    rules.require(proposal)
    collisions = merchant_collisions(proposal, rules.organization_aliases)
    selected = {
        e.id
        for e in proposal.entries
        if e.category == "other"
        and (e.id in collisions or e.source in rules.organization_review_sources)
    }
    requests, _, _ = _entity_requests(proposal, rules)
    requests = [r for r in requests if r["id"] in selected]
    by_id = {e.id: e for e in proposal.entries}
    aliases = {
        name.casefold(): group[0].casefold()
        for group in rules.organization_aliases
        for name in group
    }
    grouped = {}
    for request in requests:
        key = aliases.get(request["source"].casefold(), request["source"].casefold())
        grouped.setdefault(key, []).append(request)
    requests = [group[0] for group in grouped.values()]
    # These values were generated in this authorized pass; donor name/address values are absent.
    reserved = [
        e.forms.romanized
        for e in proposal.entries
        if e.category == "other" and e.id not in selected
    ]
    for request in requests:
        request["previous_generated_forms"] = by_id[request["id"]].forms.model_dump(
            mode="json"
        )
        request["required_new_identity"] = (
            request["source"] in rules.organization_review_sources
        )
        request["forbidden_existing_merchant_names"] = reserved
        request["context"] += (
            " Previously generated merchant may collide or claim a fictional property is established. Choose a distinct plausible target-country merchant identity. Fictional local merchant is allowed with explicit fictional rationale. Do not use any reserved name; all requested different source identities need different full names. Preserve same-source aliases. Do not invent an established branch/property/collection claim."
        )
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    reply = _revise_batch(
        requests,
        [],
        args.checkpoints,
        args.model,
        {"temperature": 0, "reasoning_effort": "medium"},
        rules,
    )
    result = proposal.model_copy(deep=True)
    target = {e.id: e for e in result.entries}
    for group in grouped.values():
        value = reply.entries[group[0]["id"]]
        for request in group:
            e = target[request["id"]]
            e.forms = LocalizedForms.model_validate(value["forms"])
            e.notes = value["notes"]
            e.reuse = None
    if merchant_collisions(result, rules.organization_aliases):
        raise ValueError(
            "Distinct source merchants still collide; checkpoint retained for targeted repair"
        )
    result.provenance = GenerationProvenance(
        model=args.model,
        model_args={"temperature": 0, "reasoning_effort": "medium"},
        prompt_version=rules.entity_version,
        call_name=rules.entity_call_name,
        input_sha256=reply.input_sha256,
        parent_map_sha256=content_digest(proposal),
    )
    write_artifact(result, args.output)


def add_banking_merchant_review_args(commands):
    """Register exact-source merchant distinctness and factual-rationale corrections."""
    parser = commands.add_parser("resolve-banking-merchant-identities")
    for argument in ("map", "output", "checkpoints", "rules"):
        parser.add_argument("--" + argument, type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.set_defaults(func=_run_merchant_review)


def entries_by_id(proposal):
    """Index a source-bound proposal without changing its entries."""
    return {e.id: e for e in proposal.entries}


class BankingCandidateRevision(LocalizationModel):
    """Complete source-bound preparation lineage; native review remains pending."""

    version: Literal["banking-candidate-revision-v1"] = "banking-candidate-revision-v1"
    baseline_map_sha256: str = Field(description="Original full proposal digest.")
    input_map_sha256: str = Field(
        description="Generated full proposal before synchronization."
    )
    output_map_sha256: str = Field(
        description="Final reconstructed full proposal digest."
    )
    rules: BankingReviewRules = Field(
        description="Exact locale rules used for final synchronization."
    )
    source_discovery: DiscoveryCheckpoint = Field(
        description="Actual source-only discovery record."
    )
    generation_calls: list[EntityCheckpoint] = Field(
        description="Actual typed authoring call checkpoints, preserving original prompt versions."
    )
    revised_entries: list[str] = Field(
        description="Source entry IDs whose complete mapping changed from baseline."
    )
    status: Literal["pending_native_review"] = "pending_native_review"


def synchronize_banking_candidate(proposal, structured, rules):
    """Refresh exact local references after bounded revisions without authoring content."""
    from tau2.domains.banking_knowledge.localization_bindings import address_component
    from tau2.domains.banking_knowledge.localization_households import (
        harmonize_banking_households,
    )
    from tau2.domains.banking_knowledge.localization_prose import (
        reconstruct_banking_prose,
    )
    from tau2.multilingual.localization import LocalizedForms

    rules.require(proposal)
    if proposal.source != structured.source:
        raise ValueError(
            "Structured revision and generated bindings have different source inventories"
        )
    result = proposal.model_copy(deep=True)
    replacements = {e.id: e for e in structured.entries if e.category != "prose"}
    result.entries = [replacements.get(e.id, e) for e in result.entries]
    from tau2.multilingual.factory.localization_artifact import (
        _automatic_user_ids,
        automatic_localization,
    )
    from tau2.multilingual.localization import SourceEntry

    normalized = []
    for entry in result.entries:
        source = SourceEntry.model_validate(
            entry.model_dump(include=set(SourceEntry.model_fields))
        )
        automatic = automatic_localization(source, result.settings)
        normalized.append(
            _automatic_user_ids(automatic, result.user_id_revision)
            if automatic is not None
            else entry
        )
    result.entries = normalized
    from tau2.multilingual.factory.localization_emails import (
        _is_email,
        localize_email_entries,
    )

    owners = {
        match[1]
        for e in result.entries
        if e.category == "identity_name"
        and (match := re.search(r"entity=([^;]+);", e.context))
    }
    email_sources = [
        e
        for e in result.source.entries
        if _is_email(e) and any(f"entity={owner};" in e.context for owner in owners)
    ]
    email_ids = {e.id for e in email_sources}
    refreshed_emails = {
        e.id: e
        for e in localize_email_entries(
            email_sources, [e for e in result.entries if e.id not in email_ids]
        )
    }
    result.entries = [refreshed_emails.get(e.id, e) for e in result.entries]
    protected = {k.casefold(): v for k, v in rules.protected_sources.items()}
    for e in result.entries:
        if e.category == "other" and e.source.casefold() in protected:
            reason = protected[e.source.casefold()]
            e.forms = LocalizedForms(
                native=e.source, romanized=e.source, english=reason
            )
            e.operation = "preserve"
            e.notes = reason
            e.reuse = None
    household = harmonize_banking_households(result)
    result = household.proposal
    entries = {e.id: e for e in result.entries}
    for b in result.prose_bindings:
        if b.source.casefold() in protected:
            b.kind = "protected"
            b.source_entry_id = None
            b.source_part = None
            b.provenance = None
            b.reason = protected[b.source.casefold()]
            b.forms = LocalizedForms(
                native=b.source, romanized=b.source, english=b.reason
            )
        elif b.kind == "reuse":
            ref = entries[b.source_entry_id]
            b.forms = (
                address_component(ref, b.source_part) if b.source_part else ref.forms
            )
    for b in result.prose_bindings:
        prose = entries[b.entry_id]
        if not re.search(
            r"daughter\s+" + re.escape(b.source) + r"\b", prose.source, re.I
        ):
            continue
        matches = [
            e
            for e in result.entries
            if e.category == "identity_name"
            and e.source.casefold().startswith(b.source.casefold() + " ")
            and _tasks(e) & _tasks(prose)
        ]
        if matches and len({(e.forms.native, e.forms.romanized) for e in matches}) == 1:
            reference = min(matches, key=lambda e: e.id)
            b.kind = "reuse"
            b.source_entry_id = reference.id
            b.source_part = None
            b.provenance = None
            b.forms = reference.forms
            b.reason = "Explicit daughter given-name alias resolves to the same task identity; retain exact complete localized name."
    result.prose_bindings = _unique_bindings(result.prose_bindings)
    return reconstruct_banking_prose(result), household


def _run_finalize_review(args):
    from tau2.multilingual.factory.database_localization import _finish_draft
    from tau2.multilingual.factory.localization_artifact import read_localization_map
    from tau2.multilingual.factory.localization_compile import (
        validate_localization_source,
    )
    from tau2.utils.artifacts import content_digest

    proposal = validate_localization_source(
        LocalizationMap.model_validate_json(args.map.read_text())
    )
    structured = validate_localization_source(
        LocalizationMap.model_validate_json(args.structured.read_text())
    )
    rules = BankingReviewRules.model_validate_json(args.rules.read_text())
    baseline = read_localization_map(args.baseline)
    input_map_sha256 = content_digest(proposal)
    discovery = DiscoveryCheckpoint.model_validate_json(args.discovery.read_text())
    proposal = _protect_foreign_fraud_identities(proposal, discovery, rules)
    reconstruction, household = synchronize_banking_candidate(
        proposal, structured, rules
    )
    result = reconstruction.proposal
    previous = {e.id: e for e in baseline.entries}
    calls = []
    for folder in args.history:
        calls.extend(
            EntityCheckpoint.model_validate_json(p.read_text())
            for p in sorted(folder.glob("*.json"))
        )
    record = BankingCandidateRevision(
        baseline_map_sha256=content_digest(baseline),
        input_map_sha256=input_map_sha256,
        output_map_sha256=content_digest(result),
        rules=rules,
        source_discovery=DiscoveryCheckpoint.model_validate_json(
            args.discovery.read_text()
        ),
        generation_calls=calls,
        revised_entries=[e.id for e in result.entries if e != previous[e.id]],
    )
    write_artifact(record, args.output.with_suffix(".revision.json"))
    write_artifact(household.revision, args.output.with_suffix(".households.json"))
    write_artifact(
        reconstruction.reconstruction, args.output.with_suffix(".reconstruction.json")
    )
    write_artifact(result, args.output.with_suffix(".snapshot.json"))
    _finish_draft(result, args.output)


def add_banking_finalize_review_args(commands):
    """Register deterministic synchronization and complete candidate lineage packaging."""
    parser = commands.add_parser("finalize-banking-candidate-review")
    for name in ("map", "structured", "rules", "baseline", "discovery", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--history", type=Path, action="append", required=True)
    parser.set_defaults(func=_run_finalize_review)


def _unique_bindings(bindings):
    unique = {}
    for binding in bindings:
        key = (binding.entry_id, binding.source.casefold())
        previous = unique.get(key)
        if previous and (previous.forms.native, previous.forms.romanized) != (
            binding.forms.native,
            binding.forms.romanized,
        ):
            raise ValueError("Conflicting bindings for the same exact source surface")
        unique.setdefault(key, binding)
    return list(unique.values())


def _protect_foreign_fraud_identities(proposal, discovery, rules):
    """Keep each remote fraud merchant together with its protected source geography."""
    from tau2.multilingual.localization import LocalizedForms

    aliases = {k.casefold(): v.casefold() for k, v in rules.source_aliases.items()}

    def canonical(value):
        return aliases.get(value.casefold(), value.casefold())

    entries = entries_by_id(proposal)
    protected = []
    for key, spans in discovery.entries.items():
        e = entries[key]
        if "role=transaction_description;" not in e.context:
            continue
        if not any(
            span.disposition == "preserve"
            and re.search(
                r"fraud|unauthorized|remote", span.role + " " + span.reason, re.I
            )
            for span in spans
        ):
            continue
        merchants = [
            span.source
            for span in spans
            if re.search(r"merchant|retail|restaurant|operator|store", span.role, re.I)
            and not re.search(r"city|location|country|state", span.role, re.I)
        ]
        for merchant in merchants:
            protected.append((_tasks(e), canonical(merchant), e.id))
    for b in proposal.prose_bindings:
        task = _tasks(entries[b.entry_id])
        evidence = next(
            (
                origin
                for tasks, name, origin in protected
                if task & tasks and canonical(b.source) == name
            ),
            None,
        )
        if evidence:
            b.kind = "protected"
            b.source_entry_id = None
            b.source_part = None
            b.provenance = None
            b.reason = (
                "Foreign fraud merchant identity retained with its source remote location; source evidence "
                + evidence
                + ". No target-country-chain overseas branch is invented."
            )
            b.forms = LocalizedForms(
                native=b.source, romanized=b.source, english=b.reason
            )
    for e in proposal.entries:
        if e.category == "other" and any(
            _tasks(e) & tasks and canonical(e.source) == name
            for tasks, name, _ in protected
        ):
            e.forms = LocalizedForms(
                native=e.source,
                romanized=e.source,
                english="Foreign fraud merchant identity preserved with its source location.",
            )
            e.notes = e.forms.english
            e.operation = "preserve"
            e.reuse = None
    return proposal


def street_premise(value: str) -> tuple[str, str]:
    """Separate the explicit premise from a street component, never its road number."""
    value = value.strip()
    for pattern in (
        r"^(.+?(?:路|街|巷|道|胡同))\s*([0-9]+(?:-[0-9]+)?)号.*$",
        r"^(.+? (?:Lu|Jie|Xiang|Dadao|Hutong))\s+([0-9]+(?:-[0-9]+)?) Hao(?:\s.*)?$",
        r"^(.+?(?:로|길))\s+([0-9]+(?:-[0-9]+)?)\s+[0-9]+(?:동\s+[0-9]+)?호$",
        r"^(.+?-(?:ro|gil))\s+([0-9]+(?:-[0-9]+)?)\s+[0-9]+(?:-dong\s+[0-9]+)?-ho$",
    ):
        marked = re.fullmatch(pattern, value, re.I)
        if marked:
            return marked[2], marked[1].strip()
    leading = re.fullmatch(r"([0-9]+(?:-[0-9]+)?)\s+(.+)", value)
    if leading:
        return leading[1], leading[2]
    hanzi = re.fullmatch(r"(.+?)([0-9]+(?:-[0-9]+)?)号", value)
    if hanzi:
        return hanzi[2], hanzi[1].strip()
    trailing = re.fullmatch(r"(.+?)\s+([0-9]+(?:-[0-9]+)?)(?:号|번지)?", value)
    if trailing:
        return trailing[2], trailing[1]
    raise ValueError(
        "Address street component requires an explicit space-separated premise number"
    )


def address_premise(parts) -> tuple[str, str]:
    """Read premise from explicit street lines, independent of full display order."""
    pieces = parts.street_lines
    if len(pieces) > 1 and re.fullmatch(r"[0-9]+(?:-[0-9]+)?", pieces[1]):
        return pieces[1], pieces[0]
    return street_premise(pieces[0])


def _run_inline_addresses(args):
    from tau2.multilingual.localization import GenerationProvenance, LocalizedForms
    from tau2.utils.artifacts import content_digest

    proposal = LocalizationMap.model_validate_json(args.map.read_text())
    rules = BankingReviewRules.model_validate_json(args.rules.read_text())
    rules.require(proposal)
    indexed = entries_by_id(proposal)
    requests = {}
    completed = []
    refs = {}
    for b in proposal.prose_bindings:
        if b.kind != "generated" or not re.match(r"^\d+\s+.+,", b.source):
            continue
        linked = {
            x.source_entry_id
            for x in proposal.prose_bindings
            if x.entry_id == b.entry_id
            and x.source_part in {"address_city", "address_city_region"}
        }
        if len(linked) != 1:
            continue
        ref = indexed[next(iter(linked))]
        key = json_content_digest({"source": b.source})
        if key in requests:
            continue
        refs[key] = ref
        if ref.forms.address is None:
            raise ValueError(
                "Inline address generation requires explicit address components"
            )
        streets = {}
        for rep in ("native", "romanized"):
            _, streets[rep] = address_premise(getattr(ref.forms.address, rep))
        components = {
            component: {
                rep: getattr(getattr(ref.forms.address, rep), component)
                for rep in ("native", "romanized")
            }
            for component in ("city", "region", "district", "postal_code", "country")
        }
        for prefix, forms in [
            ("street:", streets),
            *[(component + ":", forms) for component, forms in components.items()],
        ]:
            completed.append(
                {
                    "id": prefix + key,
                    "source": "Required existing local "
                    + prefix
                    + " for distinct premise",
                    "category": "address",
                    "forms": {
                        **forms,
                        "english": "Private exact local address components retained locally.",
                    },
                }
            )
        requests[key] = {
            "id": key,
            "source": b.source,
            "source_role_context": indexed[b.entry_id].source,
            "kind": "address",
            "context": "Generate a distinct synthetic premise/unit in the exact source-linked neighborhood. Return explicit forms.address components. Preserve the exact required street reference within street_lines and copy each required_address_references token into its named semantic component field. Do not invent other city, region, district or postcode values. Use a new premise number between 10 and 999 with the existing street in local road/premise order and a context-appropriate unit; preserve source office versus residential role. Full display order is locale-specific and rendered locally.",
            "required_street_reference": "street:" + key,
            "required_address_references": {
                component: component + ":" + key for component in components
            },
        }
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    reply = _revise_batch(
        list(requests.values()),
        completed,
        args.checkpoints,
        args.model,
        {"temperature": 0, "reasoning_effort": "medium"},
        rules,
    )
    result = proposal.model_copy(deep=True)
    for b in result.prose_bindings:
        key = json_content_digest({"source": b.source})
        if key in requests:
            forms = LocalizedForms.model_validate(reply.entries[key]["forms"])
            for rep in ("native", "romanized"):
                previous, _ = address_premise(getattr(refs[key].forms.address, rep))
                current, _ = address_premise(getattr(forms.address, rep))
                if normalized_phone(current) == normalized_phone(previous):
                    raise ValueError(
                        "Inline address must use a distinct synthetic premise"
                    )
            b.forms = forms
            b.reason = reply.entries[key]["notes"]
            b.provenance = GenerationProvenance(
                model=reply.model,
                model_args=reply.model_args,
                prompt_version=reply.prompt_version,
                call_name=rules.entity_call_name,
                input_sha256=reply.input_sha256,
                parent_map_sha256=content_digest(proposal),
            )
    write_artifact(result, args.output)


def add_banking_inline_address_args(commands):
    """Register exact local-component authoring for distinct inline premises."""
    parser = commands.add_parser("repair-banking-inline-addresses")
    for name in ("map", "rules", "output", "checkpoints"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.set_defaults(func=_run_inline_addresses)
