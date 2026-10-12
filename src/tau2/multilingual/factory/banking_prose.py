"""Fixed supplemental entity drafting and exact English reconstruction CLI."""

import json
import re
from pathlib import Path

from pydantic import Field, JsonValue

from tau2.data_model.message import SystemMessage, UserMessage
from tau2.domains.banking_knowledge.localization_prose import (
    _owner,
    _references,
    _tasks,
    reconstruct_banking_prose,
)
from tau2.domains.banking_knowledge.localization_tokens import banking_response_model
from tau2.multilingual.localization import (
    GenerationProvenance,
    LocalizationMap,
    LocalizationModel,
    LocalizedForms,
    ProseEntityBinding,
    json_content_digest,
)
from tau2.utils.artifacts import content_digest, write_artifact

VERSION = "banking-incidental-entities-v1"
PROMPT = """Localize only the requested incidental banking entities into the target language and country. These are public synthetic benchmark names and businesses. Produce plausible local given names for incidental people, full names for full-name entities, and local company names retaining the source industry and organizational meaning. For companies named after a person, a plausible local company name is sufficient; never alter financial conditions. Native and romanized forms must identify exactly the same entity. Keep romanized forms in Latin script. Do not add addresses, country prefixes, explanations or punctuation to entity values. The English form explains the candidate and its source role. All candidates remain drafts for native review. Return exactly every requested ID once using the supplied schema. Never generate instruction prose."""


class BindingDraft(LocalizationModel):
    """Checkpoint with actual call lineage and exact source-bound candidates."""

    parent_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Exact source-checked parent map."
    )
    model: str = Field(min_length=1, description="Actual entity authoring model.")
    model_args: dict[str, JsonValue] = Field(description="Recorded provider settings.")
    prompt_version: str = VERSION
    bindings: list[ProseEntityBinding] = Field(
        description="Source-bound candidates requiring native review."
    )


def _source_candidates(proposal):
    entities = [
        e
        for e in proposal.entries
        if e.category in {"identity_name", "address", "contact", "other"}
    ]
    ready, pending = [], []
    for entry in proposal.entries:
        if entry.category != "prose" or not isinstance(entry.source, str):
            continue
        text = entry.source
        chosen, ambiguous, _ = _references(entry, entities, "native")
        candidates = []
        for match in re.finditer(r'company called "([^"]+)"', text):
            candidates.append((match[1], "company"))
        for match in re.finditer(r"(?:worker named|someone named) ([A-Z][a-z]+)", text):
            candidates.append((match[1], "incidental given name"))
        match = re.match(r"\*\*Your character:\*\* ([A-Z][a-z]+ [A-Z][a-z]+),", text)
        if match and match[1].casefold() not in chosen:
            candidates.append((match[1], "full name"))
        if text.startswith("DIRECT DEPOSIT - "):
            value = text.removeprefix("DIRECT DEPOSIT - ").removesuffix(" PAYROLL")
            if (
                value
                not in {
                    "PAYROLL",
                    "DIVIDEND INCOME",
                    "PART TIME INCOME",
                    "STUDENT LOAN DISBURSEMENT",
                }
                and value.casefold() not in chosen
            ):
                candidates.append((value, "employer company"))
        for match in re.finditer(r"(?:FROM|TO) ([A-Z]\. [A-Z]+)", text):
            candidates.append((match[1], "abbreviated personal payee name"))
        for source, role in candidates:
            if source.casefold() not in chosen:
                pending.append((entry.id, source, role))
        protected = []
        if "bypass code" in text and "9K2X7M4P1N8Q3R5T6A" in text:
            protected.append(
                (
                    "9K2X7M4P1N8Q3R5T6A",
                    "Authentication bypass token remains a protocol value.",
                )
            )
        if "Seattle, Washington" in text:
            protected.append(
                (
                    "Seattle, Washington",
                    "Source geography remains unchanged; Washington here is not a surname.",
                )
            )
        for source, reason in protected:
            ready.append(
                ProseEntityBinding(
                    entry_id=entry.id,
                    source=source,
                    forms=LocalizedForms(
                        native=source, romanized=source, english=reason
                    ),
                    kind="protected",
                    reason=reason,
                )
            )
        # The explicit verification Name field identifies whose shared address the caller uses.
        name = re.search(r"- Name: ([^|\n]+)\s*\|", text)
        if name:
            owners = {
                _owner(e)
                for e in entities
                if e.category == "identity_name"
                and e.source == name[1].strip()
                and _tasks(entry) & _tasks(e)
            }
            for address in entities:
                if (
                    address.category == "address"
                    and address.source.casefold() in ambiguous
                    and _owner(address) in owners
                    and _tasks(entry) & _tasks(address)
                ):
                    ready.append(
                        ProseEntityBinding(
                            entry_id=entry.id,
                            source=address.source,
                            forms=address.forms,
                            kind="reuse",
                            source_entry_id=address.id,
                            reason="Explicit caller verification name identifies the address owner.",
                        )
                    )
    return ready, pending


def draft_bindings(
    proposal: LocalizationMap, output: Path, model: str, model_args: dict
) -> BindingDraft:
    """Generate only missing incidental entities; no donor or full prose leaves the process."""
    digest = content_digest(proposal)
    if output.exists():
        existing = BindingDraft.model_validate_json(output.read_text())
        if (
            existing.parent_sha256 != digest
            or existing.model != model
            or existing.model_args != model_args
            or existing.prompt_version != VERSION
        ):
            raise ValueError("Binding checkpoint belongs to a different parent map")
        return existing
    ready, pending = _source_candidates(proposal)
    groups = {}
    for entry_id, source, role in pending:
        key = json_content_digest({"source": source, "role": role})
        groups.setdefault(key, {"source": source, "role": role, "parents": []})[
            "parents"
        ].append(entry_id)
    if groups:
        from tau2.utils.llm_utils import generate

        response_model = banking_response_model(list(groups))
        payload = {
            "language": proposal.settings.language,
            "country": proposal.settings.country_code,
            "entries": [
                {"id": key, "source": value["source"], "role": value["role"]}
                for key, value in groups.items()
            ],
        }
        kwargs = {
            **model_args,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "banking_incidental_entities",
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            },
        }
        message = generate(
            model=model,
            messages=[
                SystemMessage(role="system", content=PROMPT),
                UserMessage(
                    role="user", content=json.dumps(payload, ensure_ascii=False)
                ),
            ],
            call_name="banking_incidental_entities",
            **kwargs,
        )
        response = response_model.model_validate_json(message.content).model_dump()[
            "entries"
        ]
        provenance = GenerationProvenance(
            model=model,
            model_args=model_args,
            prompt_version=VERSION,
            call_name="banking_incidental_entities",
            input_sha256=json_content_digest(payload),
        )
        for key, group in groups.items():
            for entry_id in group["parents"]:
                ready.append(
                    ProseEntityBinding(
                        entry_id=entry_id,
                        source=group["source"],
                        forms=LocalizedForms.model_validate(response[key]["forms"]),
                        kind="generated",
                        reason=f"Source-extracted {group['role']}; native review pending.",
                        provenance=provenance,
                    )
                )
    result = BindingDraft(
        parent_sha256=digest, model=model, model_args=model_args, bindings=ready
    )
    write_artifact(result, output)
    return result


def _load(path):
    from tau2.multilingual.factory.database_localization import FailedRegionalDraft
    from tau2.multilingual.factory.localization_artifact import read_localization_map

    data = json.loads(path.read_text())
    if data.get("schema_version") == "regional-localization-failed-v1":
        return FailedRegionalDraft.model_validate(data).proposal
    if data.get("schema_version") == "regional-localization-v1":
        return LocalizationMap.model_validate(data)
    return read_localization_map(path)


def _run(args):
    from tau2.multilingual.factory.database_localization import _finish_draft

    proposal = _load(args.map)
    from tau2.domains.banking_knowledge.localization_households import (
        harmonize_banking_households,
    )

    household = harmonize_banking_households(proposal)
    write_artifact(household.revision, args.output.with_suffix(".households.json"))
    proposal = household.proposal
    bindings = draft_bindings(
        proposal,
        args.bindings,
        args.model,
        {"temperature": 0, "reasoning_effort": "low"},
    )
    proposal = proposal.model_copy(update={"prose_bindings": bindings.bindings})
    from tau2.domains.banking_knowledge.localization_bindings import (
        bind_explicit_person_aliases,
    )

    proposal = bind_explicit_person_aliases(proposal)
    result = reconstruct_banking_prose(proposal)
    write_artifact(
        result.reconstruction, args.output.with_suffix(".reconstruction.json")
    )
    write_artifact(result.proposal, args.output.with_suffix(".snapshot.json"))
    _finish_draft(result.proposal, args.output)


def add_banking_prose_args(commands):
    """Register complete candidate reconstruction within the factory CLI."""
    from tau2.multilingual.factory.banking_candidate_review import (
        add_banking_candidate_review_args,
    )

    add_banking_candidate_review_args(commands)
    parser = commands.add_parser(
        "repair-banking-prose",
        help="Draft source-bound incidental entities and reconstruct exact English instructions.",
    )
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.set_defaults(func=_run)
