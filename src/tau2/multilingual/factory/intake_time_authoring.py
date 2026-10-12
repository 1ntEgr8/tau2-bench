"""Fixed-prompt, source-bound repairs of selected Hard intake time expressions."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated, Literal

from loguru import logger
from pydantic import Field, JsonValue, StringConstraints, model_validator

from tau2.config import (
    DEFAULT_FACTORY_LOCALIZATION_BATCH_SIZE,
    DEFAULT_FACTORY_LOCALIZATION_CONCURRENCY,
)
from tau2.data_model.message import SystemMessage, UserMessage
from tau2.multilingual.factory.intake_candidate_review import (
    LOCALE_RULES_VERSION,
    locale_rules,
)
from tau2.multilingual.factory.intake_difficulty_audit import (
    DifficultyAuditReport,
    DifficultyDecision,
)
from tau2.multilingual.factory.intake_difficulty_inventory import (
    DifficultyCase,
    DifficultyInventory,
    build_difficulty_inventory,
)
from tau2.multilingual.factory.localization_artifact import (
    read_localization_map,
    write_localization_map,
)
from tau2.multilingual.factory.localization_compile import validate_localization
from tau2.multilingual.localization import (
    GenerationProvenance,
    IntakeCallerSurface,
    LocalizationMap,
    LocalizationModel,
    RegionalSettings,
    json_content_digest,
)
from tau2.utils.artifacts import content_digest, write_artifact
from tau2.utils.llm_utils import extract_json_from_llm_response, generate

TIME_AUTHORING_PROMPT_VERSION = "intake-hard-time-authoring-v2"
TIME_AUTHORING_CALL_NAME = "intake_hard_time_authoring"
TIME_AUTHORING_PROMPT = """Author only the initial spoken time expressions for the exact Hard intake tasks supplied. This is candidate generation, NEVER native approval. Return exactly one surface per requested canonical_task_id and preserve its canonical_value byte-for-byte. Do not change any time, AM/PM truth, date, timezone, caller context, record, submitted answer, entity, decoy, or benchmark selection.

Read any source-bound review_feedback as factual repair concerns, not replacement wording or native approval. An independent audit decision can be absent when explicit review feedback binds this exact task and map. First decide feasibility: `feasible` requires an ordinary local expression with a real difficulty advantage over the Easy controls; `source_limited` means the fixed answer cannot naturally provide that advantage. In a source_limited case return the natural BASIC clock expression, with no manufactured offset arithmetic, and explain the limitation. Do not replace a simple fixed time with an unusual countdown such as fifty-five minutes before the next hour merely to make subtraction hard. A mathematically correct expression is not evidence of normal conversational usage. Keep limitation empty only when feasible; never claim native approval or change the source difficulty label.

For EACH case, read its independent audit correction_instructions and any source_blocked_reason, current native caller payload, private complete answer, clarification guidance and ALL TEN same-family Easy controls. A source_blocked finding permits only a conservative attempt at the best natural expression while preserving every stated invariant; it is NOT reclassified or cleared by this authoring step. A primary time repair means changing the initial spoken surface, NEVER the canonical clock value. Repair the cited difficulty loss with a natural local expression that requires meaningful time interpretation relative to those Easy cases. Merely using local script, removing AM/PM, or replacing a numeral with a number word does not establish Hard difficulty. A quarter/half expression is not automatically harder than the supplied Easy examples. Use normal local usage, not an awkward literal translation, invented idiom, deliberately convoluted arithmetic puzzle, or unsupported situational story. The expression must resolve to the exact supplied hour and minute; carefully verify hour rollover, twelve, and minutes-before arithmetic.

native_spoken is ONLY the initial expression in the target language and script. Omit AM/PM and morning/afternoon/evening/night words initially. Do not include a private annotation, full answer, quotation wrapper, meta-instruction, explanation or stress/pronunciation guide. The runtime, not you, appends the exact complete local time as a private anchor; the caller knows its day period and gives it when asked, and uses that full answer for read-back confirmation. romanized_spoken denotes precisely the same expression in ordinary Latin romanization and is review metadata, never a spoken native replacement. For Spanish and Portuguese use the ordinary native Latin spelling including accents in both forms.

Hindi: use natural native number/time expressions with correct local grammar, not romanized Hindi or imported stress guides. Spanish: use natural Spain usage and correct singular/plural hour grammar. Portuguese: use natural Brazilian usage, not Iberian calques. Mandarin: use natural Mandarin time interpretation, not literal foreign idioms or literary expressions selected merely to appear difficult. Korean: use natural Korean expressions, including conventional minutes-before phrasing where apt; do not calque a foreign quarter expression. These are constraints, not evidence that a particular expression is hard.

Write a concise English rationale naming the exact interpretation challenge, a specific contrast with the supplied Easy controls, and how asking for the day period resolves the remaining ambiguity. Never claim review approval or empirically proven difficulty. The same fixed canonical time must remain recoverable. If the fixed time cannot sustain any natural locally Hard wording, retain a correct natural expression and explicitly state that limitation in the rationale rather than inventing an invalid expression or claiming it Hard; the independent follow-up audit will determine whether the source selection needs review. Return only response-schema JSON."""

Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
TaskId = Annotated[str, Field(pattern=r"^intake_times_hard_[0-9]{2}$")]


class TimeReviewFeedback(LocalizationModel):
    """Supplemental factual concern bound to one task on an exact candidate map."""

    canonical_task_id: Annotated[TaskId, Field(description="Selected Hard time task.")]
    map_sha256: Annotated[
        str, Field(pattern=r"^[0-9a-f]{64}$", description="Exact reviewed map digest.")
    ]
    concern: Annotated[
        Nonempty,
        Field(
            description="Observed defect or feasibility concern, not replacement speech."
        ),
    ]


class IntakeTimeAuthoringRequest(LocalizationModel):
    """Explicit immutable inputs and authoring configuration for a bounded repair."""

    map_file: Annotated[
        Path, Field(description="Exact complete candidate map to revise.")
    ]
    audit_file: Annotated[
        Path | None,
        Field(description="Difficulty audit bound to this exact map and inventory."),
    ] = None
    review_feedback: Annotated[
        list[TimeReviewFeedback],
        Field(
            description="Exact-map factual review; required for every target without an audit."
        ),
    ] = Field(default_factory=list)
    canonical_task_ids: Annotated[
        list[TaskId],
        Field(
            min_length=1,
            max_length=10,
            description="Explicit Hard time tasks with caller-surface repair findings.",
        ),
    ]
    revision: Annotated[
        Nonempty, Field(description="New immutable draft-map revision.")
    ]
    model: Annotated[
        Nonempty,
        Field(description="Authoring model used by the shared generation seam."),
    ]
    model_args: Annotated[
        dict[str, JsonValue], Field(description="Recorded explicit model settings.")
    ]
    batch_size: Annotated[
        int,
        Field(ge=1, le=100, description="Maximum targeted tasks per authoring call."),
    ] = DEFAULT_FACTORY_LOCALIZATION_BATCH_SIZE
    max_concurrency: Annotated[
        int, Field(ge=1, le=10, description="Maximum concurrent authoring calls.")
    ] = DEFAULT_FACTORY_LOCALIZATION_CONCURRENCY

    @model_validator(mode="after")
    def distinct_targets(self):
        """Reject repeated work before making any model calls."""
        if len(set(self.canonical_task_ids)) != len(self.canonical_task_ids):
            raise ValueError("Hard time authoring requires distinct task IDs")
        reviewed = [row.canonical_task_id for row in self.review_feedback]
        if len(set(reviewed)) != len(reviewed) or not set(reviewed) <= set(
            self.canonical_task_ids
        ):
            raise ValueError("Time review feedback must uniquely target selected tasks")
        if self.audit_file is None and set(reviewed) != set(self.canonical_task_ids):
            raise ValueError(
                "Without an audit every selected time needs review feedback"
            )
        return self


class TimeAuthoringCase(LocalizationModel):
    """Exact measured time evidence with its source-bound independent repair finding."""

    case: Annotated[
        DifficultyCase,
        Field(
            description="One Hard time and all ten Easy controls; no caller identity."
        ),
    ]
    decision: Annotated[
        DifficultyDecision | None,
        Field(description="Exact independent audit correction instructions."),
    ] = None
    review_feedback: Annotated[
        TimeReviewFeedback | None,
        Field(description="Supplemental concern, never an invented audit decision."),
    ] = None

    @model_validator(mode="after")
    def bound_correction(self):
        """Only explicit time caller-surface repairs may enter this authoring path."""
        if (
            self.case.family != "times"
            or (
                self.decision is not None
                and self.case.canonical_task_id != self.decision.canonical_task_id
            )
            or (
                self.review_feedback is not None
                and self.case.canonical_task_id
                != self.review_feedback.canonical_task_id
            )
        ):
            raise ValueError(
                "Time authoring correction differs from its Hard time case"
            )
        if self.review_feedback is not None:
            return self
        if self.decision is None or not (
            self.decision.disposition == "source_blocked"
            or (
                self.decision.disposition == "revise"
                and {"caller_surface", "primary", "omission_instruction"}
                & set(self.decision.repair_targets)
            )
        ):
            raise ValueError(
                "Time authoring requires a revise caller_surface/primary/omission_instruction finding or a source blocker"
            )
        return self


class TimeAuthoringPayload(LocalizationModel):
    """Safe selected benchmark content and exact binding hashes sent to the model."""

    map_sha256: Annotated[str, Field(description="Complete parent-map digest.")]
    source_sha256: Annotated[str, Field(description="Frozen source inventory digest.")]
    inventory_sha256: Annotated[
        str, Field(description="Full difficulty evidence digest.")
    ]
    audit_sha256: Annotated[
        str | None,
        Field(description="Exact audit digest; null for explicitly reviewed targets."),
    ]
    settings: Annotated[
        RegionalSettings, Field(description="Target language and country.")
    ]
    locale_rules_version: Annotated[
        str, Field(description="Version of fixed locale constraints.")
    ]
    locale_rules: Annotated[
        str, Field(description="Country-specific naturalness and validity constraints.")
    ]
    cases: Annotated[
        list[TimeAuthoringCase],
        Field(min_length=1, description="Every task in this exact bounded batch."),
    ]


class AuthoredTimeSurface(LocalizationModel):
    """Model-authored speech only; generation provenance is assigned locally."""

    canonical_task_id: Annotated[
        TaskId, Field(description="Exact requested canonical task ID.")
    ]
    canonical_value: Annotated[
        str,
        Field(
            pattern=r"^(1[0-2]|[1-9]):[0-5][0-9] (AM|PM)$",
            description="Exact unchanged complete source time.",
        ),
    ]
    native_spoken: Annotated[
        Nonempty,
        Field(
            description="Natural initial target-script expression without day period."
        ),
    ]
    romanized_spoken: Annotated[
        Nonempty,
        Field(description="Ordinary Latin romanization of that same expression."),
    ]
    rationale: Annotated[
        Nonempty,
        Field(
            description="Specific local interpretation challenge relative to Easy controls; pending native review."
        ),
    ]
    feasibility: Annotated[
        Literal["feasible", "source_limited"],
        Field(
            description="Whether naturally Hard speech is possible for the fixed answer."
        ),
    ]
    limitation: Annotated[
        str,
        Field(description="Specific source limitation; empty only when feasible."),
    ]

    @model_validator(mode="after")
    def explicit_feasibility(self):
        """Require an honest limitation rather than silently claiming Hard success."""
        if (self.feasibility == "source_limited") != bool(self.limitation.strip()):
            raise ValueError(
                "Source-limited time requires a limitation; feasible time must not have one"
            )
        return self


class TimeAuthoringBatch(LocalizationModel):
    """Strict model response with exact ID/value coverage checked locally."""

    surfaces: Annotated[
        list[AuthoredTimeSurface],
        Field(description="Exactly one candidate surface per supplied task."),
    ]


class TimeAuthoringCheckpoint(LocalizationModel):
    """Successful response bound to exact safe evidence and explicit configuration."""

    input_sha256: Annotated[
        str,
        Field(
            pattern=r"^[0-9a-f]{64}$",
            description="Digest of request, payload and versioned prompt.",
        ),
    ]
    request: Annotated[
        IntakeTimeAuthoringRequest, Field(description="Recorded authoring settings.")
    ]
    prompt_version: Annotated[
        str, Field(description="Version of the fixed authoring prompt.")
    ]
    response: Annotated[
        TimeAuthoringBatch, Field(description="Complete source-checked model response.")
    ]


def _payload(
    request: IntakeTimeAuthoringRequest,
    inventory: DifficultyInventory,
    audit: DifficultyAuditReport | None,
) -> TimeAuthoringPayload:
    cases = {case.canonical_task_id: case for case in inventory.cases}
    decisions = (
        {decision.canonical_task_id: decision for decision in audit.decisions}
        if audit is not None
        else {}
    )
    feedback = {row.canonical_task_id: row for row in request.review_feedback}
    if any(row.map_sha256 != inventory.map_sha256 for row in feedback.values()):
        raise ValueError("Time review feedback is stale for the exact map")
    if any(
        identifier not in cases
        or (identifier not in decisions and identifier not in feedback)
        for identifier in request.canonical_task_ids
    ):
        raise ValueError(
            "Every requested Hard time must have a source-bound audit decision"
        )
    return TimeAuthoringPayload(
        map_sha256=inventory.map_sha256,
        source_sha256=inventory.source_sha256,
        inventory_sha256=content_digest(inventory),
        audit_sha256=content_digest(audit) if audit is not None else None,
        settings=inventory.settings,
        locale_rules_version=LOCALE_RULES_VERSION,
        locale_rules=locale_rules(inventory.settings),
        cases=[
            TimeAuthoringCase(
                case=cases[identifier],
                decision=decisions.get(identifier),
                review_feedback=feedback.get(identifier),
            )
            for identifier in sorted(request.canonical_task_ids)
        ],
    )


def _input_digest(
    request: IntakeTimeAuthoringRequest, payload: TimeAuthoringPayload
) -> str:
    return json_content_digest(
        {
            "request": request.model_dump(mode="json"),
            "payload": payload.model_dump(mode="json"),
            "prompt_version": TIME_AUTHORING_PROMPT_VERSION,
        }
    )


def _provenance(
    request: IntakeTimeAuthoringRequest, digest: str, parent: str
) -> GenerationProvenance:
    return GenerationProvenance(
        model=request.model,
        model_args=request.model_args,
        prompt_version=TIME_AUTHORING_PROMPT_VERSION,
        call_name=TIME_AUTHORING_CALL_NAME,
        input_sha256=digest,
        parent_map_sha256=parent,
    )


def _surfaces(
    request: IntakeTimeAuthoringRequest,
    proposal: LocalizationMap,
    payload: TimeAuthoringPayload,
    response: TimeAuthoringBatch,
) -> list[IntakeCallerSurface]:
    expected = {
        row.case.canonical_task_id: row.case.source_value for row in payload.cases
    }
    if sorted(row.canonical_task_id for row in response.surfaces) != sorted(expected):
        raise ValueError("Time authoring must return every requested task exactly once")
    if any(
        row.canonical_value != expected[row.canonical_task_id]
        for row in response.surfaces
    ):
        raise ValueError("Time authoring changed an immutable canonical value")
    period_words = {
        "hi": r"सुबह|दोपहर|शाम|रात",
        "es": r"\b(?:mañana|tarde|noche|madrugada|mediodía|medianoche)\b",
        "pt": r"\b(?:manhã|tarde|noite|madrugada|meio-dia|meia-noite)\b",
        "zh": r"凌晨|上午|中午|下午|晚上|午夜|早上|傍晚|夜里",
        "ko": r"오전|오후|아침|저녁|밤|낮|새벽",
    }[proposal.settings.language]
    if any(
        re.search(period_words, row.native_spoken, re.IGNORECASE)
        or re.search(r"\b[ap]\.?\s*m\.?\b", row.native_spoken, re.IGNORECASE)
        for row in response.surfaces
    ):
        raise ValueError("Initial authored time expression must omit its day period")
    provenance = _provenance(
        request, _input_digest(request, payload), payload.map_sha256
    )
    authored = [
        IntakeCallerSurface(
            **row.model_dump(exclude={"feasibility", "limitation"}),
            provenance=provenance,
        )
        for row in response.surfaces
    ]
    # The map boundary supplies locale-script and frozen source-truth checks for
    # both new responses and resumed checkpoints before either can be accepted.
    LocalizationMap.model_validate(
        {
            **proposal.model_dump(),
            "intake_caller_surfaces": [
                *[
                    row.model_dump()
                    for row in proposal.intake_caller_surfaces
                    if row.canonical_task_id not in expected
                ],
                *[row.model_dump() for row in authored],
            ],
        }
    )
    return authored


def _read_time_checkpoint(
    request: IntakeTimeAuthoringRequest,
    proposal: LocalizationMap,
    payload: TimeAuthoringPayload,
    checkpoints: Path,
) -> list[IntakeCallerSurface]:
    """Require completed exact-input evidence without generating missing work."""
    digest = _input_digest(request, payload)
    path = checkpoints / f"{digest}.json"
    if not path.is_file():
        raise ValueError(f"Missing completed time authoring checkpoint: {path}")
    saved = TimeAuthoringCheckpoint.model_validate_json(path.read_text())
    if (
        saved.input_sha256 != digest
        or saved.request != request
        or saved.prompt_version != TIME_AUTHORING_PROMPT_VERSION
    ):
        raise ValueError(
            "Time authoring checkpoint differs from exact source/model/prompt input"
        )
    return _surfaces(request, proposal, payload, saved.response)


def author_time_batch(
    request: IntakeTimeAuthoringRequest,
    proposal: LocalizationMap,
    payload: TimeAuthoringPayload,
    checkpoints: Path,
) -> list[IntakeCallerSurface]:
    """Generate one batch or reuse its exact-input, fully checked checkpoint."""
    digest = _input_digest(request, payload)
    path = checkpoints / f"{digest}.json"
    if path.exists():
        return _read_time_checkpoint(request, proposal, payload, checkpoints)
    reply = generate(
        model=request.model,
        messages=[
            SystemMessage(role="system", content=TIME_AUTHORING_PROMPT),
            UserMessage(role="user", content=payload.model_dump_json()),
        ],
        call_name=TIME_AUTHORING_CALL_NAME,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": TIME_AUTHORING_CALL_NAME,
                "strict": True,
                "schema": TimeAuthoringBatch.model_json_schema(),
            },
        },
        **request.model_args,
    )
    response = TimeAuthoringBatch.model_validate_json(
        extract_json_from_llm_response(reply.content or "")
    )
    surfaces = _surfaces(request, proposal, payload, response)
    write_artifact(
        TimeAuthoringCheckpoint(
            input_sha256=digest,
            request=request,
            prompt_version=TIME_AUTHORING_PROMPT_VERSION,
            response=response,
        ),
        path,
    )
    return surfaces


def revise_intake_time_surfaces(
    request: IntakeTimeAuthoringRequest, output: Path, checkpoints: Path
) -> LocalizationMap:
    """Repair only named Hard time caller surfaces, validating before writing a draft."""
    request = IntakeTimeAuthoringRequest.model_validate(request.model_dump())
    proposal = read_localization_map(request.map_file)
    if proposal.source.domain != "intake" or proposal.settings.language not in {
        "hi",
        "es",
        "pt",
        "zh",
        "ko",
    }:
        raise ValueError(
            "Hard time authoring requires a supported localized intake map"
        )
    inventory = build_difficulty_inventory(proposal)
    audit = (
        DifficultyAuditReport.model_validate_json(request.audit_file.read_text())
        if request.audit_file is not None
        else None
    )
    if audit is not None and (
        audit.map_sha256 != content_digest(proposal)
        or audit.source_sha256 != content_digest(proposal.source)
        or audit.inventory_sha256 != content_digest(inventory)
        or audit.settings != proposal.settings
        or set(audit.expected_task_ids)
        != {row.canonical_task_id for row in inventory.cases}
    ):
        raise ValueError(
            "Time authoring audit is stale or differs from the exact map/source/inventory"
        )
    if request.revision == proposal.revision:
        raise ValueError("Time authoring requires a new immutable map revision")
    payload = _payload(request, inventory, audit)
    provenance = _provenance(
        request, _input_digest(request, payload), content_digest(proposal)
    )
    targets = set(request.canonical_task_ids)
    retained = sorted(
        (
            row
            for row in proposal.intake_caller_surfaces
            if row.canonical_task_id not in targets
        ),
        key=lambda row: row.canonical_task_id,
    )
    batches = [
        payload.model_copy(
            update={"cases": payload.cases[index : index + request.batch_size]}
        )
        for index in range(0, len(payload.cases), request.batch_size)
    ]
    existing_output = output.exists()
    authored: list[IntakeCallerSurface] = []
    failures: list[str] = []
    if existing_output:
        # A saved map cannot stand in for generation evidence, including the
        # feasibility/limitation verdict retained only in each checkpoint.
        for batch in batches:
            authored.extend(
                _read_time_checkpoint(request, proposal, batch, checkpoints)
            )
    else:
        with ThreadPoolExecutor(max_workers=request.max_concurrency) as pool:
            futures = {
                pool.submit(
                    author_time_batch, request, proposal, batch, checkpoints
                ): batch
                for batch in batches
            }
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    authored.extend(future.result())
                except Exception as error:
                    identifiers = ", ".join(
                        row.case.canonical_task_id for row in batch.cases
                    )
                    failure = f"{identifiers}: {error}"
                    failures.append(failure)
                    logger.error("Hard time authoring failed for {}", failure)
    if failures:
        raise ValueError(
            "Hard time authoring incomplete; successful checkpoints retained: "
            + "; ".join(sorted(failures))
        )
    revised = proposal.model_copy(
        update={
            "revision": request.revision,
            "provenance": provenance,
            "intake_caller_surfaces": sorted(
                [*retained, *authored], key=lambda row: row.canonical_task_id
            ),
        }
    )
    revised = validate_localization(revised)
    if existing_output:
        saved = read_localization_map(output)
        if saved != revised:
            raise ValueError(
                "Existing time-authoring output differs from exact request and completed checkpoints; use a new output path"
            )
        return saved
    write_localization_map(revised, output)
    return revised
