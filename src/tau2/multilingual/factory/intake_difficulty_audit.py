"""Exhaustive, source-bound difficulty audits of the selected intake Hard frame."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
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
from tau2.multilingual.factory.intake_difficulty_inventory import (
    DifficultyCase,
    DifficultyInventory,
    build_difficulty_inventory,
)
from tau2.multilingual.factory.localization_artifact import read_localization_map
from tau2.multilingual.localization import (
    LocalizationModel,
    RegionalSettings,
    json_content_digest,
)
from tau2.utils.artifacts import content_digest, write_artifact
from tau2.utils.llm_utils import extract_json_from_llm_response, generate

AUDIT_PROMPT_VERSION = "intake-difficulty-audit-v2"
AUDIT_PROMPT = """Audit every supplied Hard intake case systematically. This is an independent model assessment, NEVER native-speaker approval. Return exactly one decision per canonical_task_id, with no missing, duplicate or extra cases. The English Hard label is NOT evidence of actual difficulty. Compare the English source to the English forms of the same-family Easy controls and assess source difficulty separately, then assess whether the localized native-script caller experience is naturally Hard relative to the TEN supplied Easy cases from the SAME family. Cite concrete localized text, the specific capture hazard, the relevant Easy comparators by task ID, and the clarification that resolves the ambiguity. Generic claims such as 'complex', 'long', 'unfamiliar' or 'preserves difficulty' are insufficient.

Read ALL supplied evidence: written answers, initial caller payload, privately known full answer, caller clarification/time guidance, decoys, pronunciation examples, and their romanized metadata. Judge actual initial speech and agent capture, not merely visual length or an inherited label. A caller must know the canonical answer and enough context to clarify unambiguously. Do not confuse intentionally omitted AM/PM with a caller who does not know the day period. Romanized metadata must identify the same entity; actual native speech must use the target language/script, without English stress guides. Written decoys may be homophones if their required normalized answers differ; intentional mispronunciations must be audibly different, not spelling-only variants. A coherent primary with broken dependencies still needs revise.

Dates: changing numeric display is not enough. Assess natural relative-calendar reasoning, ordinal weekdays, offsets and month/year boundaries against Easy examples; preserve the exact underlying date and private canonical answer. Times: a targeted caller_surface repair can replace an inadequate formatter output with a natural native expression while keeping the canonical time unchanged; the current formatter lacking a phrase is NOT a source blocker. Assess natural local number/time interpretation and day-period clarification against Easy; quarter-past alone is not automatically Hard. Phones: preserve meaningful international-prefix/extension/repeated-digit/grouping challenges with valid local shapes; ordinary domestic conversion must not erase source difficulty. Names, properties and companies: require credible local identities with real spelling, phonetic or component-boundary difficulty, never nonsense, unnatural hyphens, arbitrary length or imported syllables. Addresses: require coherent local geography and meaningful building/block/floor/unit/locality capture. Emails: preserve punctuation/digit/letter ambiguity and native symbol names with valid ASCII written addresses. Codes/plates: local format constraints must hold; never claim ambiguity for characters forbidden in that position. Medicines: conventional pharmaceutical transliterations are not themselves a defect, but assess their actual capture difficulty. Preserve ingredient, strength, units, formulation and clinical identity; assess natural pronunciation, units and decimals without changing a dose or selecting another drug.

Disposition keep requires localized_difficulty=hard, naturalness=true, known_answer_recoverable=true, concrete hazards with text evidence AND resolving clarification, a specific comparison_to_easy, and no dependency issues. Otherwise choose revise with precise correction instructions (not invented production replacement values) and repair_targets naming exactly what must change: primary, caller_surface, decoys, pronunciations, spelling_styles, omission_instruction. Use caller_surface for initial time expression plus private full-time anchor, never change canonical time. A date primary can revise its relative phrase only while preserving the exact canonical date annotation. Choose source_blocked only when the fixed source semantics/protocol cannot support a natural Hard realization without changing the frozen benchmark selection, date/time, clinical identity/dose, checksum or other invariant; explain why. An uncertain local judgment can NEVER be keep. Distinguish benchmark-source limitations from repairable localized defects. All outcomes still require native-speaker difficulty and speech review; native_review_required is always true. Return only the response-schema JSON."""

# These are representation/policy clarifications, not waivers of Hard evidence.
AUDIT_PROMPT += """

Representation and source constraints: Empty spelling_styles entries are intentional no-op placeholders under the fixed-shape localization contract when a source-only style does not apply. They are not malformed and are not a defect by themselves; never invent difficulty to fill them. English zero-as-oh or double/triple speech is NOT universally required: judge the supplied locale guidance and natural local practice rather than importing English habits.

Canonical ASCII phone/email/tool representations are not literal speech transcripts. Written ext., x, +, email punctuation and domain syntax can remain conventional protocol metadata; their presence alone is NOT English spoken leakage or a naturalness defect. A caller should render these symbols and extension markers naturally in the target language. Require correction only when actual speech instructions/examples are wrong, ambiguous or inconsistent, while preserving the written protocol contract. Native pronunciation token/reference/distortion examples still require the correct target script and real audible contrasts.

For dates, the immutable constraint is the exact canonical date under the supplied pinned clock. The source relative phrase or relation, such as 'tomorrow', is NOT itself immutable: it may be replaced by a natural harder equivalent that resolves to the same exact date. Do not mark source_blocked merely because translating the current English phrase preserves an Easy expression. Likewise, time caller expressions may change while the full canonical time and privately known day period remain fixed; formatter limits are not source blockers. If no natural Hard realization exists under the actual fixed constraints, retain source_blocked with specific evidence rather than adding awkward wording or arbitrary errors.

The coined family includes companies, employers and aftermarket brands. Require a credible local invented identity with a concrete spelling, sound or component-boundary hazard relative to the supplied Easy cases. A familiar generic business suffix, transliteration or arbitrary length alone does not establish Hard difficulty.
"""


Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Difficulty = Literal["easy", "hard", "uncertain"]
HazardTag = Literal[
    "lexical_spelling",
    "phonetic",
    "component_boundary",
    "multi_component",
    "numeric_grouping",
    "number_interpretation",
    "relative_temporal_reasoning",
    "dayperiod_clarification",
    "script_clarification",
]


class DifficultyAuditRequest(LocalizationModel):
    """Recorded runtime settings for a complete, resumable single-locale audit."""

    map_file: Annotated[Path, Field(description="Exact candidate localization map.")]
    revision: Annotated[Nonempty, Field(description="User-visible audit revision.")]
    model: Annotated[
        Nonempty, Field(description="Model passed to the shared LLM seam.")
    ]
    model_args: Annotated[
        dict[str, JsonValue],
        Field(description="Recorded explicit generation settings."),
    ]
    batch_size: Annotated[
        int, Field(ge=1, le=100, description="Number of Hard cases in each model call.")
    ] = DEFAULT_FACTORY_LOCALIZATION_BATCH_SIZE
    max_concurrency: Annotated[
        int, Field(ge=1, le=10, description="Maximum concurrent audit calls.")
    ] = DEFAULT_FACTORY_LOCALIZATION_CONCURRENCY


class DifficultyHazard(LocalizationModel):
    """A concrete capture hazard and the information needed to resolve it."""

    tag: Annotated[
        HazardTag, Field(description="Closed capture difficulty vocabulary.")
    ]
    evidence: Annotated[
        Nonempty,
        Field(description="Exact localized text and its specific capture risk."),
    ]
    resolving_clarification: Annotated[
        Nonempty, Field(description="Natural clarification yielding the known answer.")
    ]


class DifficultyDecision(LocalizationModel):
    """One accountable disposition for one frozen Hard task, never an approval."""

    canonical_task_id: Annotated[
        Nonempty, Field(description="Exact canonical task identifier from the payload.")
    ]
    disposition: Annotated[
        Literal["keep", "revise", "source_blocked"],
        Field(
            description="Keep a demonstrated Hard case, repair it, or flag source limits."
        ),
    ]
    source_difficulty: Annotated[
        Difficulty,
        Field(description="Independent source-English assessment, not its label."),
    ]
    source_assessment: Annotated[
        Nonempty,
        Field(description="Evidence for source difficulty and any source limitation."),
    ]
    localized_difficulty: Annotated[
        Difficulty,
        Field(description="Local difficulty relative to same-family Easy cases."),
    ]
    naturalness: Annotated[
        bool,
        Field(
            description="Whether the localized entity and caller speech are natural."
        ),
    ]
    known_answer_recoverable: Annotated[
        bool,
        Field(
            description="Whether caller knowledge and clarification recover exact gold."
        ),
    ]
    hazards: Annotated[
        list[DifficultyHazard], Field(description="Concrete localized capture hazards.")
    ]
    comparison_to_easy: Annotated[
        Nonempty,
        Field(description="Specific comparative evidence citing supplied Easy IDs."),
    ]
    dependency_issues: Annotated[
        list[Nonempty],
        Field(description="Defects in linked decoys, speech or guidance."),
    ]
    correction_instructions: Annotated[
        list[Nonempty],
        Field(description="Repair constraints, not authored production values."),
    ]
    repair_targets: Annotated[
        list[
            Literal[
                "primary",
                "caller_surface",
                "decoys",
                "pronunciations",
                "spelling_styles",
                "omission_instruction",
            ]
        ],
        Field(description="Exact linked pieces requiring repair, empty for keep."),
    ]
    source_blocked_reason: Annotated[
        str | None,
        Field(
            description="Why immutable source semantics prevent a natural Hard case."
        ),
    ]
    native_review_required: Annotated[
        Literal[True],
        Field(description="Model audit never substitutes for native review."),
    ]

    @model_validator(mode="after")
    def require_supported_disposition(self):
        """Do not silently certify weak, incoherent or uncertain candidates as Hard."""
        if len(set(self.repair_targets)) != len(self.repair_targets):
            raise ValueError("Repair targets must be distinct")
        if not self.comparison_to_easy.strip():
            raise ValueError("Difficulty comparison must contain evidence")
        if self.disposition == "keep":
            if not (
                self.localized_difficulty == "hard"
                and self.naturalness
                and self.known_answer_recoverable
                and self.hazards
                and not self.dependency_issues
                and not self.correction_instructions
                and not self.repair_targets
                and self.source_blocked_reason is None
            ):
                raise ValueError(
                    "keep requires natural recoverable Hard evidence and no open issues"
                )
        elif self.disposition == "revise":
            if (
                not self.correction_instructions
                or not self.repair_targets
                or self.source_blocked_reason is not None
            ):
                raise ValueError(
                    "revise requires correction instructions, repair targets and no source blocker"
                )
        elif not self.source_blocked_reason or not self.source_blocked_reason.strip():
            raise ValueError(
                "source_blocked requires an explicit immutable-source reason"
            )
        return self


class DifficultyAuditBatch(LocalizationModel):
    """Strict model response, subsequently checked against the exact requested IDs."""

    decisions: Annotated[
        list[DifficultyDecision],
        Field(description="Exactly one decision per supplied case."),
    ]


class DifficultyAuditPayload(LocalizationModel):
    """Only selected public benchmark evidence; donor identities are not included."""

    map_sha256: Annotated[str, Field(description="Digest of the full parent map.")]
    source_sha256: Annotated[str, Field(description="Digest of the source snapshot.")]
    inventory_sha256: Annotated[
        str, Field(description="Digest of the typed full inventory.")
    ]
    settings: Annotated[
        RegionalSettings, Field(description="Actual target locale settings.")
    ]
    locale_rules_version: Annotated[
        str, Field(description="Version of local validity rules.")
    ]
    locale_rules: Annotated[
        str, Field(description="Country-specific validity constraints.")
    ]
    cases: Annotated[
        list[DifficultyCase], Field(description="Hard cases and their Easy controls.")
    ]


class DifficultyAuditCheckpoint(LocalizationModel):
    """Successful source-pinned model evidence used only for the same exact input."""

    input_sha256: Annotated[
        str, Field(description="Digest of request, payload and prompt version.")
    ]
    request: Annotated[
        DifficultyAuditRequest, Field(description="Complete generation settings.")
    ]
    prompt_version: Annotated[
        str, Field(description="Version of the fixed audit prompt.")
    ]
    response: Annotated[
        DifficultyAuditBatch, Field(description="Validated complete batch response.")
    ]


class DifficultyAuditFailure(LocalizationModel):
    """An unresolved batch error accounted for by each affected canonical task ID."""

    canonical_task_id: Annotated[
        Nonempty, Field(description="Case with no valid model decision.")
    ]
    error: Annotated[
        Nonempty, Field(description="Reported local/model failure; rerun retries it.")
    ]


class DifficultyAuditReport(LocalizationModel):
    """Complete accounting including failures, never a native approval artifact."""

    request: Annotated[
        DifficultyAuditRequest, Field(description="Recorded runtime settings.")
    ]
    prompt_version: Annotated[
        str, Field(description="Version of the fixed model audit.")
    ]
    map_sha256: Annotated[str, Field(description="Exact parent candidate map digest.")]
    source_sha256: Annotated[str, Field(description="Exact canonical source digest.")]
    inventory_sha256: Annotated[str, Field(description="Full typed inventory digest.")]
    settings: Annotated[
        RegionalSettings, Field(description="Audited language/country settings.")
    ]
    expected_task_ids: Annotated[
        list[str], Field(description="All100 frozen Hard task IDs.")
    ]
    decisions: Annotated[
        list[DifficultyDecision], Field(description="Valid model assessments.")
    ]
    errors: Annotated[
        list[DifficultyAuditFailure], Field(description="All unresolved cases.")
    ]
    keep_count: Annotated[
        int, Field(ge=0, description="Number of demonstrated Hard candidates.")
    ]
    revise_count: Annotated[
        int, Field(ge=0, description="Number requiring localized revision.")
    ]
    source_blocked_count: Annotated[
        int, Field(ge=0, description="Number requiring source review.")
    ]
    error_count: Annotated[
        int, Field(ge=0, description="Number awaiting a successful model audit.")
    ]
    machine_gate_passed: Annotated[
        bool, Field(description="True only if all100 decisions are keep.")
    ]
    native_review_status: Annotated[
        Literal["pending"],
        Field(description="Independent native review remains mandatory."),
    ] = "pending"

    @model_validator(mode="after")
    def require_complete_accounting(self):
        """Every frozen task has one disposition or one explicit error, without duplicates."""
        if (
            len(self.expected_task_ids) != 100
            or len(set(self.expected_task_ids)) != 100
        ):
            raise ValueError(
                "Difficulty audit requires exactly 100 distinct Hard tasks"
            )
        observed = [d.canonical_task_id for d in self.decisions] + [
            e.canonical_task_id for e in self.errors
        ]
        if sorted(observed) != sorted(self.expected_task_ids):
            raise ValueError(
                "Difficulty report must account for every task exactly once"
            )
        actual = [
            sum(d.disposition == kind for d in self.decisions)
            for kind in ("keep", "revise", "source_blocked")
        ] + [len(self.errors)]
        if actual != [
            self.keep_count,
            self.revise_count,
            self.source_blocked_count,
            self.error_count,
        ]:
            raise ValueError("Difficulty report counts disagree with the task evidence")
        if self.machine_gate_passed != (self.keep_count == 100 and not self.errors):
            raise ValueError("Difficulty gate requires 100 supported keep decisions")
        return self


def _validate_coverage(
    response: DifficultyAuditBatch, cases: list[DifficultyCase]
) -> None:
    expected = [case.canonical_task_id for case in cases]
    if sorted(d.canonical_task_id for d in response.decisions) != sorted(expected):
        raise ValueError(
            "Difficulty audit must return every requested task exactly once"
        )
    by_id = {case.canonical_task_id: case for case in cases}
    for decision in response.decisions:
        easy_ids = [
            e.canonical_task_id
            for e in by_id[decision.canonical_task_id].easy_comparators
        ]
        if not any(
            identifier in decision.comparison_to_easy for identifier in easy_ids
        ):
            raise ValueError(
                "Difficulty comparison must cite a supplied same-family Easy task"
            )


class DifficultyAuditCacheInput(LocalizationModel):
    """Exact source-pinned input identity shared by execution and read-only inspection."""

    payload: Annotated[
        DifficultyAuditPayload, Field(description="Typed outgoing audit evidence.")
    ]
    input_sha256: Annotated[
        str, Field(description="Existing request/payload/version digest.")
    ]
    path: Annotated[Path, Field(description="Expected immutable checkpoint path.")]


def prepare_difficulty_batch(
    request: DifficultyAuditRequest,
    inventory: DifficultyInventory,
    cases: list[DifficultyCase],
    checkpoints: Path,
) -> DifficultyAuditCacheInput:
    """Reconstruct the same checkpoint identity without contacting a model."""
    payload = DifficultyAuditPayload(
        map_sha256=inventory.map_sha256,
        source_sha256=inventory.source_sha256,
        inventory_sha256=content_digest(inventory),
        settings=inventory.settings,
        locale_rules_version=LOCALE_RULES_VERSION,
        locale_rules=locale_rules(inventory.settings),
        cases=cases,
    )
    # The request is part of the identity, including revision and explicit settings.
    digest = json_content_digest(
        {
            "payload": payload.model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
            "prompt_version": AUDIT_PROMPT_VERSION,
        }
    )
    return DifficultyAuditCacheInput(
        payload=payload, input_sha256=digest, path=checkpoints / f"{digest}.json"
    )


def read_difficulty_checkpoint(
    request: DifficultyAuditRequest, prepared: DifficultyAuditCacheInput
) -> DifficultyAuditBatch:
    """Validate exact cached input identity and full response coverage without a model."""
    if not prepared.path.exists():
        raise FileNotFoundError(
            f"No completed audit checkpoint for input {prepared.input_sha256}; model assessment incomplete"
        )
    saved = DifficultyAuditCheckpoint.model_validate_json(prepared.path.read_text())
    if (
        saved.input_sha256 != prepared.input_sha256
        or saved.request != request
        or saved.prompt_version != AUDIT_PROMPT_VERSION
    ):
        raise ValueError(
            "Difficulty checkpoint differs from exact source/model/prompt input"
        )
    _validate_coverage(saved.response, prepared.payload.cases)
    return saved.response


def audit_batch(
    request: DifficultyAuditRequest,
    inventory: DifficultyInventory,
    cases: list[DifficultyCase],
    checkpoints: Path,
) -> DifficultyAuditBatch:
    """Audit one fixed batch or validate and reuse its exact-input checkpoint."""
    prepared = prepare_difficulty_batch(request, inventory, cases, checkpoints)
    payload, digest, path = prepared.payload, prepared.input_sha256, prepared.path
    if path.exists():
        return read_difficulty_checkpoint(request, prepared)
    task_ids = [case.canonical_task_id for case in cases]
    started = perf_counter()
    logger.info(
        "Intake difficulty batch started: language={} task_ids={} input_sha256={}",
        inventory.settings.language,
        task_ids,
        digest,
    )
    try:
        message = generate(
            model=request.model,
            messages=[
                SystemMessage(role="system", content=AUDIT_PROMPT),
                UserMessage(role="user", content=payload.model_dump_json()),
            ],
            call_name="intake_difficulty_audit",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "intake_difficulty_audit",
                    "strict": True,
                    "schema": DifficultyAuditBatch.model_json_schema(),
                },
            },
            **request.model_args,
        )
    finally:
        logger.info(
            "Intake difficulty batch call ended: language={} task_ids={} input_sha256={} elapsed_seconds={:.3f}",
            inventory.settings.language,
            task_ids,
            digest,
            perf_counter() - started,
        )
    response = DifficultyAuditBatch.model_validate_json(
        extract_json_from_llm_response(message.content or "")
    )
    _validate_coverage(response, cases)
    write_artifact(
        DifficultyAuditCheckpoint(
            input_sha256=digest,
            request=request,
            prompt_version=AUDIT_PROMPT_VERSION,
            response=response,
        ),
        path,
    )
    return response


def run_difficulty_audit(
    request: DifficultyAuditRequest, output: Path, checkpoints: Path
) -> DifficultyAuditReport:
    """Audit all100 cases, preserve successful checkpoints and account for failures."""
    proposal = read_localization_map(request.map_file)
    inventory = build_difficulty_inventory(proposal)
    ids = [case.canonical_task_id for case in inventory.cases]
    if len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError(
            "Difficulty inventory must contain exactly 100 distinct Hard tasks"
        )
    if output.exists():
        saved = DifficultyAuditReport.model_validate_json(output.read_text())
        if (
            saved.request != request
            or saved.prompt_version != AUDIT_PROMPT_VERSION
            or saved.inventory_sha256 != content_digest(inventory)
            or saved.map_sha256 != inventory.map_sha256
            or saved.source_sha256 != inventory.source_sha256
            or saved.expected_task_ids != ids
        ):
            raise ValueError(
                "Existing difficulty report is stale; use a new output path"
            )
        if not saved.errors:
            _validate_coverage(
                DifficultyAuditBatch(decisions=saved.decisions), inventory.cases
            )
            return saved
        raise ValueError(
            "Existing audit records errors; resume checkpoints with a new output path"
        )
    batches = [
        inventory.cases[i : i + request.batch_size]
        for i in range(0, 100, request.batch_size)
    ]
    decisions: list[DifficultyDecision] = []
    errors: list[DifficultyAuditFailure] = []
    with ThreadPoolExecutor(max_workers=request.max_concurrency) as pool:
        pending = {
            pool.submit(audit_batch, request, inventory, batch, checkpoints): batch
            for batch in batches
        }
        for future in as_completed(pending):
            batch = pending[future]
            try:
                decisions.extend(future.result().decisions)
            except Exception as error:
                logger.error(
                    "Intake difficulty audit failed for {} cases: {}", len(batch), error
                )
                errors.extend(
                    DifficultyAuditFailure(
                        canonical_task_id=c.canonical_task_id,
                        error=str(error) or type(error).__name__,
                    )
                    for c in batch
                )
    return _write_difficulty_report(request, inventory, decisions, errors, output)


def _write_difficulty_report(
    request: DifficultyAuditRequest,
    inventory: DifficultyInventory,
    decisions: list[DifficultyDecision],
    errors: list[DifficultyAuditFailure],
    output: Path,
) -> DifficultyAuditReport:
    """Serialize complete accounting; missing model evidence can never pass the gate."""
    ids = [case.canonical_task_id for case in inventory.cases]
    order = {identifier: index for index, identifier in enumerate(ids)}
    decisions.sort(key=lambda d: order[d.canonical_task_id])
    errors.sort(key=lambda e: order[e.canonical_task_id])
    kept = sum(d.disposition == "keep" for d in decisions)
    report = DifficultyAuditReport(
        request=request,
        prompt_version=AUDIT_PROMPT_VERSION,
        map_sha256=inventory.map_sha256,
        source_sha256=inventory.source_sha256,
        inventory_sha256=content_digest(inventory),
        settings=inventory.settings,
        expected_task_ids=ids,
        decisions=decisions,
        errors=errors,
        keep_count=kept,
        revise_count=sum(d.disposition == "revise" for d in decisions),
        source_blocked_count=sum(d.disposition == "source_blocked" for d in decisions),
        error_count=len(errors),
        machine_gate_passed=kept == 100,
    )
    write_artifact(report, output)
    return report


def inspect_difficulty_audit(
    request: DifficultyAuditRequest, output: Path, checkpoints: Path
) -> DifficultyAuditReport:
    """Snapshot valid checkpoints and explicit missing/invalid-case errors, with no LLM calls."""
    inventory = build_difficulty_inventory(read_localization_map(request.map_file))
    decisions: list[DifficultyDecision] = []
    errors: list[DifficultyAuditFailure] = []
    for start in range(0, len(inventory.cases), request.batch_size):
        cases = inventory.cases[start : start + request.batch_size]
        prepared = prepare_difficulty_batch(request, inventory, cases, checkpoints)
        try:
            decisions.extend(read_difficulty_checkpoint(request, prepared).decisions)
        except (OSError, ValueError) as error:
            errors.extend(
                DifficultyAuditFailure(
                    canonical_task_id=case.canonical_task_id,
                    error=f"Checkpoint inspection: {error}",
                )
                for case in cases
            )
    return _write_difficulty_report(request, inventory, decisions, errors, output)
