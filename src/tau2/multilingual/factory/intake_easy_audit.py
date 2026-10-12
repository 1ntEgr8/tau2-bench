"""Independent, source-bound audit of every selected Easy intake case."""

import csv
import hashlib
import io
import os
import tempfile
from collections import Counter
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
from tau2.domains.intake.tasks.generator import CANONICAL_BANKS
from tau2.multilingual.factory.intake_candidate_review import (
    LOCALE_RULES_VERSION,
    locale_rules,
)
from tau2.multilingual.factory.intake_difficulty_inventory import (
    DifficultyComparator,
    DifficultyInventory,
    Digest,
    Family,
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

EASY_AUDIT_PROMPT_VERSION = "intake-easy-audit-v1"
EASY_AUDIT_PROMPT = """Independently audit every supplied Easy intake case. Prior Hard audits and the source Easy label are not approval or evidence of actual ease. Return exactly one decision per supplied case canonical_task_id; family_controls are context, not additional tasks to decide. Assess the English source separately, then the localized native caller experience and all dependencies. Easy means straightforward ordinary capture for a fluent local speaker, not absence of every possible error. Use concrete supplied text as evidence; evaluate naturalness, local entity conventions, exact-answer recoverability, semantic fidelity and whether the localization has unnecessarily made an Easy task too hard. The actual ten same-family Easy controls provide context but may themselves contain defects; do not endorse a bad case merely because another control has the same problem. Never propose making Easy harder to match a Hard benchmark. Uncertainty must not become keep.

Read the exact written answer, source/native/romanized lookup payloads, decoys, pronunciation examples, omission/spelling instructions, pinned clock and caller guidance. Localized identities may differ from English personal/business/address identities by design; preserve their role, semantic task, locale plausibility and linked consistency, not foreign literal spellings. Invented company/property identities should be locally plausible, not foreign transliterations or literal source-name calques. Straightforward native personal names and coherent local addresses are preferred to conspicuous complexity. Native mispronunciation tokens/distortions use native orthography and real audible differences; romanized metadata must describe the same identity, not stress/capitalization coaching. Written decoys are a separate branch from intentional mispronunciations: a decoy may be a homophone if its required normalized answer remains distinct, but a pronunciation distortion must sound different. Do not require pronunciation examples where the source array is empty, or invent extra unsupported complication slots.

Runtime roles and sampling are essential. This is an outbound intake call: the service agent is the CALLER, and the user simulator is the CALLEE/record owner answering. Thus instructions to the user such as 'when the caller asks' or 'correct the caller' properly refer to the service agent; they are not role reversal. ONE complication kind is sampled, and spelling_styles are alternatives: only ONE nonempty style is selected, not all styles simultaneously. Omission and decoy branches are likewise conditional, not concurrent mandatory actions. Empty source arrays and empty inapplicable style strings are intentional, not defects by themselves. English instructions are internal control text, not literal spoken leakage; quoted speech examples must be natural in the target language. Do not confuse separate alternative styles with a contradiction. However, always-active guidance remains active alongside whichever single complication is sampled: flag a specific style that explicitly contradicts that guidance.

Private time knowledge is not a verbatim initial utterance. The full localized time/day period and canonical annotation in the lookup payload are privately known so the callee can clarify. Supplied time guidance can require an initial bare phrase with the day period withheld, independently of any sampled omission/style. Judge the actual initial instruction, not the full private anchor as if it must be spoken. A style explicitly requiring AM/PM on the initial answer despite active bare-initial guidance is a real cross-layer conflict; private AM/PM knowledge alone is correct. Keep immutable time gold unchanged, use ordinary local time speech, and avoid contrived arithmetic. Dates use the target local display convention while the exact canonical date and pinned clock stay fixed; changing a relative phrase is permitted only if its exact date meaning remains. Do not impose extra relative reasoning or complex offsets on Easy cases.

ASCII phone/email/tool syntax is not a speech transcript. Conventional written +, ext., x, #, domain punctuation, mg and protocol code characters are not defects just because they are Latin/ASCII. Check actual spoken instructions and local phone/country/extension shape. Preserve genuine source wrong-digit/length decoys rather than requiring every wrong answer to be a valid number; do not manufacture new invalid lengths. Local code/plate syntax must be valid; do not claim ambiguities forbidden by a position's alphabet. Do not universally force or ban English oh/double/triple: judge natural local usage and the actual context, especially ordinary medication-dose speech versus requested digit clarification. Medication transliterations are expected. Preserve every primary AND decoy clinical identity, strength, units, formulation and array position; never swap drugs or doses to change difficulty. Preserve checksums, source field types and exact protocol gold. Semantic constraints outrank ease, so a fixed source medication or code that is intrinsically hard may require source_blocked, not a different easier identity. Check number words in their actual context: ordinary hour/minute number systems (including native versus Sino-Korean numerals), calendar-year reading versus cardinal grouping, postcodes and digit blocks versus quantities, zero versus letter names, medication strengths and requested code spelling are not interchangeable. Specialist radio/military digit names or imported English-only letter/number habits need local register justification; flag explicit wrong instructions without blanket-banning customary local loans.

For each case provide source_difficulty and localized_difficulty as easy, too_hard or uncertain, concrete source_assessment, ease_assessment and evidence, naturalness, known_answer_recoverable and source_semantics_preserved. Keep requires a natural, recoverable, semantically faithful Easy case with no unresolved dependency issues, repair targets or correction instructions. Otherwise choose revise for an applicable localized defect, naming only the exact repair_targets: primary, caller_surface, decoys, pronunciations, spelling_styles, omission_instruction. Supply factual correction constraints, not invented production replacements. Do not request a primary rewrite when only a dependency needs correction. Use source_blocked when fixed semantics/protocol or missing source slots make an Easy realization infeasible, and give the precise source_blocked_reason; never silently rewrite the benchmark. A source blocker may coexist with a separately repairable dependency, which must be explicit. All model decisions require native review; native_review_required is always true. Return only response-schema JSON."""

Nonempty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EasyDifficulty = Literal["easy", "too_hard", "uncertain"]
EasyRepairTarget = Literal[
    "primary",
    "caller_surface",
    "decoys",
    "pronunciations",
    "spelling_styles",
    "omission_instruction",
]


class EasyAuditRequest(LocalizationModel):
    """Explicit model settings and exact input map for a complete Easy audit."""

    map_file: Annotated[Path, Field(description="Exact candidate localization map.")]
    revision: Annotated[Nonempty, Field(description="User-visible audit revision.")]
    model: Annotated[
        Nonempty, Field(description="Model passed to the shared LLM seam.")
    ]
    model_args: Annotated[
        dict[str, JsonValue], Field(description="Explicit recorded model arguments.")
    ]
    batch_size: Annotated[
        int, Field(ge=1, le=100, description="Easy cases per model call.")
    ] = DEFAULT_FACTORY_LOCALIZATION_BATCH_SIZE
    max_concurrency: Annotated[
        int, Field(ge=1, le=10, description="Maximum concurrent audit calls.")
    ] = DEFAULT_FACTORY_LOCALIZATION_CONCURRENCY


class EasyDecision(LocalizationModel):
    """Independent Easy assessment, never native-speaker approval."""

    canonical_task_id: Annotated[
        Nonempty, Field(description="Exact supplied Easy task ID.")
    ]
    disposition: Annotated[
        Literal["keep", "revise", "source_blocked"],
        Field(
            description="Keep, repair a localized defect, or expose fixed-source limits."
        ),
    ]
    source_difficulty: Annotated[
        EasyDifficulty,
        Field(
            description="Independent assessment of source English, not its inherited label."
        ),
    ]
    source_assessment: Annotated[
        Nonempty, Field(description="Source evidence and any immutable limitations.")
    ]
    localized_difficulty: Annotated[
        EasyDifficulty,
        Field(description="Actual local capture difficulty of the Easy task."),
    ]
    ease_assessment: Annotated[
        Nonempty,
        Field(description="Concrete local capture burden and same-family context."),
    ]
    evidence: Annotated[
        Nonempty, Field(description="Exact supplied text supporting the decision.")
    ]
    naturalness: Annotated[
        bool,
        Field(description="Whether the local identity and actual speech are natural."),
    ]
    known_answer_recoverable: Annotated[
        bool,
        Field(
            description="Whether caller knowledge and clarification recover exact gold."
        ),
    ]
    source_semantics_preserved: Annotated[
        bool,
        Field(
            description="Whether source invariants and semantic identity constraints survive localization."
        ),
    ]
    dependency_issues: Annotated[
        list[Nonempty],
        Field(
            description="Exact broken decoy, speech, omission or guidance dependencies."
        ),
    ]
    correction_instructions: Annotated[
        list[Nonempty],
        Field(
            description="Factual repair constraints, never authored replacement entities."
        ),
    ]
    repair_targets: Annotated[
        list[EasyRepairTarget],
        Field(description="Only linked pieces requiring repair; empty for keep."),
    ]
    source_blocked_reason: Annotated[
        str | None,
        Field(
            description="Why frozen source constraints prevent a natural Easy realization."
        ),
    ]
    native_review_required: Annotated[
        Literal[True],
        Field(description="Model assessment never grants native approval."),
    ]

    @model_validator(mode="after")
    def validate_disposition(self):
        """Reject unsupported keep decisions and unscoped repair requests."""
        if len(set(self.repair_targets)) != len(self.repair_targets):
            raise ValueError("Easy repair targets must be distinct")
        if self.disposition == "keep":
            if not (
                self.localized_difficulty == "easy"
                and self.naturalness
                and self.known_answer_recoverable
                and self.source_semantics_preserved
                and not self.dependency_issues
                and not self.correction_instructions
                and not self.repair_targets
                and self.source_blocked_reason is None
            ):
                raise ValueError(
                    "keep requires natural recoverable faithful Easy evidence with no open issues"
                )
        elif self.disposition == "revise":
            if (
                not self.correction_instructions
                or not self.repair_targets
                or self.source_blocked_reason is not None
            ):
                raise ValueError(
                    "revise requires correction instructions and repair targets, without a source blocker"
                )
        elif not self.source_blocked_reason or not self.source_blocked_reason.strip():
            raise ValueError(
                "source_blocked requires a specific immutable-source reason"
            )
        if self.disposition == "source_blocked" and bool(
            self.correction_instructions
        ) != bool(self.repair_targets):
            raise ValueError(
                "A separate source-blocked repair needs both instructions and targets"
            )
        return self


class EasyAuditBatch(LocalizationModel):
    """Model response validated against exact requested Easy task IDs."""

    decisions: Annotated[
        list[EasyDecision],
        Field(description="Exactly one decision per requested Easy case."),
    ]


class EasyAuditPayload(LocalizationModel):
    """Only measured Easy evidence; donor and owner identities never cross this boundary."""

    map_sha256: Digest
    source_sha256: Digest
    inventory_sha256: Digest
    settings: Annotated[
        RegionalSettings, Field(description="Actual target locale settings.")
    ]
    locale_rules_version: Annotated[
        Nonempty, Field(description="Version of local validity rules.")
    ]
    locale_rules: Annotated[
        Nonempty, Field(description="Country-specific identity and syntax constraints.")
    ]
    cases: Annotated[
        list[DifficultyComparator],
        Field(
            min_length=1,
            max_length=100,
            description="Exactly the Easy cases to decide, with all measured dependencies and guidance.",
        ),
    ]
    family_controls: Annotated[
        list[DifficultyComparator],
        Field(
            min_length=10,
            max_length=100,
            description="All ten actual Easy controls for each requested family, deduplicated once.",
        ),
    ]

    @model_validator(mode="after")
    def validate_controls(self):
        """Reject incomplete controls or case evidence inconsistent with its control."""
        controls = {c.canonical_task_id: c for c in self.family_controls}
        families = {c.family for c in self.cases}
        if len(controls) != len(self.family_controls) or Counter(
            c.family for c in self.family_controls
        ) != Counter({family: 10 for family in families}):
            raise ValueError(
                "Easy payload needs all ten distinct controls per requested family"
            )
        if len({c.canonical_task_id for c in self.cases}) != len(self.cases) or any(
            controls.get(c.canonical_task_id) != c for c in self.cases
        ):
            raise ValueError("Easy payload cases must agree with their exact controls")
        return self


class EasyAuditCheckpoint(LocalizationModel):
    """One successful response pinned to exact evidence, settings and prompt."""

    input_sha256: Digest
    request: Annotated[
        EasyAuditRequest, Field(description="Complete generation settings.")
    ]
    prompt_version: Annotated[
        Nonempty, Field(description="Fixed audit prompt version.")
    ]
    response: Annotated[
        EasyAuditBatch, Field(description="Validated exact-batch model evidence.")
    ]


class EasyAuditFailure(LocalizationModel):
    """An explicit unresolved task, never silently dropped or passed."""

    canonical_task_id: Annotated[
        Nonempty, Field(description="Easy case lacking a valid audit decision.")
    ]
    error: Annotated[
        Nonempty, Field(description="Local, checkpoint or provider failure.")
    ]


def _expected_ids() -> set[str]:
    return {
        f"intake_{family}_easy_{i:02d}"
        for family in CANONICAL_BANKS
        for i in range(1, 11)
    }


class EasyAuditReport(LocalizationModel):
    """Exactly 100 source-bound outcomes including every failure, with no native approval."""

    request: Annotated[
        EasyAuditRequest, Field(description="Recorded runtime settings.")
    ]
    prompt_version: Annotated[
        Nonempty, Field(description="Fixed audit prompt version.")
    ]
    map_sha256: Digest
    source_sha256: Digest
    inventory_sha256: Digest
    settings: Annotated[
        RegionalSettings, Field(description="Exact audited locale settings.")
    ]
    expected_task_ids: Annotated[
        list[str], Field(description="All 100 frozen Easy IDs, ten in every family.")
    ]
    decisions: Annotated[
        list[EasyDecision], Field(description="Every valid model assessment.")
    ]
    errors: Annotated[
        list[EasyAuditFailure],
        Field(description="Every unresolved case with an explicit reason."),
    ]
    keep_count: Annotated[
        int, Field(ge=0, description="Number judged naturally Easy and coherent.")
    ]
    revise_count: Annotated[
        int, Field(ge=0, description="Number requiring localized revision.")
    ]
    source_blocked_count: Annotated[
        int, Field(ge=0, description="Number requiring source feasibility review.")
    ]
    error_count: Annotated[
        int, Field(ge=0, description="Number without valid model evidence.")
    ]
    machine_gate_passed: Annotated[
        bool,
        Field(
            description="True only for 100 supported keep decisions, not native approval."
        ),
    ]
    native_review_status: Annotated[
        Literal["pending"],
        Field(description="Independent native review remains necessary."),
    ] = "pending"

    @model_validator(mode="after")
    def validate_accounting(self):
        """Reject missing, duplicate, wrong-tier outcomes or dishonest summary counts."""
        if (
            len(self.expected_task_ids) != 100
            or set(self.expected_task_ids) != _expected_ids()
        ):
            raise ValueError("Easy audit requires the exact 100-task Easy frame")
        observed = [d.canonical_task_id for d in self.decisions] + [
            e.canonical_task_id for e in self.errors
        ]
        if sorted(observed) != sorted(self.expected_task_ids):
            raise ValueError("Easy report must account for every task exactly once")
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
            raise ValueError("Easy report counts disagree with evidence")
        if self.machine_gate_passed != (self.keep_count == 100 and not self.errors):
            raise ValueError("Easy machine gate requires 100 supported keep decisions")
        return self


class EasyAuditCacheInput(LocalizationModel):
    """Same exact checkpoint identity for generation and no-call inspection."""

    payload: Annotated[EasyAuditPayload, Field(description="Typed outgoing evidence.")]
    input_sha256: Digest
    path: Annotated[Path, Field(description="Expected immutable checkpoint path.")]


def extract_easy_cases(inventory: DifficultyInventory) -> list[DifficultyComparator]:
    """Deduplicate all repeated same-family controls with strict source-bound coverage."""
    inventory = DifficultyInventory.model_validate(inventory.model_dump())
    unique: dict[str, DifficultyComparator] = {}
    for case in inventory.cases:
        for comparator in case.easy_comparators:
            easy = DifficultyComparator.model_validate(comparator.model_dump())
            if (
                easy.canonical_task_id in unique
                and unique[easy.canonical_task_id] != easy
            ):
                raise ValueError("Repeated Easy control evidence differs")
            unique[easy.canonical_task_id] = easy
    if set(unique) != _expected_ids() or Counter(
        c.family for c in unique.values()
    ) != Counter({f: 10 for f in CANONICAL_BANKS}):
        raise ValueError(
            "Easy inventory requires exactly ten selected cases per family"
        )
    return list(unique.values())


def prepare_easy_batch(
    request: EasyAuditRequest,
    inventory: DifficultyInventory,
    cases: list[DifficultyComparator],
    checkpoints: Path,
) -> EasyAuditCacheInput:
    """Prepare safe exact-map evidence and checkpoint identity without provider calls."""
    controls = extract_easy_cases(inventory)
    families = {c.family for c in cases}
    payload = EasyAuditPayload(
        map_sha256=inventory.map_sha256,
        source_sha256=inventory.source_sha256,
        inventory_sha256=content_digest(inventory),
        settings=inventory.settings,
        locale_rules_version=LOCALE_RULES_VERSION,
        locale_rules=locale_rules(inventory.settings),
        cases=cases,
        family_controls=[c for c in controls if c.family in families],
    )
    digest = json_content_digest(
        {
            "request": request.model_dump(mode="json"),
            "payload": payload.model_dump(mode="json"),
            "prompt_version": EASY_AUDIT_PROMPT_VERSION,
        }
    )
    return EasyAuditCacheInput(
        payload=payload, input_sha256=digest, path=checkpoints / f"{digest}.json"
    )


def _validate_coverage(
    response: EasyAuditBatch, cases: list[DifficultyComparator]
) -> None:
    if sorted(d.canonical_task_id for d in response.decisions) != sorted(
        c.canonical_task_id for c in cases
    ):
        raise ValueError("Easy audit must return every requested task exactly once")


def read_easy_checkpoint(
    request: EasyAuditRequest, prepared: EasyAuditCacheInput
) -> EasyAuditBatch:
    """Read only a complete exact-input response, rejecting stale or corrupt evidence."""
    if not prepared.path.exists():
        raise FileNotFoundError(
            f"No completed Easy checkpoint for {prepared.input_sha256}; assessment incomplete"
        )
    saved = EasyAuditCheckpoint.model_validate_json(prepared.path.read_text())
    if (
        saved.input_sha256 != prepared.input_sha256
        or saved.request != request
        or saved.prompt_version != EASY_AUDIT_PROMPT_VERSION
    ):
        raise ValueError("Easy checkpoint differs from exact map/model/prompt input")
    _validate_coverage(saved.response, prepared.payload.cases)
    return saved.response


def audit_easy_batch(
    request: EasyAuditRequest,
    inventory: DifficultyInventory,
    cases: list[DifficultyComparator],
    checkpoints: Path,
) -> EasyAuditBatch:
    """Audit one batch or reuse its exact checkpoint through the shared LLM seam."""
    prepared = prepare_easy_batch(request, inventory, cases, checkpoints)
    if prepared.path.exists():
        return read_easy_checkpoint(request, prepared)
    started = perf_counter()
    logger.info(
        "Easy audit started: language={} tasks={} input={}",
        inventory.settings.language,
        [c.canonical_task_id for c in cases],
        prepared.input_sha256,
    )
    try:
        message = generate(
            model=request.model,
            messages=[
                SystemMessage(role="system", content=EASY_AUDIT_PROMPT),
                UserMessage(role="user", content=prepared.payload.model_dump_json()),
            ],
            call_name="intake_easy_audit",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "intake_easy_audit",
                    "strict": True,
                    "schema": EasyAuditBatch.model_json_schema(),
                },
            },
            **request.model_args,
        )
    finally:
        logger.info(
            "Easy audit ended: input={} elapsed_seconds={:.3f}",
            prepared.input_sha256,
            perf_counter() - started,
        )
    response = EasyAuditBatch.model_validate_json(
        extract_json_from_llm_response(message.content or "")
    )
    _validate_coverage(response, cases)
    write_artifact(
        EasyAuditCheckpoint(
            input_sha256=prepared.input_sha256,
            request=request,
            prompt_version=EASY_AUDIT_PROMPT_VERSION,
            response=response,
        ),
        prepared.path,
    )
    return response


class EasyLedgerRow(LocalizationModel):
    """Model-owned CSV columns for one independently audited Easy case."""

    language: Annotated[str, Field(description="Target language.")]
    family: Annotated[Family, Field(description="Source entity family.")]
    canonical_task_id: Annotated[str, Field(description="Exact selected Easy task ID.")]
    tier: Annotated[
        Literal["easy"],
        Field(description="Frozen assignment; never inferred from a verdict."),
    ] = "easy"
    measured_entry_id: Annotated[str, Field(description="Exact primary map entry.")]
    source_value: Annotated[str, Field(description="Canonical measured source value.")]
    native_value: Annotated[str, Field(description="Localized measured answer.")]
    romanized_value: Annotated[
        str, Field(description="Same answer in romanized metadata.")
    ]
    disposition: Annotated[
        Literal["keep", "revise", "source_blocked", "audit_error"],
        Field(description="Raw model disposition or explicit missing evidence."),
    ]
    source_difficulty: Annotated[
        str, Field(description="Independent source assessment, empty for audit errors.")
    ]
    localized_difficulty: Annotated[
        str, Field(description="Local Easy assessment, empty for audit errors.")
    ]
    evidence: Annotated[
        str, Field(description="Raw model evidence or exact audit error.")
    ]
    dependency_issues: Annotated[
        str, Field(description="JSON array of raw dependency findings.")
    ]
    correction_instructions: Annotated[
        str, Field(description="JSON array of factual repair constraints.")
    ]
    repair_targets: Annotated[
        str, Field(description="JSON array of precise repair targets.")
    ]
    source_blocked_reason: Annotated[
        str, Field(description="Immutable source limitation, empty when absent.")
    ]
    native_review_status: Annotated[
        Literal["pending"], Field(description="No native approval is implied.")
    ] = "pending"


class EasyLedgerManifest(LocalizationModel):
    """Binds the human review CSV to the exact audit and evidence, not approval."""

    version: Annotated[
        Literal["intake-easy-ledger-v1"],
        Field(description="Model-backed ledger contract version."),
    ] = "intake-easy-ledger-v1"
    map_sha256: Digest
    source_sha256: Digest
    inventory_sha256: Digest
    audit_sha256: Digest
    csv_sha256: Digest
    row_count: Annotated[
        Literal[100], Field(description="All selected Easy cases.")
    ] = 100
    rows: Annotated[
        list[EasyLedgerRow],
        Field(min_length=100, max_length=100, description="Exact typed CSV rows."),
    ]

    @model_validator(mode="after")
    def validate_rows(self):
        """Manifest rows must themselves cover the exact Easy frame."""
        if {r.canonical_task_id for r in self.rows} != _expected_ids() or len(
            {r.language for r in self.rows}
        ) != 1:
            raise ValueError("Easy ledger requires one complete language frame")
        if hashlib.sha256(_render_ledger(self.rows)).hexdigest() != self.csv_sha256:
            raise ValueError("Easy ledger CSV digest differs from typed rows")
        return self


def _render_ledger(rows: list[EasyLedgerRow]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(EasyLedgerRow.model_fields))
    writer.writeheader()
    writer.writerows(row.model_dump(mode="json") for row in rows)
    return stream.getvalue().encode("utf-8")


def _write_csv(data: bytes, output: Path) -> None:
    """Publish complete CSV bytes without replacing different review evidence."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.link(temporary, output)
        except FileExistsError:
            if output.read_bytes() != data:
                raise ValueError(
                    "Refusing to overwrite a different Easy ledger"
                ) from None
        finally:
            temporary.unlink(missing_ok=True)


def easy_ledger_rows(
    report: EasyAuditReport, inventory: DifficultyInventory
) -> list[EasyLedgerRow]:
    """Build all 100 rows only when report and full evidence have identical digests."""
    import json

    report = EasyAuditReport.model_validate(report.model_dump())
    cases = extract_easy_cases(inventory)
    if (
        report.map_sha256,
        report.source_sha256,
        report.inventory_sha256,
        report.settings,
        report.expected_task_ids,
    ) != (
        inventory.map_sha256,
        inventory.source_sha256,
        content_digest(inventory),
        inventory.settings,
        [c.canonical_task_id for c in cases],
    ):
        raise ValueError("Easy report and inventory refer to different exact inputs")
    decisions = {d.canonical_task_id: d for d in report.decisions}
    errors = {e.canonical_task_id: e for e in report.errors}
    rows = []
    for case in cases:
        d = decisions.get(case.canonical_task_id)
        rows.append(
            EasyLedgerRow(
                language=inventory.settings.language,
                family=case.family,
                canonical_task_id=case.canonical_task_id,
                measured_entry_id=case.measured_entry_id,
                source_value=case.source_value,
                native_value=case.native_value,
                romanized_value=case.romanized_value,
                disposition=d.disposition if d else "audit_error",
                source_difficulty=d.source_difficulty if d else "",
                localized_difficulty=d.localized_difficulty if d else "",
                evidence=d.evidence if d else errors[case.canonical_task_id].error,
                dependency_issues=json.dumps(
                    d.dependency_issues if d else [], ensure_ascii=False
                ),
                correction_instructions=json.dumps(
                    d.correction_instructions if d else [], ensure_ascii=False
                ),
                repair_targets=json.dumps(d.repair_targets if d else []),
                source_blocked_reason=d.source_blocked_reason or "" if d else "",
            )
        )
    return rows


def _write_report(
    report: EasyAuditReport, inventory: DifficultyInventory, output: Path
) -> EasyAuditReport:
    rows = easy_ledger_rows(report, inventory)
    data = _render_ledger(rows)
    csv_path = output.with_suffix(".ledger.csv")
    if csv_path.exists() and csv_path.read_bytes() != data:
        raise ValueError("Refusing to overwrite a different Easy ledger")
    write_artifact(report, output)
    write_artifact(
        EasyLedgerManifest(
            map_sha256=report.map_sha256,
            source_sha256=report.source_sha256,
            inventory_sha256=report.inventory_sha256,
            audit_sha256=content_digest(report),
            csv_sha256=hashlib.sha256(data).hexdigest(),
            rows=rows,
        ),
        output.with_suffix(".ledger.manifest.json"),
    )
    _write_csv(data, csv_path)
    return report


def _report(
    request: EasyAuditRequest,
    inventory: DifficultyInventory,
    decisions: list[EasyDecision],
    errors: list[EasyAuditFailure],
) -> EasyAuditReport:
    ids = [c.canonical_task_id for c in extract_easy_cases(inventory)]
    order = {identifier: n for n, identifier in enumerate(ids)}
    decisions.sort(key=lambda d: order[d.canonical_task_id])
    errors.sort(key=lambda e: order[e.canonical_task_id])
    count = Counter(d.disposition for d in decisions)
    return EasyAuditReport(
        request=request,
        prompt_version=EASY_AUDIT_PROMPT_VERSION,
        map_sha256=inventory.map_sha256,
        source_sha256=inventory.source_sha256,
        inventory_sha256=content_digest(inventory),
        settings=inventory.settings,
        expected_task_ids=ids,
        decisions=decisions,
        errors=errors,
        keep_count=count["keep"],
        revise_count=count["revise"],
        source_blocked_count=count["source_blocked"],
        error_count=len(errors),
        machine_gate_passed=count["keep"] == 100,
    )


def run_easy_audit(
    request: EasyAuditRequest, output: Path, checkpoints: Path
) -> EasyAuditReport:
    """Audit the entire Easy frame, resume valid checkpoints, retain explicit failures."""
    inventory = build_difficulty_inventory(read_localization_map(request.map_file))
    cases = extract_easy_cases(inventory)
    if output.exists():
        saved = EasyAuditReport.model_validate_json(output.read_text())
        if (
            saved.request != request
            or saved.prompt_version != EASY_AUDIT_PROMPT_VERSION
            or (
                saved.map_sha256,
                saved.source_sha256,
                saved.inventory_sha256,
                saved.settings,
                saved.expected_task_ids,
            )
            != (
                inventory.map_sha256,
                inventory.source_sha256,
                content_digest(inventory),
                inventory.settings,
                [c.canonical_task_id for c in cases],
            )
        ):
            raise ValueError("Existing Easy report is stale; use a new output path")
        if saved.errors:
            raise ValueError(
                "Existing Easy report has errors; resume checkpoints with a new output path"
            )
        _validate_coverage(EasyAuditBatch(decisions=saved.decisions), cases)
        reconstructed = []
        for n in range(0, len(cases), request.batch_size):
            prepared = prepare_easy_batch(
                request, inventory, cases[n : n + request.batch_size], checkpoints
            )
            reconstructed.extend(read_easy_checkpoint(request, prepared).decisions)
        if _report(request, inventory, reconstructed, []) != saved:
            raise ValueError("Existing Easy report differs from its exact checkpoints")
        return _write_report(saved, inventory, output)
    batches = [
        cases[n : n + request.batch_size]
        for n in range(0, len(cases), request.batch_size)
    ]
    decisions: list[EasyDecision] = []
    errors: list[EasyAuditFailure] = []
    with ThreadPoolExecutor(max_workers=request.max_concurrency) as pool:
        pending = {
            pool.submit(audit_easy_batch, request, inventory, batch, checkpoints): batch
            for batch in batches
        }
        for future in as_completed(pending):
            batch = pending[future]
            try:
                decisions.extend(future.result().decisions)
            except Exception as error:
                logger.error("Easy audit failed for {} cases: {}", len(batch), error)
                errors.extend(
                    EasyAuditFailure(
                        canonical_task_id=c.canonical_task_id,
                        error=str(error) or type(error).__name__,
                    )
                    for c in batch
                )
    return _write_report(
        _report(request, inventory, decisions, errors), inventory, output
    )


def inspect_easy_audit(
    request: EasyAuditRequest, output: Path, checkpoints: Path
) -> EasyAuditReport:
    """Export valid exact checkpoints and explicit missing cases without any model calls."""
    inventory = build_difficulty_inventory(read_localization_map(request.map_file))
    cases = extract_easy_cases(inventory)
    decisions: list[EasyDecision] = []
    errors: list[EasyAuditFailure] = []
    for n in range(0, len(cases), request.batch_size):
        batch = cases[n : n + request.batch_size]
        prepared = prepare_easy_batch(request, inventory, batch, checkpoints)
        try:
            decisions.extend(read_easy_checkpoint(request, prepared).decisions)
        except (OSError, ValueError) as error:
            errors.extend(
                EasyAuditFailure(
                    canonical_task_id=c.canonical_task_id,
                    error=f"Checkpoint inspection: {error}",
                )
                for c in batch
            )
    return _write_report(
        _report(request, inventory, decisions, errors), inventory, output
    )


EasyAggregateLanguage = Literal["hi", "es", "pt", "zh", "ko"]
EASY_AGGREGATE_LANGUAGES = ("hi", "es", "pt", "zh", "ko")


class EasyAggregateInput(LocalizationModel):
    """Exact source and model provenance retained for each language's review."""

    language: Annotated[
        EasyAggregateLanguage, Field(description="One of the five required languages.")
    ]
    request: Annotated[
        EasyAuditRequest,
        Field(description="Complete model settings from the original report."),
    ]
    settings: Annotated[
        RegionalSettings, Field(description="Audited country and language settings.")
    ]
    prompt_version: Annotated[
        str, Field(description="Exact audit prompt version used.")
    ]
    map_sha256: Digest
    source_sha256: Digest
    inventory_sha256: Digest
    audit_sha256: Digest
    ledger_csv_sha256: Digest
    keep_count: Annotated[
        int, Field(ge=0, description="Raw keep outcomes in this language.")
    ]
    revise_count: Annotated[
        int, Field(ge=0, description="Raw localized-repair outcomes.")
    ]
    source_blocked_count: Annotated[
        int, Field(ge=0, description="Raw source limitations.")
    ]
    error_count: Annotated[
        int, Field(ge=0, description="Explicit missing model evidence.")
    ]


class EasyFamilyCounts(LocalizationModel):
    """One language by family cell with complete ten-case accounting."""

    language: Annotated[EasyAggregateLanguage, Field(description="Reviewed language.")]
    family: Annotated[Family, Field(description="Frozen Easy entity family.")]
    keep: Annotated[int, Field(ge=0, description="Raw keep count.")]
    revise: Annotated[int, Field(ge=0, description="Raw revision count.")]
    source_blocked: Annotated[int, Field(ge=0, description="Raw source blocker count.")]
    audit_error: Annotated[int, Field(ge=0, description="Unresolved audit count.")]

    @model_validator(mode="after")
    def require_ten_cases(self):
        """No family cell may disappear from the review matrix."""
        if self.keep + self.revise + self.source_blocked + self.audit_error != 10:
            raise ValueError("Each Easy aggregate family must account for ten cases")
        return self


class EasyAuditAggregate(LocalizationModel):
    """Five-language review summary and exact 500-row ledger, never native approval."""

    version: Annotated[
        Literal["intake-easy-aggregate-v1"],
        Field(description="Aggregate schema and accounting version."),
    ] = "intake-easy-aggregate-v1"
    source_sha256: Digest
    inputs: Annotated[
        list[EasyAggregateInput],
        Field(
            min_length=5,
            max_length=5,
            description="One exact audit provenance record per required language.",
        ),
    ]
    family_counts: Annotated[
        list[EasyFamilyCounts],
        Field(
            min_length=50,
            max_length=50,
            description="Ten families in each of five languages.",
        ),
    ]
    rows: Annotated[
        list[EasyLedgerRow],
        Field(
            min_length=500,
            max_length=500,
            description="All 500 source-selected Easy cases with original decisions.",
        ),
    ]
    csv_sha256: Digest
    native_review_status: Annotated[
        Literal["pending"], Field(description="Model review is not native approval.")
    ] = "pending"

    @model_validator(mode="after")
    def validate_complete_aggregate(self):
        """Bind summary counts and CSV bytes to every exact row and input frame."""
        languages = [i.language for i in self.inputs]
        if len(set(languages)) != 5 or set(languages) != set(EASY_AGGREGATE_LANGUAGES):
            raise ValueError(
                "Easy aggregate requires each of hi/es/pt/zh/ko exactly once"
            )
        if any(
            i.source_sha256 != self.source_sha256 or i.settings.language != i.language
            for i in self.inputs
        ):
            raise ValueError(
                "Easy aggregate inputs must share a source frame and match their language"
            )
        if {(r.language, r.canonical_task_id) for r in self.rows} != {
            (language, task) for language in languages for task in _expected_ids()
        }:
            raise ValueError(
                "Easy aggregate must include exactly 500 distinct selected cases"
            )
        actual = Counter((r.language, r.family, r.disposition) for r in self.rows)
        cells = {(c.language, c.family) for c in self.family_counts}
        if cells != {
            (language, family) for language in languages for family in CANONICAL_BANKS
        }:
            raise ValueError("Easy aggregate family matrix is incomplete or duplicated")
        for cell in self.family_counts:
            if any(
                getattr(cell, kind) != actual[cell.language, cell.family, kind]
                for kind in ("keep", "revise", "source_blocked", "audit_error")
            ):
                raise ValueError("Easy aggregate family counts differ from its rows")
        for item in self.inputs:
            by_language = Counter(
                r.disposition for r in self.rows if r.language == item.language
            )
            if [
                by_language[k]
                for k in ("keep", "revise", "source_blocked", "audit_error")
            ] != [
                item.keep_count,
                item.revise_count,
                item.source_blocked_count,
                item.error_count,
            ]:
                raise ValueError("Easy aggregate input counts differ from its rows")
        if self.csv_sha256 != hashlib.sha256(_render_ledger(self.rows)).hexdigest():
            raise ValueError("Easy aggregate CSV digest differs from typed rows")
        return self


class EasyAggregateFiles(LocalizationModel):
    """Locations of the summary/provenance JSON and separate combined ledger CSV."""

    summary_file: Annotated[
        Path, Field(description="Typed aggregate summary and provenance.")
    ]
    ledger_csv: Annotated[
        Path, Field(description="CSV containing exactly 500 Easy cases.")
    ]


def build_easy_aggregate(
    reports: list[EasyAuditReport], manifests: list[EasyLedgerManifest]
) -> EasyAuditAggregate:
    """Combine five bound report/ledger pairs without recompiling maps or calling models."""
    import json

    reports = [EasyAuditReport.model_validate(r.model_dump()) for r in reports]
    manifests = [EasyLedgerManifest.model_validate(m.model_dump()) for m in manifests]
    if (
        len(reports) != 5
        or len(manifests) != 5
        or {r.settings.language for r in reports} != set(EASY_AGGREGATE_LANGUAGES)
    ):
        raise ValueError(
            "Easy aggregate requires exactly five reports for hi/es/pt/zh/ko"
        )
    if len({r.source_sha256 for r in reports}) != 1:
        raise ValueError("Easy aggregate cannot mix source frames")
    by_audit = {m.audit_sha256: m for m in manifests}
    if len(by_audit) != 5 or set(by_audit) != {content_digest(r) for r in reports}:
        raise ValueError("Every Easy aggregate ledger must bind one exact report")
    reports.sort(key=lambda r: EASY_AGGREGATE_LANGUAGES.index(r.settings.language))
    rows, inputs = [], []
    for report in reports:
        manifest = by_audit[content_digest(report)]
        if (manifest.map_sha256, manifest.source_sha256, manifest.inventory_sha256) != (
            report.map_sha256,
            report.source_sha256,
            report.inventory_sha256,
        ) or any(r.language != report.settings.language for r in manifest.rows):
            raise ValueError("Easy aggregate ledger provenance differs from report")
        decisions = {d.canonical_task_id: d for d in report.decisions}
        errors = {e.canonical_task_id: e for e in report.errors}
        by_id = {r.canonical_task_id: r for r in manifest.rows}
        for task_id in report.expected_task_ids:
            row, d = by_id[task_id], decisions.get(task_id)
            expected = (
                (
                    d.disposition,
                    d.source_difficulty,
                    d.localized_difficulty,
                    d.evidence,
                    d.dependency_issues,
                    d.correction_instructions,
                    d.repair_targets,
                    d.source_blocked_reason or "",
                )
                if d
                else ("audit_error", "", "", errors[task_id].error, [], [], [], "")
            )
            observed = (
                row.disposition,
                row.source_difficulty,
                row.localized_difficulty,
                row.evidence,
                json.loads(row.dependency_issues),
                json.loads(row.correction_instructions),
                json.loads(row.repair_targets),
                row.source_blocked_reason,
            )
            if observed != expected or not task_id.startswith(
                f"intake_{row.family}_easy_"
            ):
                raise ValueError(
                    "Easy aggregate ledger row differs from exact report evidence"
                )
            rows.append(row)
        inputs.append(
            EasyAggregateInput(
                language=report.settings.language,
                request=report.request,
                settings=report.settings,
                prompt_version=report.prompt_version,
                map_sha256=report.map_sha256,
                source_sha256=report.source_sha256,
                inventory_sha256=report.inventory_sha256,
                audit_sha256=content_digest(report),
                ledger_csv_sha256=manifest.csv_sha256,
                keep_count=report.keep_count,
                revise_count=report.revise_count,
                source_blocked_count=report.source_blocked_count,
                error_count=report.error_count,
            )
        )
    counts = Counter((r.language, r.family, r.disposition) for r in rows)
    cells = [
        EasyFamilyCounts(
            language=language,
            family=family,
            keep=counts[language, family, "keep"],
            revise=counts[language, family, "revise"],
            source_blocked=counts[language, family, "source_blocked"],
            audit_error=counts[language, family, "audit_error"],
        )
        for language in EASY_AGGREGATE_LANGUAGES
        for family in CANONICAL_BANKS
    ]
    return EasyAuditAggregate(
        source_sha256=reports[0].source_sha256,
        inputs=inputs,
        family_counts=cells,
        rows=rows,
        csv_sha256=hashlib.sha256(_render_ledger(rows)).hexdigest(),
    )


def write_easy_aggregate(
    aggregate: EasyAuditAggregate, output: Path
) -> EasyAggregateFiles:
    """Write immutable model-backed aggregate JSON and its 500-row CSV."""
    aggregate = EasyAuditAggregate.model_validate(aggregate.model_dump())
    files = EasyAggregateFiles(
        summary_file=output, ledger_csv=output.with_suffix(".csv")
    )
    data = _render_ledger(aggregate.rows)
    if files.ledger_csv.exists() and files.ledger_csv.read_bytes() != data:
        raise ValueError("Refusing to overwrite a different Easy aggregate ledger")
    write_artifact(aggregate, output)
    _write_csv(data, files.ledger_csv)
    return files
