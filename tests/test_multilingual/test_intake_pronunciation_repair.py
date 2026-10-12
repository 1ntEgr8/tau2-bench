"""Deterministic pronunciation guides preserve actual speech stimuli and lineage."""

from pathlib import Path

import pytest

from tau2.multilingual.factory.intake_authoring import validate_pronunciation_script
from tau2.multilingual.factory.intake_pronunciation import repair_native_references
from tau2.multilingual.factory.localization_artifact import RegionalMapArtifact
from tau2.utils.utils import DATA_DIR


def test_reference_repair_preserves_tokens_distortions_and_other_metadata():
    original = RegionalMapArtifact.model_validate_json(
        (Path(DATA_DIR) / "tau2/multilingual/hi/localization/intake.json").read_text()
    )
    entry = next(
        e
        for e in original.entries
        if isinstance(e.forms.native, list)
        and e.forms.native
        and isinstance(e.forms.native[0], dict)
        and "distorted" in e.forms.native[0]
    )
    entry.forms.native[0]["reference"] = "STRESS-guide"
    entry.forms.romanized[0]["reference"] = "STRESS-guide"
    result, record = repair_native_references(original, "test-pronunciation-repair")
    assert record.changes
    assert record.status == "pending_native_review"
    for before, after in zip(original.entries, result.entries):
        expected = before.model_copy(deep=True)
        for form in ("native", "romanized"):
            values = getattr(expected.forms, form)
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, dict) and "distorted" in value:
                        value["reference"] = value["token"]
        assert expected == after
    repeated, second = repair_native_references(result, result.revision)
    assert repeated == result
    assert second.changes == []


@pytest.mark.parametrize("field", ["token", "reference", "distorted"])
def test_hindi_native_pronunciations_reject_latin_in_each_speech_field(field):
    values = [
        {
            "token": "किरण",
            "reference": "किरण",
            "distorted": "करण",
            "boundary": "word",
            "reason": "vowel_swap",
        }
    ]
    values[0][field] = "Ki-RAN"
    with pytest.raises(ValueError, match="native Hindi"):
        validate_pronunciation_script(values, "hi", "native")


def test_romanized_reference_is_plain_token_not_stress_respellings():
    values = [
        {
            "token": "Liladhar",
            "reference": "LEE-laa-dhar",
            "distorted": "Leladhar",
            "boundary": "word",
            "reason": "vowel_swap",
        }
    ]
    with pytest.raises(ValueError, match="exact token"):
        validate_pronunciation_script(values, "hi", "romanized")
    values[0]["reference"] = "Liladhar"
    validate_pronunciation_script(values, "hi", "romanized")
    validate_pronunciation_script(values, "es", "native")


@pytest.mark.parametrize("representation", ["native", "romanized"])
def test_compiler_rejects_stress_guides_before_publication(representation):
    from tau2.multilingual.factory.localization_artifact import read_localization_map
    from tau2.multilingual.factory.localization_compile import compile_localization

    proposal = read_localization_map(
        Path(DATA_DIR) / "tau2/multilingual/hi/localization/intake.json"
    )
    entry = next(
        e
        for e in proposal.entries
        if "dependency=pronunciations;" in e.context and e.forms.native
    )
    getattr(entry.forms, representation)[0]["reference"] = "STRESSED-guide"
    with pytest.raises(ValueError, match="native Hindi|exact token"):
        compile_localization(proposal)
