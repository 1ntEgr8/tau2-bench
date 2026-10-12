"""Banking address repairs preserve identity and protocol values."""

from types import SimpleNamespace

import pytest

from tau2.domains.banking_knowledge.localization_bindings import address_component
from tau2.multilingual.factory.banking_address_repair import (
    decompose_address,
    repair_address_forms,
)
from tau2.multilingual.factory.generation_config import FactoryModelConfig
from tau2.multilingual.localization import LocalizedForms


def test_korean_district_slot_is_not_silently_treated_as_city():
    forms = LocalizedForms(
        native="권선로 731 402호, 권선구, 수원시, 16571, 대한민국",
        romanized="Gwonseon-ro 731 402-ho, Gwonseon-gu, Suwon-si, 16571, Daehanminguk",
        english="Candidate",
    )
    fixed, findings = repair_address_forms(forms, "ko")
    assert fixed.native == "대한민국 수원시 권선구 권선로 731 402호 (16571)"
    assert fixed.address.native.city == "수원시"
    assert fixed.address.native.region == ""
    assert findings
    assert (
        address_component(
            SimpleNamespace(category="address", forms=fixed), "address_city"
        ).native
        == "수원시"
    )
    assert repair_address_forms(fixed, "ko")[0] == fixed


def test_city_prefix_deduplication_and_span_identity():
    forms = LocalizedForms(
        native="西安市碑林区友谊东路66号1号楼908室, 西安市, 陕西省, 710001, 中国",
        romanized="Xi'an Shi Beilin Qu Youyi Dong Lu 66 Hao 1 Hao Lou 908 Shi, Xi'an Shi, Shaanxi Sheng, 710001, Zhongguo",
        english="Candidate",
    )
    fixed, _ = repair_address_forms(forms, "zh")
    assert fixed.native == "中国陕西省西安市碑林区友谊东路66号1号楼908室，710001"
    assert fixed.romanized.count("Xi'an Shi") == 1
    assert repair_address_forms(fixed, "zh")[0] == fixed


def test_ambiguous_or_invalid_input_is_not_invented():
    with pytest.raises(ValueError):
        decompose_address("9K2X7M4P1N8Q3R5T6A", "zh", "native")
    with pytest.raises(ValueError, match="Explicit"):
        address_component(
            SimpleNamespace(
                category="address",
                forms=LocalizedForms(
                    native="street, city, province, 12345, country",
                    romanized="same",
                    english="Candidate",
                ),
            ),
            "address_city",
        )


def test_changed_display_rejects_stale_component_metadata():
    forms = LocalizedForms(
        native="Calle Mayor 12, Madrid, Madrid, 28001, España",
        romanized="Calle Mayor 12, Madrid, Madrid, 28001, España",
        english="Candidate",
    )
    fixed, _ = repair_address_forms(forms, "es")
    fixed.native = fixed.native.replace("12", "99")
    with pytest.raises(ValueError, match="component evidence"):
        repair_address_forms(fixed, "es")


def test_address_metadata_does_not_leak_private_donor_components():
    from tau2.domains.banking_knowledge.localization_tokens import (
        BankingReferenceTokens,
        banking_response_model,
        banking_response_schema,
    )

    payload = {
        "completed_mappings": [
            {
                "id": "address",
                "forms": {
                    "native": "private native",
                    "romanized": "private roman",
                    "english": "private explanation",
                    "address": {"private": "component"},
                },
            }
        ]
    }
    BankingReferenceTokens.protect(payload)
    assert "address" not in payload["completed_mappings"][0]["forms"]
    schema = banking_response_schema(banking_response_model(["address"]))
    for model in schema["$defs"].values():
        if "properties" in model:
            assert set(model["required"]) == set(model["properties"])


def test_generation_rejects_foreign_street_order_including_templates():
    from tau2.multilingual.factory import database_localization as drafting
    from tau2.multilingual.localization import (
        LocalizationSource,
        SourceDocument,
        json_content_digest,
    )
    from tests.test_multilingual.test_database_localization_draft import (
        region,
        source_entry,
    )

    entry = source_entry("street", "123 Source Street", "address", "street")
    content = {entry.id: entry.source}
    source = LocalizationSource(
        domain="retail",
        documents=[
            SourceDocument(
                id="database",
                kind="database",
                source_path="fixture.json",
                sha256=json_content_digest(content),
                content=content,
            )
        ],
        entries=[entry],
    )
    settings = region().model_copy(update={"language": "zh", "country_code": "CN"})
    request = drafting.RegionalDraftRequest(
        source=source,
        settings=settings,
        revision="test",
        generation=FactoryModelConfig(model="test", reasoning_effort="low"),
    )
    batch = drafting.BatchInput(request_sha256="a" * 64, entries=[entry], completed=[])
    for value in ("123号锦江路", "{number_0}号锦江路"):
        generated = drafting.GeneratedBatch.model_validate(
            {
                "entries": [
                    {
                        "id": entry.id,
                        "forms": {
                            "native": value,
                            "romanized": "Jinjiang Lu 123 Hao",
                            "english": "Fixture",
                        },
                        "operation": "adapt",
                        "notes": "Fixture",
                    }
                ]
            }
        )
        with pytest.raises(ValueError, match="premise/road order"):
            drafting._validate_batch(batch, generated, request=request)


def test_repair_retains_prose_lineage_and_invalidates_changed_reuse(monkeypatch):
    from tau2.multilingual.factory import localization_compile
    from tau2.multilingual.factory.banking_address_repair import (
        repair_banking_addresses,
    )
    from tau2.multilingual.localization import EntityReuseProvenance
    from tau2.utils.artifacts import ReviewApproval
    from tests.test_multilingual.test_banking_prose import entry, proposal

    address = entry(
        "address",
        "12 Main Street",
        "Calle Mayor 12, Madrid, Madrid, 28001, España",
        "address",
    )
    address.reuse = EntityReuseProvenance(
        source_domain="retail",
        source_entry_ids=["donor"],
        source_artifact_sha256="b" * 64,
        approval=ReviewApproval(content_sha256="b" * 64, reviewed_by="Fixture"),
    )
    prose = entry(
        "text",
        "Visit 12 Main Street.",
        "Visit Calle Mayor 12, Madrid, Madrid, 28001, España.",
    )
    original = proposal(address, prose)
    # Fixture has no real task document; exercise repair/reconstruction, whose
    # source and compiled task validation are covered by integration checks.
    monkeypatch.setattr(
        localization_compile, "validate_localization", lambda value: value
    )
    repaired, report = repair_banking_addresses(original)
    assert repaired.entries[0].reuse is None
    assert report.reconstruction.changes
    assert report.reconstruction.changes[0].previous == prose.forms.native
    assert (
        report.reconstruction.changes[0].reconstructed
        == repaired.entries[1].forms.native
    )
    again, repeated = repair_banking_addresses(repaired)
    assert again == repaired
    assert not repeated.changed_entry_ids
    assert not repeated.reconstruction.changes
