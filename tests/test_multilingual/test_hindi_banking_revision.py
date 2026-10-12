"""Hindi entity revision preserves evidence and excludes private donor forms."""

import pytest

from tau2.multilingual.factory.banking_candidate_review import (
    EntitySpan,
    normalized_phone,
    validate_discovered_spans,
)
from tests.test_multilingual.test_banking_prose import entry, proposal


def test_phone_normalization_detects_cross_script_collisions():
    assert normalized_phone("९८७६५४३२१०") == normalized_phone("987-654-3210")


def test_discovery_rejects_invented_source_span():
    p = proposal(entry("text", "Lives in Denver.", "Lives in Denver."))
    span = EntitySpan(
        source="Seattle",
        role="residence",
        disposition="generate",
        reference_id=None,
        reason="Test",
    )
    with pytest.raises(ValueError, match="source"):
        validate_discovered_spans(p, {"text": [span]})


def test_discovery_cannot_replace_kb_bank_identity():
    p = proposal(entry("text", "Rho-Bank customer.", "Rho-Bank customer."))
    span = EntitySpan(
        source="Rho-Bank",
        role="bank",
        disposition="generate",
        reference_id=None,
        reason="Test",
    )
    with pytest.raises(ValueError, match="protected"):
        validate_discovered_spans(p, {"text": [span]})


def test_structured_revision_masks_family_forms(monkeypatch, tmp_path):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from tau2.multilingual.factory.banking_candidate_review import (
        BankingReviewRules,
        _revise_batch,
    )

    rules = BankingReviewRules.model_validate_json(
        Path(
            "data/tau2/multilingual/hi/localization/banking_review_rules.json"
        ).read_text()
    )
    from tau2.utils import llm_utils

    def fake_generate(**kwargs):
        payload = json.loads(kwargs["messages"][1].content)
        assert "PRIVATE_FAMILY" not in json.dumps(payload)
        token = payload["completed_mappings"][0]["forms"]
        return SimpleNamespace(
            content=json.dumps(
                {
                    "entries": {
                        "name": {
                            "forms": {
                                "native": "आरव " + token["native"],
                                "romanized": "Aarav " + token["romanized"],
                                "english": "Test",
                            },
                            "operation": "adapt",
                            "notes": "Pending review",
                        }
                    }
                }
            )
        )

    monkeypatch.setattr(llm_utils, "generate", fake_generate)
    result = _revise_batch(
        [{"id": "name", "source": "Kenji Tanaka", "kind": "name"}],
        [
            {
                "id": "family",
                "source": "Required family",
                "forms": {
                    "native": "PRIVATE_FAMILY",
                    "romanized": "PRIVATE_FAMILY",
                    "english": "Private",
                },
            }
        ],
        tmp_path,
        "test",
        {},
        rules,
    )
    assert result.entries["name"]["forms"]["native"] == "आरव PRIVATE_FAMILY"


def test_address_component_uses_explicit_locality_independent_of_punctuation():
    from tau2.domains.banking_knowledge.localization_bindings import address_component

    value = entry(
        "address",
        "US address",
        "24 अशोक रोड, नई दिल्ली, दिल्ली 110001, भारत",
        "address",
        romanized="24 Ashoka Road, New Delhi, Delhi 110001, India",
    )
    from tau2.multilingual.address_format import AddressForms, AddressParts

    value.forms.address = AddressForms(
        language="hi",
        native=AddressParts(
            street_lines=["24 अशोक रोड"],
            city="नई दिल्ली",
            region="दिल्ली",
            postal_code="११०००१",
        ),
        romanized=AddressParts(
            street_lines=["24 Ashoka Road"],
            city="New Delhi",
            region="Delhi",
            postal_code="110001",
        ),
    )
    assert address_component(value, "address_city").romanized == "New Delhi"
    value.forms.native = "24 अशोक रोड, नई दिल्ली, दिल्ली, 110001, भारत"
    value.forms.romanized = "24 Ashoka Road, New Delhi, Delhi, 110001, India"
    assert (
        address_component(value, "address_city_region").romanized == "New Delhi, Delhi"
    )


def test_merchant_collision_guard_respects_explicit_source_aliases():
    from tau2.multilingual.factory.banking_candidate_review import merchant_collisions

    p = proposal(
        entry("a", "Comcast", "JioFiber", "other"),
        entry("b", "XFINITY", "JioFiber", "other"),
        entry("c", "Other utility", "JioFiber", "other"),
    )
    assert merchant_collisions(p, [["Comcast", "XFINITY"]]) == {"a", "b", "c"}
    p.entries.pop()
    assert not merchant_collisions(p, [["Comcast", "XFINITY"]])


def test_address_component_rejects_invalid_part_and_preserves_unicode_digits():
    from tau2.domains.banking_knowledge.localization_bindings import address_component

    value = entry(
        "address",
        "US address",
        "24 अशोक रोड, नई दिल्ली, दिल्ली ११०००१, भारत",
        "address",
        romanized="24 Ashoka Road, New Delhi, Delhi 110001, India",
    )
    from tau2.multilingual.address_format import AddressForms, AddressParts

    value.forms.address = AddressForms(
        language="hi",
        native=AddressParts(
            street_lines=["24 अशोक रोड"],
            city="नई दिल्ली",
            region="दिल्ली",
            postal_code="११०००१",
        ),
        romanized=AddressParts(
            street_lines=["24 Ashoka Road"],
            city="New Delhi",
            region="Delhi",
            postal_code="110001",
        ),
    )
    assert address_component(value, "address_region").native == "दिल्ली"
    assert value.forms.address.native.postal_code == "११०००१"
    with pytest.raises(ValueError, match="Unknown"):
        address_component(value, "ignored")
    value.forms.address = None
    with pytest.raises(ValueError, match="Explicit address components required"):
        address_component(value, "address_city")


def test_absent_address_component_metadata_does_not_change_old_serialization():
    from tau2.multilingual.localization import LocalizedForms, ProseEntityBinding

    b = ProseEntityBinding(
        entry_id="text",
        source="SOURCE",
        forms=LocalizedForms(native="SOURCE", romanized="SOURCE", english="test"),
        kind="protected",
        reason="protocol",
    )
    assert "source_part" not in b.model_dump(mode="json")
    from tau2.domains.banking_knowledge.localization_bindings import (
        validate_prose_bindings,
    )

    p = proposal(entry("text", "SOURCE", "SOURCE"))
    p.prose_bindings = [b.model_copy(update={"source_part": "address_city"})]
    with pytest.raises(ValueError, match="requires a reuse"):
        validate_prose_bindings(p)


def test_protected_financial_service_and_terminal_digits_remain_source_bound():
    from pathlib import Path

    from tau2.multilingual.factory.banking_candidate_review import (
        BankingReviewRules,
        DiscoveryCheckpoint,
        _binding_plan,
    )

    rules = BankingReviewRules.model_validate_json(
        Path(
            "data/tau2/multilingual/hi/localization/banking_review_rules.json"
        ).read_text()
    )
    rules.span_overrides = {}
    p = proposal(entry("text", "EveryonePay transfer; QUICKCASH ATM #7749.", "old"))
    discovery = DiscoveryCheckpoint(
        input_sha256="a" * 64,
        model="test",
        model_args={},
        prompt_version="test",
        entries={
            "text": [
                EntitySpan(
                    source="EveryonePay",
                    role="payment service",
                    disposition="generate",
                    reference_id=None,
                    reason="Name",
                ),
                EntitySpan(
                    source="QUICKCASH ATM #7749",
                    role="merchant",
                    disposition="generate",
                    reference_id=None,
                    reason="Name",
                ),
            ]
        },
    )
    ready, requests, members = _binding_plan(p, discovery, rules)
    assert ready[0].kind == "protected"
    assert ready[0].forms.native == "EveryonePay"
    assert requests[0]["source"] == "QUICKCASH ATM"
    assert members[requests[0]["id"]][0][1].source == "QUICKCASH ATM"


def test_inline_friend_address_cannot_be_replaced_by_home_city_only():
    from tau2.multilingual.factory.banking_candidate_review import _residence_binding

    e = entry("text", "A friend lives at 2156 Pine Avenue, Portland, OR 97205.", "old")
    p = proposal(e)
    span = EntitySpan(
        source="2156 Pine Avenue, Portland, OR 97205",
        role="friend current residence address",
        disposition="generate",
        reference_id=None,
        reason="Across town",
    )
    assert _residence_binding(p, e, span) is None


def test_residence_requires_person_evidence_not_coincidental_city(monkeypatch):
    from tau2.multilingual.factory import banking_candidate_review as review

    e = entry("text", "Caller lives in Seattle.", "old")
    address = entry(
        "address",
        "1 Lane, Seattle, WA 98001",
        "1 Road, Pune, Maharashtra, 411004, India",
        "address",
        owner="other-user",
    )
    p = proposal(e, address)
    monkeypatch.setattr(review, "source_person_owners", lambda p, e: {"actual-user"})
    span = EntitySpan(
        source="Seattle",
        role="residence city",
        disposition="address_city",
        reference_id="address",
        reason="Bad same-city guess",
    )
    with pytest.raises(ValueError, match="person/account evidence"):
        review._residence_binding(p, e, span)


def test_duplicate_discovery_surfaces_merge_only_when_forms_agree():
    from tau2.multilingual.factory.banking_candidate_review import _unique_bindings
    from tau2.multilingual.localization import LocalizedForms, ProseEntityBinding

    b = ProseEntityBinding(
        entry_id="text",
        source="Merchant",
        forms=LocalizedForms(native="VALUE", romanized="VALUE", english="test"),
        kind="protected",
        reason="test",
    )
    assert _unique_bindings([b, b.model_copy()]) == [b]
    other = b.model_copy(deep=True)
    other.forms.native = "CONFLICT"
    with pytest.raises(ValueError, match="Conflicting"):
        _unique_bindings([b, other])


def test_remote_fraud_merchant_protection_stays_with_source_task():
    from pathlib import Path

    from tau2.multilingual.factory.banking_candidate_review import (
        BankingReviewRules,
        DiscoveryCheckpoint,
        _protect_foreign_fraud_identities,
    )
    from tau2.multilingual.localization import LocalizedForms, ProseEntityBinding

    remote = entry(
        "remote",
        "SHOP MIAMI FL",
        "Indian Shop MIAMI FL",
        role="transaction_description",
    )
    local = entry(
        "local", "SHOP", "Indian Shop", task=1, role="transaction_description"
    )
    p = proposal(remote, local)
    p.prose_bindings = [
        ProseEntityBinding(
            entry_id=e.id,
            source="SHOP",
            forms=LocalizedForms(
                native="Indian Shop", romanized="Indian Shop", english="test"
            ),
            kind="generated",
            reason="test",
            provenance=p.provenance,
        )
        for e in (remote, local)
    ]
    discovery = DiscoveryCheckpoint(
        input_sha256="a" * 64,
        model="test",
        model_args={},
        prompt_version="test",
        entries={
            "remote": [
                EntitySpan(
                    source="SHOP",
                    role="merchant",
                    disposition="generate",
                    reference_id=None,
                    reason="Retail",
                ),
                EntitySpan(
                    source="MIAMI FL",
                    role="remote fraud location",
                    disposition="preserve",
                    reference_id=None,
                    reason="Fraud city",
                ),
            ]
        },
    )
    rules = BankingReviewRules.model_validate_json(
        Path(
            "data/tau2/multilingual/hi/localization/banking_review_rules.json"
        ).read_text()
    )
    result = _protect_foreign_fraud_identities(p, discovery, rules)
    assert result.prose_bindings[0].forms.native == "SHOP"
    assert result.prose_bindings[1].forms.native == "Indian Shop"
