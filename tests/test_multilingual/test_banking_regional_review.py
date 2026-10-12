"""Locale review keeps geographic components and target speech representations."""

import pytest

from tau2.domains.banking_knowledge.localization_bindings import address_component
from tau2.multilingual.address_format import AddressForms, AddressParts, format_locality
from tau2.multilingual.factory.banking_candidate_review import (
    address_premise,
    street_premise,
)
from tests.test_multilingual.test_banking_prose import entry


@pytest.mark.parametrize(
    "address,city,region",
    [
        ("Calle de Prueba 12, Madrid, Madrid, 28001, España", "Madrid", "Madrid"),
        ("Rua Teste 12, São Paulo, SP, 01310-100, Brasil", "São Paulo", "SP"),
        (
            "시험로 12, 서울특별시, 서울특별시, 03000, 대한민국",
            "서울특별시",
            "서울특별시",
        ),
        ("测试路 12号, 北京市, 北京市, 100000, 中国", "北京市", "北京市"),
        ("12 Test Road, Delhi, Delhi 110001, India", "Delhi", "Delhi"),
    ],
)
def test_locality_is_derived_exactly_from_existing_address(address, city, region):
    value = entry("address", "Source", address, category="address")
    parts = AddressParts(street_lines=["Road 12"], city=city, region=region)
    value.forms.address = AddressForms(language="es", native=parts, romanized=parts)
    assert address_component(value, "address_city").native == city
    assert address_component(value, "address_region").native == region
    assert address_component(value, "address_city_region").native == format_locality(
        parts, "es", "native"
    )


@pytest.mark.parametrize(
    "value,number,street",
    [
        ("12 Test Road", "12", "Test Road"),
        ("Calle Test 34", "34", "Calle Test"),
        ("도로14길 8-2", "8-2", "도로14길"),
        ("测试路 12号", "12", "测试路"),
        ("金桥二路22号", "22", "金桥二路"),
        ("권선로 731 402호", "731", "권선로"),
        ("Gwonseon-ro 731 402-ho", "731", "Gwonseon-ro"),
        ("도로14길 8-2 101동 402호", "8-2", "도로14길"),
    ],
)
def test_premise_extraction_preserves_embedded_road_numbers(value, number, street):
    assert street_premise(value) == (number, street)


def test_separate_brazilian_premise_preserves_road_number():
    assert address_premise(
        AddressParts(
            street_lines=["Rua XV de Novembro", "139", "Apto. 418"],
            city="Curitiba",
            region="PR",
            postal_code="80020-310",
            country="Brasil",
        )
    ) == ("139", "Rua XV de Novembro")


def test_compact_family_names_require_explicit_source_review():
    from pathlib import Path

    from tau2.multilingual.factory.banking_candidate_review import BankingReviewRules

    rules = BankingReviewRules.model_validate_json(
        Path(
            "data/tau2/multilingual/hi/localization/banking_review_rules.json"
        ).read_text()
    )
    rules.family_name_position = "first"
    assert rules.family_token("刘 宇轩", "native") == "刘"
    with pytest.raises(ValueError, match="source-verified"):
        rules.family_token("刘宇轩", "native")
    rules.compact_family_name_length = 1
    assert rules.family_token("刘宇轩", "native") == "刘"
    rules.compact_family_name_length = 2
    assert rules.family_token("欧阳明", "native") == "欧阳"


def test_merchant_collision_checks_native_script_and_honors_source_aliases():
    from tau2.multilingual.factory.banking_candidate_review import merchant_collisions
    from tests.test_multilingual.test_banking_prose import proposal

    left = entry("left", "Source One", "초록마을", "other", romanized="Chorok Maeul")
    right = entry("right", "Source Two", "초록마을", "other", romanized="Chorocmaeul")
    value = proposal(left, right)
    assert merchant_collisions(value, []) == {"left", "right"}
    assert merchant_collisions(value, [["Source One", "Source Two"]]) == set()


def test_bounded_business_repair_reserves_untargeted_prose_without_donor_values():
    from pathlib import Path

    from tau2.multilingual.factory.banking_candidate_review import (
        BankingReviewRules,
        reserved_prose_business_names,
    )
    from tau2.multilingual.localization import ProseEntityBinding
    from tests.test_multilingual.test_banking_prose import proposal

    rules = BankingReviewRules.model_validate_json(
        Path(
            "data/tau2/multilingual/hi/localization/banking_review_rules.json"
        ).read_text()
    )
    rules.source_aliases = {"CHASE": "CHASE BANK"}
    merchant = entry("merchant", "Source Shop", "가게", "other", romanized="Gage")
    competitor = entry("competitor", "Pursuit", "도담은행", romanized="Dodam Bank")
    alias = entry("alias", "CHASE", "새서울은행", romanized="Sae Seoul Bank")
    donor = entry(
        "donor", "Private Person", "PRIVATE_NATIVE", romanized="PRIVATE_LATIN"
    )
    ready = [
        ProseEntityBinding(
            entry_id=e.id,
            source=e.source,
            forms=e.forms,
            kind=kind,
            reason="Test scoped reference.",
        )
        for e, kind in [
            (competitor, "generated"),
            (alias, "generated"),
            (donor, "reuse"),
        ]
    ]
    reserved = reserved_prose_business_names(
        proposal(merchant), ready, "CHASE BANK", rules
    )
    assert set(reserved) == {"가게", "Gage", "도담은행", "Dodam Bank"}
