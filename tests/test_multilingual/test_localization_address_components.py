"""Street repairs conserve exact components and never reinterpret units or PINs."""

import pytest

from tau2.multilingual.factory.localization_address_components import repair_street_line


@pytest.mark.parametrize(
    "language,representation,before,after",
    [
        ("es", "native", "616 Calle de Alcalá", "Calle de Alcalá 616"),
        (
            "es",
            "native",
            "548 Gran Via de les Corts Catalanes",
            "Gran Via de les Corts Catalanes 548",
        ),
        ("pt", "native", "993 Rua das Acácias", "Rua das Acácias, 993"),
        ("zh", "native", "986号锦江路", "锦江路986号"),
        ("zh", "romanized", "986 Hao Jinjiang Lu", "Jinjiang Lu 986 Hao"),
        ("zh", "native", "975 干将东路", "干将东路975号"),
        ("zh", "romanized", "975 Ganjiang Dong Lu", "Ganjiang Dong Lu 975 Hao"),
        ("ko", "native", "570 종로", "종로 570"),
        ("ko", "romanized", "570 Jong-ro", "Jong-ro 570"),
    ],
)
def test_known_road_components_reorder_without_numeric_changes(
    language, representation, before, after
):
    assert repair_street_line(before, language, representation) == after
    assert repair_street_line(after, language, representation) == after


@pytest.mark.parametrize(
    "language,value",
    [
        ("hi", "986 प्रभात रोड"),
        ("en", "986 Sunset Drive"),
        ("es", "Calle 9, 123"),
        ("es", "28005"),
        ("zh", "986室"),
        ("ko", "570호"),
        ("pt", "993 Caixa Postal"),
        ("zh", "9K2X7M4P1N8Q3R5T6A"),
        ("es", "12B Unknown Place"),
    ],
)
def test_valid_other_locale_and_ambiguous_nonroad_values_are_not_reinterpreted(
    language, value
):
    assert repair_street_line(value, language, "native") == value


def test_source_bound_prose_changes_but_unrelated_people_and_ids_do_not():
    from test_localization_user_ids import _entry, _proposal

    from tau2.multilingual.factory.localization_address_components import (
        repair_address_components,
    )
    from tau2.utils.artifacts import content_digest

    proposal = _proposal()
    road = _entry(
        "616 Maple Street",
        ["street"],
        category="address",
        romanized="616 Calle de Alcalá",
    )
    road.context = "role=address1; entity=alice; source_address={}"
    prose = _entry(
        "Your address is 616 Maple Street; keep ZIP 12345.",
        ["prose"],
        category="prose",
        romanized="Your address is 616 Calle de Alcalá; keep ZIP 54321.",
    )
    prose.context = "Scenario/evaluation prose for alice; source-bound fixture"
    proposal.entries.extend([road, prose])
    fixed, report = repair_address_components(proposal)
    assert fixed.entries[:3] == proposal.entries[:3]
    assert (
        fixed.entries[-1].forms.native
        == "Your address is Calle de Alcalá 616; keep ZIP 54321."
    )
    assert fixed.source == proposal.source
    assert fixed.provenance == proposal.provenance
    assert fixed.address_revision.parent_map_sha256 == content_digest(proposal)
    assert len(report.changes) == 2
    assert not report.review_findings
    again, second_report = repair_address_components(fixed)
    assert again == fixed
    assert not second_report.changes


def test_unbound_equal_looking_address_in_prose_is_reported_not_replaced():
    from test_localization_user_ids import _entry, _proposal

    from tau2.multilingual.factory.localization_address_components import (
        repair_address_components,
    )

    proposal = _proposal()
    road = _entry(
        "616 Maple Street",
        ["street"],
        category="address",
        romanized="616 Calle de Alcalá",
    )
    road.context = "role=address1; entity=alice; source_address={}"
    unrelated = _entry(
        "Your friend lives at 1 Elm Street.",
        ["prose"],
        category="prose",
        romanized="Your friend lives at 616 Calle de Alcalá.",
    )
    proposal.entries.extend([road, unrelated])
    fixed, report = repair_address_components(proposal)
    assert fixed.entries[-1] == unrelated
    assert len(report.review_findings) == 2


def test_explicit_tool_refresh_refuses_changed_database(tmp_path, monkeypatch):
    from test_localization_user_ids import _proposal

    from tau2.multilingual.factory import localization_inventory
    from tau2.multilingual.factory.localization_address_components import (
        read_address_repair_input,
    )
    from tau2.multilingual.factory.localization_artifact import (
        AuthoredLocalization,
        RegionalMapArtifact,
        _manifest,
    )

    proposal = _proposal()
    artifact = RegionalMapArtifact(
        revision=proposal.revision,
        settings=proposal.settings,
        source=_manifest(proposal.source),
        entries=[
            AuthoredLocalization.model_validate(
                entry.model_dump(include=set(AuthoredLocalization.model_fields))
            )
            for entry in proposal.entries
            if entry.category != "protocol"
        ],
        provenance=proposal.provenance,
    )
    path = tmp_path / "map.json"
    path.write_text(artifact.model_dump_json())
    changed = proposal.source.model_copy(deep=True)
    changed.documents[0].sha256 = "f" * 64
    monkeypatch.setattr(
        localization_inventory, "inventory_localization", lambda *args: changed
    )
    with pytest.raises(ValueError, match="cannot change database"):
        read_address_repair_input(path, True)


@pytest.mark.parametrize(
    "canonical,localized",
    [("112 Main Street", "112 测试路"), ("12 Main Street", "12 测试路")],
)
def test_prefix_collision_and_another_owner_never_rewrite_their_address(
    canonical, localized
):
    from test_localization_user_ids import _entry, _proposal

    from tau2.multilingual.factory.localization_address_components import (
        repair_address_components,
    )

    proposal = _proposal()
    proposal.settings.language = "zh"
    road = _entry(
        "12 Main Street", ["street"], category="address", romanized="12 测试路"
    )
    road.context = "role=address1; entity=alice; source_address={}"
    prose = _entry(
        f"Deliver to {canonical}.",
        ["prose"],
        category="prose",
        romanized=f"Deliver to {localized}.",
    )
    prose.context = "Scenario/evaluation prose for bob; source-bound fixture"
    proposal.entries.extend([road, prose])
    fixed, report = repair_address_components(proposal)
    assert fixed.entries[-1] == prose
    if canonical == "12 Main Street":
        assert report.review_findings


def test_tool_refresh_rejects_unlisted_semantic_change(tmp_path, monkeypatch):
    from test_localization_user_ids import _proposal

    from tau2.multilingual.factory import localization_inventory
    from tau2.multilingual.factory.localization_address_components import (
        ToolEvidenceTransition,
        read_address_repair_input,
    )
    from tau2.multilingual.factory.localization_artifact import (
        AuthoredLocalization,
        RegionalMapArtifact,
        _manifest,
    )
    from tau2.multilingual.localization import SourceDocument

    proposal = _proposal()
    proposal.source.documents.append(
        SourceDocument(
            id="tool_text:tools.py",
            kind="tool_text",
            source_path="../src/tau2/domains/retail/tools.py",
            sha256="1" * 64,
            content=[],
        )
    )
    artifact = RegionalMapArtifact(
        revision=proposal.revision,
        settings=proposal.settings,
        source=_manifest(proposal.source),
        entries=[
            AuthoredLocalization.model_validate(
                entry.model_dump(include=set(AuthoredLocalization.model_fields))
            )
            for entry in proposal.entries
            if entry.category != "protocol"
        ],
        provenance=proposal.provenance,
    )
    path = tmp_path / "map.json"
    path.write_text(artifact.model_dump_json())
    changed = proposal.source.model_copy(deep=True)
    changed.documents[-1].sha256 = "2" * 64
    monkeypatch.setattr(
        localization_inventory, "inventory_localization", lambda *args: changed
    )
    transition = ToolEvidenceTransition(
        domain="retail",
        document_id="tool_text:tools.py",
        before_sha256="1" * 64,
        after_sha256="3" * 64,
        reason="Inspected fixture transition, not permission for another hash.",
    )
    for transitions in [None, [transition]]:
        with pytest.raises(ValueError, match="exact inspected transition"):
            read_address_repair_input(path, True, transitions)
