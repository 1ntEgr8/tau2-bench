"""Regional name evidence stays source-bound and cannot assign ambiguous genders."""

import pytest

from tau2.multilingual.factory.localization_gender import GenderNameResult
from tau2.multilingual.factory.regional_name_gender import (
    apply_name_assessments,
    name_surface_groups,
)
from tests.test_multilingual.test_banking_prose import entry, proposal


def regional(*entries):
    mapping = proposal(*entries)
    mapping.source.domain = "retail"
    return mapping


def given(identifier, source="Juan", native="娟", owner="person-a"):
    return entry(
        identifier,
        source,
        native,
        category="identity_name",
        role="first_name",
        owner=owner,
        romanized="Juan",
    )


def test_repeated_surfaces_share_assessment_without_losing_owner_coverage():
    first, second = given("a"), given("b", owner="person-b")
    family = entry("family", "Li", "李", category="identity_name", role="last_name")
    mapping = regional(first, second, family)
    groups = name_surface_groups(mapping)
    assert len(groups) == 1
    assert groups[0].entry_ids == ["a", "b"]
    other_language = mapping.model_copy(deep=True)
    for e in other_language.entries[:2]:
        e.forms.native = e.forms.romanized = "Juan"
    other = name_surface_groups(other_language)[0]
    assert other.source_id == groups[0].source_id
    assert other.request.id != groups[0].request.id


def test_owner_source_context_is_not_pooled_with_namesakes():
    first, second = given("a"), given("b", owner="person-b")
    first.context += ' source_identity={"first_name":"Juan","last_name":"Li"}'
    second.context += ' source_identity={"first_name":"Juan","last_name":"Wang"}'
    text = entry("context", "Juan Li is his son's father.", "Unchanged context.")
    groups = name_surface_groups(regional(first, second, text))
    assert len(groups) == 2
    assert groups[0].source_id != groups[1].source_id


def test_unknown_sources_stay_unchanged_and_evidence_binds_each_owner():
    first, second = given("a", source="Lei"), given("b", source="Lei", owner="person-b")
    second.forms.english = "Owner-specific explanation."
    family = entry("family", "Li", "李", category="identity_name", role="last_name")
    mapping = regional(first, second, family)
    groups = name_surface_groups(mapping)
    request = groups[0].request
    result = GenderNameResult(
        id=request.id,
        source_gender="unknown",
        target_gender="female",
        decision="unresolved",
        forms=request.forms.model_dump(),
        reason="Source convention uncertain.",
    )
    applied = apply_name_assessments(mapping, groups, [result], "test")
    updated, unresolved = applied.proposal, applied.unresolved_entry_ids
    assert unresolved == ["a", "b"]
    assert updated.source == mapping.source
    assert [e.forms for e in updated.entries] == [e.forms for e in mapping.entries]
    for e in updated.entries:
        e.name_gender.require(e.forms)
    assert updated.entries[-1].name_gender.source_gender == "unknown"
    with pytest.raises(ValueError, match="ambiguous source"):
        apply_name_assessments(
            mapping, groups, [result.model_copy(update={"decision": "revise"})], "test"
        )
    stale = mapping.model_copy(deep=True)
    stale.entries[0].forms.native = "Changed"
    with pytest.raises(ValueError, match="Stale"):
        apply_name_assessments(stale, groups, [result], "test")
    with pytest.raises(ValueError, match="coverage"):
        apply_name_assessments(mapping, groups, [], "test")


def test_product_attribute_contexts_are_not_merged():
    from tau2.multilingual.factory.database_localization import _draft_groups

    a = entry("armrest", "none", "none", category="product_option")
    b = entry("backlight", "none", "none", category="product_option")
    a.context = "Product option armrest"
    b.context = "Product option backlight"
    assert len(_draft_groups([a, b])) == 2


def test_same_full_name_does_not_supply_another_owners_pronouns():
    first, second = given("a"), given("b", owner="person-b")
    for e in (first, second):
        e.context += ' source_identity={"first_name":"Juan","last_name":"Li"}'
    text = entry("context", "Juan Li is his son's father.", "Unchanged context.")
    groups = name_surface_groups(regional(first, second, text))
    assert len(groups) == 1
    assert all("father" not in c for c in groups[0].request.context)
