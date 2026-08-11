"""
The World Bank adapter, tested against the recorded payload shape.

The download itself is boilerplate and untestable without a network; the mapping
onto the contract is where the decisions live, so that is what is pinned here.
"""

from __future__ import annotations

import pytest

from climate_attribution.data import GDP, SOVEREIGN
from climate_attribution.data.adapters import UnexpectedPayload, parse_worldbank

from .fixtures.worldbank_payload import EMPTY_PAYLOAD, ERROR_PAYLOAD, GDP_PAYLOAD


def parse(**overrides):
    fields = {"measure": GDP, "unit": "USD", "basis": "nominal"}
    fields.update(overrides)
    return parse_worldbank(GDP_PAYLOAD, **fields)


def test_rows_become_observations_keyed_by_iso3_and_year():
    observations = parse()
    keyed = {(o.entity_id, o.period): o.value for o in observations}
    assert keyed[("USA", "2023")] == pytest.approx(27_360_935_000_000)
    assert keyed[("DEU", "2023")] == pytest.approx(4_456_081_000_000)


def test_vintage_defaults_to_the_payloads_last_updated_date():
    """Exactly the provenance the contract wants, and the API supplies it."""
    assert parse()[0].vintage == "worldbank-2025-07-01"


def test_vintage_can_be_overridden_for_an_archived_file():
    assert parse(vintage="worldbank-archive-2019")[0].vintage == (
        "worldbank-archive-2019"
    )


def test_missing_vintage_is_refused_rather_than_invented():
    payload = [{"page": 1, "total": 0}, []]
    with pytest.raises(UnexpectedPayload, match="lastupdated"):
        parse_worldbank(payload, measure=GDP, unit="USD")


def test_null_values_are_skipped_not_zero_filled():
    assert ("DEU", "2024") not in {(o.entity_id, o.period) for o in parse()}


def test_rows_without_an_iso3_code_are_dropped():
    assert all(o.entity_id for o in parse())
    assert "Not classified" not in {o.entity_id for o in parse()}


def test_aggregates_can_be_filtered_out_by_entity():
    """Regional aggregates carry valid ISO3 codes, so filtering is the caller's."""
    assert "EAS" in {o.entity_id for o in parse()}
    sovereigns = parse(entities=["USA", "DEU"])
    assert {o.entity_id for o in sovereigns} == {"USA", "DEU"}


def test_periods_can_be_filtered():
    assert {o.period for o in parse(periods=["2023"])} == {"2023"}


def test_unit_and_basis_are_carried_through_unchanged():
    record = parse(unit="USD", basis="ppp")[0]
    assert record.unit == "USD"
    assert record.basis == "ppp"
    assert record.entity_type == SOVEREIGN


def test_unit_is_required_because_guessing_is_how_units_go_wrong():
    with pytest.raises(TypeError):
        parse_worldbank(GDP_PAYLOAD, measure=GDP)  # no unit


def test_an_api_error_response_is_surfaced():
    with pytest.raises(UnexpectedPayload, match="error"):
        parse_worldbank(ERROR_PAYLOAD, measure=GDP, unit="USD")


def test_an_empty_result_set_is_not_an_error():
    assert parse_worldbank(EMPTY_PAYLOAD, measure=GDP, unit="USD") == []


def test_a_malformed_payload_is_rejected_clearly():
    with pytest.raises(UnexpectedPayload, match="two-element"):
        parse_worldbank({"not": "a list"}, measure=GDP, unit="USD")
