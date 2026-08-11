"""The data contract: what it insists on, and why."""

from __future__ import annotations

import pytest

from climate_attribution.data import GDP, SOVEREIGN, Holding, Observation
from climate_attribution.data.schema import UNKNOWN_VINTAGE


def observation(**overrides) -> Observation:
    fields = {
        "entity_id": "USA",
        "entity_type": SOVEREIGN,
        "measure": GDP,
        "period": "2023",
        "value": 27_360.0,
        "unit": "USD_bn",
        "vintage": "worldbank-2025-07-01",
        "basis": "nominal",
    }
    fields.update(overrides)
    return Observation(**fields)


def test_observation_carries_units_basis_and_vintage():
    record = observation()
    assert record.unit == "USD_bn"
    assert record.basis == "nominal"
    assert record.vintage == "worldbank-2025-07-01"


def test_vintage_is_required():
    """
    The point of the whole contract: provenance is cheap now and unrecoverable
    later. You cannot look at a GDP figure six months on and work out which
    release produced it.
    """
    with pytest.raises(ValueError, match="vintage"):
        observation(vintage="")


def test_unknown_vintage_is_allowed_but_must_be_stated():
    """Recording ignorance is still better than dropping the field."""
    record = observation(vintage=UNKNOWN_VINTAGE)
    assert record.vintage == UNKNOWN_VINTAGE


def test_identifying_fields_must_be_populated():
    for field in ("entity_id", "entity_type", "measure", "period", "unit"):
        with pytest.raises(ValueError, match=field):
            observation(**{field: ""})


def test_observations_are_hashable_and_comparable():
    """Frozen records so a panel can index them without defensive copying."""
    assert observation() == observation()
    assert len({observation(), observation()}) == 1


def test_observation_key_ignores_value_and_provenance():
    assert observation().key == observation(value=1.0, vintage="other").key


def test_holding_requires_an_entity_and_a_period():
    with pytest.raises(ValueError, match="entity_id"):
        Holding(entity_id="", period="2023", weight=0.5)
    with pytest.raises(ValueError, match="period"):
        Holding(entity_id="USA", period="", weight=0.5)


def test_holding_has_no_vintage():
    """
    Holdings are portfolio facts, known exactly; observations are measured by
    someone else and get revised. Only the latter needs provenance.
    """
    assert not hasattr(Holding(entity_id="USA", period="2023", weight=0.5), "vintage")
