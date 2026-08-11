"""Panel assembly: aligning contract records into the arrays the engine wants."""

from __future__ import annotations

import pytest

from climate_attribution.data import (
    GDP,
    SCOPE_1,
    SOVEREIGN,
    Holding,
    MissingObservation,
    Observation,
    Panel,
)

# Mt CO2e over $bn GDP reads as tCO2e per $m once scaled by 1e6/1e3.
MT_PER_BN_TO_T_PER_M = 1000.0

BOOK = {
    #        w_t0  w_t1   E_t0   E_t1   GDP_t0  GDP_t1
    "AAA": (0.50, 0.40, 100.0, 90.0, 1000.0, 1100.0),
    "BBB": (0.50, 0.60, 50.0, 55.0, 500.0, 500.0),
}


def build_panel() -> Panel:
    holdings, observations = [], []
    for entity_id, row in BOOK.items():
        for index, period in enumerate(("t0", "t1")):
            holdings.append(
                Holding(entity_id=entity_id, period=period, weight=row[index])
            )
            observations.append(
                Observation(
                    entity_id=entity_id, entity_type=SOVEREIGN, measure=SCOPE_1,
                    period=period, value=row[2 + index], unit="MtCO2e",
                    vintage="edgar-2025",
                )
            )
            observations.append(
                Observation(
                    entity_id=entity_id, entity_type=SOVEREIGN, measure=GDP,
                    period=period, value=row[4 + index], unit="USD_bn",
                    vintage="worldbank-2025-07-01", basis="nominal",
                )
            )
    return Panel(holdings, observations)


# ── Lookup ───────────────────────────────────────────────────────────────────


def test_entities_follow_the_order_holdings_were_supplied():
    assert build_panel().entities("t0") == ("AAA", "BBB")


def test_weights_are_returned_per_period():
    panel = build_panel()
    assert panel.weights("t0") == {"AAA": 0.50, "BBB": 0.50}
    assert panel.weights("t1") == {"AAA": 0.40, "BBB": 0.60}


def test_series_selects_one_measure_for_the_held_entities():
    assert build_panel().series(GDP, "t0") == {"AAA": 1000.0, "BBB": 500.0}


def test_total_sums_several_measures():
    """Scope 1 + Scope 2 is the common case; here Scope 1 alone."""
    assert build_panel().total([SCOPE_1], "t1") == {"AAA": 90.0, "BBB": 55.0}


def test_unknown_period_has_no_entities():
    assert build_panel().entities("t9") == ()


# ── Intensity and contribution ───────────────────────────────────────────────


def test_intensity_divides_and_scales():
    panel = build_panel()
    intensity = panel.intensity([SCOPE_1], GDP, "t0", scale=MT_PER_BN_TO_T_PER_M)
    assert intensity == pytest.approx({"AAA": 100.0, "BBB": 100.0})


def test_contribution_is_weight_times_intensity():
    panel = build_panel()
    contribution = panel.contribution(
        [SCOPE_1], GDP, "t0", scale=MT_PER_BN_TO_T_PER_M
    )
    assert sum(contribution.values()) == pytest.approx(100.0)

    later = panel.contribution([SCOPE_1], GDP, "t1", scale=MT_PER_BN_TO_T_PER_M)
    assert sum(later.values()) == pytest.approx(0.4 * (90 / 1100 * 1000) + 0.6 * 110.0)


def test_scale_is_never_guessed():
    """
    Units are compared, not converted. Leaving scale at its default gives Mt per
    $bn, which is a thousandth of tCO2e per $m -- a silently wrong answer if the
    caller assumed otherwise, which is why validate reports the units in play.
    """
    panel = build_panel()
    assert panel.intensity([SCOPE_1], GDP, "t0")["AAA"] == pytest.approx(0.1)


# ── Failure modes ────────────────────────────────────────────────────────────


def test_missing_observation_names_the_entities():
    panel = Panel(
        [Holding(entity_id="AAA", period="t0", weight=1.0)],
        [
            Observation(
                entity_id="AAA", entity_type=SOVEREIGN, measure=GDP, period="t0",
                value=1000.0, unit="USD_bn", vintage="v1",
            )
        ],
    )
    with pytest.raises(MissingObservation, match="AAA"):
        panel.series(SCOPE_1, "t0")


def test_a_gap_is_never_filled_with_zero():
    """
    Zero-filling a denominator would be rejected by the engine; zero-filling a
    numerator would silently understate the portfolio. Both are worse than
    stopping.
    """
    panel = build_panel()
    with pytest.raises(MissingObservation):
        panel.series("scope3", "t0")


def test_duplicate_observations_are_rejected():
    record = Observation(
        entity_id="AAA", entity_type=SOVEREIGN, measure=GDP, period="t0",
        value=1000.0, unit="USD_bn", vintage="v1",
    )
    with pytest.raises(ValueError, match="duplicate observation"):
        Panel([], [record, record])


def test_duplicate_holdings_are_rejected():
    holding = Holding(entity_id="AAA", period="t0", weight=0.5)
    with pytest.raises(ValueError, match="duplicate holding"):
        Panel([holding, holding], [])


def test_the_same_entity_may_repeat_across_periods():
    panel = build_panel()
    assert panel.observation("AAA", GDP, "t0").value == 1000.0
    assert panel.observation("AAA", GDP, "t1").value == 1100.0
