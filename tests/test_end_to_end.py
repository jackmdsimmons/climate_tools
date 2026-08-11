"""
The whole path: contract records -> panel -> partition -> decomposition.

This is the test that proves the pieces connect, and that the engine still needs
no sovereign-specific code to attribute a sovereign portfolio.
"""

from __future__ import annotations

import pytest

from climate_attribution import Block, Level, classify, decompose_blocks
from climate_attribution.data import (
    GDP,
    SCOPE_1,
    SOVEREIGN,
    Holding,
    Observation,
    Panel,
    occupancy,
    validate,
)
from climate_attribution.partition import nested_weight_drivers

# Mt CO2e over $bn GDP, read as tCO2e per $m.
SCALE = 1000.0

#          region  w_t0  w_t1  E_t0   E_t1   GDP_t0  GDP_t1
BOOK = {
    "USA": ("AMER", 0.35, 0.30, 5900.0, 5750.0, 27000.0, 28200.0),
    "DEU": ("EMEA", 0.25, 0.25, 675.0, 640.0, 4500.0, 4600.0),
    "IND": ("APAC", 0.15, 0.20, 3900.0, 4050.0, 3700.0, 3950.0),
    "BRA": ("AMER", 0.15, 0.15, 1300.0, 1270.0, 2200.0, 2300.0),
    "ZAF": ("EMEA", 0.10, 0.00, 480.0, 470.0, 400.0, 415.0),
    "POL": ("EMEA", 0.00, 0.10, 330.0, 325.0, 850.0, 900.0),
}


def build_panel() -> Panel:
    holdings, observations = [], []
    for entity_id, row in BOOK.items():
        for index, period in enumerate(("t0", "t1")):
            weight = row[1 + index]
            if weight > 0.0:
                holdings.append(
                    Holding(entity_id=entity_id, period=period, weight=weight)
                )
            observations.append(
                Observation(
                    entity_id=entity_id, entity_type=SOVEREIGN, measure=SCOPE_1,
                    period=period, value=row[3 + index], unit="MtCO2e",
                    vintage="edgar-2025",
                )
            )
            observations.append(
                Observation(
                    entity_id=entity_id, entity_type=SOVEREIGN, measure=GDP,
                    period=period, value=row[5 + index], unit="USD_bn",
                    vintage="worldbank-2025-07-01", basis="nominal",
                )
            )
    return Panel(holdings, observations)


def attribute(panel: Panel):
    contribution_t0 = panel.contribution([SCOPE_1], GDP, "t0", scale=SCALE)
    contribution_t1 = panel.contribution([SCOPE_1], GDP, "t1", scale=SCALE)
    partition = classify(contribution_t0, contribution_t1)
    held = partition.survivors

    weights_t0 = panel.weights("t0")
    weights_t1 = panel.weights("t1")
    intensity_t0 = panel.intensity([SCOPE_1], GDP, "t0", held, scale=SCALE)
    intensity_t1 = panel.intensity([SCOPE_1], GDP, "t1", held, scale=SCALE)
    regions = [BOOK[e][0] for e in held]

    weight_drivers = nested_weight_drivers(
        [weights_t0[e] for e in held],
        [weights_t1[e] for e in held],
        [Level("region_allocation", regions, regions)],
        labels=held,
        selection_name="country_selection",
    )

    return partition, decompose_blocks(
        [
            Block(
                name="Exited",
                drivers={"contribution": (
                    [contribution_t0[e] for e in partition.leavers],
                    [contribution_t1.get(e, 0.0) for e in partition.leavers],
                )},
                labels=partition.leavers,
            ),
            Block(
                name="Entered",
                drivers={"contribution": (
                    [contribution_t0.get(e, 0.0) for e in partition.entrants],
                    [contribution_t1[e] for e in partition.entrants],
                )},
                labels=partition.entrants,
            ),
            Block(
                name="Held throughout",
                drivers={
                    **weight_drivers,
                    "carbon intensity": (
                        [intensity_t0[e] for e in held],
                        [intensity_t1[e] for e in held],
                    ),
                },
                labels=held,
            ),
        ]
    )


# ── The path ─────────────────────────────────────────────────────────────────


def test_the_panel_validates_clean():
    findings = validate(build_panel(), ["t0", "t1"], measures=[SCOPE_1, GDP])
    assert findings == []


def test_contract_to_decomposition_reconciles_exactly():
    panel = build_panel()
    partition, result = attribute(panel)

    assert partition.leavers == ("ZAF",)
    assert partition.entrants == ("POL",)
    assert partition.survivors == ("USA", "DEU", "IND", "BRA")

    result.check_additivity()
    assert result.residual == pytest.approx(0.0, abs=1e-9)
    assert result.total_t0 == pytest.approx(
        sum(panel.contribution([SCOPE_1], GDP, "t0", scale=SCALE).values())
    )
    assert result.total_t1 == pytest.approx(
        sum(panel.contribution([SCOPE_1], GDP, "t1", scale=SCALE).values())
    )


def test_the_engine_needs_no_sovereign_specific_code():
    """
    The claim this whole test file exists to pin. Nothing imported from
    climate_attribution.core or .partition knows what a country is -- the
    sovereign case is a GDP denominator and a region grouping, both passed in
    as ordinary data.
    """
    _, result = attribute(build_panel())
    assert set(result.effects) == {
        "Exited :: contribution",
        "Entered :: contribution",
        "Held throughout :: reallocation",
        "Held throughout :: region_allocation",
        "Held throughout :: country_selection",
        "Held throughout :: carbon intensity",
    }


def test_divestment_is_reported_separately_from_reallocation():
    """A sold country lands in its own block, never inside the allocation bar."""
    _, result = attribute(build_panel())
    effects = result.effects
    assert effects["Exited :: contribution"] < 0.0
    assert effects["Entered :: contribution"] > 0.0


def test_waterfall_opens_and_closes_on_the_portfolio_kpi():
    panel = build_panel()
    _, result = attribute(panel)
    steps = result.waterfall()
    assert steps[0][0] == "Portfolio t0"
    assert steps[-1][0] == "Portfolio t1"
    assert sum(v for _, v in steps[1:-1]) == pytest.approx(result.delta)


# ── Diagnostics stay opt-in ──────────────────────────────────────────────────


def test_region_grouping_has_healthy_occupancy():
    panel = build_panel()
    held = classify(
        panel.contribution([SCOPE_1], GDP, "t0", scale=SCALE),
        panel.contribution([SCOPE_1], GDP, "t1", scale=SCALE),
    ).survivors
    report = occupancy(
        [panel.weights("t0")[e] for e in held], [BOOK[e][0] for e in held]
    )
    # AMER holds USA and BRA, so selection can speak; EMEA and APAC are singletons.
    assert report.cells == 3
    assert report.singleton_cells == 2
    assert 0.0 < report.silent_share < 1.0


def test_a_vintage_change_surfaces_only_when_asked():
    """
    Rebasing GDP mid-comparison does not disturb the decomposition -- it still
    reconciles exactly, which is precisely the danger. Only validate() reports it.
    """
    panel = build_panel()
    rebased = Panel(
        panel.holdings,
        [
            Observation(
                entity_id=o.entity_id, entity_type=o.entity_type, measure=o.measure,
                period=o.period, value=o.value * (1.89 if _rebased(o) else 1.0),
                unit=o.unit, basis=o.basis,
                vintage="worldbank-2026-07-01" if _rebased(o) else o.vintage,
            )
            for o in panel.observations
        ],
    )

    _, result = attribute(rebased)
    result.check_additivity()  # still perfect arithmetic

    findings = validate(rebased, ["t0", "t1"], measures=[SCOPE_1, GDP])
    assert [f.code for f in findings] == ["vintage_change"]
    assert findings[0].entities == ("IND",)


def _rebased(observation: Observation) -> bool:
    return (
        observation.entity_id == "IND"
        and observation.measure == GDP
        and observation.period == "t1"
    )
