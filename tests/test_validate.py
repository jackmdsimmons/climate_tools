"""Opt-in checks, and the occupancy diagnostic for over-nested groupings."""

from __future__ import annotations

import pytest

from climate_attribution.data import (
    ERROR,
    GDP,
    SCOPE_1,
    SOVEREIGN,
    WARNING,
    Holding,
    Observation,
    Panel,
    errors,
    occupancy,
    validate,
)
from climate_attribution.data.schema import UNKNOWN_VINTAGE


def obs(entity_id, measure, period, value, **overrides) -> Observation:
    fields = {
        "entity_id": entity_id,
        "entity_type": SOVEREIGN,
        "measure": measure,
        "period": period,
        "value": value,
        "unit": "USD_bn" if measure == GDP else "MtCO2e",
        "vintage": "v1",
        "basis": "nominal" if measure == GDP else "",
    }
    fields.update(overrides)
    return Observation(**fields)


def panel_with(holdings, observations) -> Panel:
    return Panel(holdings, observations)


def clean_panel() -> Panel:
    holdings, observations = [], []
    for period in ("t0", "t1"):
        for entity_id, weight in (("AAA", 0.6), ("BBB", 0.4)):
            holdings.append(Holding(entity_id=entity_id, period=period, weight=weight))
            observations.append(obs(entity_id, GDP, period, 1000.0))
            observations.append(obs(entity_id, SCOPE_1, period, 100.0))
    return panel_with(holdings, observations)


def codes(findings) -> set[str]:
    return {f.code for f in findings}


# ── Nothing fires on a clean panel ───────────────────────────────────────────


def test_a_clean_panel_produces_no_findings():
    assert validate(clean_panel(), ["t0", "t1"], measures=[GDP, SCOPE_1]) == []


def test_validation_is_never_triggered_by_assembly():
    """
    Building a panel and computing contributions must stay silent. Checks that
    fire on their own train people to ignore them.
    """
    panel = clean_panel()
    assert panel.contribution([SCOPE_1], GDP, "t0")  # no warnings, no findings
    assert validate(panel, ["t0"], measures=[GDP]) == []


# ── Individual checks ────────────────────────────────────────────────────────


def test_weights_not_summing_to_one_is_a_warning_not_an_error():
    """Wrong for an equity book, entirely normal for one holding cash."""
    holdings = [Holding(entity_id="AAA", period="t0", weight=0.85)]
    findings = validate(panel_with(holdings, [obs("AAA", GDP, "t0", 1.0)]), ["t0"])
    assert "weight_sum" in codes(findings)
    assert all(f.severity == WARNING for f in findings if f.code == "weight_sum")


def test_negative_weight_is_an_error():
    holdings = [
        Holding(entity_id="AAA", period="t0", weight=1.2),
        Holding(entity_id="BBB", period="t0", weight=-0.2),
    ]
    findings = validate(panel_with(holdings, []), ["t0"])
    negative = [f for f in findings if f.code == "negative_weight"]
    assert negative and negative[0].severity == ERROR
    assert negative[0].entities == ("BBB",)


def test_missing_measure_for_a_held_entity_is_reported():
    holdings = [Holding(entity_id="AAA", period="t0", weight=1.0)]
    findings = validate(
        panel_with(holdings, [obs("AAA", GDP, "t0", 1.0)]),
        ["t0"],
        measures=[GDP, SCOPE_1],
    )
    missing = [f for f in findings if f.code == "missing_observation"]
    assert missing and missing[0].entities == ("AAA",)


def test_coverage_is_only_checked_for_measures_in_use():
    holdings = [Holding(entity_id="AAA", period="t0", weight=1.0)]
    findings = validate(
        panel_with(holdings, [obs("AAA", GDP, "t0", 1.0)]), ["t0"], measures=[GDP]
    )
    assert "missing_observation" not in codes(findings)


def test_mixed_units_within_a_measure_is_an_error():
    """A megatonne read as a tonne gives a clean residual and a wrong answer."""
    holdings = [Holding(entity_id="AAA", period="t0", weight=1.0)]
    observations = [
        obs("AAA", SCOPE_1, "t0", 100.0, unit="MtCO2e"),
        obs("BBB", SCOPE_1, "t0", 100.0, unit="tCO2e"),
    ]
    findings = validate(
        panel_with(holdings, observations), ["t0"], measures=[SCOPE_1]
    )
    mixed = [f for f in findings if f.code == "mixed_units"]
    assert mixed and mixed[0].severity == ERROR


def test_mixed_bases_within_a_measure_is_an_error():
    """Nominal and PPP GDP are not comparable across entities."""
    holdings = [Holding(entity_id="AAA", period="t0", weight=1.0)]
    observations = [
        obs("AAA", GDP, "t0", 1000.0, basis="nominal"),
        obs("BBB", GDP, "t0", 2000.0, basis="ppp"),
    ]
    findings = validate(panel_with(holdings, observations), ["t0"], measures=[GDP])
    assert "mixed_bases" in codes(findings)


def test_inconsistent_entity_type_is_an_error():
    holdings = [Holding(entity_id="AAA", period="t0", weight=1.0)]
    observations = [
        obs("AAA", GDP, "t0", 1000.0),
        obs("AAA", SCOPE_1, "t0", 100.0, entity_type="corporate"),
    ]
    findings = validate(panel_with(holdings, observations), ["t0"])
    assert "inconsistent_entity_type" in codes(findings)


# ── The rebasing detector ────────────────────────────────────────────────────


def test_a_vintage_change_between_periods_is_reported():
    """
    The GDP rebasing case. Nigeria's 2014 rebasing raised measured GDP ~89% with
    no change in real activity, which reads as a 47% fall in carbon intensity
    and lands entirely on the intensity effect.
    """
    holdings = [
        Holding(entity_id="NGA", period=p, weight=1.0) for p in ("t0", "t1")
    ]
    observations = [
        obs("NGA", GDP, "t0", 270.0, vintage="worldbank-2013-12-01"),
        obs("NGA", GDP, "t1", 510.0, vintage="worldbank-2014-12-01"),
    ]
    findings = validate(
        panel_with(holdings, observations), ["t0", "t1"], measures=[GDP]
    )
    change = [f for f in findings if f.code == "vintage_change"]
    assert change and change[0].entities == ("NGA",)
    assert change[0].severity == WARNING


def test_a_stable_vintage_across_periods_is_not_reported():
    assert "vintage_change" not in codes(
        validate(clean_panel(), ["t0", "t1"], measures=[GDP])
    )


def test_unknown_vintage_is_flagged_because_restatement_becomes_undetectable():
    holdings = [Holding(entity_id="AAA", period="t0", weight=1.0)]
    observations = [obs("AAA", GDP, "t0", 1.0, vintage=UNKNOWN_VINTAGE)]
    findings = validate(panel_with(holdings, observations), ["t0"], measures=[GDP])
    assert "unknown_vintage" in codes(findings)


def test_errors_filters_out_the_judgement_calls():
    holdings = [
        Holding(entity_id="AAA", period="t0", weight=0.85),
        Holding(entity_id="BBB", period="t0", weight=-0.05),
    ]
    findings = validate(panel_with(holdings, []), ["t0"])
    assert codes(errors(findings)) == {"negative_weight"}


# ── Occupancy diagnostic ─────────────────────────────────────────────────────


def test_occupancy_reports_a_healthy_single_level_grouping():
    report = occupancy([0.25, 0.25, 0.25, 0.25], ["a", "a", "b", "b"])
    assert report.cells == 2
    assert report.singleton_cells == 0
    assert report.silent_share == pytest.approx(0.0)


def test_occupancy_exposes_a_taxonomy_artifact():
    """
    Three levels over six holdings puts every name in its own cell, so the
    deepest selection effect is structurally zero -- exactly what the earlier
    three-level demo produced, and not a finding about the portfolio.
    """
    weights = [0.2, 0.1, 0.25, 0.25, 0.1, 0.1]
    region = ["EU", "EU", "EU", "US", "US", "US"]
    sector = ["energy", "energy", "tech", "energy", "tech", "tech"]
    sub = ["oil", "renew", "semis", "oil", "semis", "software"]

    report = occupancy(weights, region, sector, sub)
    assert report.depth == 3
    assert report.cells == 6
    assert report.singleton_cells == 6
    assert report.silent_share == pytest.approx(1.0)
    assert "structurally zero" in report.summary()


def test_occupancy_improves_as_levels_are_removed():
    weights = [0.2, 0.1, 0.25, 0.25, 0.1, 0.1]
    region = ["EU", "EU", "EU", "US", "US", "US"]
    sector = ["energy", "energy", "tech", "energy", "tech", "tech"]

    deep = occupancy(weights, region, sector)
    shallow = occupancy(weights, region)
    assert shallow.silent_share < deep.silent_share
    assert shallow.singleton_cells == 0


def test_occupancy_with_no_levels_is_a_single_cell():
    report = occupancy([0.5, 0.5])
    assert report.cells == 1
    assert report.silent_share == pytest.approx(0.0)


def test_occupancy_label_count_must_match():
    with pytest.raises(ValueError, match="labels"):
        occupancy([0.5, 0.5], ["only-one"])
