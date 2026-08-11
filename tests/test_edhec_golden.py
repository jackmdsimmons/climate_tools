"""
Golden-master tests: reproduce the published attribution results from
Bouchet (2025), EDHEC Research Insights Summer 2025, exhibits 7, 9 and 10a.

These are the tests that matter. Synthetic additivity checks prove the engine is
self-consistent; only these prove it computes what the literature computes.
Effects in the paper are printed to one decimal place, so they are asserted to
0.05 where the published value is exact to that precision.
"""

from __future__ import annotations

import pytest

from climate_attribution import Block, classify, decompose_blocks, lmdi_decompose
from climate_attribution.partition import nested_weight_drivers

from .fixtures import edhec_portfolio as fx

PUBLISHED_TOL = 0.05

# Exhibit 5 gives BS-LI's t1 quantity as a whole number of instruments (19,384),
# which lands the published weights a few parts per million off round figures.
ROUNDED_WEIGHT_TOL = 1e-4


# ── Headline metrics ─────────────────────────────────────────────────────────


def test_portfolio_weights_match_exhibit_5():
    """Published weights: 10/30/30/30% at t0, 0/20/50/30% at t1."""
    assert fx.weights("t0") == pytest.approx([0.10, 0.30, 0.30, 0.30])
    assert fx.weights("t1") == pytest.approx(
        [0.0, 0.20, 0.50, 0.30], abs=ROUNDED_WEIGHT_TOL
    )


def test_sector_weights_match_exhibit_5():
    """Brown sector falls from 40% to 20%; green rises from 60% to 80%."""
    for period, brown, green in (("t0", 0.40, 0.60), ("t1", 0.20, 0.80)):
        w = fx.as_map(fx.weights(period))
        assert w["BS-HI"] + w["BS-LI"] == pytest.approx(
            brown, abs=ROUNDED_WEIGHT_TOL
        )
        assert w["GS-HI"] + w["GS-LI"] == pytest.approx(
            green, abs=ROUNDED_WEIGHT_TOL
        )


def test_company_intensities():
    """Scope 1+2 per $m revenue. BS-LI holds intensity flat as revenue grows."""
    assert fx.intensity("t0") == pytest.approx([1000.0, 375.0, 200.0, 75.0])
    assert fx.intensity("t1") == pytest.approx([875.0, 375.0, 175.0, 62.5])


def test_portfolio_intensity_matches_published_headline():
    """295.0 -> 181.2 tCO2e/$m (p.11)."""
    assert fx.weighted_intensity("t0") == pytest.approx(
        fx.PUBLISHED_INTENSITY_T0, abs=PUBLISHED_TOL
    )
    assert fx.weighted_intensity("t1") == pytest.approx(
        fx.PUBLISHED_INTENSITY_T1, abs=PUBLISHED_TOL
    )


def test_absolute_emissions_match_published_headline():
    """19,667 -> 14,945 tCO2e (exhibit 7)."""
    assert sum(fx.absolute_emissions("t0")) == pytest.approx(
        fx.PUBLISHED_ABSOLUTE_T0, abs=0.5
    )
    assert sum(fx.absolute_emissions("t1")) == pytest.approx(
        fx.PUBLISHED_ABSOLUTE_T1, abs=0.5
    )


# ── Exhibit 7: absolute emissions, step 1 only ───────────────────────────────


def test_exhibit_7_absolute_emissions_by_subset():
    """
    Partition-only model: each subset carries a single driver, its own absolute
    emissions. Published bars: brown equity -6,292, GS-HI +2,125, GS-LI -555.

    This exercises the single-driver path with a leaver whose t1 contribution is
    zero -- the case where a naive log ratio is undefined.
    """
    e0 = fx.as_map(fx.absolute_emissions("t0"))
    e1 = fx.as_map(fx.absolute_emissions("t1"))

    groups = {
        "Equity - brown sector": ("BS-HI", "BS-LI"),
        "Debt - green sector, GS-HI": ("GS-HI",),
        "Debt - green sector, GS-LI": ("GS-LI",),
    }
    blocks = [
        Block(
            name=name,
            drivers={"asset emissions": (
                [e0[i] for i in ids], [e1[i] for i in ids],
            )},
            labels=ids,
        )
        for name, ids in groups.items()
    ]

    result = decompose_blocks(blocks)
    effects = result.effects

    assert effects["Equity - brown sector :: asset emissions"] == pytest.approx(
        -6292.0, abs=0.5
    )
    assert effects["Debt - green sector, GS-HI :: asset emissions"] == pytest.approx(
        2125.0, abs=0.5
    )
    assert effects["Debt - green sector, GS-LI :: asset emissions"] == pytest.approx(
        -555.0, abs=0.5
    )
    assert result.total_t1 == pytest.approx(fx.PUBLISHED_ABSOLUTE_T1, abs=0.5)
    result.check_additivity()


# ── Exhibit 9: intensity by divestment, allocation, price, company intensity ──


def test_exhibit_9_intensity_by_quantity_price_and_intensity():
    """
    Published effects: divestment -100.0, asset allocation +10.1,
    price fluctuations -10.4, company emissions intensity -13.5.

    Survivors carry four drivers -- quantity, price, inverse portfolio value and
    company intensity -- whose product is weight * intensity. The paper reports
    price and portfolio value together as "price fluctuations".
    """
    contributions_t0 = fx.as_map(fx.intensity_contribution("t0"))
    contributions_t1 = fx.as_map(fx.intensity_contribution("t1"))
    partition = classify(contributions_t0, contributions_t1)

    assert partition.survivors == ("BS-LI", "GS-HI", "GS-LI")
    assert partition.leavers == ("BS-HI",)
    assert partition.entrants == ()

    kept = partition.survivors
    inverse_value = [1.0 / fx.portfolio_value("t0")] * len(kept), \
                    [1.0 / fx.portfolio_value("t1")] * len(kept)

    blocks = [
        Block(
            name="Divested assets",
            drivers={"asset contribution to intensity": (
                [contributions_t0[i] for i in partition.leavers],
                [contributions_t1[i] for i in partition.leavers],
            )},
            labels=partition.leavers,
        ),
        Block(
            name="Remaining assets",
            drivers={
                "asset quantity": (
                    fx.subset(fx.quantity("t0"), kept),
                    fx.subset(fx.quantity("t1"), kept),
                ),
                "price": (
                    fx.subset(fx.price("t0"), kept),
                    fx.subset(fx.price("t1"), kept),
                ),
                "portfolio value": inverse_value,
                "emissions intensity scope 12": (
                    fx.subset(fx.intensity("t0"), kept),
                    fx.subset(fx.intensity("t1"), kept),
                ),
            },
            labels=kept,
        ),
    ]

    result = decompose_blocks(blocks)
    effects = result.effects

    assert effects["Divested assets :: asset contribution to intensity"] == (
        pytest.approx(-100.0, abs=PUBLISHED_TOL)
    )
    assert effects["Remaining assets :: asset quantity"] == pytest.approx(
        10.1, abs=PUBLISHED_TOL
    )
    # Price and portfolio value are charted as a single "price fluctuations" bar.
    price_fluctuations = (
        effects["Remaining assets :: price"]
        + effects["Remaining assets :: portfolio value"]
    )
    assert price_fluctuations == pytest.approx(-10.4, abs=PUBLISHED_TOL)
    assert effects["Remaining assets :: emissions intensity scope 12"] == (
        pytest.approx(-13.5, abs=PUBLISHED_TOL)
    )

    assert result.total_t0 == pytest.approx(fx.PUBLISHED_INTENSITY_T0, abs=PUBLISHED_TOL)
    assert result.total_t1 == pytest.approx(fx.PUBLISHED_INTENSITY_T1, abs=PUBLISHED_TOL)
    result.check_additivity()


# ── Exhibit 10a: intensity by sector allocation and stock selection ──────────


def test_exhibit_10a_sector_allocation_and_stock_selection():
    """
    Published effects: divestment -100.0, reallocation +19.6, sector allocation
    -30.2, stock selection +10.3, emissions intensity -13.5.

    This is the footnote-9 case. Survivor weights are renormalised within the
    survivor subset, so the +19.6 reallocation effect absorbs the subset growing
    from 90% to 100% of the portfolio, and BS-LI's sector allocation effect is not
    skewed by BS-HI's exclusion.
    """
    contributions_t0 = fx.as_map(fx.intensity_contribution("t0"))
    contributions_t1 = fx.as_map(fx.intensity_contribution("t1"))
    partition = classify(contributions_t0, contributions_t1)
    kept = partition.survivors

    # The paper's equity vocabulary is passed explicitly rather than relying on
    # the library's neutral defaults, which are shared with sovereign books.
    sectors = [fx.SECTORS[i] for i in kept]
    weight_drivers = nested_weight_drivers(
        fx.subset(fx.weights("t0"), kept),
        fx.subset(fx.weights("t1"), kept),
        sectors,
        sectors,
        labels=kept,
        allocation_name="sector_allocation",
        selection_name="stock_selection",
    )

    blocks = [
        Block(
            name="Divested assets",
            drivers={"asset contribution to intensity": (
                [contributions_t0[i] for i in partition.leavers],
                [contributions_t1[i] for i in partition.leavers],
            )},
            labels=partition.leavers,
        ),
        Block(
            name="Remaining assets",
            drivers={
                **weight_drivers,
                "emissions intensity scope 12": (
                    fx.subset(fx.intensity("t0"), kept),
                    fx.subset(fx.intensity("t1"), kept),
                ),
            },
            labels=kept,
        ),
    ]

    result = decompose_blocks(blocks)
    effects = result.effects

    assert effects["Divested assets :: asset contribution to intensity"] == (
        pytest.approx(-100.0, abs=PUBLISHED_TOL)
    )
    assert effects["Remaining assets :: reallocation"] == pytest.approx(
        19.6, abs=PUBLISHED_TOL
    )
    assert effects["Remaining assets :: sector_allocation"] == pytest.approx(
        -30.2, abs=PUBLISHED_TOL
    )
    assert effects["Remaining assets :: stock_selection"] == pytest.approx(
        10.3, abs=PUBLISHED_TOL
    )
    assert effects["Remaining assets :: emissions intensity scope 12"] == (
        pytest.approx(-13.5, abs=PUBLISHED_TOL)
    )
    result.check_additivity()


def test_renormalisation_is_what_makes_exhibit_10a_work():
    """
    Without the survivor-subset renormalisation, the sector allocation effect is
    wrong -- this is the specific defect footnote 9 warns about.

    Using raw portfolio weights, the brown sector's weight within the surviving
    names appears to fall from 33.3% to 20.0% only because BS-HI was sold. The
    naive decomposition folds that divestment into sector allocation instead of
    reporting it separately.
    """
    contributions_t0 = fx.as_map(fx.intensity_contribution("t0"))
    contributions_t1 = fx.as_map(fx.intensity_contribution("t1"))
    kept = classify(contributions_t0, contributions_t1).survivors

    # Naive: sector weights taken over the whole portfolio, no group driver.
    naive_sector_t0, naive_sector_t1 = [], []
    for period, out in (("t0", naive_sector_t0), ("t1", naive_sector_t1)):
        w = fx.as_map(fx.weights(period))
        totals = {"brown": w["BS-HI"] + w["BS-LI"], "green": w["GS-HI"] + w["GS-LI"]}
        out.extend(totals[fx.SECTORS[i]] for i in kept)

    naive = lmdi_decompose(
        {
            "sector_allocation": (naive_sector_t0, naive_sector_t1),
            "stock_selection": (
                [fx.as_map(fx.weights("t0"))[i] / s
                 for i, s in zip(kept, naive_sector_t0)],
                [fx.as_map(fx.weights("t1"))[i] / s
                 for i, s in zip(kept, naive_sector_t1)],
            ),
            "emissions intensity scope 12": (
                fx.subset(fx.intensity("t0"), kept),
                fx.subset(fx.intensity("t1"), kept),
            ),
        },
        labels=kept,
    )

    # The naive decomposition is internally consistent -- it still adds up -- so
    # nothing flags the error. It just answers a different question.
    naive.check_additivity()

    # Both investor-facing effects are misstated. Sector allocation absorbs part
    # of the divestment; stock selection absorbs the rest, overstating the
    # manager's within-sector picks roughly fourfold.
    assert naive.effects["sector_allocation"] == pytest.approx(-37.2, abs=0.5)
    assert naive.effects["stock_selection"] == pytest.approx(36.9, abs=0.5)

    assert abs(naive.effects["sector_allocation"] - (-30.2)) > 5.0
    assert abs(naive.effects["stock_selection"] - 10.3) > 20.0
