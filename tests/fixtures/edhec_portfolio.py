"""
The fictitious four-company portfolio from Bouchet (2025), used as a golden
master for the decomposition engine.

  Bouchet, V. (2025). Attribution Analysis of Equity Portfolio Emissions:
  Examining and Integrating Existing Frameworks. Scientific Portfolio (an EDHEC
  Venture), published in EDHEC Research Insights, Summer 2025 (IPE supplement),
  pp. 8-14. Exhibits 4, 5, 7, 9 and 10.

Two companies sit in a carbon-intensive ("brown") sector and two in a low-carbon
("green") sector, one high-intensity and one low-intensity in each. Over the
period the manager fully divests BS-HI, trims BS-LI, and reallocates to GS-HI.

Everything below is transcribed from exhibits 4 and 5; the derived quantities in
this module reproduce the paper's published headline figures:

    emissions intensity (Scope 1+2 per revenue)   295.0  ->  181.2 tCO2e/$m
    absolute emissions (via EVIC footprint)      19,667  ->  14,945 tCO2e

Two transcription notes, both resolved by reconciling against the paper's own
published weights:

  * Exhibit 5 lists GS-HI and GS-LI as debt instruments and shows debt prices
    unchanged, yet gives GS-LI an instrument price of 1,500 at t1. Only the 1,500
    reproduces the paper's stated t1 weights of 20/50/30%, so it is used here.
  * The body text on p.11 says absolute emissions start at 19,677 tCO2e while
    exhibit 7 shows 19,667. The exhibit is right: 19,666.67 falls out of the
    underlying data, and the exhibit's bars sum to the stated close of 14,945.
"""

from __future__ import annotations

# Instrument order used by every driver array built from this fixture.
IDS: tuple[str, ...] = ("BS-HI", "BS-LI", "GS-HI", "GS-LI")

SECTORS: dict[str, str] = {
    "BS-HI": "brown",
    "BS-LI": "brown",
    "GS-HI": "green",
    "GS-LI": "green",
}

# ── Exhibit 4: changes in company variables ──────────────────────────────────
# Emissions in tCO2e, revenue and EVIC in $m, production in tonnes.

COMPANIES: dict[str, dict[str, float]] = {
    "BS-HI": {
        "scope1_t0": 75_000_000.0, "scope1_t1": 75_000_000.0,
        "scope2_t0": 25_000_000.0, "scope2_t1": 12_500_000.0,
        "revenue_t0": 100_000.0,   "revenue_t1": 100_000.0,
        "production_t0": 100.0,    "production_t1": 100.0,
        "evic_t0": 150_000.0,      "evic_t1": 170_000.0,
    },
    "BS-LI": {
        "scope1_t0": 25_000_000.0, "scope1_t1": 50_000_000.0,
        "scope2_t0": 12_500_000.0, "scope2_t1": 6_250_000.0,
        "revenue_t0": 100_000.0,   "revenue_t1": 150_000.0,
        "production_t0": 100.0,    "production_t1": 100.0,
        "evic_t0": 150_000.0,      "evic_t1": 180_000.0,
    },
    "GS-HI": {
        "scope1_t0": 15_000_000.0, "scope1_t1": 15_000_000.0,
        "scope2_t0": 5_000_000.0,  "scope2_t1": 2_500_000.0,
        "revenue_t0": 100_000.0,   "revenue_t1": 100_000.0,
        "production_t0": 100.0,    "production_t1": 100.0,
        "evic_t0": 150_000.0,      "evic_t1": 180_000.0,
    },
    "GS-LI": {
        "scope1_t0": 5_000_000.0,  "scope1_t1": 5_000_000.0,
        "scope2_t0": 2_500_000.0,  "scope2_t1": 1_250_000.0,
        "revenue_t0": 100_000.0,   "revenue_t1": 100_000.0,
        "production_t0": 100.0,    "production_t1": 100.0,
        "evic_t0": 150_000.0,      "evic_t1": 250_000.0,
    },
}

# ── Exhibit 5: changes in the portfolio ──────────────────────────────────────
# Quantity of instruments held and price per instrument, in $.

HOLDINGS: dict[str, dict[str, float]] = {
    "BS-HI": {"quantity_t0": 10_000.0, "quantity_t1": 0.0,
              "price_t0": 1_000.0, "price_t1": 1_200.0},
    "BS-LI": {"quantity_t0": 30_000.0, "quantity_t1": 19_384.0,
              "price_t0": 1_000.0, "price_t1": 1_300.0},
    "GS-HI": {"quantity_t0": 30_000.0, "quantity_t1": 63_000.0,
              "price_t0": 1_000.0, "price_t1": 1_000.0},
    "GS-LI": {"quantity_t0": 30_000.0, "quantity_t1": 25_200.0,
              "price_t0": 1_000.0, "price_t1": 1_500.0},
}

# Published headline figures, for assertion in tests.
PUBLISHED_INTENSITY_T0 = 295.0
PUBLISHED_INTENSITY_T1 = 181.2
PUBLISHED_ABSOLUTE_T0 = 19_667.0
PUBLISHED_ABSOLUTE_T1 = 14_945.0


# ── Derived quantities ───────────────────────────────────────────────────────


def quantity(period: str) -> list[float]:
    """Number of instruments held, per instrument."""
    return [HOLDINGS[i][f"quantity_{period}"] for i in IDS]


def price(period: str) -> list[float]:
    """Price per instrument."""
    return [HOLDINGS[i][f"price_{period}"] for i in IDS]


def market_value(period: str) -> list[float]:
    """Value of each holding, in $ -- the units of `quantity` times `price`."""
    return [
        HOLDINGS[i][f"quantity_{period}"] * HOLDINGS[i][f"price_{period}"]
        for i in IDS
    ]


def portfolio_value(period: str) -> float:
    """Total portfolio value, in $."""
    return sum(market_value(period))


def weights(period: str) -> list[float]:
    """Financial weight of each holding in the portfolio."""
    total = portfolio_value(period)
    return [v / total for v in market_value(period)]


def intensity(period: str, scope: str = "scope12") -> list[float]:
    """Company emissions intensity in tCO2e per $m of revenue."""
    out = []
    for i in IDS:
        company = COMPANIES[i]
        if scope == "scope1":
            emissions = company[f"scope1_{period}"]
        elif scope == "scope2":
            emissions = company[f"scope2_{period}"]
        elif scope == "scope12":
            emissions = company[f"scope1_{period}"] + company[f"scope2_{period}"]
        else:
            raise ValueError(f"unknown scope {scope!r}")
        out.append(emissions / company[f"revenue_{period}"])
    return out


def footprint(period: str) -> list[float]:
    """Company emissions per $m of EVIC."""
    return [
        (COMPANIES[i][f"scope1_{period}"] + COMPANIES[i][f"scope2_{period}"])
        / COMPANIES[i][f"evic_{period}"]
        for i in IDS
    ]


def weighted_intensity(period: str) -> float:
    """Portfolio emissions intensity: sum of weight * company intensity."""
    return sum(w * i for w, i in zip(weights(period), intensity(period)))


def absolute_emissions(period: str) -> list[float]:
    """
    Absolute emissions financed by each holding: market value * footprint.

    Market value is converted from $ to $m so it lines up with EVIC's units.
    """
    return [
        (v / 1e6) * f for v, f in zip(market_value(period), footprint(period))
    ]


def intensity_contribution(period: str) -> list[float]:
    """Each holding's contribution to portfolio intensity: weight * intensity."""
    return [w * i for w, i in zip(weights(period), intensity(period))]


def as_map(values: list[float]) -> dict[str, float]:
    """Pair a per-instrument array with its ids."""
    return dict(zip(IDS, values))


def subset(values: list[float], ids: tuple[str, ...]) -> list[float]:
    """Select the entries for `ids`, preserving the order of `ids`."""
    lookup = as_map(values)
    return [lookup[i] for i in ids]
