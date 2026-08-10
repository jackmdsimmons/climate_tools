# climate_attribution

Constituent-level attribution of changes in portfolio climate KPIs, using an
LMDI (logarithmic mean Divisia index) decomposition.

**Status: phases 0-1 complete.** The metric-agnostic engine and the partitioning
layer are built and validated against published results. No metric presets yet —
see the roadmap below.

## The problem

A weighted-average climate KPI is a sum over constituents of a product of factors:

```
KPI_p = sum_j  w_j * (E_j / A_j)
```

with `E_j` an emissions numerator (Scope 1, 1+2, 1+2+3) and `A_j` an activity
denominator (revenue for corporates, GDP for sovereigns, EVIC for financed
emissions). When the KPI moves, you want to know how much came from reallocating
capital, how much from constituents actually decarbonising, and how much from
things outside the investor's control.

LMDI attributes that change additively and exactly:

```
dKPI_p = E_D1 + E_D2 + ... + E_DN

E_Dn = sum_j  L(M_j,t1, M_j,t0) * ln( D_n,j,t1 / D_n,j,t0 )

L(x, y) = (x - y) / (ln x - ln y)          the logarithmic mean
```

No residual, no interaction term, and no sensitivity to the order the drivers are
listed in — which is what lets the number of factors stay flexible.

## Why partitioning comes first

LMDI's log ratio is undefined when a constituent's contribution is zero in either
period. That is not a rare corner: it is every position bought, every position
sold, and every constituent with a missing or zero denominator. The standard
convention `L(x, 0) = 0` keeps the arithmetic running but gives those instruments
*no* effect from any driver, so the decomposition silently stops adding up.

The fix is structural, not numerical, and follows the three-step model in Bouchet
(2025):

1. **Partition** the universe into disjoint subsets *before* choosing drivers —
   survivors, entrants, leavers, and any further grouping you want (sector, asset
   class, divestment-list vs. other).
2. **Choose drivers per subset.** The driver chain may differ between subsets.
   Leavers and entrants get a single driver — their whole contribution — whose
   effect is exactly the change in that contribution, with no logarithm taken.
   Survivors carry the full chain.
3. **Choose the attribution method** and apply it uniformly.

There is a second, subtler requirement. When survivors are decomposed into sector
allocation and stock selection, their weights must be renormalised *within the
survivor subset*, with the subset's share of the portfolio carried as its own
`reallocation` driver:

```
w_j = w_RI * ws_s(j) * wis_j
```

Without it, a divested holding leaks into the sector allocation effect of the
names that stayed. `tests/test_edhec_golden.py` pins this down: on the reference
portfolio the naive version reports sector allocation of -37.2 instead of -30.2
and stock selection of +36.9 instead of +10.3 — while still passing an additivity
check, because it is internally consistent. It just answers a different question.

## Usage

```python
from climate_attribution import Block, classify, decompose_blocks
from climate_attribution.partition import nested_weight_drivers

# Step 1 -- partition on contribution, not on identifier membership.
partition = classify(contributions_t0, contributions_t1)
kept = partition.survivors

# Step 2 -- drivers per subset. Weights renormalised within the survivors.
weight_drivers = nested_weight_drivers(weights_t0, weights_t1, sectors)

result = decompose_blocks([
    Block(
        name="Divested assets",
        drivers={"asset contribution": (divested_t0, divested_t1)},
        labels=partition.leavers,
    ),
    Block(
        name="Remaining assets",
        drivers={**weight_drivers, "emissions intensity": (intensity_t0, intensity_t1)},
        labels=kept,
    ),
])

result.check_additivity()
for label, value in result.waterfall():
    print(f"{label:<50}{value:>10.2f}")
```

Two properties are worth knowing when composing driver sets:

- Because effects are sums of logs, **splitting a driver into multiplicative
  factors splits its effect additively**, and merging two drivers merges their
  effects. Grouping drivers for reporting is exact, not approximate.
- The engine **raises `NonPositiveDriver` rather than clamping or dropping** a
  zero or negative value, and the error names the offending instruments and points
  at the partition API. Silence here is the defect this package exists to avoid.

## Validation

`tests/test_edhec_golden.py` reproduces the published attribution results for the
fictitious four-company portfolio in Bouchet (2025), exhibits 4, 5, 7, 9 and 10a:

| Exhibit 10a step                | computed | published |
|---------------------------------|---------:|----------:|
| Portfolio t0                    |   295.00 |     295.0 |
| Divested assets — contribution  |  -100.00 |    -100.0 |
| Remaining — reallocation        |    19.59 |      19.6 |
| Remaining — sector allocation   |   -30.21 |     -30.2 |
| Remaining — stock selection     |    10.35 |      10.3 |
| Remaining — emissions intensity |   -13.48 |     -13.5 |
| Portfolio t1                    |   181.25 |     181.2 |

Exhibit 9's alternative driver chain (quantity, price, portfolio value, company
intensity) and exhibit 7's absolute-emissions partition are reproduced too. Every
decomposition asserts a zero residual as an invariant.

Two transcription notes are recorded in `tests/fixtures/edhec_portfolio.py`: the
paper's exhibit 5 gives GS-LI an instrument price inconsistent with its stated
instrument type (the published weights resolve it), and the body text's 19,677
tCO2e opening figure disagrees with exhibit 7's 19,667 (the exhibit is right).

## Roadmap

- [x] **Phase 0 — core engine.** Logarithmic mean, metric-agnostic LMDI over
      arbitrary driver sets, additivity as an enforced invariant, strict refusal
      on non-positive values.
- [x] **Phase 1 — partitioning.** Contribution-based classification of survivors,
      entrants and leavers; blocks with per-subset driver chains; survivor-subset
      weight renormalisation.
- [ ] **Phase 2 — sovereign single KPI.** GDP-denominated intensity, two-driver
      weight × intensity decomposition. Sovereign-specific edge cases: country
      entry/exit, GDP rebasing, PPP vs. nominal basis, emissions/GDP vintage lag.
- [ ] **Phase 3 — nested allocation.** Region and income-group control for
      sovereigns; GICS sector control for corporates.
- [ ] **Phase 4 — scope disaggregation and inflation.** Scope 1 vs. Scope 2 as
      separate additive drivers; physical-production intensity vs. revenue
      intensity, separating real decarbonisation from price inflation.
- [ ] **Phase 5 — corporates and blended portfolios.** Metric registry (WACI,
      footprint, absolute emissions, financed emissions), mixed sovereign +
      corporate books.
- [ ] **Phase 6 — reporting and multi-period.** Waterfall output, period chaining
      with data-revision effects kept separate from real change, CLI.

## References

- Bouchet, V. (2025). *Attribution Analysis of Equity Portfolio Emissions:
  Examining and Integrating Existing Frameworks.* Scientific Portfolio (an EDHEC
  Venture). Summarised in EDHEC Research Insights, Summer 2025 (IPE supplement),
  pp. 8-14.
- Ang, B.W. (2015). *LMDI Decomposition Approach: A Guide for Implementation.*
  Energy Policy 86: 233-238.
- Simmons, J., M. Jain, E. Bourne and J. Kooroshy (2022). *Decarbonization in
  Equity Benchmarks: Smoke Still Rising.* FTSE Russell.
- Nagy, Z., G. Giese and X. Wang (2023). *A Framework for Attributing Changes in
  Portfolio Carbon Footprint.* Journal of Portfolio Management 49(8): 163-184.
- NZAOA (2023). *Understanding the Drivers of Investment Portfolio
  Decarbonisation.* Net-Zero Asset Owner Alliance.
