# climate_attribution

Constituent-level attribution of changes in portfolio climate KPIs, using an
LMDI (logarithmic mean Divisia index) decomposition.

**Status: phases 0-2 complete.** The metric-agnostic engine, the partitioning
layer, and the data contract are built and validated against published results.
No metric presets yet — see the roadmap below.

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

There is a second, subtler requirement. When survivors are decomposed into
allocation and selection, their weights must be renormalised *within the survivor
subset*, with the subset's share of the portfolio carried as its own
`reallocation` driver:

```
w_j = w_RI * wg_g(j) * wig_j
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
# Levels are generic and can be nested to any depth: GICS sector for corporates,
# region then income band for sovereigns. Each level carries membership for both
# periods so a reclassification is caught rather than assumed away.
weight_drivers = nested_weight_drivers(
    weights_t0, weights_t1,
    [Level("sector_allocation", sectors_t0, sectors_t1)],
    labels=kept,
    selection_name="stock_selection",
)

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

## The data contract

Sources are adapters onto a vendor-neutral core. Two flat record types:
`Holding` (a portfolio fact, known exactly) and `Observation` (a fact about the
world, measured by someone else and revised without warning).

```python
from climate_attribution.data import Holding, Observation, Panel, validate

panel = Panel(holdings, observations)

# Opt-in. Nothing here runs as a side effect of assembly or decomposition.
for finding in validate(panel, ["t0", "t1"], measures=["scope1", "gdp"]):
    print(finding)

contributions = panel.contribution(["scope1"], "gdp", "t0", scale=1000.0)
```

Three `Observation` fields exist solely to stop silently wrong arithmetic:

- **`unit`** — mixing tonnes with megatonnes gives a clean residual and a wrong
  answer. Units are compared, never converted; `Panel.intensity` takes an
  explicit `scale` rather than guessing.
- **`basis`** — nominal vs PPP GDP, EVIC vs market cap. Two entities on different
  bases are not comparable.
- **`vintage`** — required, no default. It cannot be recovered later: you cannot
  look at a GDP figure six months on and determine which release produced it. A
  rebasing can move a country's measured GDP by 80% with no change in real
  activity, which reads as a 45% fall in carbon intensity and lands squarely on
  the intensity effect. `validate` reports vintage changes between periods; the
  decomposition itself still reconciles perfectly, which is exactly the danger.

Nothing in the contract is entity-specific. `test_end_to_end.py` pins the claim
that a sovereign portfolio needs no sovereign-specific code — a GDP denominator
and a region grouping, both passed in as ordinary data.

### Diagnostics are opt-in

`occupancy()` reports how much of a grouping can actually carry a selection
effect. Selection is silent in a group of one: a lone member's share of its own
cell is 100% in both periods, so its effect is zero by construction. As levels
are added, cells multiply and occupancy collapses until the reported selection
effect describes the taxonomy rather than the portfolio.

```
>>> occupancy(weights, region, sector, subsector).summary()
'3 level(s), 6 cells, 6 holding a single member; 100.0% of weight sits where
 selection is structurally zero'
```

It is never called automatically. One grouping level is defensible, two
sometimes; deeper is a diagnostic, not a result.

## Nesting order is a modelling choice

Levels nest to any depth — `region > sector > subsector` gives a driver per
level, all reconciling exactly. But with two *crossed* dimensions there is no
neutral decomposition. Nesting region inside sector and sector inside region both
add up to the same total while attributing different amounts to each dimension:

```
NEST: region -> sector          NEST: sector -> region
   region             -3.58        sector            -19.85
   sector_in_region  -13.55        region_in_sector    2.73
   within_finest      -1.48        within_finest      -1.48
   intensity         -10.40        intensity         -10.40
   sum               -29.00        sum               -29.00
```

The region effect flips sign. Whichever dimension is nested first absorbs the
shared variation — the same phenomenon that gives Brinson attribution its
interaction term.

This is **not** covered by LMDI's order-invariance. LMDI is invariant to the
order drivers are *listed* in; nesting order changes the driver values themselves,
before any decomposition happens. `test_nesting_order_changes_the_attribution`
pins this deliberately so it is not later "fixed" as a bug.

Two consequences the API enforces: driver names encode the nesting
(`sector_in_region`, never a bare `sector` that could be mistaken for a
nested-first effect), and the ordering is explicit in the caller's `levels`
argument. Where there is no primary dimension, prefer separate single-level
decompositions as alternative lenses and do not add their effects together.

## Two properties worth knowing when composing driver sets

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

### What is not verified

`sources/worldbank/fetch_worldbank.py` has **never been run against the live
API** — every climate and economic data host was blocked by egress policy in the
environment it was written in. The download is unverified boilerplate; run it
once before trusting it.

The mapping onto the contract *is* tested, against
`tests/fixtures/worldbank_payload.py`, which records the response shape from the
API's published documentation rather than from a captured call. If the live
response differs, that fixture is the single place to correct — the adapter and
its tests follow from it.

## Roadmap

- [x] **Phase 0 — core engine.** Logarithmic mean, metric-agnostic LMDI over
      arbitrary driver sets, additivity as an enforced invariant, strict refusal
      on non-positive values.
- [x] **Phase 1 — partitioning.** Contribution-based classification of survivors,
      entrants and leavers; blocks with per-subset driver chains; survivor-subset
      weight renormalisation, with group membership required for both periods so
      that a reclassification is rejected rather than assumed away.
- [x] **Phase 2 — data contract and validation.** Entity-generic records
      (`entity_id`, `entity_type`, `unit`, `basis`, `vintage`) so provenance is
      captured at ingest while it is still recoverable; checks for vintage
      consistency, weight coverage, and mixed denominator bases. A sovereign
      worked example serves as the regression test — the engine needs no
      sovereign-specific code, which is the property that test pins down.
      Includes a cell-occupancy diagnostic reporting how much of a selection
      effect is structurally zero, since selection is silent in a group of one
      and deep nestings quietly become artifacts of the taxonomy. **All
      diagnostics are opt-in** — never run by default, never warn unbidden.
      A World Bank adapter and `sources/worldbank/` fetcher show how a source
      maps onto the contract; the adapter is tested against a recorded payload,
      the download is not (see Validation).
- [ ] **Phase 3 — denominator factoring.** Split the activity denominator into
      real growth, inflation, FX and data-revision drivers. This is where the
      GDP-rebasing and nominal-vs-PPP problems become their own waterfall bars
      instead of contaminating the intensity effect.
- [ ] **Phase 4 — scope disaggregation.** Scope 1 vs. Scope 2 as separate
      additive drivers; physical-production intensity vs. revenue intensity.
- [ ] **Phase 5 — blended books.** Metric presets (WACI, footprint, absolute
      emissions, financed emissions) as configuration rather than code paths.
      Encodes the one genuinely entity-type-dependent rule: PCAF sovereign and
      corporate financed emissions overlap, since a company's Scope 1 already
      sits inside its host country's territorial inventory, so the two cannot be
      summed into a single KPI.
- [ ] **Phase 6 — reporting and multi-period.** Waterfall output, period chaining,
      CLI.

Note that "corporate vs. sovereign" is deliberately absent as a phase. The engine
is entity-agnostic and the grouping layer takes arbitrary labels, so asset class
is configuration — which denominator, which grouping, which numerator taxonomy —
not a code path.

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
