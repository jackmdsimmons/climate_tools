"""Partitioning, block composition, and survivor-subset renormalisation."""

from __future__ import annotations

import numpy as np
import pytest

from climate_attribution import (
    Block,
    Level,
    classify,
    decompose_blocks,
    lmdi_decompose,
)
from climate_attribution.partition import (
    GroupReclassified,
    nested_weight_drivers,
    nested_weights,
)


# ── Classification ───────────────────────────────────────────────────────────


def test_classify_splits_survivors_entrants_and_leavers():
    partition = classify(
        {"A": 0.5, "B": 0.5}, {"B": 0.4, "C": 0.6}
    )
    assert partition.survivors == ("B",)
    assert partition.leavers == ("A",)
    assert partition.entrants == ("C",)


def test_a_zero_weight_holding_still_listed_at_t1_is_a_leaver():
    """
    The specific trap: the identifier is present in both periods, so membership
    testing on keys alone would call it a survivor and hand LMDI a log of zero.
    """
    partition = classify({"SOLD": 0.25}, {"SOLD": 0.0})
    assert partition.leavers == ("SOLD",)
    assert partition.survivors == ()


def test_tolerance_treats_dust_positions_as_absent():
    holdings = {"DUST": 1e-9, "REAL": 0.5}
    # A dust position survives by default -- and would drag a near-zero value
    # into a log ratio, so callers holding residual positions should set a tol.
    assert classify(holdings, holdings).survivors == ("DUST", "REAL")
    assert classify(holdings, holdings, tol=1e-6).survivors == ("REAL",)


def test_instrument_absent_in_both_periods_is_excluded_entirely():
    partition = classify({"GONE": 0.0, "HELD": 1.0}, {"GONE": 0.0, "HELD": 1.0})
    assert partition.all_ids == ("HELD",)


def test_classification_ordering_is_stable():
    partition = classify({"B": 1.0, "A": 1.0}, {"A": 1.0, "B": 1.0})
    assert partition.survivors == ("B", "A")


def test_negative_tolerance_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        classify({"A": 1.0}, {"A": 1.0}, tol=-1.0)


def test_partition_summary_reads_clearly():
    partition = classify({"A": 1.0, "B": 1.0}, {"B": 1.0, "C": 1.0})
    assert partition.summary() == "1 survivors, 1 entrants, 1 leavers"


# ── Renormalisation ──────────────────────────────────────────────────────────


def test_nested_weights_multiply_back_to_the_original_weights():
    weights = [0.2, 0.3, 0.4]
    groups = ["brown", "green", "green"]
    subset, group, within = nested_weights(weights, groups)
    for i, w in enumerate(weights):
        assert subset[i] * group[i] * within[i] == pytest.approx(w)


def test_subset_factor_carries_the_subsets_share_of_the_portfolio():
    subset, group, within = nested_weights([0.2, 0.3, 0.4], ["a", "b", "b"])
    assert subset[0] == pytest.approx(0.9)
    assert group == pytest.approx([0.2 / 0.9, 0.7 / 0.9, 0.7 / 0.9])
    assert within == pytest.approx([1.0, 3.0 / 7.0, 4.0 / 7.0])


def test_nested_weight_drivers_are_positive_and_ready_for_lmdi():
    groups = ["brown", "green", "green"]
    drivers = nested_weight_drivers(
        [0.3, 0.3, 0.3], [0.2, 0.5, 0.3], [Level("sector", groups, groups)]
    )
    for start, end in drivers.values():
        assert all(v > 0 for v in start)
        assert all(v > 0 for v in end)


def test_driver_names_come_from_the_level_names():
    """A sovereign waterfall must not come out labelled 'stock selection'."""
    groups = ["EMEA", "APAC"]
    drivers = nested_weight_drivers(
        [0.5, 0.5], [0.4, 0.6], [Level("region", groups, groups)]
    )
    assert set(drivers) == {"reallocation", "region", "within_region"}


def test_driver_names_can_be_set_to_match_the_context():
    groups = ["energy", "utilities"]
    drivers = nested_weight_drivers(
        [0.5, 0.5], [0.4, 0.6],
        [Level("sector_allocation", groups, groups)],
        selection_name="stock_selection",
    )
    assert set(drivers) == {"reallocation", "sector_allocation", "stock_selection"}


def test_no_levels_gives_reallocation_and_pure_selection():
    drivers = nested_weight_drivers([0.4, 0.5], [0.3, 0.6])
    assert set(drivers) == {"reallocation", "selection"}
    assert drivers["reallocation"][0] == pytest.approx([0.9, 0.9])
    assert drivers["selection"][0] == pytest.approx([4 / 9, 5 / 9])


def test_duplicate_driver_names_are_rejected():
    groups = ["a", "b"]
    with pytest.raises(ValueError, match="unique"):
        nested_weight_drivers(
            [0.5, 0.5], [0.4, 0.6],
            [Level("reallocation", groups, groups)],
        )


def test_empty_subset_is_rejected():
    with pytest.raises(ValueError, match="zero total weight"):
        nested_weights([0.0, 0.0], ["a", "b"])


def test_empty_group_is_rejected_with_actionable_guidance():
    with pytest.raises(ValueError, match="own block"):
        nested_weights([0.5, 0.0], ["a", "b"])


def test_group_label_count_must_match_weights():
    with pytest.raises(ValueError, match="group labels"):
        nested_weights([0.5, 0.5], ["only-one"])


def test_mismatched_period_lengths_are_rejected():
    groups = ["a", "b"]
    with pytest.raises(ValueError, match="same length"):
        nested_weight_drivers([0.5, 0.5], [1.0], [Level("g", groups, groups)])


# ── Multi-level nesting ──────────────────────────────────────────────────────


NESTED_BOOK = {
    #          region  sector    w_t0  w_t1
    "A": ("EU", "energy", 0.20, 0.24),
    "E": ("EU", "energy", 0.10, 0.06),
    "B": ("EU", "tech", 0.25, 0.20),
    "C": ("US", "energy", 0.25, 0.20),
    "D": ("US", "tech", 0.20, 0.30),
}
NESTED_INTENSITY_T0 = [500.0, 520.0, 50.0, 400.0, 60.0]
NESTED_INTENSITY_T1 = [450.0, 500.0, 45.0, 420.0, 55.0]


def nested_book():
    labels = list(NESTED_BOOK)
    regions = [NESTED_BOOK[k][0] for k in labels]
    sectors = [NESTED_BOOK[k][1] for k in labels]
    w0 = [NESTED_BOOK[k][2] for k in labels]
    w1 = [NESTED_BOOK[k][3] for k in labels]
    return labels, regions, sectors, w0, w1


def test_any_depth_of_nesting_multiplies_back_to_the_weights():
    _, regions, sectors, w0, _ = nested_book()
    for levels in ((), (regions,), (regions, sectors), (sectors, regions)):
        factors = nested_weights(w0, *levels)
        assert len(factors) == len(levels) + 2
        product = np.ones(len(w0))
        for factor in factors:
            product = product * factor
        assert product == pytest.approx(w0)


def test_deeper_levels_are_keyed_cumulatively():
    """
    Keying on the bare label would pool 'energy' across EU and US and break the
    telescoping. The share must be within the parent group only.
    """
    _, regions, sectors, w0, _ = nested_book()
    _, region_share, sector_in_region, within = nested_weights(w0, regions, sectors)

    # EU energy is 0.30 of EU's 0.55, not of the 0.55 total energy weight.
    assert sector_in_region[0] == pytest.approx(0.30 / 0.55)
    # Within EU energy, A holds 0.20 of 0.30.
    assert within[0] == pytest.approx(0.20 / 0.30)
    assert region_share[0] == pytest.approx(0.55 / 1.0)


def test_nesting_names_encode_the_hierarchy():
    _, regions, sectors, w0, w1 = nested_book()
    drivers = nested_weight_drivers(
        w0, w1,
        [Level("region", regions, regions), Level("sector", sectors, sectors)],
    )
    assert list(drivers) == [
        "reallocation", "region", "sector_in_region", "within_sector"
    ]


def test_nesting_order_changes_the_attribution():
    """
    Pinned deliberately: this is a property of crossed dimensions, not a bug.
    Whichever dimension is nested first absorbs the shared variation, so the
    region effect here flips sign depending on the order. LMDI's invariance to
    driver *listing* order does not extend to *nesting* order, because nesting
    changes the driver values before any decomposition happens.
    """
    labels, regions, sectors, w0, w1 = nested_book()

    def run(levels):
        drivers = nested_weight_drivers(w0, w1, levels, labels=labels)
        drivers["intensity"] = (NESTED_INTENSITY_T0, NESTED_INTENSITY_T1)
        result = lmdi_decompose(drivers, labels=labels)
        result.check_additivity()
        return result

    region_first = run(
        [Level("region", regions, regions), Level("sector", sectors, sectors)]
    )
    sector_first = run(
        [Level("sector", sectors, sectors), Level("region", regions, regions)]
    )

    # Both reconcile to the same total.
    assert region_first.delta == pytest.approx(sector_first.delta)
    assert region_first.explained == pytest.approx(sector_first.explained)

    # The region effect flips sign with the nesting order.
    assert region_first.effects["region"] == pytest.approx(-3.58, abs=0.05)
    assert sector_first.effects["region_in_sector"] == pytest.approx(2.73, abs=0.05)

    # What the two orderings agree on: the finest cell is region x sector either
    # way, so anything below it is untouched.
    assert region_first.effects["within_sector"] == pytest.approx(
        sector_first.effects["within_region"]
    )
    assert region_first.effects["intensity"] == pytest.approx(
        sector_first.effects["intensity"]
    )


# ── Reclassification ─────────────────────────────────────────────────────────


def test_reclassified_instrument_is_rejected_not_assumed_away():
    """
    Routine for sovereign books grouped by income band, which are revised
    annually. The decomposition cannot represent a member moving between groups,
    so it must refuse rather than silently adopt one label.
    """
    with pytest.raises(GroupReclassified) as excinfo:
        nested_weight_drivers(
            [0.5, 0.5],
            [0.4, 0.6],
            [Level("income_band", ["lower-middle", "high"], ["upper-middle", "high"])],
            labels=["IDN", "DEU"],
        )
    message = str(excinfo.value)
    assert "IDN" in message
    assert "income_band" in message
    assert "lower-middle" in message and "upper-middle" in message
    assert "DEU" not in message


def test_reclassification_is_caught_at_any_level():
    groups = ["EMEA", "EMEA"]
    with pytest.raises(GroupReclassified, match="sector"):
        nested_weight_drivers(
            [0.5, 0.5], [0.4, 0.6],
            [
                Level("region", groups, groups),
                Level("sector", ["energy", "tech"], ["utilities", "tech"]),
            ],
        )


def test_passing_the_same_groups_twice_asserts_fixed_membership():
    groups = ["a", "b"]
    drivers = nested_weight_drivers(
        [0.5, 0.5], [0.4, 0.6], [Level("group", groups, groups)]
    )
    assert drivers["group"][0] == pytest.approx([0.5, 0.5])


def test_reclassification_error_survives_without_labels():
    with pytest.raises(GroupReclassified, match="#0"):
        nested_weight_drivers(
            [0.5, 0.5], [0.4, 0.6], [Level("g", ["a", "b"], ["c", "b"])]
        )


def test_group_label_lengths_must_match_each_other():
    with pytest.raises(ValueError, match="same length"):
        nested_weight_drivers(
            [0.5, 0.5], [0.4, 0.6], [Level("g", ["a", "b"], ["a"])]
        )


def test_label_count_must_match_instruments():
    groups = ["a", "b"]
    with pytest.raises(ValueError, match="labels"):
        nested_weight_drivers(
            [0.5, 0.5], [0.4, 0.6],
            [Level("g", groups, groups)],
            labels=["only-one"],
        )


# ── Block composition ────────────────────────────────────────────────────────


def simple_blocks() -> list[Block]:
    return [
        Block(
            name="Divested",
            drivers={"contribution": ([40.0], [0.0])},
            labels=["SOLD"],
        ),
        Block(
            name="Remaining",
            drivers={
                "weight": ([0.6, 0.4], [0.7, 0.3]),
                "intensity": ([100.0, 200.0], [90.0, 190.0]),
            },
            labels=["KEPT-A", "KEPT-B"],
        ),
    ]


def test_blocks_with_different_driver_counts_still_add_up():
    """The whole point: a leaver's block need not share the survivors' chain."""
    result = decompose_blocks(simple_blocks())
    result.check_additivity()
    assert result.residual == pytest.approx(0.0, abs=1e-12)


def test_effects_are_namespaced_by_block():
    result = decompose_blocks(simple_blocks())
    assert "Divested :: contribution" in result.effects
    assert "Remaining :: weight" in result.effects
    assert "Remaining :: intensity" in result.effects


def test_totals_span_every_block():
    result = decompose_blocks(simple_blocks())
    assert result.total_t0 == pytest.approx(40.0 + 0.6 * 100.0 + 0.4 * 200.0)
    assert result.total_t1 == pytest.approx(0.7 * 90.0 + 0.3 * 190.0)


def test_waterfall_opens_and_closes_on_the_portfolio_totals():
    result = decompose_blocks(simple_blocks())
    steps = result.waterfall()
    assert steps[0] == ("Portfolio t0", result.total_t0)
    assert steps[-1] == ("Portfolio t1", result.total_t1)
    assert sum(v for _, v in steps[1:-1]) == pytest.approx(result.delta)


def test_overlapping_blocks_are_rejected():
    blocks = [
        Block(name="One", drivers={"c": ([1.0], [2.0])}, labels=["X"]),
        Block(name="Two", drivers={"c": ([3.0], [4.0])}, labels=["X"]),
    ]
    with pytest.raises(ValueError, match="disjoint"):
        decompose_blocks(blocks)


def test_duplicate_block_names_are_rejected():
    blocks = [
        Block(name="Same", drivers={"c": ([1.0], [2.0])}, labels=["X"]),
        Block(name="Same", drivers={"c": ([3.0], [4.0])}, labels=["Y"]),
    ]
    with pytest.raises(ValueError, match="duplicate block name"):
        decompose_blocks(blocks)


def test_no_blocks_is_rejected():
    with pytest.raises(ValueError, match="at least one block"):
        decompose_blocks([])


def test_empty_block_contributes_nothing():
    """A period with no entrants should not need special-casing by the caller."""
    blocks = [
        Block(name="Entrants", drivers={"contribution": ([], [])}, labels=[]),
        Block(
            name="Remaining",
            drivers={
                "weight": ([1.0], [1.0]),
                "intensity": ([100.0], [90.0]),
            },
            labels=["KEPT"],
        ),
    ]
    result = decompose_blocks(blocks)
    assert result.effects["Entrants :: contribution"] == 0.0
    assert result.delta == pytest.approx(-10.0)
