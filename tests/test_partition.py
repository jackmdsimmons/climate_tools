"""Partitioning, block composition, and survivor-subset renormalisation."""

from __future__ import annotations

import pytest

from climate_attribution import Block, classify, decompose_blocks
from climate_attribution.partition import nested_weight_drivers, nested_weights


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
    sectors = ["brown", "green", "green"]
    group, sector, within = nested_weights(weights, sectors)
    for i, w in enumerate(weights):
        assert group[i] * sector[i] * within[i] == pytest.approx(w)


def test_group_factor_carries_the_subsets_share_of_the_portfolio():
    group, sector, within = nested_weights([0.2, 0.3, 0.4], ["a", "b", "b"])
    assert group[0] == pytest.approx(0.9)
    assert sector == pytest.approx([0.2 / 0.9, 0.7 / 0.9, 0.7 / 0.9])
    assert within == pytest.approx([1.0, 3.0 / 7.0, 4.0 / 7.0])


def test_nested_weight_drivers_are_positive_and_ready_for_lmdi():
    drivers = nested_weight_drivers(
        [0.3, 0.3, 0.3], [0.2, 0.5, 0.3], ["brown", "green", "green"]
    )
    assert set(drivers) == {"reallocation", "sector_allocation", "stock_selection"}
    for start, end in drivers.values():
        assert all(v > 0 for v in start)
        assert all(v > 0 for v in end)


def test_empty_subset_is_rejected():
    with pytest.raises(ValueError, match="zero total weight"):
        nested_weights([0.0, 0.0], ["a", "b"])


def test_empty_sector_is_rejected_with_actionable_guidance():
    with pytest.raises(ValueError, match="own block"):
        nested_weights([0.5, 0.0], ["a", "b"])


def test_sector_label_count_must_match_weights():
    with pytest.raises(ValueError, match="sector labels"):
        nested_weights([0.5, 0.5], ["only-one"])


def test_mismatched_period_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        nested_weight_drivers([0.5, 0.5], [1.0], ["a", "b"])


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
