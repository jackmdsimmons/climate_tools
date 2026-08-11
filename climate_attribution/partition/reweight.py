"""
Weight renormalisation within a subset of the portfolio.

When survivors are analysed separately from divested holdings, their allocation
and selection effects must be measured against weights that have been
renormalised *within the survivor subset*:

    w_j = w_RI * wg_g(j) * wig_j

    w_RI       total portfolio weight of the retained subset
    wg_g(j)    weight of group g within that subset
    wig_j      weight of instrument j within its group, within that subset

Without this, a divested holding distorts the allocation effect of the names
that stayed. From Bouchet (2025), footnote 9:

    "A driver capturing the weight change of remaining instruments relative to
    divested ones is introduced, isolating sector allocation and stock selection
    effects for retained stocks. Without this, BS-LI's sector allocation effect
    would be skewed by BS-HI's exclusion."

The `w_RI` factor is not bookkeeping -- it is a reported effect in its own right
(the "reallocation effect"), capturing the subset growing from 90% to 100% of the
portfolio as the divested holding is sold.

"Group" is deliberately generic: GICS sector for a corporate book, region or
income band for a sovereign one, asset class for a blended one. Pass the driver
names that match your context -- the defaults are neutral so that a sovereign
waterfall does not come out labelled "stock selection".
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class GroupReclassified(ValueError):
    """
    Raised when an instrument's group membership differs between the two periods.

    A reclassification is a genuine analytical decision, not a relabelling: the
    decomposition has no way to express an instrument that was in one group at t0
    and another at t1, so silently adopting either label misattributes the effect.
    """


def nested_weights(
    weights: Sequence[float],
    groups: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Factor subset weights into (subset, group-within-subset, member-within-group).

    Args:
        weights: portfolio weight of each instrument in the subset. These are
            weights in the *whole* portfolio, not pre-normalised ones -- the
            subset factor is what carries the subset's share.
        groups: group label per instrument, same length as `weights`.

    Returns:
        Three arrays whose elementwise product equals `weights`.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be 1-D")
    if len(groups) != w.size:
        raise ValueError(f"got {len(groups)} group labels for {w.size} instruments")

    subset_total = float(w.sum())
    if subset_total <= 0.0:
        raise ValueError(
            "subset has zero total weight; it cannot carry a weight decomposition"
        )

    group_totals = np.zeros(w.size, dtype=float)
    for label in set(groups):
        mask = np.array([g == label for g in groups])
        total = float(w[mask].sum())
        if total <= 0.0:
            raise ValueError(
                f"group {label!r} has zero total weight within the subset; "
                "drop the group or move its instruments to their own block"
            )
        group_totals[mask] = total

    subset = np.full(w.size, subset_total)
    group_share = group_totals / subset_total
    within_group = w / group_totals
    return subset, group_share, within_group


def nested_weight_drivers(
    weights_t0: Sequence[float],
    weights_t1: Sequence[float],
    groups_t0: Sequence[str],
    groups_t1: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    subset_name: str = "reallocation",
    allocation_name: str = "group_allocation",
    selection_name: str = "within_group_selection",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Build the three weight drivers for a subset, ready to pass to `lmdi_decompose`.

    Group membership is required for *both* periods so that a reclassification is
    detected rather than assumed away. Passing the same sequence twice is the
    correct way to assert that membership is fixed.

    Args:
        weights_t0, weights_t1: portfolio weights of the subset's members.
        groups_t0, groups_t1: group label per member in each period.
        labels: optional instrument identifiers, used in error messages.
        subset_name: driver name for the subset's share of the portfolio.
        allocation_name: driver name for group weight within the subset.
            Corporate books usually want "sector_allocation"; sovereign books
            "region_allocation" or similar.
        selection_name: driver name for member weight within its group.
            Corporate books usually want "stock_selection".

    Raises:
        GroupReclassified: if any instrument changes group between periods.
    """
    if len(weights_t0) != len(weights_t1):
        raise ValueError("weights_t0 and weights_t1 must be the same length")
    if len(groups_t0) != len(groups_t1):
        raise ValueError("groups_t0 and groups_t1 must be the same length")

    _reject_reclassified(groups_t0, groups_t1, labels)

    subset_0, group_0, within_0 = nested_weights(weights_t0, groups_t0)
    subset_1, group_1, within_1 = nested_weights(weights_t1, groups_t1)

    return {
        subset_name: (subset_0, subset_1),
        allocation_name: (group_0, group_1),
        selection_name: (within_0, within_1),
    }


# ── Internals ────────────────────────────────────────────────────────────────


def _reject_reclassified(
    groups_t0: Sequence[str],
    groups_t1: Sequence[str],
    labels: Sequence[str] | None,
) -> None:
    """Fail on membership changes the decomposition cannot represent."""
    if labels is not None and len(labels) != len(groups_t0):
        raise ValueError(
            f"got {len(labels)} labels for {len(groups_t0)} instruments"
        )

    moved = [
        (
            str(labels[i]) if labels is not None else f"#{i}",
            groups_t0[i],
            groups_t1[i],
        )
        for i in range(len(groups_t0))
        if groups_t0[i] != groups_t1[i]
    ]
    if not moved:
        return

    shown = "\n".join(
        f"  {name}: {before!r} -> {after!r}" for name, before, after in moved[:5]
    )
    if len(moved) > 5:
        shown += f"\n  ... ({len(moved)} total)"

    raise GroupReclassified(
        "Group membership changed between periods:\n"
        + shown
        + "\n\nThe weight decomposition holds membership fixed, so adopting either "
        "label would misattribute the change between allocation and selection. "
        "This is routine for sovereign books grouped by income band, which are "
        "revised annually. Either move the reclassified instruments to their own "
        "block, or pick one scheme for both periods and document the choice -- "
        "then pass it as both `groups_t0` and `groups_t1`."
    )
