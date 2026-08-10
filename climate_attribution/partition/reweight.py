"""
Weight renormalisation within a subset of the portfolio.

When survivors are analysed separately from divested holdings, their sector
allocation and stock selection effects must be measured against weights that
have been renormalised *within the survivor subset*:

    w_j = w_RI * ws_s(j) * wis_j

    w_RI       total portfolio weight of the retained subset
    ws_s(j)    weight of sector s within that subset
    wis_j      weight of instrument j within its sector, within that subset

Without this, a divested holding distorts the sector allocation effect of the
names that stayed. From Bouchet (2025), footnote 9:

    "A driver capturing the weight change of remaining instruments relative to
    divested ones is introduced, isolating sector allocation and stock selection
    effects for retained stocks. Without this, BS-LI's sector allocation effect
    would be skewed by BS-HI's exclusion."

The `w_RI` factor is not bookkeeping -- it is a reported effect in its own right
(the "reallocation effect"), capturing the subset growing from 90% to 100% of the
portfolio as the divested holding is sold.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def nested_weights(
    weights: Sequence[float],
    sectors: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Factor subset weights into (group, sector-within-group, instrument-within-sector).

    Args:
        weights: portfolio weight of each instrument in the subset. These are
            weights in the *whole* portfolio, not pre-normalised ones -- the
            group factor is what carries the subset's share.
        sectors: sector label per instrument, same length as `weights`.

    Returns:
        Three arrays whose elementwise product equals `weights`.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be 1-D")
    if len(sectors) != w.size:
        raise ValueError(
            f"got {len(sectors)} sector labels for {w.size} instruments"
        )

    group_total = float(w.sum())
    if group_total <= 0.0:
        raise ValueError(
            "subset has zero total weight; it cannot carry a weight decomposition"
        )

    sector_totals = np.zeros(w.size, dtype=float)
    for label in set(sectors):
        mask = np.array([s == label for s in sectors])
        total = float(w[mask].sum())
        if total <= 0.0:
            raise ValueError(
                f"sector {label!r} has zero total weight within the subset; "
                "drop the sector or move its instruments to their own block"
            )
        sector_totals[mask] = total

    group = np.full(w.size, group_total)
    sector_share = sector_totals / group_total
    within_sector = w / sector_totals
    return group, sector_share, within_sector


def nested_weight_drivers(
    weights_t0: Sequence[float],
    weights_t1: Sequence[float],
    sectors: Sequence[str],
    *,
    group_name: str = "reallocation",
    sector_name: str = "sector_allocation",
    within_name: str = "stock_selection",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Build the three weight drivers for a subset, ready to pass to `lmdi_decompose`.

    Sector membership is taken as fixed across the period. A reclassified
    instrument is a genuine edge case, not a relabelling: put it in its own block
    rather than letting it silently move between sectors mid-decomposition.
    """
    if len(weights_t0) != len(weights_t1):
        raise ValueError("weights_t0 and weights_t1 must be the same length")

    group_0, sector_0, within_0 = nested_weights(weights_t0, sectors)
    group_1, sector_1, within_1 = nested_weights(weights_t1, sectors)

    return {
        group_name: (group_0, group_1),
        sector_name: (sector_0, sector_1),
        within_name: (within_0, within_1),
    }
