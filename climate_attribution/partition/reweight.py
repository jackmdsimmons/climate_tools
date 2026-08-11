"""
Weight renormalisation within a subset of the portfolio.

A portfolio weight is factored into a chain of nested shares, each one "this
thing's share of the level above it":

    w_j = W_subset * (W_L1 / W_subset) * (W_L1L2 / W_L1) * ... * (w_j / W_L1..Lk)

Every denominator cancels against the next numerator, so the chain collapses back
to w_j. With one grouping level that is the familiar three-factor split:

    w_j = w_RI * wg_g(j) * wig_j

    w_RI       total portfolio weight of the retained subset
    wg_g(j)    weight of group g within that subset
    wig_j      weight of instrument j within its group, within that subset

Renormalising *within the subset* is what stops a divested holding from
distorting the allocation effect of the names that stayed. From Bouchet (2025),
footnote 9:

    "A driver capturing the weight change of remaining instruments relative to
    divested ones is introduced, isolating sector allocation and stock selection
    effects for retained stocks. Without this, BS-LI's sector allocation effect
    would be skewed by BS-HI's exclusion."

The `w_RI` factor is not bookkeeping -- it is a reported effect in its own right
(the "reallocation effect"), capturing the subset growing from 90% to 100% of the
portfolio as the divested holding is sold.

Levels are deliberately generic: GICS sector for a corporate book, region or
income band for a sovereign one, asset class for a blended one. Level names are
required rather than defaulted, because they end up as driver names in the
report and the nesting they describe is a modelling choice -- see below.


NESTING ORDER IS A MODELLING CHOICE
-----------------------------------
With two crossed dimensions there is no neutral decomposition. Nesting region
inside sector and nesting sector inside region both reconcile exactly to the same
total, but they attribute different amounts -- sometimes different signs -- to
each dimension. Whichever dimension is nested first absorbs the shared variation.

This is *not* something LMDI's order-invariance protects against. LMDI is
invariant to the order drivers are listed in, because the effects are computed
from the driver values. Nesting order changes the driver values themselves,
before any decomposition happens. It is the same phenomenon that gives Brinson
attribution its interaction term.

Two consequences the API enforces:

  * Driver names encode the nesting ("sector_in_region", not bare "sector"), so a
    nested-second effect cannot be mistaken for a nested-first one in a report.
  * The ordering is explicit in the caller's `levels` argument rather than
    inferred, and should be documented alongside the results.

Where there is genuinely no primary dimension, prefer running separate
single-level decompositions as alternative lenses, and do not add their effects
together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Level:
    """
    One grouping level, with its membership labels in each period.

    Membership is required for both periods so that a reclassification is
    detected rather than assumed away. Passing the same sequence twice is the
    correct way to assert that membership is fixed.
    """

    name: str
    t0: Sequence[str]
    t1: Sequence[str]


class GroupReclassified(ValueError):
    """
    Raised when an instrument's group membership differs between the two periods.

    A reclassification is a genuine analytical decision, not a relabelling: the
    decomposition has no way to express an instrument that was in one group at t0
    and another at t1, so silently adopting either label misattributes the effect.
    """


def nested_weights(
    weights: Sequence[float],
    *levels: Sequence[str],
) -> list[np.ndarray]:
    """
    Factor subset weights into a chain of nested shares.

    Args:
        weights: portfolio weight of each instrument in the subset. These are
            weights in the *whole* portfolio, not pre-normalised ones -- the
            first factor is what carries the subset's share, and normalising it
            away is what makes a divestment leak into the allocation effect.
        *levels: one label sequence per grouping level, coarsest first. Each must
            be the same length as `weights`. Pass none for no grouping at all.

    Returns:
        `len(levels) + 2` arrays whose elementwise product equals `weights`:
        the subset total, one share per level, then each member's share of its
        finest group.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be 1-D")
    for depth, labels in enumerate(levels):
        if len(labels) != w.size:
            raise ValueError(
                f"level {depth} has {len(labels)} group labels "
                f"for {w.size} instruments"
            )

    subset_total = float(w.sum())
    if subset_total <= 0.0:
        raise ValueError(
            "subset has zero total weight; it cannot carry a weight decomposition"
        )

    factors: list[np.ndarray] = [np.full(w.size, subset_total)]
    parent_totals = np.full(w.size, subset_total)

    for depth in range(len(levels)):
        # The key is cumulative: keying on the bare label would pool, say,
        # "energy" across every region and break the telescoping.
        keys = list(zip(*levels[: depth + 1]))
        totals = np.zeros(w.size, dtype=float)
        for key in set(keys):
            mask = np.array([k == key for k in keys])
            total = float(w[mask].sum())
            if total <= 0.0:
                shown = " > ".join(str(part) for part in key)
                raise ValueError(
                    f"group {shown!r} has zero total weight within the subset; "
                    "drop the group or move its instruments to their own block"
                )
            totals[mask] = total
        factors.append(totals / parent_totals)
        parent_totals = totals

    factors.append(w / parent_totals)
    return factors


def nested_weight_drivers(
    weights_t0: Sequence[float],
    weights_t1: Sequence[float],
    levels: Sequence[Level] = (),
    *,
    labels: Sequence[str] | None = None,
    subset_name: str = "reallocation",
    selection_name: str | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Build the weight drivers for a subset, ready to pass to `lmdi_decompose`.

    Args:
        weights_t0, weights_t1: portfolio weights of the subset's members.
        levels: grouping levels, coarsest first. The order is a modelling choice
            that changes the attribution -- see the module docstring.
        labels: optional instrument identifiers, used in error messages.
        subset_name: driver name for the subset's share of the portfolio.
        selection_name: driver name for member weight within its finest group.
            Defaults to "within_<finest level>". Corporate books analysing a
            single sector level usually want "stock_selection".

    Returns:
        One driver per factor, named so the nesting is legible: the first level
        keeps its own name, deeper levels are named "<level>_in_<parent>".

    Raises:
        GroupReclassified: if any instrument changes group between periods.
    """
    levels = tuple(levels)
    n = len(weights_t0)
    if n != len(weights_t1):
        raise ValueError("weights_t0 and weights_t1 must be the same length")
    for level in levels:
        if len(level.t0) != len(level.t1):
            raise ValueError(
                f"level {level.name!r}: t0 and t1 group labels "
                "must be the same length"
            )

    _reject_reclassified(levels, labels, n)

    factors_t0 = nested_weights(weights_t0, *(level.t0 for level in levels))
    factors_t1 = nested_weights(weights_t1, *(level.t1 for level in levels))

    names = [subset_name, *_level_driver_names(levels)]
    if selection_name is not None:
        names.append(selection_name)
    elif levels:
        names.append(f"within_{levels[-1].name}")
    else:
        names.append("selection")

    if len(set(names)) != len(names):
        raise ValueError(f"driver names must be unique, got {names}")

    return {
        name: (start, end)
        for name, start, end in zip(names, factors_t0, factors_t1)
    }


# ── Internals ────────────────────────────────────────────────────────────────


def _level_driver_names(levels: tuple[Level, ...]) -> list[str]:
    """Name each level so its position in the nesting is visible."""
    return [
        level.name if depth == 0 else f"{level.name}_in_{levels[depth - 1].name}"
        for depth, level in enumerate(levels)
    ]


def _reject_reclassified(
    levels: tuple[Level, ...],
    labels: Sequence[str] | None,
    n: int,
) -> None:
    """Fail on membership changes the decomposition cannot represent."""
    if labels is not None and len(labels) != n:
        raise ValueError(f"got {len(labels)} labels for {n} instruments")

    moved = [
        (
            level.name,
            str(labels[i]) if labels is not None else f"#{i}",
            level.t0[i],
            level.t1[i],
        )
        for level in levels
        for i in range(n)
        if level.t0[i] != level.t1[i]
    ]
    if not moved:
        return

    shown = "\n".join(
        f"  {level}: {name} moved {before!r} -> {after!r}"
        for level, name, before, after in moved[:5]
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
        "then pass it as both `t0` and `t1` on the Level."
    )
