"""
Step 1 of the three-step attribution model: split the universe into disjoint
subsets before any driver is chosen.

    P = union of P_k,   P_k intersect P_l = empty for k != l

This is the structural fix for LMDI's zero-value problem. Rather than patching
the log ratio for an instrument that entered or left the portfolio, the instrument
is placed in a subset that carries a *different, shorter* driver chain -- in the
limit, a single driver, whose effect is just the change in its contribution.

Note that membership is decided on contribution, not on whether an identifier
appears in the data. A holding that is still listed at t1 with a weight of zero
is a leaver, and treating it as a survivor is precisely the bug this module
exists to prevent.

Reference:
  Bouchet, V. (2025). Attribution Analysis of Equity Portfolio Emissions:
  Examining and Integrating Existing Frameworks. Scientific Portfolio / EDHEC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Partition:
    """A universe split into instruments that stayed, arrived, and left."""

    survivors: tuple[str, ...]
    entrants: tuple[str, ...]
    leavers: tuple[str, ...]

    @property
    def all_ids(self) -> tuple[str, ...]:
        return self.survivors + self.entrants + self.leavers

    def __post_init__(self) -> None:
        seen = self.all_ids
        if len(set(seen)) != len(seen):
            raise ValueError("partition subsets must be disjoint")

    def summary(self) -> str:
        return (
            f"{len(self.survivors)} survivors, "
            f"{len(self.entrants)} entrants, "
            f"{len(self.leavers)} leavers"
        )


def classify(
    contributions_t0: Mapping[str, float],
    contributions_t1: Mapping[str, float],
    *,
    tol: float = 0.0,
) -> Partition:
    """
    Classify instruments by presence in each period.

    Args:
        contributions_t0: instrument id -> contribution (or weight) at t0.
        contributions_t1: instrument id -> contribution (or weight) at t1.
            Ids may appear in one mapping, the other, or both.
        tol: magnitudes at or below this are treated as absent. Use a small
            positive value to stop dust positions from entering a log ratio.

    Returns:
        A `Partition`. Ordering follows first appearance in t0 then t1, so
        results are stable across runs.
    """
    if tol < 0.0:
        raise ValueError("tol must be non-negative")

    ordered: list[str] = list(contributions_t0)
    ordered += [k for k in contributions_t1 if k not in contributions_t0]

    survivors: list[str] = []
    entrants: list[str] = []
    leavers: list[str] = []

    for key in ordered:
        held_t0 = abs(contributions_t0.get(key, 0.0)) > tol
        held_t1 = abs(contributions_t1.get(key, 0.0)) > tol
        if held_t0 and held_t1:
            survivors.append(key)
        elif held_t1:
            entrants.append(key)
        elif held_t0:
            leavers.append(key)
        # Absent in both periods: not part of the analysis at all.

    return Partition(
        survivors=tuple(survivors),
        entrants=tuple(entrants),
        leavers=tuple(leavers),
    )
