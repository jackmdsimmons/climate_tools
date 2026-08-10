"""
Core LMDI (logarithmic mean Divisia index) decomposition engine.

A portfolio-level climate metric is written as a sum over instruments of a
*product* of drivers:

    M_p,t = sum_j M_j,t = sum_j prod_n D_n,j,t

LMDI attributes the change in M_p to each driver additively and exactly -- no
interaction term, no residual, and no sensitivity to the order in which drivers
are listed:

    dM_p = E_D1 + E_D2 + ... + E_DN

    E_Dn = sum_j L(M_j,t1, M_j,t0) * ln( D_n,j,t1 / D_n,j,t0 )

This module is deliberately metric-agnostic: it knows nothing about emissions,
weights or sectors. Callers supply named drivers whose product reconstructs each
instrument's contribution, which is what makes "a flexible number of multiples"
(WACI = sum of w * E / R, footprint, absolute emissions, scope splits) all the
same computation.

Two properties worth knowing when composing driver sets:

  * Because effects are sums of logs, splitting one driver into two multiplicative
    factors splits its effect additively -- and merging two drivers into their
    product merges their effects. Grouping drivers for reporting is therefore
    free and exact (see `test_lmdi.py::test_merging_drivers_sums_their_effects`).
  * The decomposition is exact only where every driver is strictly positive in
    both periods. Zero or negative values are not a numerical nuisance to be
    clamped; they mean the instrument does not belong in this driver chain. The
    engine raises `NonPositiveDriver` and directs you to partition instead.

References:
  Ang, B.W. (2015). LMDI Decomposition Approach: A Guide for Implementation.
    Energy Policy 86: 233-238.
  Bouchet, V. (2025). Attribution Analysis of Equity Portfolio Emissions:
    Examining and Integrating Existing Frameworks. Scientific Portfolio / EDHEC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .logmean import log_mean_array

# Additivity is exact in real arithmetic; this is the floating-point allowance.
RESIDUAL_TOL = 1e-9


class NonPositiveDriver(ValueError):
    """
    Raised when a multi-driver decomposition is asked to take the log of a
    non-positive value -- typically an instrument entering, leaving, or carrying
    a zero/missing denominator.
    """


@dataclass(frozen=True)
class Decomposition:
    """Result of decomposing one homogeneous group of instruments."""

    effects: dict[str, float]
    total_t0: float
    total_t1: float
    labels: tuple[str, ...]

    @property
    def delta(self) -> float:
        """Actual change in the metric over the period."""
        return self.total_t1 - self.total_t0

    @property
    def explained(self) -> float:
        """Sum of the attributed driver effects."""
        return sum(self.effects.values())

    @property
    def residual(self) -> float:
        """Unexplained change. Exactly zero for a valid LMDI decomposition."""
        return self.delta - self.explained

    def check_additivity(self, tol: float = RESIDUAL_TOL) -> None:
        """Raise if the effects fail to reconstruct the change in the metric."""
        scale = max(1.0, abs(self.delta), abs(self.total_t0))
        if abs(self.residual) > tol * scale:
            raise AssertionError(
                f"LMDI additivity violated: delta={self.delta!r}, "
                f"explained={self.explained!r}, residual={self.residual!r}"
            )


def lmdi_decompose(
    drivers: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    *,
    labels: Sequence[str] | None = None,
) -> Decomposition:
    """
    Decompose the change in sum_j prod_n D_n,j into one effect per driver.

    Args:
        drivers: ordered mapping of driver name -> (values at t0, values at t1).
            Each pair of sequences is indexed by instrument, all the same length.
            The product of the drivers must equal the instrument's contribution
            to the portfolio metric.
        labels: optional instrument identifiers, used in error messages.

    Returns:
        A `Decomposition` whose effects sum exactly to the change in the metric.

    Raises:
        NonPositiveDriver: if two or more drivers are supplied and any driver
            value is not strictly positive. Give those instruments their own
            single-driver block via `climate_attribution.partition` instead.
    """
    if not drivers:
        raise ValueError("at least one driver is required")

    names = list(drivers)
    t0: dict[str, np.ndarray] = {}
    t1: dict[str, np.ndarray] = {}
    n_instruments: int | None = None

    for name in names:
        start, end = drivers[name]
        a = np.asarray(start, dtype=float)
        b = np.asarray(end, dtype=float)
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError(f"driver {name!r}: expected 1-D sequences")
        if a.shape != b.shape:
            raise ValueError(
                f"driver {name!r}: t0 has {a.size} values but t1 has {b.size}"
            )
        if n_instruments is None:
            n_instruments = a.size
        elif a.size != n_instruments:
            raise ValueError(
                f"driver {name!r} covers {a.size} instruments, "
                f"but {names[0]!r} covers {n_instruments}"
            )
        t0[name], t1[name] = a, b

    assert n_instruments is not None
    if n_instruments == 0:
        return Decomposition(
            effects={name: 0.0 for name in names},
            total_t0=0.0,
            total_t1=0.0,
            labels=(),
        )

    if labels is None:
        ids = tuple(f"#{i}" for i in range(n_instruments))
    else:
        ids = tuple(str(x) for x in labels)
        if len(ids) != n_instruments:
            raise ValueError(
                f"got {len(ids)} labels for {n_instruments} instruments"
            )

    contribution_t0 = np.ones(n_instruments, dtype=float)
    contribution_t1 = np.ones(n_instruments, dtype=float)
    for name in names:
        contribution_t0 *= t0[name]
        contribution_t1 *= t1[name]

    # A single driver IS the contribution, so its effect is the change itself.
    # No logarithm is taken, which is exactly why entrants and leavers are
    # representable -- this is the escape hatch the partition layer relies on.
    if len(names) == 1:
        only = names[0]
        return Decomposition(
            effects={only: float(np.sum(contribution_t1 - contribution_t0))},
            total_t0=float(contribution_t0.sum()),
            total_t1=float(contribution_t1.sum()),
            labels=ids,
        )

    _reject_non_positive(names, t0, t1, ids)

    weight = log_mean_array(contribution_t1, contribution_t0)
    effects = {
        name: float(np.sum(weight * np.log(t1[name] / t0[name]))) for name in names
    }

    result = Decomposition(
        effects=effects,
        total_t0=float(contribution_t0.sum()),
        total_t1=float(contribution_t1.sum()),
        labels=ids,
    )
    result.check_additivity()
    return result


# ── Internals ────────────────────────────────────────────────────────────────


def _reject_non_positive(
    names: list[str],
    t0: Mapping[str, np.ndarray],
    t1: Mapping[str, np.ndarray],
    labels: tuple[str, ...],
) -> None:
    """Fail loudly, and usefully, on values LMDI's log ratio cannot accept."""
    problems: list[str] = []
    for name in names:
        for period, values in (("t0", t0[name]), ("t1", t1[name])):
            bad = np.flatnonzero(~(values > 0.0))
            if bad.size:
                shown = ", ".join(
                    f"{labels[i]}={values[i]:g}" for i in bad[:5]
                )
                if bad.size > 5:
                    shown += f", ... ({bad.size} total)"
                problems.append(f"  driver {name!r} at {period}: {shown}")

    if not problems:
        return

    raise NonPositiveDriver(
        "LMDI takes the log of each driver's ratio, which requires strictly "
        "positive values in both periods. Offending values:\n"
        + "\n".join(problems)
        + "\n\nA zero usually means the instrument entered or left the portfolio, "
        "or its denominator (revenue, GDP, EVIC) is missing. Do not clamp or drop "
        "it -- that is what silently breaks additivity. Instead partition the "
        "universe first and give these instruments their own single-driver block:\n"
        "    from climate_attribution.partition import classify, Block, decompose_blocks\n"
        "A single-driver block measures their effect exactly, as the change in "
        "their contribution."
    )
